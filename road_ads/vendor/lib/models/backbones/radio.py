import copy
import json
import logging
import math
import os
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from timm.models import clean_state_dict

if __package__ in (None, ""):
    import sys

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from lib.models.backbones.lora_dino_v3 import LoRAAdapter, LoRAQKV
    from lib.models.backbones.peft import get_pyramid_feature, set_requires_grad, set_train
    from lib.models.backbones.reins import LoRAReins
else:
    from .lora_dino_v3 import LoRAAdapter, LoRAQKV
    from .peft import get_pyramid_feature, set_requires_grad, set_train
    from .reins import LoRAReins


work_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

_LOCAL_RADIO_CHECKPOINTS = {
    "c-radio_v4-h": "c-radio_v4-h_half.pth.tar",
    "c-radio_v4-so400m": "c-radio_v4-so400m_half.pth.tar",
}


def _default_out_indices(num_layers: int) -> List[int]:
    if num_layers <= 0:
        return []
    if num_layers == 24:
        return [7, 11, 15, 23]
    if num_layers == 32:
        return [7, 15, 23, 31]
    if num_layers <= 4:
        return list(range(num_layers))
    return [
        max(0, min(num_layers - 1, math.ceil(num_layers * ratio) - 1))
        for ratio in (0.25, 0.5, 0.75, 1.0)
    ]


def _load_radio_hubconf():
    if __package__ in (None, ""):
        from lib.models.backbones.radio_modules import hubconf
    else:
        from .radio_modules import hubconf

    return hubconf


def _import_radio_module(module_name: str, attr_name: str):
    if __package__ in (None, ""):
        module = __import__(f"lib.models.backbones.radio_modules.{module_name}", fromlist=[attr_name])
    else:
        module = __import__(f"{__package__}.radio_modules.{module_name}", fromlist=[attr_name])
    return getattr(module, attr_name)


def _dtype_from_string(name: Optional[str]):
    if name is None:
        return None
    if name.startswith("torch."):
        name = name[len("torch.") :]
    return getattr(torch, name)


def _torch_load_weights(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_existing_path(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return path

    candidates = [
        os.path.join(work_root, path),
        os.path.join(work_root, path.lstrip("./")),
        os.path.join(work_root, "train", path),
    ]
    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            return candidate
    return path


def _resolve_radio_version(version: Optional[str], pretrained=None) -> str:
    if isinstance(pretrained, dict):
        version = (
            pretrained.get("radio")
            or pretrained.get("checkpoint")
            or pretrained.get("backbone")
            or pretrained.get("model")
            or version
        )
    elif isinstance(pretrained, str):
        version = pretrained

    version = version or "c-radio_v4-so400m"
    if version in _LOCAL_RADIO_CHECKPOINTS:
        local_path = os.path.join(work_root, "pretrained", "radio", _LOCAL_RADIO_CHECKPOINTS[version])
        if os.path.exists(local_path):
            return local_path
    return _resolve_existing_path(version)


def _resolve_converted_paths(radio_config: Dict, pretrained=None):
    config_path = radio_config.get("converted_config", None)
    weights_path = radio_config.get("converted_weights", None)

    if isinstance(pretrained, dict):
        config_path = pretrained.get("radio_config", config_path)
        weights_path = pretrained.get("radio_weights", weights_path)

    if config_path is not None:
        config_path = _resolve_existing_path(config_path)
    if weights_path is not None:
        weights_path = _resolve_existing_path(weights_path)

    if config_path is None and weights_path is None:
        version = radio_config.get("version", None)
        if isinstance(pretrained, dict):
            version = (
                pretrained.get("radio")
                or pretrained.get("checkpoint")
                or pretrained.get("backbone")
                or pretrained.get("model")
                or version
            )
        elif isinstance(pretrained, str):
            version = pretrained

        if version is not None:
            source_path = _resolve_radio_version(version)
            for suffix in (".pth.tar", ".tar", ".pth"):
                if source_path.endswith(suffix):
                    prefix = source_path[: -len(suffix)]
                    candidate_config = prefix + "_config.json"
                    candidate_weights = prefix + "_weights.pth"
                    if os.path.exists(candidate_config) and os.path.exists(candidate_weights):
                        return candidate_config, candidate_weights
                    break

    return config_path, weights_path


def _load_converted_radio_model(
    config_path: str,
    weights_path: str,
    adaptor_names=None,
    vitdet_window_size: Optional[int] = None,
):
    adaptor_registry = _import_radio_module("radio.adaptor_registry", "adaptor_registry")
    RadioResource = _import_radio_module("radio.common", "RadioResource")
    Resolution = _import_radio_module("radio.common", "Resolution")
    configure_damp_from_args = _import_radio_module("radio.enable_damp", "configure_damp_from_args")
    configure_spectral_reparam_from_args = _import_radio_module(
        "radio.enable_spectral_reparam", "configure_spectral_reparam_from_args"
    )
    disable_spectral_reparam = _import_radio_module("radio.enable_spectral_reparam", "disable_spectral_reparam")
    FeatureNormalizer = _import_radio_module("radio.feature_normalizer", "FeatureNormalizer")
    IntermediateFeatureNormalizer = _import_radio_module("radio.feature_normalizer", "IntermediateFeatureNormalizer")
    get_default_conditioner = _import_radio_module("radio.input_conditioner", "get_default_conditioner")
    RADIOModel = _import_radio_module("radio.radio_model", "RADIOModel")
    create_model_from_args = _import_radio_module("radio.radio_model", "create_model_from_args")
    VitDetArgs = _import_radio_module("radio.vitdet", "VitDetArgs")
    apply_vitdet_arch = _import_radio_module("radio.vitdet", "apply_vitdet_arch")

    hubconf = _load_radio_hubconf()

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    weights = _torch_load_weights(weights_path)
    state_dict = weights["state_dict"] if isinstance(weights, dict) and "state_dict" in weights else weights

    args_dict = copy.deepcopy(config["args"])
    args_dict["dtype"] = _dtype_from_string(args_dict.get("dtype"))
    args = Namespace(**args_dict)

    resource_dict = config.get("resource", {})
    preferred_resolution = resource_dict.get("preferred_resolution", None)
    if preferred_resolution is not None:
        preferred_resolution = Resolution(*preferred_resolution)
    resource = RadioResource(
        url=config.get("source", weights_path),
        patch_size=resource_dict.get("patch_size", None),
        max_resolution=resource_dict.get("max_resolution", None),
        preferred_resolution=preferred_resolution,
        supports_vitdet=resource_dict.get("supports_vitdet", True),
        vitdet_num_windowed=resource_dict.get("vitdet_num_windowed", None),
        vitdet_num_global=resource_dict.get("vitdet_num_global", None),
    )

    mod = create_model_from_args(args)
    mod_state_dict = hubconf.get_prefix_state_dict(state_dict, "base_model.")

    if getattr(args, "spectral_reparam", False):
        configure_spectral_reparam_from_args(mod, args, state_dict_guidance=mod_state_dict)
    if getattr(args, "damp", None):
        configure_damp_from_args(mod, args)

    state_dict = clean_state_dict(state_dict)
    key_warn = mod.load_state_dict(mod_state_dict, strict=False)
    if key_warn.missing_keys:
        logging.getLogger().warning(f"Missing keys in RADIO converted state dict: {key_warn.missing_keys}")
    if key_warn.unexpected_keys:
        logging.getLogger().warning(f"Unexpected keys in RADIO converted state dict: {key_warn.unexpected_keys}")

    if getattr(args, "spectral_reparam", False):
        disable_spectral_reparam(mod)
        args.spectral_reparam = False

    conditioner = get_default_conditioner()
    conditioner.load_state_dict(hubconf.get_prefix_state_dict(state_dict, "input_conditioner."))

    dtype = getattr(args, "dtype", torch.float32)
    mod.to(dtype=dtype)
    conditioner.dtype = dtype

    cls_token_per_teacher = getattr(args, "cls_token_per_teacher", True)
    if cls_token_per_teacher:
        name_to_idx_map = {}
        for idx, teacher in enumerate(args.teachers):
            if teacher.get("use_summary", True):
                name = teacher["name"]
                if name not in name_to_idx_map:
                    name_to_idx_map[name] = idx
        summary_idxs = torch.tensor(sorted(name_to_idx_map.values()), dtype=torch.int64)
    else:
        summary_idxs = torch.tensor([0], dtype=torch.int64)

    if adaptor_names is None:
        adaptor_names = []
    elif isinstance(adaptor_names, str):
        adaptor_names = [adaptor_names]

    adaptors = {}
    for adaptor_name in adaptor_names:
        for teacher_idx, teacher_conf in enumerate(args.teachers):
            if teacher_conf["name"] == adaptor_name:
                break
        else:
            raise ValueError(f'Unable to find the specified adaptor name. Known names: {list(t["name"] for t in args.teachers)}')

        adaptor_state = {}
        for key, value in state_dict.items():
            prefix_idx_head = f"_heads.{teacher_idx}"
            prefix_name_head = f"_heads.{adaptor_name}"
            prefix_idx_feat = f"_feature_projections.{teacher_idx}"
            prefix_name_feat = f"_feature_projections.{adaptor_name}"
            if key.startswith(prefix_idx_head):
                adaptor_state["summary" + key[len(prefix_idx_head) :]] = value
            elif key.startswith(prefix_name_head):
                adaptor_state["summary" + key[len(prefix_name_head) :]] = value
            elif key.startswith(prefix_idx_feat):
                adaptor_state["feature" + key[len(prefix_idx_feat) :]] = value
            elif key.startswith(prefix_name_feat):
                adaptor_state["feature" + key[len(prefix_name_feat) :]] = value

        adaptor = adaptor_registry.create_adaptor(teacher_conf["type"], args, teacher_conf, adaptor_state)
        adaptor.head_idx = teacher_conf.get("token_slot", teacher_idx) if cls_token_per_teacher else 0
        adaptors[adaptor_name] = adaptor

    feat_norm_sd = hubconf.get_prefix_state_dict(state_dict, "_feature_normalizer.")
    feature_normalizer = None
    if feat_norm_sd:
        feature_normalizer = FeatureNormalizer(feat_norm_sd["mean"].shape[0], dtype=dtype)
        feature_normalizer.load_state_dict(feat_norm_sd)

    inter_feat_norm_sd = hubconf.get_prefix_state_dict(state_dict, "_intermediate_feature_normalizer.")
    inter_feature_normalizer = None
    if inter_feat_norm_sd:
        inter_feature_normalizer = IntermediateFeatureNormalizer(
            *inter_feat_norm_sd["means"].shape[:2],
            rot_per_layer=inter_feat_norm_sd["rotation"].ndim == 3,
            dtype=dtype,
        )
        inter_feature_normalizer.load_state_dict(inter_feat_norm_sd)

    radio = RADIOModel(
        mod,
        conditioner,
        summary_idxs=summary_idxs,
        patch_size=resource.patch_size,
        max_resolution=resource.max_resolution,
        window_size=vitdet_window_size,
        preferred_resolution=resource.preferred_resolution,
        adaptors=adaptors,
        feature_normalizer=feature_normalizer,
        inter_feature_normalizer=inter_feature_normalizer,
    )

    if vitdet_window_size is not None:
        apply_vitdet_arch(
            mod,
            VitDetArgs(
                vitdet_window_size,
                radio.num_summary_tokens,
                num_windowed=resource.vitdet_num_windowed,
                num_global=resource.vitdet_num_global,
            ),
        )

    return radio


def _load_checkpoint_state(pretrained):
    checkpoint = torch.load(pretrained, map_location="cpu")
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if "model" in checkpoint:
        return checkpoint["model"]
    if "adapter" in checkpoint:
        return checkpoint["adapter"]
    if "reins" in checkpoint:
        return checkpoint["reins"]
    if "lora" in checkpoint:
        return checkpoint["lora"]
    return checkpoint


def _strip_adapter_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = (
        "adapter.",
        "backbone.adapter.",
        "module.backbone.adapter.",
        "model.adapter.",
        "reins.",
        "backbone.reins.",
        "model.reins.",
        "lora.",
        "backbone.lora.",
        "model.lora.",
    )
    adapter_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
                break
        adapter_state_dict[new_key] = value
    return adapter_state_dict


def _module_param_dtype(module: nn.Module) -> torch.dtype:
    for param in module.parameters():
        return param.dtype
    return torch.float32


class RADIOBackbone(nn.Module):
    def __init__(self, backbone_config=None, pretrained=None):
        super().__init__()
        backbone_config = backbone_config or {}
        radio_config = copy.deepcopy(backbone_config.get("radio_config", {}))

        self.logger = logging.getLogger()
        self.backbone_config = backbone_config
        self.radio_config = radio_config
        self.save_whole_backbone = True

        converted_config, converted_weights = _resolve_converted_paths(radio_config, pretrained)
        if converted_config is not None or converted_weights is not None:
            if converted_config is None or converted_weights is None:
                raise ValueError("RADIO converted loading requires both converted_config and converted_weights.")
            version = converted_weights
            self.radio = _load_converted_radio_model(
                converted_config,
                converted_weights,
                adaptor_names=radio_config.get("adaptor_names", None),
                vitdet_window_size=radio_config.get("vitdet_window_size", None),
            )
        else:
            version = _resolve_radio_version(radio_config.get("version"), pretrained)
            hubconf = _load_radio_hubconf()
            self.radio = hubconf.radio_model(
                version=version,
                progress=radio_config.get("progress", True),
                adaptor_names=radio_config.get("adaptor_names", None),
                vitdet_window_size=radio_config.get("vitdet_window_size", None),
            )

        self.pyramid_scales = radio_config.get("pyramid_scales", [4, 2, 1, 0.5])
        self.intermediate_norm = radio_config.get("intermediate_norm", True)
        self.intermediate_aggregation = radio_config.get("intermediate_aggregation", "sparse")
        self.norm_alpha_scheme = radio_config.get("norm_alpha_scheme", "post-alpha")
        self.output_float = radio_config.get("output_float", True)
        self.input_mode = radio_config.get("input_mode", "imagenet_norm")
        self.clip_input = radio_config.get("clip_input", True)

        input_mean = radio_config.get("input_mean", (123.675, 116.28, 103.53))
        input_std = radio_config.get("input_std", (58.395, 57.12, 57.375))
        self.register_buffer("input_mean", torch.tensor(input_mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor(input_std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        self.patch_size = self.radio.patch_size
        self.embed_dim = self.radio.embed_dim
        self.num_features = self.embed_dim
        self.blocks = self._get_blocks()
        self.num_layers = len(self.blocks)
        self.num_summary_tokens = int(getattr(self.radio, "num_summary_tokens", 1) or 0)
        self.out_indices = radio_config.get("out_indices")
        if self.out_indices is None:
            self.out_indices = _default_out_indices(self.num_layers)

        self.logger.info(f"Loaded RADIO backbone from {version}.")
        self.logger.info(
            f"RADIO config: embed_dim={self.embed_dim}, patch_size={self.patch_size}, "
            f"num_layers={self.num_layers}, out_indices={self.out_indices}"
        )

    def _get_blocks(self) -> List[nn.Module]:
        blocks = getattr(self.radio, "blocks", None)
        if blocks is None:
            blocks = getattr(getattr(self.radio, "model", None), "blocks", None)
        if blocks is None:
            return []
        return list(blocks)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode in ("imagenet_norm", "normalized", "norm"):
            x = x * self.input_std.to(dtype=x.dtype, device=x.device) + self.input_mean.to(dtype=x.dtype, device=x.device)
            x = x / 255.0
        elif self.input_mode in ("0_255", "255", "uint8"):
            x = x / 255.0
        elif self.input_mode in ("0_1", "01", "radio"):
            pass
        else:
            raise ValueError(
                f"Unsupported RADIO input_mode={self.input_mode}. "
                "Use one of: imagenet_norm, 0_255, 0_1."
            )

        if self.clip_input:
            x = x.clamp(0.0, 1.0)
        return x

    def _cast_feature(self, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x is not None and self.output_float:
            return x.float()
        return x

    def _forward_radio(self, x: torch.Tensor):
        if hasattr(self.radio, "forward_intermediates"):
            final, outs = self.radio.forward_intermediates(
                x,
                indices=self.out_indices,
                norm=self.intermediate_norm,
                stop_early=False,
                output_fmt="NCHW",
                intermediates_only=False,
                aggregation=self.intermediate_aggregation,
                norm_alpha_scheme=self.norm_alpha_scheme,
            )
            summary = final.summary
        else:
            summary, features = self.radio(x, feature_fmt="NCHW")
            num_outs = len(self.pyramid_scales)
            outs = [features for _ in range(num_outs)]
        outs = [self._cast_feature(out).contiguous() for out in outs]
        summary = self._cast_feature(summary)
        return outs, summary

    def forward(self, x, masks=None):
        del masks
        radio_input = self._prepare_input(x)
        outs, summary = self._forward_radio(radio_input)
        return get_pyramid_feature(outs, self.pyramid_scales), outs, summary

    def freeze_params(self):
        for _, param in self.named_parameters():
            param.requires_grad = False
        self.logger.info("Freeze all params in RADIO.")


class PureRADIO(RADIOBackbone):
    def __init__(self, backbone_config=None, pretrained=None):
        super().__init__(backbone_config=backbone_config, pretrained=pretrained)

        self.freeze_grad = (backbone_config or {}).get("freeze_grad", False)
        if self.freeze_grad:
            self.freeze_params()
            self.save_whole_backbone = False

        self.train(True)

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_grad", False):
            self.radio.eval()
            for _, param in self.named_parameters():
                param.requires_grad = False
        return self


class RADIO(PureRADIO):
    pass


class FrozenRADIO(PureRADIO):
    def __init__(self, backbone_config=None, pretrained=None):
        backbone_config = copy.deepcopy(backbone_config or {})
        backbone_config["freeze_grad"] = True
        super().__init__(backbone_config=backbone_config, pretrained=pretrained)


class ReinsRADIO(RADIOBackbone):
    def __init__(self, backbone_config=None, pretrained=None):
        super().__init__(backbone_config=backbone_config, pretrained=pretrained)

        backbone_config = backbone_config or {}
        self.enable_adapter = False
        self._reins_hook_handles = []
        reins_config = copy.deepcopy(backbone_config.get("reins_config", None))
        if reins_config is not None:
            self.enable_adapter = True
            self.save_whole_backbone = False
            reins_config.setdefault("embed_dims", self.embed_dim)
            reins_config.setdefault("num_layers", self.num_layers)
            reins_config.setdefault("patch_size", self.patch_size or 16)
            self.adapter = LoRAReins(**reins_config)
            self._register_reins_hooks()

        if isinstance(pretrained, dict) and pretrained.get("adapter") and self.enable_adapter:
            self.load_adapter_pretrained(pretrained["adapter"])

        self.train(True)

    def _register_reins_hooks(self):
        if not self.blocks:
            raise ValueError("RADIO model does not expose transformer blocks; ReinsRADIO cannot attach adapter hooks.")

        non_adapter_layers = getattr(self.adapter, "non_adapter_layers", 0)
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx < non_adapter_layers:
                continue
            handle = block.register_forward_hook(self._make_reins_hook(layer_idx))
            self._reins_hook_handles.append(handle)

    def _make_reins_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output):
                return output
            output_dtype = output.dtype
            adapter_dtype = _module_param_dtype(self.adapter)
            output_for_adapter = output.to(adapter_dtype)
            return self.adapter(
                output_for_adapter,
                layer_idx,
                batch_first=True,
                has_cls_token=self.num_summary_tokens > 0,
                num_register_token=max(self.num_summary_tokens - 1, 0),
            ).to(output_dtype)

        return hook

    def load_adapter_pretrained(self, pretrained, strict=True):
        state_dict = _strip_adapter_prefixes(_load_checkpoint_state(pretrained))
        missing_keys, unexpected_keys = self.adapter.load_state_dict(state_dict, strict=strict)
        self.logger.info(f"Loaded RADIO Reins adapter checkpoint from {pretrained}")
        if missing_keys:
            self.logger.warning(f"Missing {len(missing_keys)} adapter keys: {missing_keys}")
        if unexpected_keys:
            self.logger.warning(f"Unexpected {len(unexpected_keys)} adapter keys: {unexpected_keys}")

    def save_adapter(self, path):
        if self.enable_adapter:
            torch.save({"adapter": self.adapter.state_dict()}, path)
            self.logger.info(f"[ReinsRADIO] Adapter saved to {path}")
        else:
            self.logger.warning("[ReinsRADIO] No adapter to save")

    def get_adapter_state_dict(self):
        return self.adapter.state_dict() if self.enable_adapter else {}

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        super().train(mode)
        if self.enable_adapter:
            set_requires_grad(self, ["adapter"])
            set_train(self, ["adapter"])
        return self


class LoRARADIO(RADIOBackbone):
    def __init__(self, backbone_config=None, pretrained=None):
        super().__init__(backbone_config=backbone_config, pretrained=pretrained)

        backbone_config = backbone_config or {}
        self.enable_adapter = False
        lora_config = copy.deepcopy(backbone_config.get("lora_config", None))
        if lora_config is not None:
            self.enable_adapter = True
            self.save_whole_backbone = False
            lora_config.setdefault("embed_dim", self.embed_dim)
            lora_config.setdefault("num_layers", self.num_layers)
            self.adapter = LoRAAdapter(**lora_config)

        if isinstance(pretrained, dict) and pretrained.get("adapter") and self.enable_adapter:
            self.load_adapter_pretrained(pretrained["adapter"])

        if self.enable_adapter:
            self._replace_attention_layers()

        self.train(True)

    def _replace_attention_layers(self):
        replaced = 0
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx < self.adapter.non_adapter_layers:
                continue
            attn = getattr(block, "attn", None)
            original_qkv = getattr(attn, "qkv", None)
            if original_qkv is None:
                continue
            lora_qkv = LoRAQKV(
                original_qkv=original_qkv,
                layer_idx=layer_idx,
                embed_dim=original_qkv.in_features,
                non_adapter_layers=self.adapter.non_adapter_layers,
            )
            lora_qkv.forward = _partial_lora_qkv_forward(lora_qkv, self._forward_lora_adapter)
            attn.qkv = lora_qkv
            replaced += 1

        if replaced == 0:
            raise ValueError("No RADIO attention qkv layers were replaced; LoRARADIO cannot enable LoRA.")
        self.logger.info(f"[LoRARADIO] Replaced {replaced} RADIO attention qkv layers with LoRA wrappers.")

    def _forward_lora_adapter(self, x, layer_idx):
        input_dtype = x.dtype
        adapter_dtype = _module_param_dtype(self.adapter)
        deltas = self.adapter.forward(x.to(adapter_dtype), layer_idx)
        return tuple(delta.to(input_dtype) if delta is not None else None for delta in deltas)

    def load_adapter_pretrained(self, pretrained, strict=True):
        state_dict = _strip_adapter_prefixes(_load_checkpoint_state(pretrained))
        missing_keys, unexpected_keys = self.adapter.load_state_dict(state_dict, strict=strict)
        self.logger.info(f"Loaded RADIO LoRA adapter checkpoint from {pretrained}")
        if missing_keys:
            self.logger.warning(f"Missing {len(missing_keys)} adapter keys: {missing_keys}")
        if unexpected_keys:
            self.logger.warning(f"Unexpected {len(unexpected_keys)} adapter keys: {unexpected_keys}")

    def save_adapter(self, path):
        if self.enable_adapter:
            torch.save({"adapter": self.adapter.state_dict()}, path)
            self.logger.info(f"[LoRARADIO] Adapter saved to {path}")
        else:
            self.logger.warning("[LoRARADIO] No adapter to save")

    def get_adapter_state_dict(self):
        return self.adapter.state_dict() if self.enable_adapter else {}

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            return self

        for param in self.parameters():
            param.requires_grad = False
        if self.enable_adapter:
            for param in self.adapter.parameters():
                param.requires_grad = True
            self.adapter.train(True)
        return self


def _partial_lora_qkv_forward(lora_qkv: LoRAQKV, adapter_func):
    def forward(x):
        return LoRAQKV.forward(lora_qkv, x, adapter_func=adapter_func)

    return forward


def get_std_pure_radio_h(freeze_grad=True):
    return PureRADIO(
        backbone_config={
            "freeze_grad": freeze_grad,
            "radio_config": {
                "version": "c-radio_v4-h",
                "out_indices": [7, 15, 23, 31],
            },
        }
    )


def get_std_reins_radio_h():
    return ReinsRADIO(
        backbone_config={
            "radio_config": {
                "version": "c-radio_v4-h",
                "out_indices": [7, 15, 23, 31],
            },
            "reins_config": {
                "lora_dim": 16,
                "non_adapter_layers": 0,
                "token_length": 100,
                "link_token_to_query": True,
            },
        }
    )


def get_std_lora_radio_h():
    return LoRARADIO(
        backbone_config={
            "radio_config": {
                "version": "c-radio_v4-h",
                "out_indices": [7, 15, 23, 31],
            },
            "lora_config": {
                "lora_rank": 16,
                "non_adapter_layers": 0,
                "lora_alpha": 16.0,
                "lora_dropout": 0.0,
                "target_modules": ["q", "v"],
            },
        }
    )


def _count_parameters(model: nn.Module):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def _shape_text(tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return "None"
    return f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"


def _assert_finite(name: str, tensors):
    if torch.is_tensor(tensors):
        tensors = [tensors]
    for idx, tensor in enumerate(tensors):
        if tensor is None:
            continue
        if not torch.isfinite(tensor).all().item():
            raise RuntimeError(f"{name}[{idx}] contains NaN or Inf.")


def _make_smoke_input(batch_size: int, height: int, width: int, input_mode: str, device: torch.device) -> torch.Tensor:
    image_255 = torch.rand(batch_size, 3, height, width, device=device) * 255.0
    if input_mode in ("imagenet_norm", "normalized", "norm"):
        mean = torch.tensor((123.675, 116.28, 103.53), dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor((58.395, 57.12, 57.375), dtype=torch.float32, device=device).view(1, 3, 1, 1)
        return (image_255 - mean) / std
    if input_mode in ("0_255", "255", "uint8"):
        return image_255
    if input_mode in ("0_1", "01", "radio"):
        return image_255 / 255.0
    raise ValueError(f"Unsupported input_mode={input_mode}")


def _resolve_smoke_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device)


def _set_radio_runtime_dtype(model: nn.Module, dtype: torch.dtype):
    radio = getattr(model, "radio", None)
    input_conditioner = getattr(radio, "input_conditioner", None)
    if input_conditioner is not None and hasattr(input_conditioner, "dtype"):
        input_conditioner.dtype = dtype


def _build_smoke_model(args):
    radio_config = {
        "version": args.version,
        "input_mode": args.input_mode,
        "pyramid_scales": args.pyramid_scales,
    }
    if args.out_indices is not None:
        radio_config["out_indices"] = args.out_indices
    if args.converted_config:
        radio_config["converted_config"] = args.converted_config
    if args.converted_weights:
        radio_config["converted_weights"] = args.converted_weights
    if args.vitdet_window_size is not None:
        radio_config["vitdet_window_size"] = args.vitdet_window_size
    if args.adaptor_names:
        radio_config["adaptor_names"] = args.adaptor_names

    backbone_config = {"radio_config": radio_config}
    pretrained = {}
    if args.pretrained:
        pretrained["radio"] = args.pretrained
    if args.converted_config:
        pretrained["radio_config"] = args.converted_config
    if args.converted_weights:
        pretrained["radio_weights"] = args.converted_weights
    if not pretrained:
        pretrained = None

    if args.variant == "freeze":
        model_cls = FrozenRADIO
    elif args.variant == "pure":
        model_cls = PureRADIO
        backbone_config["freeze_grad"] = args.freeze_grad
    elif args.variant == "radio":
        model_cls = RADIO
        backbone_config["freeze_grad"] = args.freeze_grad
    elif args.variant == "reins":
        model_cls = ReinsRADIO
        backbone_config["reins_config"] = {
            "lora_dim": args.reins_lora_dim,
            "non_adapter_layers": args.non_adapter_layers,
            "token_length": args.reins_token_length,
            "link_token_to_query": args.link_token_to_query,
        }
    elif args.variant == "lora":
        model_cls = LoRARADIO
        backbone_config["lora_config"] = {
            "lora_rank": args.lora_rank,
            "non_adapter_layers": args.non_adapter_layers,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.lora_targets,
        }
    else:
        raise ValueError(f"Unsupported variant={args.variant}")

    return model_cls(backbone_config=backbone_config, pretrained=pretrained)


def _parse_smoke_args():
    parser = ArgumentParser(description="Smoke test RADIO backbone loading and one forward pass.")
    parser.add_argument("--variant", default="freeze", choices=["radio", "pure", "freeze", "reins", "lora"])
    parser.add_argument("--version", default="c-radio_v4-so400m", help="RADIO alias or checkpoint path.")
    parser.add_argument("--pretrained", default=None, help="Explicit RADIO .pth/.pth.tar checkpoint path.")
    parser.add_argument("--converted-config", default=None, help="Converted RADIO JSON config path.")
    parser.add_argument("--converted-weights", default=None, help="Converted RADIO tensor weights path.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    parser.add_argument("--out-indices", type=int, nargs="+", default=None)
    parser.add_argument("--pyramid-scales", type=float, nargs="+", default=[4, 2, 1, 0.5])
    parser.add_argument("--input-mode", default="imagenet_norm", choices=["imagenet_norm", "0_255", "0_1"])
    parser.add_argument("--model-dtype", default="keep", choices=["keep", "float32", "float16", "bfloat16"])
    parser.add_argument("--eval", action="store_true", help="Run the smoke forward in eval mode.")
    parser.add_argument("--no-forward", action="store_true", help="Only instantiate the model.")
    parser.add_argument("--freeze-grad", action="store_true", help="Freeze params for pure/radio variants.")
    parser.add_argument("--vitdet-window-size", type=int, default=None)
    parser.add_argument("--adaptor-names", nargs="*", default=None)
    parser.add_argument("--non-adapter-layers", type=int, default=0)
    parser.add_argument("--reins-lora-dim", type=int, default=4)
    parser.add_argument("--reins-token-length", type=int, default=8)
    parser.add_argument("--link-token-to-query", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=4.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", nargs="+", default=["q", "v"])
    return parser.parse_args()


def _main():
    args = _parse_smoke_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    torch.manual_seed(0)

    device = _resolve_smoke_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model = _build_smoke_model(args)
    model_dtype = None if args.model_dtype == "keep" else _dtype_from_string(args.model_dtype)
    if model_dtype is None:
        model = model.to(device)
    else:
        model = model.to(device=device, dtype=model_dtype)
        _set_radio_runtime_dtype(model, model_dtype)
    if args.eval:
        model.eval()
    else:
        model.train(True)

    total_params, trainable_params = _count_parameters(model)
    print(f"RADIO smoke variant: {args.variant}")
    print(f"device: {device}")
    print(
        "model: "
        f"embed_dim={model.embed_dim}, patch_size={model.patch_size}, "
        f"num_layers={model.num_layers}, num_summary_tokens={model.num_summary_tokens}"
    )
    print(f"out_indices: {model.out_indices}")
    print(f"params: total={total_params / 1e6:.2f}M, trainable={trainable_params / 1e6:.2f}M")

    if args.variant == "freeze" and trainable_params != 0:
        raise RuntimeError(f"FrozenRADIO should have 0 trainable params, got {trainable_params}.")

    if args.no_forward:
        print("skip forward: --no-forward is set")
        return

    height, width = args.image_size
    x = _make_smoke_input(args.batch_size, height, width, args.input_mode, device)
    with torch.no_grad():
        pyramid_features, vit_features, summary = model(x)

    print("pyramid features:")
    for idx, feat in enumerate(pyramid_features):
        print(f"  pyramid[{idx}]: {_shape_text(feat)}")
    print("vit features:")
    for idx, feat in enumerate(vit_features):
        print(f"  vit[{idx}]: {_shape_text(feat)}")
    print(f"summary: {_shape_text(summary)}")

    _assert_finite("pyramid_features", pyramid_features)
    _assert_finite("vit_features", vit_features)
    _assert_finite("summary", summary)

    expected = len(model.out_indices)
    if len(vit_features) != expected:
        raise RuntimeError(f"Expected {expected} vit features, got {len(vit_features)}.")
    if len(pyramid_features) != len(args.pyramid_scales):
        raise RuntimeError(f"Expected {len(args.pyramid_scales)} pyramid features, got {len(pyramid_features)}.")

    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"cuda peak memory: {peak_mem_gb:.2f} GiB")

    print("RADIO smoke test passed.")


if __name__ == "__main__":
    _main()
