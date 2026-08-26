from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch import nn
from torch.nn import functional as F

from deeplab.model import Res_Deeplab

from .config import resolve_repo_path


VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"


class NativeDeepLabV2(nn.Module):
    """Official ADS DeepLabV2-ResNet101 segmentor."""

    def __init__(self, num_classes: int, pretrained: str | None = None):
        super().__init__()
        self.model = Res_Deeplab(num_classes=num_classes)
        if pretrained:
            self.load_pretrained(pretrained)

    def load_pretrained(self, path: str) -> None:
        checkpoint_path = resolve_repo_path(path)
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Native ADS checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        current = self.model.state_dict()
        loaded = {}
        for key, value in state.items():
            key = key.removeprefix("module.")
            key = key.removeprefix("Scale.")
            if key.startswith("layer5."):
                continue
            if key in current and current[key].shape == value.shape:
                loaded[key] = value
        missing, unexpected = self.model.load_state_dict(loaded, strict=False)
        if not loaded:
            raise RuntimeError(f"No compatible weights found in {checkpoint_path}")
        print(
            f"Loaded {len(loaded)} native backbone tensors from {checkpoint_path} "
            f"({len(missing)} missing, {len(unexpected)} unexpected)."
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        logits = self.model(image)
        return F.interpolate(
            logits, size=image.shape[-2:], mode="bilinear",
            align_corners=True)

    def optimizer_groups(self, base_lr: float) -> List[Dict[str, object]]:
        backbone_modules = [
            self.model.conv1, self.model.bn1, self.model.layer1,
            self.model.layer2, self.model.layer3, self.model.layer4,
        ]
        backbone = [
            parameter
            for module in backbone_modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        classifier = [
            parameter for parameter in self.model.layer5.parameters()
            if parameter.requires_grad
        ]
        return [
            {"params": backbone, "lr": base_lr, "lr_scale": 1.0},
            {"params": classifier, "lr": base_lr * 10.0, "lr_scale": 10.0},
        ]


def _activate_vendor() -> None:
    vendor = str(VENDOR_ROOT)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)


def _daformer_decoder_params() -> Dict[str, object]:
    return {
        "embed_dims": 256,
        "embed_cfg": {"type": "mlp", "act_cfg": None, "norm_cfg": None},
        "embed_neck_cfg": {"type": "mlp", "act_cfg": None, "norm_cfg": None},
        "fusion_cfg": {
            "type": "aspp",
            "sep": True,
            "dilations": (1, 6, 12, 18),
            "pool": False,
            "act_cfg": {"type": "ReLU"},
            "norm_cfg": {"type": "BN", "requires_grad": True},
        },
    }


class DINOv3ReINHRDA(nn.Module):
    """Self-contained DINOv3-B + ReIN + HRDA segmentor used by TC-ADA."""

    def __init__(self, num_classes: int, pretrained: str):
        super().__init__()
        _activate_vendor()
        from lib.models.backbones.reins_dino_v3 import ReinsDINOv3
        from lib.models.decode_heads.hrda_head import HRDAHead
        from lib.models.segmentors.hrda_encoder_decoder import HRDAEncoderDecoder

        pretrained_path = resolve_repo_path(pretrained)
        if pretrained_path is None or not pretrained_path.is_file():
            raise FileNotFoundError(f"DINOv3-B checkpoint not found: {pretrained_path}")

        backbone_config = {
            "reins_config": {
                "lora_dim": 16,
                "num_layers": 12,
                "non_adapter_layers": 0,
                "embed_dims": 768,
                "patch_size": 16,
                "token_length": 100,
                "link_token_to_query": False,
            },
            "dinov3_config": {
                "img_size": 512,
                "patch_size": 16,
                "pos_embed_rope_rescale_coords": 2.0,
                "pos_embed_rope_dtype": "fp32",
                "embed_dim": 768,
                "depth": 12,
                "num_heads": 12,
                "ffn_ratio": 4.0,
                "qkv_bias": True,
                "layerscale_init": 1e-5,
                "ffn_layer": "mlp",
                "ffn_bias": True,
                "proj_bias": True,
                "n_storage_tokens": 4,
                "mask_k_bias": True,
                "out_indices": [2, 5, 8, 11],
            },
        }
        backbone = ReinsDINOv3(
            backbone_config=backbone_config,
            pretrained={"dinov3": str(pretrained_path)},
        )
        decoder_config = {
            "in_channels": [768, 768, 768, 768],
            "in_index": [0, 1, 2, 3],
            "channels": 256,
            "dropout_ratio": 0.1,
            "num_classes": num_classes,
            "norm_cfg": {"type": "BN", "requires_grad": True},
            "align_corners": False,
            "loss_decode": {
                "type": "CrossEntropyLoss",
                "use_sigmoid": False,
                "loss_weight": 1.0,
            },
            "single_scale_head": "DAFormerHead",
            "interpolate": False,
            "decoder_params": _daformer_decoder_params(),
            "lr_loss_weight": 0.0,
            "hr_loss_weight": 0.1,
            "scales": [0.5, 1.0],
            "attention_embed_dim": 256,
            "attention_classwise": True,
            "enable_hr_crop": True,
            "hr_slide_inference": True,
            "hr_slide_overlapping": True,
            "hr_crop_size": [512, 512],
            "crop_coord_divisible": 8,
            "blur_hr_crop": False,
            "feature_scale": 0.5,
            "fixed_attention": None,
            "debug_output_attention": False,
        }
        decode_head = HRDAHead(decoder_config)
        self.model = HRDAEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            test_cfg={"mode": "slide", "stride": [512, 512], "crop_size": [1024, 1024]},
        )

    def _eval_crop(self, image: torch.Tensor) -> torch.Tensor:
        # Memory-safe equivalent of HRDA's non-batched high-resolution slide.
        # The vendored implementation concatenates all nine 512x512 crops,
        # which can exceed 24 GB even at evaluation time.
        head = self.model.decode_head
        low_image = F.interpolate(
            image, scale_factor=0.5, mode="bilinear", align_corners=False)
        low_features = self.model.extract_unscaled_feat(low_image)
        low_logits = head.head(low_features)
        attention = torch.sigmoid(head.scale_attention(low_features))

        batch, _, height, width = image.shape
        crop_h, crop_w = 512, 512
        stride_h, stride_w = 256, 256
        rows = max(math.ceil((height - crop_h) / stride_h) + 1, 1)
        cols = max(math.ceil((width - crop_w) / stride_w) + 1, 1)
        out_h, out_w = height // head.os, width // head.os
        high_logits = image.new_zeros(
            (batch, self.model.num_classes, out_h, out_w))
        counts = image.new_zeros((batch, 1, out_h, out_w))

        for row in range(rows):
            for col in range(cols):
                y2 = min(row * stride_h + crop_h, height)
                x2 = min(col * stride_w + crop_w, width)
                y1 = max(y2 - crop_h, 0)
                x1 = max(x2 - crop_w, 0)
                features = self.model.extract_unscaled_feat(
                    image[:, :, y1:y2, x1:x2])
                crop_logits = head.head(features)
                fy1, fy2 = y1 // head.os, y2 // head.os
                fx1, fx2 = x1 // head.os, x2 // head.os
                high_logits[:, :, fy1:fy2, fx1:fx2] += crop_logits
                counts[:, :, fy1:fy2, fx1:fx2] += 1

        high_logits = high_logits / counts.clamp_min(1)
        low_logits = (1.0 - attention) * low_logits
        low_logits = F.interpolate(
            low_logits, size=high_logits.shape[-2:], mode="bilinear",
            align_corners=False)
        attention = F.interpolate(
            attention, size=high_logits.shape[-2:], mode="bilinear",
            align_corners=False)
        fused = attention * high_logits + low_logits
        return F.interpolate(
            fused, size=image.shape[-2:], mode="bilinear",
            align_corners=False)

    def _slide_inference(self, image: torch.Tensor) -> torch.Tensor:
        crop_h, crop_w = 1024, 1024
        stride_h, stride_w = 512, 512
        batch, _, height, width = image.shape
        rows = max(math.ceil((height - crop_h) / stride_h) + 1, 1)
        cols = max(math.ceil((width - crop_w) / stride_w) + 1, 1)
        predictions = image.new_zeros(
            (batch, self.model.num_classes, height, width))
        counts = image.new_zeros((batch, 1, height, width))
        for row in range(rows):
            for col in range(cols):
                y2 = min(row * stride_h + crop_h, height)
                x2 = min(col * stride_w + crop_w, width)
                y1 = max(y2 - crop_h, 0)
                x1 = max(x2 - crop_w, 0)
                crop_logits = self._eval_crop(image[:, :, y1:y2, x1:x2])
                predictions[:, :, y1:y2, x1:x2] += crop_logits
                counts[:, :, y1:y2, x1:x2] += 1
        return predictions / counts.clamp_min(1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if self.training:
            features, _ = self.model._forward_train_features(image)
            output = self.model.decode_head.forward_test(features)
            return F.interpolate(
                output, size=image.shape[-2:], mode="bilinear",
                align_corners=False)
        if image.shape[-2] <= 1024 and image.shape[-1] <= 1024:
            return self._eval_crop(image)
        return self._slide_inference(image)

    def optimizer_groups(self, base_lr: float) -> List[Dict[str, object]]:
        groups: Dict[tuple, List[nn.Parameter]] = {}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            lr_scale = 10.0 if "decode_head" in name else 1.0
            no_decay = parameter.ndim == 1 or any(
                token in name.lower() for token in ("norm", "token", "pos_embed")
            )
            key = (lr_scale, no_decay)
            groups.setdefault(key, []).append(parameter)
        return [
            {
                "params": parameters,
                "lr": base_lr * lr_scale,
                "lr_scale": lr_scale,
                "weight_decay": 0.0 if no_decay else None,
            }
            for (lr_scale, no_decay), parameters in groups.items()
        ]


def build_segmentor(config: Dict[str, object]) -> nn.Module:
    family = str(config["family"]).lower()
    if family == "native":
        return NativeDeepLabV2(
            int(config["num_classes"]), config.get("pretrained"))
    if family == "vfm":
        return DINOv3ReINHRDA(
            int(config["num_classes"]), str(config["pretrained"]))
    raise ValueError(f"Unknown model family: {family}")


def trainable_parameter_count(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable
