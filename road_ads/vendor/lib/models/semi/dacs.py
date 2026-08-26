# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
import logging
from copy import deepcopy
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules.dropout import _DropoutNd
from timm.layers import DropPath
import numpy as np
from tqdm import tqdm

from .segmentor import SemiSegmentor, get_module
from .masking_consistency_module import MaskingConsistencyModule

from lib.models.model_utils.funcs import downscale_label_ratio, add_prefix, crop
from lib.models.model_utils.dacs_transforms import (get_class_masks, get_context_class_masks, get_mean_std_self, strong_transform, strong_transform_wo_mix)
from lib.models.model_utils.visualization import (
    get_debug_palette,
    save_debug_images,
    save_debug_mic_images,
    save_debug_tkm_images,
    save_debug_images_s,
    save_debug_depth_images,
    save_cls_debug_images,
    save_debug_hrda_images,
    save_debug_sadg_images,
    save_weather_aligned_mix_debug_images,
)
from lib.models.segmentors import *
from lib.loss.losses import BCEWithLogitsLoss2d, BCEWithLogitsLoss2d_Batch_Weighted, BCEWithLogitsLoss2d_Batch_Patch_Weighted, parse_losses
from train.vplf import (
    VFMPrototypePseudoLabelFilter,
    append_vplf_class_stats_csv,
    compute_vplf_class_stats,
    get_vplf_config,
    save_vplf_debug_images,
    select_vfm_feature_map,
)
from train.sadg import (
    SADG_DEFAULT_AUG_TYPES,
    SADGAugmentor,
    logit_kl_consistency,
    normalize_sadg_aug_type,
)

from utils.classes import CLASSES
from utils.util import (
    ACDC_EVAL_SCENES,
    compute_ious_from_hist,
    get_acdc_scene_from_path,
    get_dataset_eval_splits,
)
dataset_class = CLASSES['cityscapes']

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True

def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm

class DACS(SemiSegmentor):
    """DACS semi trainer wrapper.

    DACS 半监督语义分割训练封装器。

    The class owns semi-specific state such as EMA teacher, pseudo-label
    thresholds, class mixing, MIC, and feature distillation.

    该类维护半监督训练相关状态，例如 EMA teacher、伪标签阈值、类别混合、
    MIC 以及特征蒸馏配置。
    """

    # Human-readable names for feature-distillation modes.
    # 特征蒸馏类型的可读名称，用于日志输出。
    FDIST_TYPE_NAMES = {
        1: 'DACS Original',
        2: 'DINOv3 Support',
        3: 'Feature Similarity',
    }

    # Mapping from PEFT backbones to frozen teacher backbones for distillation.
    # 特征蒸馏冻结模型使用的 backbone 类型映射：将 PEFT backbone 转为 pure backbone。
    FZNET_BACKBONE_TYPE_MAPPING = {
        'AdapterReinsDINOv3': 'PureDINOv3',
        'AdapterDINOv3': 'PureDINOv3',
        'AdapterLoraDINOv3': 'PureDINOv3',
        'LoraDINOv3': 'PureDINOv3',
        'ReinsDINOv3': 'PureDINOv3',
    }

    @staticmethod
    def _is_master_process():
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    
    def __init__(self, cfg):
        super(DACS, self).__init__(deepcopy(cfg.model))
        self.local_iter = 0
        self.max_iters = cfg.max_iters
        self.logger = logging.getLogger()
        
        self.cfg = cfg
        self.img_mean = self.cfg.data['rgb_mean']
        self.img_std = self.cfg.data['rgb_std']
        self.class_set = self.cfg.data.get(
            'class_set', self.cfg.data.get('dataset_name', None))
        self.dataset_class = CLASSES.get(self.class_set, CLASSES['cityscapes'])
        self.debug_palette = get_debug_palette(self.num_classes, self.class_set)
        self.crop_size = self.cfg.data['source']['cropsize']
        self.semi_cfg = self.cfg.semi
        
        self.source_only = self.semi_cfg['source_only']
        self.ema_begin_iter = self.semi_cfg.get('ema_begin_iter', 0)  # 0 for always
        self.semi_begin_iter = self.semi_cfg.get('semi_begin_iter', 0)  # 0 for always
        self.alpha = self.semi_cfg['alpha']  # 0.999
        self.pseudo_threshold = self.semi_cfg['pseudo_threshold'] # 0.968
        self.psweight_ignore_top = self.semi_cfg['pseudo_weight_ignore_top']  # 15
        self.psweight_ignore_bottom = self.semi_cfg['pseudo_weight_ignore_bottom']  # 120
        self._init_offline_teacher_pseudo_config()
        
        self.share_src_backward = self.semi_cfg.get('share_src_backward', False)
        
        self._init_feature_distillation_config()
        self._init_vplf_config()
        
        
        self.enable_token_masking = self.cfg.model['token_mask_ratio'] is not None
        self.mix = self.semi_cfg['mix']  # class
        self.src_mix_cls_ratio_init = self.semi_cfg.get('src_mix_cls_ratio_init', 0.5)
        self.src_mix_cls_ratio_final = self.semi_cfg.get('src_mix_cls_ratio_final', 0.5)
        self.ratio_mode = self.semi_cfg.get('ratio_mode', 'constant')
        self.get_context_class_mask = self.semi_cfg.get('get_context_class_mask', False)
        self.blur = self.semi_cfg['blur']  # True
        self.color_jitter_s = self.semi_cfg['color_jitter_strength']
        self.color_jitter_p = self.semi_cfg['color_jitter_probability']
        
        # style consistency
        self.style_consistency_lambda = self.semi_cfg.get('style_consistency_lambda', 0)
        
        self.mask_mode = self.semi_cfg['mask_mode']
        self.enable_masking = self.mask_mode is not None
        self.debug_img_interval = int(self.semi_cfg.get('debug_img_interval', 0) or 0)
        self.print_grad_magnitude = self.semi_cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None
        self.stage2_sadg_debug_data = None
        self.weather_mix_debug_data = None
        self._weather_mix_warned_missing_path = False
        self._weather_mix_warned_unknown_path = False
        self._init_stage2_sadg_config()
        self._init_weather_aligned_mix_config()
        self._init_grad_conflict_debug_config()
        self._init_stage2_mix_loss_control_config()
        self._init_adapter_grad_conflict_config()

        self.class_probs = {}
        
        self.logger.info(f'Building EMA model for DACS...')
        ema_cfg = self._prepare_ema_model_cfg()
        if not self.source_only:
            self.ema_model = self.build_model(ema_cfg)
        self.logger.info(f'EMA model built successfully. \n')
        
        # mic
        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg)

        # feat loss / VPLF frozen VFM
        # 特征蒸馏 / VPLF 共用冻结 VFM 模型。
        needs_frozen_vfm = self.enable_fdist or self.vfm_pl_filter_enabled
        if needs_frozen_vfm:
            fznet_model_cfg = self._prepare_fznet_model_cfg(self.cfg.model)
            self.fznet_model = self.build_model(fznet_model_cfg)
            for param in self.get_fz_model().parameters():
                param.requires_grad = False
            self.get_fz_model().eval()
        else:
            self.fznet_model = None

        if self.enable_fdist:
            self.build_project_head()
        else:
            self.proj_head = None

        self.vfm_pl_filter = None
        if self.vfm_pl_filter_enabled:
            self.vfm_pl_filter = VFMPrototypePseudoLabelFilter(self.vplf_cfg)
            self.vplf_class_stats_path = os.path.join(
                self.cfg.respth, 'vplf_class_stats.csv')
            self.vplf_debug_img_interval = self.vplf_cfg['debug_img_interval']
            if self.vplf_debug_img_interval is None:
                self.vplf_debug_img_interval = self.debug_img_interval
            self.vplf_class_stats_interval = self.vplf_cfg['class_stats_interval']
            if self.vplf_class_stats_interval is None:
                self.vplf_class_stats_interval = self.vplf_cfg['log_interval']
            self.vplf_debug_data = None
            self.logger.info(
                '[VPLF] Enabled: proto_path=%s, feature_source=%s, '
                'feature_layer=%s, mode=%s, temp=%.4f',
                self.vplf_cfg['proto_path'],
                self.vplf_cfg['feature_source'],
                self.vplf_cfg['feature_layer'],
                self.vplf_cfg['filter_mode'],
                self.vplf_cfg['proto_temp'],
            )
            self.logger.info(
                '[VPLF] Debug images interval=%s, class CSV interval=%s, csv=%s',
                self.vplf_debug_img_interval,
                self.vplf_class_stats_interval,
                self.vplf_class_stats_path,
            )
        else:
            self.vplf_class_stats_path = None
            self.vplf_debug_img_interval = 0
            self.vplf_class_stats_interval = 0
            self.vplf_debug_data = None
            
        if self.get_model().auxiliary_head is not None:
            self.with_aux_head = True
            self.aux_loss_weight = self.cfg.model['aux_head']['loss_decode']['loss_weight']
            self.debug_imgs = self.cfg.model['aux_head'].get('debug_imgs', False)
        else:
            self.with_aux_head = False
            self.aux_loss_weight = 1
            self.debug_imgs = False

    @staticmethod
    def _as_tuple(value):
        if value is None:
            return tuple()
        if isinstance(value, str):
            return (value,)
        return tuple(value)

    def _init_grad_conflict_debug_config(self):
        """Read optional source-vs-mix gradient conflict diagnostics."""
        self.semi_grad_conflict_debug = bool(
            self.semi_cfg.get('semi_grad_conflict_debug', False))
        self.semi_grad_conflict_interval = int(
            self.semi_cfg.get('semi_grad_conflict_interval', 500))
        self.semi_grad_conflict_warmup = int(
            self.semi_cfg.get('semi_grad_conflict_warmup', 0))
        self.semi_grad_conflict_param_keywords = self._as_tuple(
            self.semi_cfg.get(
                'semi_grad_conflict_param_keywords',
                ('adapter', 'injector', 'reins', 'lora', 'decode_head')))
        self._grad_conflict_params = None
        self._grad_conflict_source_grads = None
        self.grad_conflict_csv_path = os.path.join(
            self.cfg.respth, 'grad_conflict_stats.csv')
        if self.semi_grad_conflict_debug:
            self.logger.info(
                '[GradConflict] Enabled: interval=%d, warmup=%d, keywords=%s',
                self.semi_grad_conflict_interval,
                self.semi_grad_conflict_warmup,
                self.semi_grad_conflict_param_keywords,
            )

    def _should_collect_grad_conflict(self):
        """Return whether this iteration should sample branch gradients."""
        if not self.semi_grad_conflict_debug:
            return False
        interval = max(1, self.semi_grad_conflict_interval)
        iter_id = self.local_iter + 1
        return iter_id >= self.semi_grad_conflict_warmup and iter_id % interval == 0

    def _get_grad_conflict_params(self):
        """Select trainable adapter/decoder parameters for gradient diagnostics."""
        if self._grad_conflict_params is not None:
            return self._grad_conflict_params

        selected = []
        keywords = tuple(k for k in self.semi_grad_conflict_param_keywords if k)
        for name, param in self.get_model().named_parameters():
            if not param.requires_grad:
                continue
            if keywords and not any(keyword in name for keyword in keywords):
                continue
            selected.append((name, param))

        if not selected:
            selected = [
                (name, param)
                for name, param in self.get_model().named_parameters()
                if param.requires_grad
            ]
            self.logger.warning(
                '[GradConflict] No parameters matched keywords=%s; '
                'falling back to all trainable parameters.',
                keywords,
            )

        self._grad_conflict_params = selected
        self.logger.info(
            '[GradConflict] Tracking %d trainable parameters.',
            len(selected),
        )
        return self._grad_conflict_params

    @staticmethod
    def _grad_conflict_group_name(param_name):
        """Group parameter names into coarse modules for easier reading."""
        if 'decode_head' in param_name:
            return 'decoder'
        if any(key in param_name for key in ('adapter', 'injector', 'reins', 'lora')):
            return 'adapter'
        return 'other'

    def _capture_grad_conflict_grads(self, loss):
        """Compute detached CPU gradients for a loss without touching .grad."""
        if loss is None or not getattr(loss, 'requires_grad', False):
            return None
        named_params = self._get_grad_conflict_params()
        if not named_params:
            return None

        names, params = zip(*named_params)
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        captured = []
        for name, grad in zip(names, grads):
            grad_cpu = None
            if grad is not None:
                grad_cpu = grad.detach().float().cpu()
            captured.append((name, grad_cpu))
        return captured

    @staticmethod
    def _accumulate_grad_stats(src_grads, mix_grads):
        """Return cosine/norm statistics for two named gradient lists."""
        groups = {
            'all': {
                'dot': 0.0,
                'src_norm_sq': 0.0,
                'mix_norm_sq': 0.0,
                'param_count': 0,
                'src_active': 0,
                'mix_active': 0,
            }
        }
        mix_by_name = dict(mix_grads or [])
        for name, src_grad in src_grads or []:
            group_name = DACS._grad_conflict_group_name(name)
            if group_name not in groups:
                groups[group_name] = {
                    'dot': 0.0,
                    'src_norm_sq': 0.0,
                    'mix_norm_sq': 0.0,
                    'param_count': 0,
                    'src_active': 0,
                    'mix_active': 0,
                }
            mix_grad = mix_by_name.get(name)
            for group in (groups['all'], groups[group_name]):
                group['param_count'] += 1
                if src_grad is not None:
                    group['src_norm_sq'] += float(torch.sum(src_grad * src_grad).item())
                    group['src_active'] += 1
                if mix_grad is not None:
                    group['mix_norm_sq'] += float(torch.sum(mix_grad * mix_grad).item())
                    group['mix_active'] += 1
                if src_grad is not None and mix_grad is not None:
                    group['dot'] += float(torch.sum(src_grad * mix_grad).item())

        log_vars = {}
        eps = 1e-12
        for group_name, group in groups.items():
            src_norm = math.sqrt(max(group['src_norm_sq'], 0.0))
            mix_norm = math.sqrt(max(group['mix_norm_sq'], 0.0))
            if src_norm > 0 and mix_norm > 0:
                cosine = group['dot'] / (src_norm * mix_norm + eps)
            else:
                cosine = 0.0
            prefix = f'grad_conflict_{group_name}'
            log_vars[f'{prefix}_cosine'] = float(cosine)
            log_vars[f'{prefix}_src_norm'] = float(src_norm)
            log_vars[f'{prefix}_mix_norm'] = float(mix_norm)
            log_vars[f'{prefix}_src_to_mix_norm_ratio'] = float(
                src_norm / (mix_norm + eps))
            log_vars[f'{prefix}_is_negative'] = float(cosine < 0)
            log_vars[f'{prefix}_param_count'] = float(group['param_count'])
            log_vars[f'{prefix}_src_active_ratio'] = float(
                group['src_active'] / max(1, group['param_count']))
            log_vars[f'{prefix}_mix_active_ratio'] = float(
                group['mix_active'] / max(1, group['param_count']))
        return log_vars

    def _build_grad_conflict_log_vars(self, mix_loss):
        """Compare current mix gradients with the cached source gradients."""
        if not self._should_collect_grad_conflict():
            return {}
        if self._grad_conflict_source_grads is None:
            return {}
        mix_grads = self._capture_grad_conflict_grads(mix_loss)
        if mix_grads is None:
            return {}
        log_vars = self._accumulate_grad_stats(
            self._grad_conflict_source_grads,
            mix_grads,
        )
        self._grad_conflict_source_grads = None
        self._append_grad_conflict_csv(log_vars)
        return log_vars

    def _append_grad_conflict_csv(self, log_vars):
        """Append gradient conflict diagnostics to a CSV for quick analysis."""
        if not log_vars or not self._is_master_process():
            return
        os.makedirs(os.path.dirname(self.grad_conflict_csv_path), exist_ok=True)
        write_header = not os.path.exists(self.grad_conflict_csv_path)
        keys = sorted(log_vars.keys())
        with open(self.grad_conflict_csv_path, 'a', encoding='utf-8') as f:
            if write_header:
                f.write('iter,' + ','.join(keys) + '\n')
            values = [str(self.local_iter + 1)]
            values.extend(f'{float(log_vars[key]):.8g}' for key in keys)
            f.write(','.join(values) + '\n')

    def _init_stage2_mix_loss_control_config(self):
        """Read optional Stage-2 mix loss scheduling and confidence weights."""
        self.mix_loss_weight = float(self.semi_cfg.get('mix_loss_weight', 1.0))
        self.mix_loss_ramp_enabled = bool(
            self.semi_cfg.get('mix_loss_ramp_enabled', False))
        self.mix_loss_ramp_start = float(
            self.semi_cfg.get('mix_loss_ramp_start', 0.5))
        self.mix_loss_ramp_end = float(
            self.semi_cfg.get('mix_loss_ramp_end', 1.0))
        self.mix_loss_ramp_start_iter = int(
            self.semi_cfg.get('mix_loss_ramp_start_iter', 0))
        self.mix_loss_ramp_end_iter = int(
            self.semi_cfg.get('mix_loss_ramp_end_iter', self.max_iters))
        self.mix_loss_ramp_mode = str(
            self.semi_cfg.get('mix_loss_ramp_mode', 'linear')).lower()

        self.mix_confidence_weight_enabled = bool(
            self.semi_cfg.get('mix_confidence_weight_enabled', False))
        self.mix_confidence_weight_mode = str(
            self.semi_cfg.get('mix_confidence_weight_mode', 'power')).lower()
        self.mix_confidence_gamma = float(
            self.semi_cfg.get('mix_confidence_gamma', 2.0))
        self.mix_confidence_min_weight = float(
            self.semi_cfg.get('mix_confidence_min_weight', 0.25))
        self.mix_confidence_max_weight = float(
            self.semi_cfg.get('mix_confidence_max_weight', 1.0))
        self.mix_confidence_threshold = float(
            self.semi_cfg.get('mix_confidence_threshold', self.pseudo_threshold))
        if self.mix_loss_ramp_enabled:
            self.logger.info(
                '[MixRamp] Enabled: %.3f -> %.3f, iter %d-%d, mode=%s',
                self.mix_loss_ramp_start,
                self.mix_loss_ramp_end,
                self.mix_loss_ramp_start_iter,
                self.mix_loss_ramp_end_iter,
                self.mix_loss_ramp_mode,
            )
        if self.mix_confidence_weight_enabled:
            self.logger.info(
                '[MixConfidence] Enabled: mode=%s, gamma=%.3f, clamp=(%.3f, %.3f), threshold=%.3f',
                self.mix_confidence_weight_mode,
                self.mix_confidence_gamma,
                self.mix_confidence_min_weight,
                self.mix_confidence_max_weight,
                self.mix_confidence_threshold,
            )

    def _get_mix_loss_weight(self):
        """Return the current scalar multiplier for the mix loss."""
        if not self.mix_loss_ramp_enabled:
            return self.mix_loss_weight

        iter_id = self.local_iter + 1
        start_iter = self.mix_loss_ramp_start_iter
        end_iter = max(start_iter + 1, self.mix_loss_ramp_end_iter)
        if iter_id <= start_iter:
            progress = 0.0
        elif iter_id >= end_iter:
            progress = 1.0
        else:
            progress = float(iter_id - start_iter) / float(end_iter - start_iter)

        if self.mix_loss_ramp_mode == 'linear':
            factor = progress
        elif self.mix_loss_ramp_mode == 'exp1':
            factor = 1.0 - math.exp(-4.0 * progress)
        elif self.mix_loss_ramp_mode == 'exp2':
            factor = (math.exp(4.0 * progress) - 1.0) / (math.exp(4.0) - 1.0)
        else:
            raise ValueError(
                "Invalid mix_loss_ramp_mode. Choose from 'linear', 'exp1', or 'exp2'.")
        return self.mix_loss_ramp_start + (
            self.mix_loss_ramp_end - self.mix_loss_ramp_start) * factor

    def _apply_mix_confidence_weight(self, pseudo_weight, pseudo_conf):
        """Down-weight target pseudo pixels in ClassMix using teacher confidence."""
        if not self.mix_confidence_weight_enabled or pseudo_conf is None:
            return pseudo_weight, {}

        conf = pseudo_conf.detach().float().clamp(0.0, 1.0)
        if self.mix_confidence_weight_mode == 'power':
            conf_weight = conf.pow(max(1e-6, self.mix_confidence_gamma))
        elif self.mix_confidence_weight_mode == 'threshold_linear':
            denom = max(1e-6, 1.0 - self.mix_confidence_threshold)
            conf_weight = ((conf - self.mix_confidence_threshold) / denom).clamp(0.0, 1.0)
            conf_weight = conf_weight.pow(max(1e-6, self.mix_confidence_gamma))
        elif self.mix_confidence_weight_mode == 'hard':
            conf_weight = conf.ge(self.mix_confidence_threshold).float()
        else:
            raise ValueError(
                "Invalid mix_confidence_weight_mode. Choose from 'power', "
                "'threshold_linear', or 'hard'.")

        conf_weight = conf_weight.clamp(
            self.mix_confidence_min_weight,
            self.mix_confidence_max_weight,
        )
        weighted_pseudo = pseudo_weight * conf_weight.to(
            device=pseudo_weight.device,
            dtype=pseudo_weight.dtype,
        )
        return weighted_pseudo, {
            'mix_confidence_weight_enabled': 1.0,
            'mix_confidence_mean': float(conf.mean().item()),
            'mix_confidence_weight_mean': float(conf_weight.mean().item()),
            'mix_confidence_weight_min': float(conf_weight.min().item()),
            'mix_confidence_weight_max': float(conf_weight.max().item()),
            'mix_confidence_pseudo_weight_mean': float(
                pseudo_weight.detach().float().mean().item()),
            'mix_confidence_final_weight_mean': float(
                weighted_pseudo.detach().float().mean().item()),
        }

    def _init_adapter_grad_conflict_config(self):
        """Read optional adapter-only source/mix gradient conflict handling."""
        self.adapter_grad_conflict_enabled = bool(
            self.semi_cfg.get('adapter_grad_conflict_enabled', False))
        self.adapter_grad_conflict_mode = str(
            self.semi_cfg.get('adapter_grad_conflict_mode', 'project_mix')).lower()
        self.adapter_grad_conflict_min_cos = float(
            self.semi_cfg.get('adapter_grad_conflict_min_cos', 0.0))
        self.adapter_grad_conflict_param_keywords = self._as_tuple(
            self.semi_cfg.get(
                'adapter_grad_conflict_param_keywords',
                ('adapter', 'injector', 'reins', 'lora')))
        self._adapter_grad_conflict_params = None
        self._adapter_grad_conflict_source_grads = None
        if self.adapter_grad_conflict_enabled:
            self.logger.info(
                '[AdapterGradConflict] Enabled: mode=%s, min_cos=%.3f, keywords=%s',
                self.adapter_grad_conflict_mode,
                self.adapter_grad_conflict_min_cos,
                self.adapter_grad_conflict_param_keywords,
            )

    def _get_adapter_grad_conflict_params(self):
        if self._adapter_grad_conflict_params is not None:
            return self._adapter_grad_conflict_params

        keywords = tuple(k for k in self.adapter_grad_conflict_param_keywords if k)
        selected = []
        for name, param in self.get_model().named_parameters():
            if not param.requires_grad:
                continue
            if keywords and not any(keyword in name for keyword in keywords):
                continue
            selected.append((name, param))

        if not selected:
            self.logger.warning(
                '[AdapterGradConflict] No parameters matched keywords=%s; disabled for this run.',
                keywords,
            )
        self._adapter_grad_conflict_params = selected
        return selected

    def _capture_adapter_grad_conflict_grads(self, loss):
        if not self.adapter_grad_conflict_enabled:
            return None
        if loss is None or not getattr(loss, 'requires_grad', False):
            return None
        named_params = self._get_adapter_grad_conflict_params()
        if not named_params:
            return None

        names, params = zip(*named_params)
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        captured = []
        for name, param, grad in zip(names, params, grads):
            captured.append((
                name,
                param,
                grad.detach().clone() if grad is not None else None,
            ))
        return captured

    @staticmethod
    def _named_grad_norm_sq(named_grads):
        norm_sq = 0.0
        for _, _, grad in named_grads or []:
            if grad is not None:
                norm_sq += float(torch.sum(grad.float() * grad.float()).item())
        return norm_sq

    def _apply_adapter_grad_conflict(self, mix_grads):
        """Project source/mix conflicting gradients only on adapter parameters."""
        if not self.adapter_grad_conflict_enabled:
            return {}
        src_grads = self._adapter_grad_conflict_source_grads
        self._adapter_grad_conflict_source_grads = None
        if not src_grads or not mix_grads:
            return {}

        src_by_name = {name: grad for name, _, grad in src_grads}
        mix_by_name = {name: grad for name, _, grad in mix_grads}
        dot = 0.0
        for name, src_grad in src_by_name.items():
            mix_grad = mix_by_name.get(name)
            if src_grad is not None and mix_grad is not None:
                dot += float(torch.sum(src_grad.float() * mix_grad.float()).item())

        eps = 1e-12
        src_norm_sq = self._named_grad_norm_sq(src_grads)
        mix_norm_sq = self._named_grad_norm_sq(mix_grads)
        src_norm = math.sqrt(max(src_norm_sq, 0.0))
        mix_norm = math.sqrt(max(mix_norm_sq, 0.0))
        cosine = dot / (src_norm * mix_norm + eps) if src_norm > 0 and mix_norm > 0 else 0.0
        should_project = dot < 0 and cosine < self.adapter_grad_conflict_min_cos and \
            src_norm_sq > 0 and mix_norm_sq > 0

        if should_project:
            if self.adapter_grad_conflict_mode == 'project_mix':
                mix_coeff = dot / (src_norm_sq + eps)
                src_coeff = 0.0
            elif self.adapter_grad_conflict_mode == 'project_src':
                src_coeff = dot / (mix_norm_sq + eps)
                mix_coeff = 0.0
            else:
                raise ValueError(
                    "Invalid adapter_grad_conflict_mode. Choose from "
                    "'project_mix' or 'project_src'.")
        else:
            src_coeff = 0.0
            mix_coeff = 0.0

        adjusted_params = 0
        with torch.no_grad():
            for name, param, mix_grad in mix_grads:
                src_grad = src_by_name.get(name)
                if src_grad is None and mix_grad is None:
                    continue
                if param.grad is None:
                    continue
                src_tensor = torch.zeros_like(param.grad)
                mix_tensor = torch.zeros_like(param.grad)
                if src_grad is not None:
                    src_tensor = src_grad.to(
                        device=param.grad.device,
                        dtype=param.grad.dtype,
                    )
                if mix_grad is not None:
                    mix_tensor = mix_grad.to(
                        device=param.grad.device,
                        dtype=param.grad.dtype,
                    )

                new_src = src_tensor
                new_mix = mix_tensor
                if should_project:
                    if self.adapter_grad_conflict_mode == 'project_mix':
                        new_mix = mix_tensor - mix_coeff * src_tensor
                    elif self.adapter_grad_conflict_mode == 'project_src':
                        new_src = src_tensor - src_coeff * mix_tensor

                other_grad = param.grad - src_tensor - mix_tensor
                param.grad.copy_(other_grad + new_src + new_mix)
                adjusted_params += 1

        return {
            'adapter_grad_conflict_cosine': float(cosine),
            'adapter_grad_conflict_applied': float(should_project),
            'adapter_grad_conflict_src_norm': float(src_norm),
            'adapter_grad_conflict_mix_norm': float(mix_norm),
            'adapter_grad_conflict_src_to_mix_norm_ratio': float(src_norm / (mix_norm + eps)),
            'adapter_grad_conflict_adjusted_params': float(adjusted_params),
        }

    def _init_stage2_sadg_config(self):
        """Read optional Stage-2 SADG source-branch regularization settings."""
        self.stage2_sadg_enabled = bool(
            self.semi_cfg.get('stage2_sadg_enabled', False))
        aug_types = self.semi_cfg.get(
            'stage2_sadg_aug_types',
            self.semi_cfg.get('sadg_aug_types', SADG_DEFAULT_AUG_TYPES))
        aug_types = tuple(
            normalize_sadg_aug_type(aug_type)
            for aug_type in self._as_tuple(aug_types)
        )
        self.stage2_sadg_cfg = {
            'gamma': self.semi_cfg.get('stage2_sadg_gamma', 2.0),
            'tau': self.semi_cfg.get('stage2_sadg_tau', 2.0),
            'lambda_aug': self.semi_cfg.get('stage2_sadg_lambda_aug', 1.0),
            'lambda_cons': self.semi_cfg.get('stage2_sadg_lambda_cons', 0.2),
            'aug_types': aug_types,
            'debug_img_interval': self.semi_cfg.get(
                'stage2_sadg_debug_img_interval', self.debug_img_interval),
        }
        self.stage2_sadg_augmentor = None
        if self.stage2_sadg_enabled:
            self.stage2_sadg_augmentor = SADGAugmentor(
                self.img_mean,
                self.img_std,
                aug_types=self.stage2_sadg_cfg['aug_types'])
            self.logger.info('[Stage2 SADG] Enabled: %s', self.stage2_sadg_cfg)

    def _init_weather_aligned_mix_config(self):
        """Read optional target-weather aligned source stylization for ClassMix."""
        self.weather_aligned_mix_enabled = bool(
            self.semi_cfg.get('weather_aligned_mix_enabled', False))
        self.weather_aligned_mix_target_gt = bool(
            self.semi_cfg.get('weather_aligned_mix_target_gt', True))
        self.weather_aligned_mix_default_weather = self.semi_cfg.get(
            'weather_aligned_mix_default_weather', 'normal')
        self.weather_aligned_mix_severity = self.semi_cfg.get(
            'weather_aligned_mix_severity',
            {'fog': 0.6, 'rain': 0.6, 'snow': 0.6, 'night': 0.7, 'normal': 0.0})
        self.weather_aligned_mix_random_severity = bool(
            self.semi_cfg.get('weather_aligned_mix_random_severity', False))
        self.weather_aligned_mix_severity_range = self.semi_cfg.get(
            'weather_aligned_mix_severity_range', (0.4, 0.8))
        self.weather_aligned_mix_severity_mode = str(
            self.semi_cfg.get('weather_aligned_mix_severity_mode', 'fixed')).lower()
        self.weather_aligned_mix_target_severity_range = self.semi_cfg.get(
            'weather_aligned_mix_target_severity_range', (0.35, 0.85))
        self.weather_aligned_mix_target_severity_by_weather = self.semi_cfg.get(
            'weather_aligned_mix_target_severity_by_weather', {})
        self.weather_aligned_mix_target_severity_blend = float(
            self.semi_cfg.get('weather_aligned_mix_target_severity_blend', 1.0))
        self.weather_aligned_mix_confidence_gamma = float(
            self.semi_cfg.get('weather_aligned_mix_confidence_gamma', 1.0))
        active_weathers = self.semi_cfg.get(
            'weather_aligned_mix_active_weathers', None)
        self.weather_aligned_mix_active_weathers = None
        if active_weathers is not None:
            self.weather_aligned_mix_active_weathers = {
                str(weather).lower()
                for weather in self._as_tuple(active_weathers)
            }
        self.weather_aligned_mix_aug_map = self.semi_cfg.get(
            'weather_aligned_mix_aug_map',
            {'fog': 'fog', 'rain': 'rain', 'snow': 'snow',
             'night': 'night', 'normal': 'normal'})
        self.weather_aligned_mix_debug_img_interval = self.semi_cfg.get(
            'weather_aligned_mix_debug_img_interval', self.debug_img_interval)
        self.weather_aligned_mix_scene_lookup = None

        aug_values = []
        for aug_type in self.weather_aligned_mix_aug_map.values():
            if aug_type in (None, 'none', 'normal'):
                continue
            aug_values.append(normalize_sadg_aug_type(aug_type))
        self.weather_aligned_mix_augmentor = None
        if self.weather_aligned_mix_enabled:
            self.weather_aligned_mix_augmentor = SADGAugmentor(
                self.img_mean,
                self.img_std,
                aug_types=aug_values or ('fog', 'rain', 'snow', 'night'))
            self.logger.info(
                '[WeatherMix] Enabled: target_gt=%s, severity=%s, '
                'severity_mode=%s, active_weathers=%s, map=%s',
                self.weather_aligned_mix_target_gt,
                self.weather_aligned_mix_severity,
                self.weather_aligned_mix_severity_mode,
                self.weather_aligned_mix_active_weathers,
                self.weather_aligned_mix_aug_map)

    def _init_feature_distillation_config(self):
        """Read feature-distillation options from `cfg.semi`.

        从 `cfg.semi` 中读取特征蒸馏相关配置，并规范化多层设置。
        """
        self.fdist_lambda = self.semi_cfg['imnet_feature_dist_lambda']
        self.fdist_type = self.semi_cfg.get('imnet_feature_dist_type', 1)
        self.fdist_layers = self.semi_cfg.get('imnet_feature_dist_layers', [-1])
        self.fdist_classes = self.semi_cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = self.semi_cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.fdist_sim_type = self.semi_cfg.get('imnet_feature_dist_sim_type', 'cosine')
        self.fdist_sim_temp = self.semi_cfg.get('imnet_feature_dist_sim_temp', 0.07)

        if not isinstance(self.fdist_layers, list):
            self.fdist_layers = [self.fdist_layers]

        if self.enable_fdist:
            self._log_feature_distillation_config()

    def _init_vplf_config(self):
        """Read VPLF options and keep default Stage-2 behavior unchanged.

        读取 VPLF 配置，并保证默认关闭时 Stage-2 行为不变。
        """
        self.vplf_cfg = get_vplf_config(self.semi_cfg)
        self.vfm_pl_filter_enabled = bool(self.vplf_cfg['enabled'])
        if self.source_only and self.vfm_pl_filter_enabled:
            self.logger.warning('[VPLF] source_only=True, disable VPLF for this run.')
            self.vfm_pl_filter_enabled = False
            self.vplf_cfg['enabled'] = False

        if not self.vfm_pl_filter_enabled:
            return

        if self.vplf_cfg['feature_source'] != 'frozen_vfm':
            raise ValueError(
                'Only vfm_feature_source="frozen_vfm" is supported in the '
                'first VPLF implementation.')
        if not self.vplf_cfg['proto_path']:
            raise ValueError('vfm_proto_path must be set when vfm_pl_filter_enabled=True.')

        self.logger.info('[VPLF] VFM-guided pseudo-label filtering is enabled.')
        self.logger.info('[VPLF] Config: %s', self.vplf_cfg)

    def _log_feature_distillation_config(self):
        """Log active feature-distillation settings.

        将当前启用的特征蒸馏设置写入日志，方便确认实验配置。
        """
        self.logger.info(f'[DACS] Feature distillation enabled:')
        self.logger.info(
            f'  - Type: {self.FDIST_TYPE_NAMES.get(self.fdist_type, "Unknown")} '
            f'(type={self.fdist_type})')
        self.logger.info(f'  - Layers: {self.fdist_layers} ({len(self.fdist_layers)} layer(s))')
        self.logger.info(f'  - Lambda: {self.fdist_lambda}')
        if self.fdist_type == 3:
            self.logger.info(f'  - Similarity type: {self.fdist_sim_type}')
            self.logger.info(f'  - Temperature: {self.fdist_sim_temp}')
        self.logger.info(f'  - Classes: {self.fdist_classes if self.fdist_classes else "All"}')

    def _prepare_ema_model_cfg(self):
        """Prepare model config for the EMA teacher.

        准备 EMA teacher 的模型配置，优先使用 EMA 专用预训练权重字段。
        """
        ema_cfg = deepcopy(self.cfg.model)
        if 'ema_backbone_pretrained' in ema_cfg:
            ema_cfg['backbone_pretrained'] = ema_cfg.pop('ema_backbone_pretrained')
        if 'ema_decoder_pretrained' in ema_cfg:
            ema_cfg['decoder_pretrained'] = ema_cfg.pop('ema_decoder_pretrained')
        return ema_cfg

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_fz_model(self):
        return get_module(self.fznet_model)
    
    def _prepare_fznet_model_cfg(self, model_cfg):
        """Prepare the frozen teacher config used by feature distillation.

        准备用于特征蒸馏的冻结 teacher 模型配置。

        The conversion keeps the student decoder shape but removes PEFT-only
        backbone options so the frozen network behaves like a pure DINOv3 model.

        该转换保留 student 的 decoder 结构，同时移除 PEFT 专用 backbone 配置，
        让冻结网络按 pure DINOv3 模型工作。

        Args:
            model_cfg: Original model config dictionary.
                原始模型配置字典。

        Returns:
            dict: Converted frozen-model config.
                转换后的冻结模型配置。
        """
        fznet_cfg = deepcopy(model_cfg)
        
        # Convert PEFT backbone type to the matching frozen pure backbone.
        # 将 PEFT backbone 类型转换为对应的冻结 pure backbone。
        backbone_type = fznet_cfg['backbone'].get('type', '')
        
        if backbone_type in self.FZNET_BACKBONE_TYPE_MAPPING:
            mapped_backbone_type = self.FZNET_BACKBONE_TYPE_MAPPING[backbone_type]
            self.logger.info(f'[DACS] Converting backbone type: {backbone_type} -> {mapped_backbone_type}')
            fznet_cfg['backbone']['type'] = mapped_backbone_type
        elif backbone_type == 'PureDINOv3':
            self.logger.info(f'[DACS] Backbone type is already PureDINOv3, no conversion needed')
        else:
            self.logger.warning(f'[DACS] Unknown backbone type: {backbone_type}, keeping original type')
        
        # Strip trainable PEFT options from backbone_config.
        # 从 backbone_config 中移除可训练 PEFT 选项。
        backbone_config = fznet_cfg['backbone'].get('backbone_config', {})
        
        if backbone_config:
            # Remove PEFT-specific configs that the frozen model should not use.
            # 移除冻结模型不应使用的 PEFT 专用配置。
            peft_configs_to_remove = ['reins_config', 'adapter_config', 'lora_config']
            removed_configs = []
            
            for config_key in peft_configs_to_remove:
                if config_key in backbone_config:
                    removed_configs.append(config_key)
                    del backbone_config[config_key]
            
            if removed_configs:
                self.logger.info(f'[DACS] Removed PEFT configs: {removed_configs}')
            
            # Force the distillation model to stay frozen.
            # 强制特征蒸馏模型保持冻结。
            if 'freeze_grad' not in backbone_config:
                backbone_config['freeze_grad'] = True
                self.logger.info(f'[DACS] Added freeze_grad=True to backbone_config')
            
            # Keep an explicit warning when the required pure-backbone config is absent.
            # 当 pure backbone 所需配置缺失时保留显式告警。
            if 'dinov3_config' not in backbone_config:
                self.logger.warning(f'[DACS] dinov3_config not found in backbone_config')
        
        # Keep only backbone pretraining needed by the frozen DINOv3 model.
        # 只保留冻结 DINOv3 模型需要的 backbone 预训练权重。
        backbone_pretrained = fznet_cfg.get('backbone_pretrained', None)
        
        if backbone_pretrained is not None:
            if isinstance(backbone_pretrained, dict):
                # Keep DINOv3 weights and drop adapter/lora/reins weights.
                # 保留 DINOv3 权重，移除 adapter/lora/reins 等权重。
                if 'dinov3' in backbone_pretrained:
                    dinov3_pretrained = backbone_pretrained['dinov3']
                    fznet_cfg['backbone_pretrained'] = {'dinov3': dinov3_pretrained}
                    self.logger.info(f'[DACS] Kept only dinov3 pretrained: {dinov3_pretrained}')
                else:
                    # The whole value may already point to DINOv3 weights.
                    # 没有 dinov3 键时，整个值可能已经是 DINOv3 权重路径。
                    self.logger.warning(f'[DACS] dinov3 key not found in backbone_pretrained, keeping original')
            elif isinstance(backbone_pretrained, str):
                # Normalize a direct path into the dict format expected downstream.
                # 将直接路径规范化为下游期望的字典格式。
                fznet_cfg['backbone_pretrained'] = {'dinov3': backbone_pretrained}
                self.logger.info(f'[DACS] Converted string pretrained to dict format')
        
        # Decoder weights should be learned by the student, not copied into fznet.
        # decoder 权重应由 student 学习，不复制到 fznet。
        if 'decoder_pretrained' in fznet_cfg:
            original_decoder_pretrained = fznet_cfg['decoder_pretrained']
            fznet_cfg['decoder_pretrained'] = None
            if original_decoder_pretrained is not None:
                self.logger.info(f'[DACS] Cleared decoder_pretrained (was: {original_decoder_pretrained})')
        
        # Remove EMA-only pretrained fields from the frozen-model config.
        # 从冻结模型配置中移除仅供 EMA 使用的预训练字段。
        ema_keys_to_remove = ['ema_backbone_pretrained', 'ema_decoder_pretrained']
        for key in ema_keys_to_remove:
            if key in fznet_cfg:
                del fznet_cfg[key]
                self.logger.info(f'[DACS] Removed {key} from fznet_cfg')
        
        # Other configs such as decode_head, aux_head, train_cfg and test_cfg stay unchanged.
        # decode_head、aux_head、train_cfg、test_cfg 等其他配置保持原样。

        # Log the converted config summary for experiment reproducibility.
        # 记录转换后的配置摘要，便于复现实验。
        self.logger.info(f'[DACS] Fznet model config prepared:')
        self.logger.info(f'  - Backbone type: {fznet_cfg["backbone"]["type"]}')
        self.logger.info(f'  - Freeze grad: {backbone_config.get("freeze_grad", "Not set")}')
        self.logger.info(f'  - Backbone pretrained: {fznet_cfg.get("backbone_pretrained", "None")}')
        self.logger.info(f'  - Decoder pretrained: {fznet_cfg.get("decoder_pretrained", "None")}')
        
        return fznet_cfg

    def _detach_ema_weights(self):
        if self.source_only:
            return
        
        
        for param in self.get_ema_model().parameters():
            param.detach_()
        self.logger.info('EMA model parameters detached.')

    def _init_ema_weights(self):
        if self.source_only:
            return
        
        self.logger.info('Initializing EMA model weights from the current model weights...')
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()
                
    def _update_ema(self, iter):
        if self.source_only:
            return
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]
    
    def eval_ema(self, val_loader, rescale=True):
        """对ema_model（教师模型）在验证集上推理并计算IoU和mIoU"""
        self.logger.info(f'Start evaluating EMA model, mode: {self.test_cfg["mode"]}')
        
        # # 设置ema模型为评估模式
        was_training = self.get_ema_model().training
        self.logger.info(f'EMA model training mode: {was_training}')
        self.get_ema_model().eval()
        
        hist = np.zeros((self.num_classes, self.num_classes))
        split_order, eval_splits, split_name = get_dataset_eval_splits(val_loader.dataset)
        split_hists = None
        split_counts = None
        if eval_splits is not None:
            split_hists = {
                split: np.zeros((self.num_classes, self.num_classes))
                for split in split_order
            }
            split_counts = {split: 0 for split in split_order}
            self.logger.info(f'{split_name} split-wise EMA evaluation enabled: all/' + '/'.join(split_order))
        self.last_eval_ious_by_scene = None
        self.last_eval_split_order = split_order
        self.last_eval_scene_counts = None
        sample_offset = 0
        
        device_predictions = []
        device_labels = []
        
        # set Dropout and DropPath to eval mode
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        try:
            with torch.no_grad():
                for val_data in tqdm(val_loader, total=len(val_loader)):
                    im = val_data['im'].cuda(non_blocking=True)
                    lb = val_data['lb']

                    im_input = {'img': im, 'lb_shape': lb.size()[1:3]}
                    
                    # 用ema模型推理
                    pred_logit = self.get_ema_model().inference(im_input, rescale=rescale)
                    
                    if isinstance(pred_logit, dict):
                        pred_logit = pred_logit.get('seg_logits', pred_logit.get('S'))

                    pred = torch.argmax(pred_logit, dim=1)
                    if split_hists is not None:
                        pred_np = pred.cpu().numpy()
                        lb_np = lb.numpy()
                        batch_size = pred_np.shape[0]
                        for batch_idx in range(batch_size):
                            split_idx = sample_offset + batch_idx
                            if split_idx >= len(eval_splits):
                                continue
                            split = eval_splits[split_idx]
                            if split not in split_hists:
                                continue
                            split_hists[split] += self.compute_hist(
                                pred_np[batch_idx:batch_idx + 1],
                                lb_np[batch_idx:batch_idx + 1])
                            split_counts[split] += 1
                        sample_offset += batch_size
                    
                    device_predictions.append(pred)
                    device_labels.append(lb)
                    
                    if len(device_predictions) >= 10:
                        self._process_batch_hist(device_predictions, device_labels, hist)
                        device_predictions.clear()
                        device_labels.clear()
                
                if device_predictions:
                    self._process_batch_hist(device_predictions, device_labels, hist)
                    
        finally:
            # pass
            if was_training:
                self.get_ema_model().train()
                # set requires_grad to False for all parameters
                for param in self.get_ema_model().parameters():
                    param.detach_()

        denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
        iu = np.diag(hist) / np.maximum(denominator, 1) * 100
        mean_iu = np.nanmean(iu)
        if split_hists is not None:
            self.last_eval_ious_by_scene = {
                split: compute_ious_from_hist(split_hists[split])
                for split in split_order
            }
            self.last_eval_scene_counts = split_counts
        
        del device_predictions, device_labels, hist
        torch.cuda.empty_cache()

        return mean_iu, iu
    
    def get_src_cls_mix_ratio(self):
        """
        Calculate the source class mix ratio based on the current iteration.

        This function computes the source class mix ratio, which decays over the
        training iterations. The decay mode can be linear, exponential (fast to slow),
        or exponential (slow to fast).

        Returns:
            float: The calculated source class mix ratio.
        """
        init_value = self.src_mix_cls_ratio_init
        final_value = self.src_mix_cls_ratio_final
        current_iter = self.local_iter
        total_iter = self.max_iters
        mode = self.ratio_mode

        if mode == "constant":
            return init_value
        elif mode == "linear":
            return init_value + (final_value - init_value) * (current_iter / total_iter)
        elif mode in ["exp1", "exp2"]:
            # 共同的参数和变量
            progress = current_iter / total_iter  # 训练进度 [0, 1]
            alpha = 4.0  # 控制衰减速度
            
            if mode == "exp1":  # Exponential decay (fast to slow) - 先快后慢
                # exp_factor 从 0 快速增加到接近 1，然后缓慢接近 1（先快后慢）
                exp_factor = 1 - math.exp(-alpha * progress)
            else:  # mode == "exp2"  # Exponential decay (slow to fast) - 先慢后快
                # exp_factor 从 0 缓慢增加，然后快速增加到接近 1（先慢后快）
                exp_factor = (math.exp(alpha * progress) - 1) / (math.exp(alpha) - 1)
            
            return init_value + (final_value - init_value) * exp_factor
        else:
            raise ValueError("Invalid mode. Choose from 'linear', 'exp1', or 'exp2'.")

    def train_step(self, data_batch, valid_pseudo_mask=None):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        log_vars = self.forward_train_step(data_batch, valid_pseudo_mask)
        # torch.cuda.empty_cache()
        return log_vars

    def masked_feat_dist(self, f1, f2, mask=None):
        """Compute masked L2 feature distance with empty/invalid-loss guards.

        计算带 mask 的 L2 特征距离，并处理空 mask 或 NaN/Inf loss。
        """
        feat_diff = f1 - f2
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        if mask is not None:
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]  # [mask_valid_index]
        
        # ⚠️ 检查是否有有效元素，避免空 tensor 的 mean
        if pw_feat_dist.numel() == 0:
            return torch.tensor(0.0, device=f1.device, requires_grad=True)
        
        result = torch.mean(pw_feat_dist)
        
        # ⚠️ 检查结果是否为 NaN，如果是则返回 0
        if torch.isnan(result) or torch.isinf(result):
            return torch.tensor(0.0, device=f1.device, requires_grad=True)
        
        return result
    
    def compute_feature_similarity(self, feat1, feat2, sim_type='cosine', temperature=0.07):
        """Compute pairwise feature similarity matrix.

        计算特征之间的两两相似性矩阵。

        Args:
            feat1: First feature map, shape `(B, C, H, W)`.
                第一个特征图，形状为 `(B, C, H, W)`。
            feat2: Second feature map, shape `(B, C, H, W)`.
                第二个特征图，形状为 `(B, C, H, W)`。
            sim_type: Similarity type, either `'cosine'` or `'l2'`.
                相似性类型，可选 `'cosine'` 或 `'l2'`。
            temperature: Temperature used to scale similarity values.
                用于缩放相似性的温度系数。

        Returns:
            Tensor: Similarity matrix, shape `(B, H*W, H*W)`.
                相似性矩阵，形状为 `(B, H*W, H*W)`。
        """
        B, C, H, W = feat1.shape
        
        # Reshape to (B, C, H*W)
        feat1_flat = feat1.view(B, C, -1)  # (B, C, H*W)
        feat2_flat = feat2.view(B, C, -1)  # (B, C, H*W)
        
        if sim_type == 'cosine':
            # L2 归一化
            feat1_norm = F.normalize(feat1_flat, p=2, dim=1)  # (B, C, H*W)
            feat2_norm = F.normalize(feat2_flat, p=2, dim=1)  # (B, C, H*W)
            
            # 计算余弦相似性: (B, H*W, H*W)
            sim_matrix = torch.bmm(feat1_norm.transpose(1, 2), feat2_norm)  # (B, H*W, H*W)
            
        elif sim_type == 'l2':
            # 计算L2距离的负数作为相似性
            # 扩展维度: feat1 -> (B, H*W, 1, C), feat2 -> (B, 1, H*W, C)
            feat1_expand = feat1_flat.transpose(1, 2).unsqueeze(2)  # (B, H*W, 1, C)
            feat2_expand = feat2_flat.transpose(1, 2).unsqueeze(1)  # (B, 1, H*W, C)
            
            # 计算L2距离: (B, H*W, H*W)
            l2_dist = torch.norm(feat1_expand - feat2_expand, p=2, dim=3)  # (B, H*W, H*W)
            sim_matrix = -l2_dist  # 负距离作为相似性
            
        else:
            raise ValueError(f"Unknown similarity type: {sim_type}")
        
        # 应用温度缩放
        sim_matrix = sim_matrix / temperature
        
        return sim_matrix
    
    def feature_similarity_loss(self, sim_student, sim_teacher, mask=None):
        """Compute MSE loss between student and teacher similarity matrices.

        计算 student 与 teacher 相似性矩阵之间的 MSE loss。

        Args:
            sim_student: Student similarity matrix, shape `(B, H*W, H*W)`.
                student 相似性矩阵，形状为 `(B, H*W, H*W)`。
            sim_teacher: Teacher similarity matrix, shape `(B, H*W, H*W)`.
                teacher 相似性矩阵，形状为 `(B, H*W, H*W)`。
            mask: Optional class mask, shape `(B, H, W)`.
                可选类别 mask，形状为 `(B, H, W)`。

        Returns:
            Tensor: Similarity distillation loss.
                相似性蒸馏 loss。
        """
        B = sim_student.shape[0]
        
        if mask is not None:
            # 将mask展平并扩展
            mask_flat = mask.view(B, -1)  # (B, H*W)
            
            # 创建二维mask: (B, H*W, H*W)
            mask_2d = mask_flat.unsqueeze(2) * mask_flat.unsqueeze(1)  # (B, H*W, H*W)
            
            # 只计算有效区域的损失
            valid_elements = mask_2d.sum()
            if valid_elements > 0:
                # MSE loss
                loss = F.mse_loss(
                    sim_student * mask_2d, 
                    sim_teacher * mask_2d, 
                    reduction='sum'
                ) / valid_elements
            else:
                loss = torch.tensor(0.0, device=sim_student.device)
        else:
            # 全局MSE loss
            loss = F.mse_loss(sim_student, sim_teacher, reduction='mean')
        
        return loss

    def _is_hrda_multiscale_feature_mode(self):
        """Return whether the current model exposes HRDA multi-scale features.

        判断当前模型是否处于 HRDA 多尺度特征模式。
        """
        return isinstance(self.get_model(), HRDAEncoderDecoder) and \
            self.get_model().feature_scale in self.get_model().feature_scale_all_strs

    @staticmethod
    def _select_feature_output(feat):
        """Normalize model feature output by selecting the feature-list payload.

        规范化模型特征输出：当输出为 tuple 时取其中的特征列表部分。
        """
        if isinstance(feat, tuple):
            return feat[1]
        return feat

    def _extract_frozen_features(self, img, select_feature_output=False):
        """Extract detached features from the frozen distillation model.

        从冻结蒸馏模型中提取 detached 特征。
        """
        with torch.no_grad():
            self.get_fz_model().eval()
            feat_fznet = self.get_fz_model().extract_feat(img)
            if select_feature_output:
                feat_fznet = self._select_feature_output(feat_fznet)
            feat_fznet = [f.detach() for f in feat_fznet]
        return feat_fznet

    def _extract_vplf_target_feature(self, img):
        """Extract the configured frozen VFM feature map for VPLF.

        为 VPLF 提取配置指定的冻结 VFM 特征图。
        """
        with torch.no_grad():
            self.get_fz_model().eval()
            features = self.get_fz_model().extract_feat(img)
            feature = select_vfm_feature_map(features, self.vplf_cfg['feature_layer'])
        return feature.detach()

    def _apply_vplf_filter(self, tar_img, pseudo_label, pseudo_conf, pseudo_weight):
        """Multiply target pseudo weights by VFM prototype reliability.

        使用 VFM 原型可靠性权重重新加权目标域伪标签权重。
        """
        if not self.vfm_pl_filter_enabled:
            return pseudo_weight, {}

        target_feature = self._extract_vplf_target_feature(tar_img)
        original_weight = pseudo_weight.detach().clone()
        final_weight, vfm_weight, sim_yhat, disagree, vplf_log_vars = self.vfm_pl_filter(
            target_feature,
            pseudo_label,
            pseudo_conf,
            pseudo_weight,
        )

        iteration = self.local_iter + 1
        stats_interval = int(self.vplf_class_stats_interval or 0)
        if self._is_master_process() and stats_interval > 0 and iteration % stats_interval == 0:
            class_rows = compute_vplf_class_stats(
                pseudo_label,
                pseudo_conf,
                original_weight,
                vfm_weight,
                final_weight,
                sim_yhat,
                disagree,
                class_names=getattr(self.vfm_pl_filter, 'class_names', None),
            )
            append_vplf_class_stats_csv(
                self.vplf_class_stats_path,
                iteration,
                class_rows,
            )

        debug_interval = int(self.vplf_debug_img_interval or 0)
        if self._is_master_process() and debug_interval > 0 and iteration % debug_interval == 0:
            self.vplf_debug_data = {
                'pseudo_label': pseudo_label.detach(),
                'pseudo_conf': pseudo_conf.detach(),
                'original_weight': original_weight.detach(),
                'vfm_weight': vfm_weight.detach(),
                'final_weight': final_weight.detach(),
                'sim_yhat': sim_yhat.detach(),
                'disagree': disagree.detach(),
            }
        else:
            self.vplf_debug_data = None

        log_interval = int(self.vplf_cfg['log_interval'])
        if self._is_master_process() and log_interval > 0 and iteration % log_interval == 0:
            self.logger.info(
                '[VPLF] iter=%d pseudo_conf=%.4f orig_w=%.4f '
                'vfm_w=%.4f final_w=%.4f sim_yhat=%.4f '
                'disagree=%.4f keep=%.4f->%.4f',
                iteration,
                vplf_log_vars['vplf_pseudo_conf_mean'],
                vplf_log_vars['vplf_original_pseudo_weight_mean'],
                vplf_log_vars['vplf_vfm_weight_mean'],
                vplf_log_vars['vplf_final_pseudo_weight_mean'],
                vplf_log_vars['vplf_vfm_sim_yhat_mean'],
                vplf_log_vars['vplf_vfm_disagree_rate'],
                vplf_log_vars['vplf_original_keep_rate'],
                vplf_log_vars['vplf_final_keep_rate'],
            )

        return final_weight, vplf_log_vars

    def _build_fdist_mask(self, gt, feature, layer_idx, scale_idx=None):
        """Build a class-filter mask for one feature-distillation layer.

        为单个特征蒸馏层构造类别过滤 mask。
        """
        fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
        gt_rescaled = gt.clone()
        if scale_idx is not None and scale_idx in HRDAEncoderDecoder.last_train_crop_box:
            gt_rescaled = crop(gt_rescaled, HRDAEncoderDecoder.last_train_crop_box[scale_idx])

        scale_factor = gt_rescaled.shape[-1] // feature.shape[-1]
        gt_rescaled = downscale_label_ratio(
            gt_rescaled,
            scale_factor,
            self.fdist_scale_min_ratio,
            self.num_classes,
            255,
        ).long().detach()
        fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)

        is_first_layer = layer_idx == self.fdist_layers[0]
        is_first_scale = scale_idx is None or scale_idx == 0
        if is_first_layer and is_first_scale:
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled

        return fdist_mask

    def _parse_fdist_loss(self, total_loss, img):
        """Average, weight, sanitize, and parse a feature-distillation loss.

        对特征蒸馏 loss 做多层平均、乘权重、防 NaN/Inf，并解析为日志格式。
        """
        avg_loss = total_loss / len(self.fdist_layers)
        avg_loss = self.fdist_lambda * avg_loss

        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
            avg_loss = torch.tensor(0.0, device=img.device, requires_grad=True)

        feat_loss, feat_log = parse_losses({'froz_feat_dist_loss': avg_loss})
        feat_log.pop('loss', None)
        return feat_loss, feat_log
    
    # ✅ DACS 原始版本 (type=1)
    def calc_feat_dist_dacs(self, img, gt, feat=None):
        """Calculate original DACS feature distillation loss.

        计算原始 DACS 特征蒸馏 loss，支持多层特征。

        Args:
            img: Input images. 输入图像。
            gt: Ground-truth labels. 真实标签。
            feat: Student feature list. student 模型提取的特征列表。

        Returns:
            tuple: Feature-distillation loss and scalar log dict.
                特征蒸馏 loss 以及标量日志字典。
        """
        total_feat_dist = 0
        feat_fznet = self._extract_frozen_features(img)

        # HRDA multi-scale features. / HRDA 多尺度特征。
        if self._is_hrda_multiscale_feature_mode():
            for layer_idx in self.fdist_layers:
                layer_feat = [f[layer_idx] for f in feat]
                layer_feat_fznet = [f[layer_idx] for f in feat_fznet]

                layer_dist = 0
                for s in range(len(layer_feat_fznet)):
                    if self.fdist_classes is not None:
                        fdist_mask = self._build_fdist_mask(
                            gt,
                            layer_feat[s],
                            layer_idx,
                            scale_idx=s,
                        )
                        fd_s = self.masked_feat_dist(
                            layer_feat[s],
                            layer_feat_fznet[s],
                            fdist_mask,
                        )
                        layer_dist += fd_s
                    else:
                        raise NotImplementedError

                total_feat_dist += layer_dist

        # Single-scale features. / 单尺度特征。
        else:
            for layer_idx in self.fdist_layers:
                if self.fdist_classes is not None:
                    fdist_mask = self._build_fdist_mask(gt, feat[layer_idx], layer_idx)
                    layer_dist = self.masked_feat_dist(
                        feat[layer_idx],
                        feat_fznet[layer_idx],
                        fdist_mask,
                    )
                else:
                    layer_dist = self.masked_feat_dist(feat[layer_idx], feat_fznet[layer_idx])

                total_feat_dist += layer_dist

        return self._parse_fdist_loss(total_feat_dist, img)
    
    # ✅ 支持 DINOv3 的版本 (type=2)
    def calc_feat_dist_dinov3(self, img, gt, feat=None):
        """Calculate DINOv3-aware feature distillation loss.

        计算适配 DINOv3 特征格式的特征蒸馏 loss，支持多层特征。

        Args:
            img: Input images. 输入图像。
            gt: Ground-truth labels, or `None` for target-domain features.
                真实标签；目标域特征蒸馏时可为 `None`。
            feat: Student feature output. student 模型提取的特征输出。

        Returns:
            tuple: Feature-distillation loss and scalar log dict.
                特征蒸馏 loss 以及标量日志字典。
        """
        total_feat_dist = 0
        feat = self._select_feature_output(feat)
        feat_fznet = self._extract_frozen_features(img, select_feature_output=True)
        proj_head = getattr(self, "proj_head", None)

        # HRDA multi-scale features. / HRDA 多尺度特征。
        if self._is_hrda_multiscale_feature_mode():
            for layer_idx in self.fdist_layers:
                layer_feat = [f[layer_idx] for f in feat]
                layer_feat_fznet = [f[layer_idx] for f in feat_fznet]

                if isinstance(proj_head, torch.nn.Module):
                    layer_feat = [proj_head(f) for f in layer_feat]
                    layer_feat_fznet = [F.normalize(f, p=2, dim=1, eps=1e-12) for f in layer_feat_fznet]

                layer_dist = 0
                for s in range(len(layer_feat)):
                    if self.fdist_classes is not None and gt is not None:
                        fdist_mask = self._build_fdist_mask(
                            gt,
                            layer_feat[s],
                            layer_idx,
                            scale_idx=s,
                        )
                        fd_s = self.masked_feat_dist(
                            layer_feat[s],
                            layer_feat_fznet[s],
                            fdist_mask,
                        )
                        layer_dist += fd_s
                    else:
                        # Target-domain path without labels.
                        # 目标域无标签路径。
                        if isinstance(proj_head, torch.nn.Module):
                            fd_s = proj_head.align_loss(layer_feat[s], layer_feat_fznet[s])
                        else:
                            fd_s = self.masked_feat_dist(layer_feat[s], layer_feat_fznet[s])
                        layer_dist += fd_s

                total_feat_dist += layer_dist

        # Single-scale features. / 单尺度特征。
        else:
            for layer_idx in self.fdist_layers:
                layer_feat = feat[layer_idx]
                layer_feat_fznet = feat_fznet[layer_idx]

                if isinstance(proj_head, torch.nn.Module):
                    layer_feat = proj_head(layer_feat)
                    layer_feat_fznet = F.normalize(layer_feat_fznet, p=2, dim=1, eps=1e-12)

                if self.fdist_classes is not None and gt is not None:
                    fdist_mask = self._build_fdist_mask(gt, layer_feat, layer_idx)
                    layer_dist = self.masked_feat_dist(layer_feat, layer_feat_fznet, fdist_mask)
                else:
                    # Target-domain path without labels.
                    # 目标域无标签路径。
                    if isinstance(proj_head, torch.nn.Module):
                        layer_dist = proj_head.align_loss(layer_feat, layer_feat_fznet)
                    else:
                        layer_dist = self.masked_feat_dist(layer_feat, layer_feat_fznet)

                total_feat_dist += layer_dist

        return self._parse_fdist_loss(total_feat_dist, img)
    
    # ✅ 新增: 特征相似性蒸馏版本 (type=3)
    def calc_feat_dist_similarity(self, img, gt, feat=None):
        """Calculate feature-similarity distillation loss.

        计算基于特征相似性关系的蒸馏 loss。

        Instead of matching raw feature values, this mode matches pairwise
        feature relationships so features can adapt to segmentation while their
        internal similarity structure stays stable.

        该模式不直接约束特征值本身，而是约束特征之间的两两相似性关系，
        让特征能适配分割任务，同时保持内部关系结构稳定。

        Args:
            img: Input images. 输入图像。
            gt: Ground-truth labels, or `None` for target-domain features.
                真实标签；目标域特征蒸馏时可为 `None`。
            feat: Student feature output. student 模型提取的特征输出。

        Returns:
            tuple: Feature-similarity distillation loss and scalar log dict.
                特征相似性蒸馏 loss 以及标量日志字典。
        """
        total_sim_loss = 0
        feat = self._select_feature_output(feat)
        feat_fznet = self._extract_frozen_features(img, select_feature_output=True)
        proj_head = getattr(self, "proj_head", None)

        # HRDA multi-scale features. / HRDA 多尺度特征。
        if self._is_hrda_multiscale_feature_mode():
            for layer_idx in self.fdist_layers:
                layer_feat = [f[layer_idx] for f in feat]
                layer_feat_fznet = [f[layer_idx] for f in feat_fznet]

                if isinstance(proj_head, torch.nn.Module):
                    layer_feat = [proj_head(f) for f in layer_feat]
                    layer_feat_fznet = [proj_head(f) for f in layer_feat_fznet]

                layer_sim_loss = 0
                for s in range(len(layer_feat)):
                    sim_student = self.compute_feature_similarity(
                        layer_feat[s], layer_feat[s],
                        sim_type=self.fdist_sim_type,
                        temperature=self.fdist_sim_temp,
                    )

                    with torch.no_grad():
                        sim_teacher = self.compute_feature_similarity(
                            layer_feat_fznet[s], layer_feat_fznet[s],
                            sim_type=self.fdist_sim_type,
                            temperature=self.fdist_sim_temp,
                        ).detach()

                    if self.fdist_classes is not None and gt is not None:
                        fdist_mask = self._build_fdist_mask(
                            gt,
                            layer_feat[s],
                            layer_idx,
                            scale_idx=s,
                        )
                    else:
                        fdist_mask = None

                    sim_loss_s = self.feature_similarity_loss(
                        sim_student,
                        sim_teacher,
                        fdist_mask,
                    )
                    layer_sim_loss += sim_loss_s

                total_sim_loss += layer_sim_loss

        # Single-scale features. / 单尺度特征。
        else:
            for layer_idx in self.fdist_layers:
                layer_feat = feat[layer_idx]
                layer_feat_fznet = feat_fznet[layer_idx]

                if isinstance(proj_head, torch.nn.Module):
                    layer_feat = proj_head(layer_feat)
                    layer_feat_fznet = proj_head(layer_feat_fznet)

                sim_student = self.compute_feature_similarity(
                    layer_feat, layer_feat,
                    sim_type=self.fdist_sim_type,
                    temperature=self.fdist_sim_temp,
                )

                with torch.no_grad():
                    sim_teacher = self.compute_feature_similarity(
                        layer_feat_fznet, layer_feat_fznet,
                        sim_type=self.fdist_sim_type,
                        temperature=self.fdist_sim_temp,
                    ).detach()

                if self.fdist_classes is not None and gt is not None:
                    fdist_mask = self._build_fdist_mask(gt, layer_feat, layer_idx)
                else:
                    fdist_mask = None

                layer_sim_loss = self.feature_similarity_loss(
                    sim_student,
                    sim_teacher,
                    fdist_mask,
                )
                total_sim_loss += layer_sim_loss

        return self._parse_fdist_loss(total_sim_loss, img)
    
    # ✅ 统一的特征蒸馏接口
    def calc_feat_dist(self, img, gt, feat=None):
        """Dispatch feature distillation to the configured implementation.

        根据配置中的蒸馏类型调用对应的特征蒸馏实现。

        Args:
            img: Input images. 输入图像。
            gt: Labels for source-domain distillation, or `None` for target.
                源域蒸馏标签；目标域蒸馏时可为 `None`。
            feat: Student feature output. student 模型提取的特征输出。

        Returns:
            tuple: Feature-distillation loss and scalar log dict.
                特征蒸馏 loss 以及标量日志字典。
        """
        assert self.enable_fdist, "Feature distillation is not enabled"
        
        fdist_methods = {
            1: self.calc_feat_dist_dacs,
            2: self.calc_feat_dist_dinov3,
            3: self.calc_feat_dist_similarity,
        }
        if self.fdist_type not in fdist_methods:
            raise ValueError(f"Unknown fdist_type: {self.fdist_type}. Must be 1, 2, or 3.")

        return fdist_methods[self.fdist_type](img, gt, feat)
    
    def update_debug_state(self):
        """Toggle debug collection on student, EMA teacher, and MIC modules.

        根据当前 iteration 为 student、EMA teacher 和 MIC 模块开关调试输出。
        """
        debug = self._is_master_process() and self.debug_img_interval > 0 and \
            (self.local_iter + 1) % self.debug_img_interval == 0
        self.get_model().decode_head.debug = debug
        if not self.source_only:
            self.get_ema_model().decode_head.debug = debug
        if self.mic is not None:
            self.mic.debug = debug
            
    def style_consistency(self, x):
        # Obtained from: https://github.com/HeliosZhao/SHADE/blob/ea4214ad4eaa1ba2210656bff315afb6c6f50e28/train.py#L424  # noqa
        outputs_sm = F.softmax(x, dim=1)
        # 2B,C,H,W first B is x, last B is x_new
        B = outputs_sm.shape[0] // 2
        im_prob = outputs_sm[:B]
        aug_prob = outputs_sm[B:]

        aug_prob = aug_prob.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        im_prob = im_prob.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

        p_mixture = torch.clamp((aug_prob + im_prob) / 2., 1e-7, 1).log()
        consistency_loss = self.style_consistency_lambda * (
            F.kl_div(p_mixture, aug_prob, reduction='batchmean') +
            F.kl_div(p_mixture, im_prob, reduction='batchmean')) / 2.

        sc_loss, sc_log = parse_losses(
            {'style_consistency_loss': consistency_loss})
        sc_log.pop('loss', None)

        return sc_loss, sc_log
    
    def get_pseudo_label_and_weight(self, logits):
        """
        Generate pseudo labels and weights based on the logits.

        This function computes the pseudo labels and their corresponding weights
        from the logits. It also identifies the regions where the pseudo probability
        is greater than or equal to the threshold.

        Args:
            logits (torch.Tensor): Logits from the model, shape [batch, channels, height, width].

        Returns:
            pseudo_label (torch.Tensor): Pseudo labels, shape [batch, height, width].
            pseudo_weight (torch.Tensor): Weights for the pseudo labels, shape [batch, height, width].
            ps_large_p (torch.Tensor): Binary mask indicating regions with high pseudo probability, shape [batch, height, width].
            pseudo_prob (torch.Tensor): EMA confidence of the selected pseudo label.
                EMA teacher 对所选伪标签类别的置信度。
        """
        # Apply softmax to logits to get probabilities
        ema_softmax = torch.softmax(logits.detach(), dim=1)  # [batch, channels, height, width]

        # Get the maximum probability and corresponding label for each pixel
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)  # [batch, height, width], [batch, height, width]

        # Identify regions where the pseudo probability is greater than or equal to the threshold
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1  # [batch, height, width]

        # Calculate the weight for the pseudo labels for each sample in the batch
        batch_size = logits.size(0)
        pseudo_weight = torch.zeros_like(pseudo_prob)  # Initialize pseudo_weight with zeros
        for i in range(batch_size):
            ps_size = pseudo_label[i].numel()  # Number of elements in the pseudo label for the current sample
            weight = torch.sum(ps_large_p[i]).item() / ps_size  # Calculate weight for the current sample
            pseudo_weight[i] = weight * torch.ones_like(pseudo_prob[i])  # Assign weight to the corresponding sample
        
        # old batch-wise pseudo weight calculation
        """
        ps_size = np.size(np.array(pseudo_label.cpu()))
        num_pixels_higher_than_threshold = torch.sum(ps_large_p).item()
        pseudo_weight = num_pixels_higher_than_threshold / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=logits.device)
        """
        
        return pseudo_label, pseudo_weight, ps_large_p, pseudo_prob
    
    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask
        return pseudo_weight

    def _update_teacher_and_mic_state(self):
        """Update EMA teacher weights and MIC teacher weights for this iteration.

        更新当前 iteration 的 EMA teacher 权重以及 MIC 模块中的 teacher 权重。
        """
        if self.local_iter == 0:
            self._detach_ema_weights()

        if self.local_iter == self.ema_begin_iter:
            self._init_ema_weights()

        if self.local_iter > self.ema_begin_iter:
            self._update_ema(self.local_iter - self.ema_begin_iter)

        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

    def _build_strong_parameters(self, batch_size, dev):
        """Build shared strong-augmentation parameters for DACS mixing.

        构建 DACS 混合增强中复用的强增强参数。
        """
        means, stds = get_mean_std_self(self.img_mean, self.img_std, batch_size, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # Same normalization for the batch. / batch 内使用相同归一化参数。
            'std': stds[0].unsqueeze(0),
        }
        return means, stds, strong_parameters

    def _disable_ema_stochastic_layers(self):
        """Disable stochastic layers in the EMA teacher during pseudo labeling.

        在生成伪标签时关闭 EMA teacher 中的随机层，保证 teacher 输出稳定。
        """
        for module in self.get_ema_model().modules():
            if isinstance(module, _DropoutNd):
                module.training = False
            if isinstance(module, DropPath):
                module.training = False

    def _init_offline_teacher_pseudo_config(self):
        """Read the optional offline strong-teacher pseudo-label settings."""
        cfg = self.semi_cfg.get('offline_teacher_pseudo', {}) or {}
        self.offline_teacher_pseudo_enabled = bool(cfg.get('enabled', False))
        self.offline_teacher_pseudo_mode = str(
            cfg.get('mode', 'teacher_then_ema')).lower()
        self.offline_teacher_pseudo_weight = float(cfg.get('weight', 1.0))
        self.offline_teacher_confidence_value = float(
            cfg.get('confidence_value', 0.95))
        self.offline_teacher_use_continuous_confidence = bool(
            cfg.get('use_continuous_confidence', False))
        self.offline_teacher_confidence_power = float(
            cfg.get('confidence_power', 1.0))
        self.offline_teacher_class_weights = {
            int(key): float(value)
            for key, value in cfg.get('class_weights', {}).items()
        }
        self.offline_teacher_min_coverage = float(
            cfg.get('min_coverage', 0.0))
        valid_modes = ('teacher_only', 'teacher_then_ema')
        if self.offline_teacher_pseudo_mode not in valid_modes:
            raise ValueError(
                'offline_teacher_pseudo.mode must be one of '
                f'{valid_modes}, got {self.offline_teacher_pseudo_mode!r}')
        if not 0.0 <= self.offline_teacher_confidence_value <= 1.0:
            raise ValueError(
                'offline_teacher_pseudo.confidence_value must be in [0, 1]')
        if self.offline_teacher_confidence_power <= 0:
            raise ValueError(
                'offline_teacher_pseudo.confidence_power must be positive')
        if self.offline_teacher_pseudo_enabled:
            self.logger.info(
                '[OfflineTeacher] Enabled: mode=%s, weight=%.3f, '
                'confidence_value=%.3f, continuous=%s, power=%.3f, '
                'class_weights=%s, min_coverage=%.3f',
                self.offline_teacher_pseudo_mode,
                self.offline_teacher_pseudo_weight,
                self.offline_teacher_confidence_value,
                self.offline_teacher_use_continuous_confidence,
                self.offline_teacher_confidence_power,
                self.offline_teacher_class_weights,
                self.offline_teacher_min_coverage,
            )

    def _merge_offline_teacher_pseudo(
            self, pseudo_label, pseudo_weight, pseudo_mask, pseudo_conf,
            offline_teacher_label, offline_teacher_confidence=None):
        """Fuse confidence-filtered offline labels with the online EMA state."""
        if not self.offline_teacher_pseudo_enabled:
            return pseudo_label, pseudo_weight, pseudo_mask, pseudo_conf, {}
        if offline_teacher_label is None:
            raise ValueError(
                'Offline teacher pseudo labels are enabled, but the target '
                'batch does not contain them. Check data.target(_unlabeled).'
                'offline_teacher_pseudo.')

        teacher_label = offline_teacher_label.detach().long()
        if teacher_label.dim() == 4 and teacher_label.shape[1] == 1:
            teacher_label = teacher_label.squeeze(1)
        if teacher_label.shape[-2:] != pseudo_label.shape[-2:]:
            teacher_label = F.interpolate(
                teacher_label.unsqueeze(1).float(),
                size=pseudo_label.shape[-2:],
                mode='nearest',
            ).squeeze(1).long()
        teacher_label = teacher_label.to(device=pseudo_label.device)
        teacher_valid = (
            teacher_label.ge(0)
            & teacher_label.lt(int(self.num_classes))
        )
        coverage = teacher_valid.float().mean()
        ema_valid = pseudo_mask.bool()
        overlap = teacher_valid & ema_valid
        if overlap.any():
            agreement = (
                teacher_label[overlap] == pseudo_label[overlap]
            ).float().mean()
        else:
            agreement = coverage.new_tensor(0.0)

        logs = {
            'offline_teacher_coverage': float(coverage.item()),
            'offline_teacher_ema_overlap': float(overlap.float().mean().item()),
            'offline_teacher_ema_agreement': float(agreement.item()),
        }
        if coverage.item() < self.offline_teacher_min_coverage:
            logs['offline_teacher_skipped_low_coverage'] = 1.0
            return pseudo_label, pseudo_weight, pseudo_mask, pseudo_conf, logs

        if self.offline_teacher_use_continuous_confidence:
            if offline_teacher_confidence is None:
                raise ValueError(
                    'Continuous offline-teacher confidence is enabled, but '
                    'the target batch has no aligned confidence map.')
            confidence_value = offline_teacher_confidence.detach()
            if confidence_value.dim() == 4 and confidence_value.shape[1] == 1:
                confidence_value = confidence_value.squeeze(1)
            if confidence_value.shape[-2:] != pseudo_label.shape[-2:]:
                confidence_value = F.interpolate(
                    confidence_value.unsqueeze(1).float(),
                    size=pseudo_label.shape[-2:],
                    mode='nearest',
                ).squeeze(1)
            confidence_value = confidence_value.to(
                device=pseudo_label.device, dtype=pseudo_conf.dtype)
            if confidence_value.max().item() > 1.0:
                confidence_value = confidence_value / 255.0
            confidence_value = confidence_value.clamp(0.0, 1.0)
        else:
            confidence_value = pseudo_conf.new_full(
                pseudo_conf.shape,
                self.offline_teacher_confidence_value,
            )

        teacher_pixel_weight = (
            confidence_value.pow(self.offline_teacher_confidence_power)
            * self.offline_teacher_pseudo_weight)
        if self.offline_teacher_class_weights:
            class_weight = pseudo_weight.new_ones(int(self.num_classes))
            for class_id, value in self.offline_teacher_class_weights.items():
                if 0 <= class_id < int(self.num_classes):
                    class_weight[class_id] = value
            safe_teacher_label = teacher_label.clamp(
                min=0, max=int(self.num_classes) - 1)
            teacher_pixel_weight = (
                teacher_pixel_weight * class_weight[safe_teacher_label])
        teacher_pixel_weight = torch.where(
            teacher_valid,
            teacher_pixel_weight,
            torch.zeros_like(teacher_pixel_weight),
        )
        valid_confidence = confidence_value[teacher_valid]
        logs['offline_teacher_confidence_mean'] = float(
            valid_confidence.mean().item()) if valid_confidence.numel() else 0.0
        logs['offline_teacher_weight_mean_valid'] = float(
            teacher_pixel_weight[teacher_valid].mean().item()
        ) if teacher_valid.any() else 0.0
        if self.offline_teacher_pseudo_mode == 'teacher_only':
            merged_label = torch.where(teacher_valid, teacher_label, pseudo_label)
            merged_weight = teacher_pixel_weight
            merged_mask = teacher_valid
            merged_conf = torch.where(
                teacher_valid,
                confidence_value,
                torch.zeros_like(pseudo_conf),
            )
            fallback_ratio = coverage.new_tensor(0.0)
        else:
            merged_label = torch.where(teacher_valid, teacher_label, pseudo_label)
            merged_weight = torch.where(
                teacher_valid,
                teacher_pixel_weight,
                pseudo_weight,
            )
            merged_mask = teacher_valid | ema_valid
            merged_conf = torch.where(
                teacher_valid,
                confidence_value,
                pseudo_conf,
            )
            fallback_ratio = (~teacher_valid & ema_valid).float().mean()

        logs.update({
            'offline_teacher_fallback_ratio': float(fallback_ratio.item()),
            'offline_teacher_final_weight_mean': float(
                merged_weight.float().mean().item()),
            'offline_teacher_active': 1.0,
        })
        return (
            merged_label,
            merged_weight,
            merged_mask,
            merged_conf,
            logs,
        )

    def _generate_target_pseudo_state(
            self, tar_img, valid_pseudo_mask, seg_debug,
            offline_teacher_label=None, offline_teacher_confidence=None):
        """Generate target pseudo labels, weights, VPLF logs, and debug state.

        生成目标域伪标签、伪标签权重、VPLF 日志以及可选的深度调试图。
        """
        self._disable_ema_stochastic_layers()

        with torch.no_grad():
            ema_logits = self.get_ema_model().generate_pseudo_label(tar_img)
        if isinstance(ema_logits, (tuple, list)):
            ema_logits = ema_logits[0]

        seg_debug['Target'] = self.get_ema_model().decode_head.debug_output
        dep_tar = None
        if hasattr(self.get_ema_model().backbone, 'color_depth_map'):
            dep_tar = self.get_ema_model().backbone.color_depth_map

        pseudo_label, pseudo_weight, pseudo_mask, pseudo_conf = self.get_pseudo_label_and_weight(ema_logits)
        del ema_logits

        pseudo_label, pseudo_weight, pseudo_mask, pseudo_conf, offline_logs = \
            self._merge_offline_teacher_pseudo(
                pseudo_label,
                pseudo_weight,
                pseudo_mask,
                pseudo_conf,
                offline_teacher_label,
                offline_teacher_confidence,
            )

        pseudo_weight = self.filter_valid_pseudo_region(pseudo_weight, valid_pseudo_mask)
        pseudo_weight, vplf_log_vars = self._apply_vplf_filter(
            tar_img,
            pseudo_label,
            pseudo_conf,
            pseudo_weight,
        )
        vplf_log_vars.update(offline_logs)
        return pseudo_label, pseudo_weight, pseudo_mask, pseudo_conf, dep_tar, vplf_log_vars

    def _forward_stage2_sadg_loss(self, src_img, src_seg_lbl, src_seg_pred,
                                  batch_size):
        """Apply optional SADG regularization to the semi source branch."""
        self.stage2_sadg_debug_data = None
        if not self.stage2_sadg_enabled:
            return {}, 0.0

        aug_img, aug_info = self.stage2_sadg_augmentor(src_img)
        severity = float(aug_info['severity'])
        alpha = math.exp(-float(self.stage2_sadg_cfg['gamma']) * severity)
        beta = 1.0 - alpha

        aug_pred_results = self.get_model().forward_train(
            (aug_img, src_seg_lbl),
            return_feat=False,
        )
        aug_logits = aug_pred_results.pop('seg_logits', None)
        aug_loss, _ = parse_losses(aug_pred_results)

        zero = aug_loss.new_zeros(())
        if self.stage2_sadg_cfg['lambda_cons'] > 0 and src_seg_pred is not None:
            kl_cons = logit_kl_consistency(
                src_seg_pred,
                aug_logits,
                self.stage2_sadg_cfg['tau'])
        else:
            kl_cons = zero

        weighted_aug_loss = (
            float(self.stage2_sadg_cfg['lambda_aug']) * alpha * aug_loss)
        weighted_cons_loss = (
            float(self.stage2_sadg_cfg['lambda_cons']) * beta * kl_cons)
        sadg_loss = weighted_aug_loss + weighted_cons_loss
        if sadg_loss.requires_grad:
            sadg_loss.backward()

        debug_interval = int(self.stage2_sadg_cfg['debug_img_interval'] or 0)
        if self._is_master_process() and debug_interval > 0 and \
                (self.local_iter + 1) % debug_interval == 0:
            src_debug_logits = src_seg_pred
            if isinstance(src_debug_logits, (tuple, list)):
                src_debug_logits = src_debug_logits[0]
            aug_debug_logits = aug_logits
            if isinstance(aug_debug_logits, (tuple, list)):
                aug_debug_logits = aug_debug_logits[0]
            self.stage2_sadg_debug_data = {
                'aug_img': aug_img.detach(),
                'src_logits': src_debug_logits.detach(),
                'aug_logits': aug_debug_logits.detach()
                if aug_debug_logits is not None else None,
                'aug_info': {
                    'aug_type': aug_info['aug_type'],
                    'severity': severity,
                    'alpha': alpha,
                    'beta': beta,
                },
            }

        aug_type_id = SADG_DEFAULT_AUG_TYPES.index(aug_info['aug_type'])
        return {
            'stage2_sadg_ce_aug': float(aug_loss.detach().item()),
            'stage2_sadg_kl_cons': float(kl_cons.detach().item()),
            'stage2_sadg_alpha': float(alpha),
            'stage2_sadg_beta': float(beta),
            'stage2_sadg_severity': float(severity),
            'stage2_sadg_lambda_aug': float(self.stage2_sadg_cfg['lambda_aug']),
            'stage2_sadg_lambda_cons': float(self.stage2_sadg_cfg['lambda_cons']),
            'stage2_sadg_loss_aug': float(weighted_aug_loss.detach().item()),
            'stage2_sadg_loss_cons': float(weighted_cons_loss.detach().item()),
            'stage2_sadg_loss': float(sadg_loss.detach().item()),
            'stage2_sadg_aug_type_id': float(aug_type_id),
        }, float(sadg_loss.detach().item())

    def _infer_target_weather_types(self, target_img_paths, batch_size):
        if not self.weather_aligned_mix_target_gt:
            return [self.weather_aligned_mix_default_weather] * batch_size
        if target_img_paths is None:
            if not self._weather_mix_warned_missing_path:
                self.logger.warning(
                    '[WeatherMix] target image paths are unavailable; '
                    'falling back to default weather=%s.',
                    self.weather_aligned_mix_default_weather)
                self._weather_mix_warned_missing_path = True
            return [self.weather_aligned_mix_default_weather] * batch_size

        weather_types = []
        unknown_paths = []
        for idx in range(batch_size):
            if isinstance(target_img_paths, (list, tuple)):
                path = target_img_paths[idx] if idx < len(target_img_paths) else ''
            else:
                path = target_img_paths
            scene = self._infer_weather_type_from_path(path)
            if scene is None:
                unknown_paths.append(str(path))
            weather_types.append(scene or self.weather_aligned_mix_default_weather)
        if unknown_paths and not self._weather_mix_warned_unknown_path:
            self.logger.warning(
                '[WeatherMix] Could not parse target weather for %d/%d paths. '
                'Examples: %s. Falling back to default weather=%s.',
                len(unknown_paths),
                batch_size,
                unknown_paths[:3],
                self.weather_aligned_mix_default_weather)
            self._weather_mix_warned_unknown_path = True
        return weather_types

    def _infer_weather_type_from_path(self, path):
        rel_key = self._get_acdc_aggregate_rel_key(path)
        basename = os.path.basename(str(path))
        lookup = self._get_weather_scene_lookup(path)
        if rel_key and rel_key in lookup:
            return lookup[rel_key]
        if basename in lookup:
            return lookup[basename]

        scene = get_acdc_scene_from_path(path)
        if scene is not None:
            return scene

        return None

    @staticmethod
    def _infer_acdc_data_root_from_path(path):
        norm_path = os.path.abspath(str(path)).replace('\\', '/')
        parts = norm_path.split('/')
        for marker in ('rgb_anon', 'rgb_anno'):
            if marker in parts:
                return '/'.join(parts[:parts.index(marker)])
        return None

    @staticmethod
    def _get_acdc_aggregate_rel_key(path):
        norm_path = str(path).replace('\\', '/')
        parts = norm_path.split('/')
        for marker in ('rgb_anon', 'rgb_anno'):
            if marker not in parts:
                continue
            idx = parts.index(marker)
            if idx + 2 < len(parts) and parts[idx + 1] in ('train', 'val', 'test'):
                return '/'.join(parts[idx + 2:])
        return None

    def _get_weather_scene_lookup(self, sample_path=None):
        if self.weather_aligned_mix_scene_lookup is not None:
            return self.weather_aligned_mix_scene_lookup

        lookup = {}
        target_cfg = self.cfg.data.get('target', {})
        data_root = target_cfg.get('data_root', '')
        candidate_data_roots = []
        sample_root = self._infer_acdc_data_root_from_path(sample_path) \
            if sample_path is not None else None
        if sample_root:
            candidate_data_roots.append(sample_root)
        if data_root:
            candidate_data_roots.append(os.path.abspath(str(data_root)))

        searched_roots = []
        for data_root in candidate_data_roots:
            if not data_root or data_root in searched_roots:
                continue
            searched_roots.append(data_root)
            for scene in ACDC_EVAL_SCENES:
                for split in ('train', 'val', 'test'):
                    for rgb_dir in ('rgb_anon', 'rgb_anno'):
                        candidate_roots = (
                            os.path.join(data_root, rgb_dir, scene, split),
                            os.path.join(data_root, rgb_dir, split, scene),
                        )
                        for candidate_root in candidate_roots:
                            if not os.path.isdir(candidate_root):
                                continue
                            for root, _, filenames in os.walk(candidate_root):
                                for filename in filenames:
                                    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                        continue
                                    full_path = os.path.join(root, filename)
                                    rel_key = os.path.relpath(
                                        full_path, candidate_root).replace('\\', '/')
                                    lookup.setdefault(rel_key, scene)
                                    lookup.setdefault(filename, scene)

        self.weather_aligned_mix_scene_lookup = lookup
        if self.weather_aligned_mix_enabled:
            self.logger.info(
                '[WeatherMix] Built ACDC aggregate-to-weather lookup with %d entries from roots=%s.',
                len(lookup),
                searched_roots)
        return lookup

    def _get_weather_mix_fixed_severity(self, weather_type):
        severity_cfg = self.weather_aligned_mix_severity
        if isinstance(severity_cfg, dict):
            return float(severity_cfg.get(
                weather_type,
                severity_cfg.get(self.weather_aligned_mix_default_weather, 0.0)))
        return float(severity_cfg)

    def _get_weather_mix_range(self, weather_type, range_cfg):
        if isinstance(range_cfg, dict):
            range_cfg = range_cfg.get(
                weather_type,
                range_cfg.get(self.weather_aligned_mix_default_weather, (0.0, 0.0)))
        low, high = range_cfg
        return float(low), float(high)

    def _severity_from_proxy(self, weather_type, proxy):
        low, high = self._get_weather_mix_range(
            weather_type,
            self.weather_aligned_mix_target_severity_by_weather
            or self.weather_aligned_mix_target_severity_range)
        proxy = max(0.0, min(1.0, float(proxy)))
        return low + (high - low) * proxy

    def _blend_weather_mix_severity(self, weather_type, target_severity):
        blend = max(0.0, min(1.0, self.weather_aligned_mix_target_severity_blend))
        if blend >= 1.0:
            return target_severity
        base = self._get_weather_mix_fixed_severity(weather_type)
        return (1.0 - blend) * base + blend * target_severity

    def _denormalize_weather_mix_img(self, img):
        mean = torch.as_tensor(
            self.img_mean, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        std = torch.as_tensor(
            self.img_std, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
        return torch.clamp((img * std + mean) / 255.0, 0.0, 1.0)

    def _estimate_target_weather_severity_proxy(self, weather_type, target_img):
        if target_img is None:
            return 0.5

        with torch.no_grad():
            rgb = self._denormalize_weather_mix_img(target_img.detach())
            luma = (
                0.299 * rgb[:, 0:1] +
                0.587 * rgb[:, 1:2] +
                0.114 * rgb[:, 2:3])
            brightness = float(luma.mean().item())
            contrast = float(torch.clamp(luma.std() * 4.0, 0.0, 1.0).item())
            saturation = float(torch.clamp(
                (rgb.max(dim=1, keepdim=True).values -
                 rgb.min(dim=1, keepdim=True).values).mean(),
                0.0,
                1.0).item())

        darkness = 1.0 - brightness
        low_contrast = 1.0 - contrast
        low_saturation = 1.0 - saturation
        if weather_type == 'night':
            proxy = 0.65 * darkness + 0.25 * low_saturation + 0.10 * low_contrast
        elif weather_type == 'rain':
            proxy = 0.40 * darkness + 0.30 * low_saturation + 0.30 * contrast
        elif weather_type in ('fog', 'snow'):
            proxy = 0.45 * low_contrast + 0.35 * low_saturation + 0.20 * brightness
        else:
            proxy = (
                0.35 * low_contrast +
                0.25 * low_saturation +
                0.25 * darkness +
                0.15 * brightness)
        return max(0.0, min(1.0, float(proxy)))

    def _estimate_weather_mix_confidence_proxy(self, pseudo_weight):
        if pseudo_weight is None:
            return 0.5
        with torch.no_grad():
            confidence = float(torch.clamp(
                pseudo_weight.detach().float().mean(),
                0.0,
                1.0).item())
        gamma = max(1e-6, self.weather_aligned_mix_confidence_gamma)
        return (1.0 - confidence) ** gamma

    def _get_weather_mix_severity(self, weather_type, target_img=None,
                                  pseudo_weight=None):
        mode = self.weather_aligned_mix_severity_mode
        if self.weather_aligned_mix_random_severity or mode == 'random':
            low, high = self._get_weather_mix_range(
                weather_type, self.weather_aligned_mix_severity_range)
            return random.uniform(float(low), float(high))

        if mode in ('target', 'target_image', 'target_severity'):
            proxy = self._estimate_target_weather_severity_proxy(
                weather_type, target_img)
            return self._blend_weather_mix_severity(
                weather_type,
                self._severity_from_proxy(weather_type, proxy))

        if mode in ('confidence', 'pseudo_confidence', 'target_confidence'):
            proxy = self._estimate_weather_mix_confidence_proxy(pseudo_weight)
            return self._blend_weather_mix_severity(
                weather_type,
                self._severity_from_proxy(weather_type, proxy))

        return self._get_weather_mix_fixed_severity(weather_type)

    def _is_weather_mix_active(self, weather_type):
        if self.weather_aligned_mix_active_weathers is None:
            return True
        return str(weather_type).lower() in self.weather_aligned_mix_active_weathers

    def _prepare_weather_aligned_source_mix(self, src_img, target_img_paths,
                                            batch_size, target_img=None,
                                            pseudo_weight=None):
        """Stylize source images to target weather before ClassMix."""
        self.weather_mix_debug_data = None
        if not self.weather_aligned_mix_enabled:
            return src_img, {}

        weather_types = self._infer_target_weather_types(target_img_paths, batch_size)
        weather_src_img = src_img.clone()
        aug_types, severities, active_mask = [], [], []
        num_augmented = 0
        num_unknown = 0
        num_gated = 0

        for idx, weather_type in enumerate(weather_types):
            if not self._is_weather_mix_active(weather_type):
                aug_types.append('normal')
                severities.append(0.0)
                active_mask.append(False)
                num_gated += 1
                continue

            aug_type = self.weather_aligned_mix_aug_map.get(weather_type, weather_type)
            if aug_type in (None, 'none', 'normal'):
                aug_types.append('normal')
                severities.append(0.0)
                active_mask.append(False)
                if weather_type == self.weather_aligned_mix_default_weather:
                    num_unknown += 1
                continue

            aug_type = normalize_sadg_aug_type(aug_type)
            target_img_i = target_img[idx:idx + 1] if target_img is not None else None
            pseudo_weight_i = (
                pseudo_weight[idx:idx + 1]
                if pseudo_weight is not None else None)
            severity = max(0.0, min(1.0, self._get_weather_mix_severity(
                weather_type,
                target_img=target_img_i,
                pseudo_weight=pseudo_weight_i)))
            weather_src_img[idx:idx + 1] = self.weather_aligned_mix_augmentor.apply(
                src_img[idx:idx + 1],
                aug_type,
                severity)
            aug_types.append(aug_type)
            severities.append(float(severity))
            active_mask.append(True)
            num_augmented += 1

        mean_severity = float(np.mean(severities)) if severities else 0.0
        self.weather_mix_debug_data = {
            'src_original': src_img.detach(),
            'src_weather': weather_src_img.detach(),
            'weather_types': list(weather_types),
            'aug_types': list(aug_types),
            'severities': list(severities),
            'active_mask': list(active_mask),
            'severity_mode': self.weather_aligned_mix_severity_mode,
            'active_weathers': list(self.weather_aligned_mix_active_weathers)
            if self.weather_aligned_mix_active_weathers is not None else None,
            'target_paths': list(target_img_paths)
            if isinstance(target_img_paths, (list, tuple)) else [str(target_img_paths)],
        }
        return weather_src_img, {
            'weather_mix_enabled': 1.0,
            'weather_mix_aug_ratio': float(num_augmented) / max(1, batch_size),
            'weather_mix_unknown_ratio': float(num_unknown) / max(1, batch_size),
            'weather_mix_gated_ratio': float(num_gated) / max(1, batch_size),
            'weather_mix_mean_severity': mean_severity,
        }

    def _build_class_mix_batch(self, src_img, src_seg_lbl, tar_img, pseudo_label,
                               pseudo_weight, pseudo_conf, strong_parameters,
                               batch_size, dev, target_img_paths=None):
        """Create Class-Mix images, labels, weights, and debug masks.

        构造 Class-Mix 后的图像、标签、权重以及调试所需的混合 mask。
        """
        mix_img, mix_seg_lbl = [None] * batch_size, [None] * batch_size
        target_mix_weight, mix_conf_log_vars = self._apply_mix_confidence_weight(
            pseudo_weight,
            pseudo_conf,
        )
        mix_seg_weight = target_mix_weight.clone()
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)
        src_img_for_mix, weather_mix_log_vars = self._prepare_weather_aligned_source_mix(
            src_img,
            target_img_paths,
            batch_size,
            target_img=tar_img,
            pseudo_weight=pseudo_weight)

        src_mix_ratio = self.get_src_cls_mix_ratio()
        if self.get_context_class_mask:
            mix_masks, num_class_choice = get_context_class_masks(
                src_seg_lbl.unsqueeze(1),
                class_ratio=src_mix_ratio,
                num_classes=self.num_classes,
            )
        else:
            mix_masks, num_class_choice = get_class_masks(
                src_seg_lbl.unsqueeze(1),
                class_ratio=src_mix_ratio,
            )

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mix_img[i], mix_seg_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((src_img_for_mix[i], tar_img[i])),
                target=torch.stack((src_seg_lbl[i], pseudo_label[i])),
            )
            _, mix_seg_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], target_mix_weight[i])),
            )

        del gt_pixel_weight
        mix_img = torch.cat(mix_img)
        mix_seg_lbl = torch.cat(mix_seg_lbl)
        if self.weather_mix_debug_data is not None:
            self.weather_mix_debug_data['mix_img'] = mix_img.detach()
            self.weather_mix_debug_data['mix_masks'] = [mask.detach() for mask in mix_masks]
        mix_build_log_vars = dict(weather_mix_log_vars)
        mix_build_log_vars.update(mix_conf_log_vars)
        return (
            mix_img,
            mix_seg_lbl,
            mix_seg_weight,
            mix_masks,
            num_class_choice,
            src_mix_ratio,
            mix_build_log_vars,
        )

    def _forward_mix_loss(self, mix_img, mix_seg_lbl, mix_seg_weight, seg_debug):
        """Run the student on mixed samples and backpropagate the mix loss.

        使用混合样本运行 student，并反向传播 mix loss。
        """
        mix_pred_results = self.get_model().forward_train(
            (mix_img, mix_seg_lbl.squeeze(1)),
            seg_weight=mix_seg_weight,
            return_feat=False,
            loss_key='unsup',
        )
        seg_debug['Mix'] = self.get_model().decode_head.debug_output

        dep_mix = None
        if hasattr(self.get_model().backbone, 'color_depth_map'):
            dep_mix = self.get_model().backbone.color_depth_map

        mix_seg_pred = mix_pred_results.pop('seg_logits', None)
        mix_seg_loss, mix_log_vars = parse_losses(mix_pred_results)
        mix_loss_weight = float(self._get_mix_loss_weight())
        weighted_mix_seg_loss = mix_seg_loss * mix_loss_weight
        grad_conflict_log_vars = self._build_grad_conflict_log_vars(
            weighted_mix_seg_loss)
        adapter_conflict_mix_grads = self._capture_adapter_grad_conflict_grads(
            weighted_mix_seg_loss)
        weighted_mix_seg_loss.backward()
        grad_conflict_log_vars.update(
            self._apply_adapter_grad_conflict(adapter_conflict_mix_grads))
        grad_conflict_log_vars.update({
            'mix_schedule_loss_weight': mix_loss_weight,
            'mix_schedule_loss_raw': float(mix_seg_loss.detach().item()),
            'mix_schedule_loss_weighted': float(
                weighted_mix_seg_loss.detach().item()),
        })
        return (
            mix_seg_pred,
            mix_log_vars,
            weighted_mix_seg_loss.item(),
            dep_mix,
            grad_conflict_log_vars,
        )

    def _forward_token_masking_loss(self, tar_img, pseudo_label, pseudo_weight,
                                    strong_parameters, batch_size):
        """Run token-masking consistency loss on augmented target images.

        在增强后的目标域图像上计算 token masking 一致性损失。
        """
        target_img_aug = [None] * batch_size
        for i in range(batch_size):
            target_img_aug[i], _ = strong_transform_wo_mix(
                strong_parameters,
                data=tar_img[i].unsqueeze(0),
            )
        target_img_aug = torch.cat(target_img_aug, dim=0)

        token_masked_results = self.get_model().forward_train(
            (target_img_aug, pseudo_label),
            seg_weight=pseudo_weight,
            return_feat=False,
            enable_token_masking=True,
            loss_key='unsup',
        )
        token_masked_pred = token_masked_results.pop('seg_logits', None)
        self._last_token_masked_pred = token_masked_pred

        token_masked_seg_loss, token_masked_log_vars = parse_losses(token_masked_results)
        token_masked_seg_loss.backward()
        return token_masked_log_vars

    def _forward_mic_loss(self, src_img, src_seg_lbl, tar_img, valid_pseudo_mask,
                          pseudo_label, pseudo_weight, seg_debug):
        """Run masked image consistency loss and collect its debug outputs.

        计算 masked image consistency 损失，并收集对应调试输出。
        """
        masked_loss_dict = self.mic(
            self.get_model(),
            src_img,
            src_seg_lbl,
            tar_img,
            valid_pseudo_mask,
            pseudo_label,
            pseudo_weight,
            loss_key='unsup',
        )
        masked_loss, masked_log_vars = parse_losses(masked_loss_dict)
        seg_debug.update(self.mic.debug_output)
        masked_loss.backward()
        return masked_log_vars, masked_loss.item()

    def _forward_source_supervised_loss(self, src_img, src_seg_lbl, seg_debug):
        """Run the supervised source-domain loss and its immediate backward pass.

        计算源域监督损失，并按配置立即反向传播或暂存到共享 backward 中。
        """
        if self.style_consistency_lambda > 0 and \
                self.get_model().backbone.style_hallucination is not None:
            src_seg_lbl = torch.cat((src_seg_lbl, src_seg_lbl), dim=0)

        src_pred_results = self.get_model().forward_train(
            (src_img, src_seg_lbl),
            return_feat=self.enable_fdist,
        )
        src_feat = src_pred_results.pop('features', None)
        src_seg_pred = src_pred_results.pop('seg_logits', None)

        seg_debug['Source'] = self.get_model().decode_head.debug_output
        src_seg_loss, src_log_vars = parse_losses(src_pred_results)
        if self._should_collect_grad_conflict():
            self._grad_conflict_source_grads = self._capture_grad_conflict_grads(
                src_seg_loss)
        if self.adapter_grad_conflict_enabled:
            self._adapter_grad_conflict_source_grads = \
                self._capture_adapter_grad_conflict_grads(src_seg_loss)

        src_loss = None
        seg_grads = None
        if not self.share_src_backward:
            src_seg_loss.backward(retain_graph=self.enable_fdist)
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                seg_grads = [
                    p.grad.detach().clone() for p in params
                    if p.grad is not None
                ]
                grad_mag = calc_grad_magnitude(seg_grads)
                self.logger.info(f'Seg. Grad.: {grad_mag}')
        else:
            src_loss = src_seg_loss

        return {
            'src_seg_lbl': src_seg_lbl,
            'src_feat': src_feat,
            'src_seg_pred': src_seg_pred,
            'src_seg_loss': src_seg_loss,
            'src_loss': src_loss,
            'seg_grads': seg_grads,
            'log_vars': src_log_vars,
            'loss_value': src_seg_loss.item(),
        }

    def forward_train_step(self, data_batch, valid_pseudo_mask=None):
        """Run one DACS training iteration and return scalar log values.

        执行一次 DACS 训练 iteration，并返回用于日志记录的标量字典。

        Args:
            data_batch: Tuple of source image/label and target image/label:
                `(src_img, src_seg_lbl, tar_img, tar_seg_lbl)`.
                源域与目标域 batch，格式为
                `(src_img, src_seg_lbl, tar_img, tar_seg_lbl)`。
            valid_pseudo_mask: Optional mask for filtering target pseudo labels.
                可选的目标域伪标签有效区域 mask。

        Returns:
            dict: Scalar loss and training metrics for logging.
                用于日志记录的标量 loss 与训练指标。
        """
        offline_teacher_confidence = None
        if len(data_batch) == 6:
            (src_img, src_seg_lbl, tar_img, tar_seg_lbl, target_img_paths,
             offline_teacher_confidence) = data_batch
        elif len(data_batch) == 5:
            src_img, src_seg_lbl, tar_img, tar_seg_lbl, target_img_paths = data_batch
        else:
            src_img, src_seg_lbl, tar_img, tar_seg_lbl = data_batch
            target_img_paths = None
        if self.source_only:
            tar_img = None
            tar_seg_lbl = None
            target_img_paths = None
        log_vars = {}
        batch_size = src_img.shape[0]
        dev = src_img.device
        src_img_base = src_img
        src_seg_lbl_base = src_seg_lbl
        self._grad_conflict_source_grads = None
        self._adapter_grad_conflict_source_grads = None

        self._update_teacher_and_mic_state()
        self.update_debug_state()
        seg_debug = {}

        semi_enabled = not self.source_only and self.local_iter >= self.semi_begin_iter

        means, stds, strong_parameters = self._build_strong_parameters(batch_size, dev)

        dep_tar, dep_mix = None, None
        
        # Source-domain supervised branch.
        # 源域监督训练分支。
        source_state = self._forward_source_supervised_loss(
            src_img,
            src_seg_lbl,
            seg_debug,
        )
        src_seg_lbl = source_state['src_seg_lbl']
        src_feat = source_state['src_feat']
        src_seg_pred = source_state['src_seg_pred']
        src_seg_loss = source_state['src_seg_loss']
        src_loss = source_state['src_loss']
        seg_grads = source_state['seg_grads']
        log_vars.update(add_prefix(source_state['log_vars'], 'src'))

        # Total loss is used for logging only.
        # total_loss 只用于日志记录。
        total_loss_value = source_state['loss_value']

        stage2_sadg_log_vars, stage2_sadg_loss_value = self._forward_stage2_sadg_loss(
            src_img_base,
            src_seg_lbl_base,
            src_seg_pred,
            batch_size)
        log_vars.update(stage2_sadg_log_vars)
        total_loss_value += stage2_sadg_loss_value
        
        
        # ✅ 特征蒸馏损失
        if self.enable_fdist:
            if self.style_consistency_lambda > 0:
                src_img = torch.cat((src_img, src_img), dim=0)
            else:
                src_img = src_img
            
            # 源域特征蒸馏 (type=1,2,3 都执行)
            src_feat_loss, src_feat_log = self.calc_feat_dist(src_img, src_seg_lbl.unsqueeze(1), src_feat)
            log_vars.update(add_prefix(src_feat_log, 'src'))
            
            
            if self.share_src_backward:
                src_loss = src_loss + src_feat_loss
            else:
                src_feat_loss.backward()
            
            total_loss_value += src_feat_loss.item()
            
            # ✅ 目标域特征蒸馏 (type=2,3 时执行)
            if self.fdist_type in [2, 3]:
                tar_feat = self.get_model().extract_feat(tar_img)
                if isinstance(tar_feat, tuple):
                    tar_feat = tar_feat[1]
                tar_feat_loss, tar_feat_log = self.calc_feat_dist(tar_img, None, tar_feat)
                log_vars.update(add_prefix(tar_feat_log, 'tar'))
                tar_feat_loss.backward()
                total_loss_value += tar_feat_loss.item()

            if self.print_grad_magnitude and seg_grads is not None:
                params = self.get_model().backbone.parameters()
                fd_grads = [p.grad.detach() for p in params if p.grad is not None]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                self.logger.info(f'Fdist Grad.: {grad_mag}')
        
        
        # style consistency
        if self.style_consistency_lambda > 0:
            src_seg_logits = src_seg_pred[0] if isinstance(src_seg_pred, (tuple, list)) else src_seg_pred
            sc_loss, sc_log = self.style_consistency(src_seg_logits)
            assert self.share_src_backward
            src_loss = src_loss + sc_loss
            log_vars.update(add_prefix(sc_log, 'src'))
            total_loss_value += sc_loss.item()
            
        if self.share_src_backward:
            src_loss.backward()
            del src_loss
        
        del src_feat, src_seg_loss
        if self.enable_fdist:
            del src_feat_loss
        if self.style_consistency_lambda > 0:
            del sc_loss
        
        if self.source_only:
            self.local_iter += 1
            log_vars['total_loss'] = total_loss_value
            return log_vars
        
        # mix loss
        pseudo_label, pseudo_weight = None, None
        if semi_enabled:
            # Generate target pseudo labels and build mixed source-target samples.
            # 生成目标域伪标签，并构造 source-target 混合样本。
            pseudo_label, pseudo_weight, pseudo_mask, pseudo_conf, dep_tar, vplf_log_vars = self._generate_target_pseudo_state(
                tar_img,
                valid_pseudo_mask,
                seg_debug,
                offline_teacher_label=tar_seg_lbl,
                offline_teacher_confidence=offline_teacher_confidence,
            )
            log_vars.update(vplf_log_vars)
            (
                mix_img,
                mix_seg_lbl,
                mix_seg_weight,
                mix_masks,
                num_class_choice,
                src_mix_ratio,
                weather_mix_log_vars,
            ) = self._build_class_mix_batch(
                src_img,
                src_seg_lbl,
                tar_img,
                pseudo_label,
                pseudo_weight,
                pseudo_conf,
                strong_parameters,
                batch_size,
                dev,
                target_img_paths,
            )
            log_vars.update(weather_mix_log_vars)
            log_vars['src_mix_ratio'] = src_mix_ratio

            (
                mix_seg_pred,
                mix_log_vars,
                mix_loss_value,
                dep_mix,
                grad_conflict_log_vars,
            ) = self._forward_mix_loss(
                mix_img,
                mix_seg_lbl,
                mix_seg_weight,
                seg_debug,
            )
            log_vars.update(add_prefix(mix_log_vars, 'mix'))
            log_vars.update(grad_conflict_log_vars)
            total_loss_value += mix_loss_value

        # token masking loss
        if semi_enabled and self.enable_token_masking:
            token_masked_log_vars = self._forward_token_masking_loss(
                tar_img,
                pseudo_label,
                pseudo_weight,
                strong_parameters,
                batch_size,
            )
            log_vars.update(add_prefix(token_masked_log_vars, 'token_masked'))
            
        
        # Masked Training
        if semi_enabled and self.enable_masking and self.mask_mode.startswith('separate'):
            masked_log_vars, masked_loss_value = self._forward_mic_loss(
                src_img,
                src_seg_lbl,
                tar_img,
                valid_pseudo_mask,
                pseudo_label,
                pseudo_weight,
                seg_debug,
            )
            log_vars.update(add_prefix(masked_log_vars, 'masked'))
            total_loss_value += masked_loss_value
            
        log_vars['total_loss'] = total_loss_value
            
        # save the debug images
        if self._is_master_process() and semi_enabled and self.debug_img_interval > 0 and \
                (self.local_iter + 1) % self.debug_img_interval == 0:
            self._save_debug_visualization(
                batch_size, means, stds, self.dataset_class,
                src_img, tar_img, mix_img,
                src_seg_lbl, tar_seg_lbl, pseudo_label, pseudo_weight, pseudo_mask, mix_seg_lbl,
                num_class_choice, mix_masks, mix_seg_weight,
                src_seg_pred, mix_seg_pred, dep_tar, dep_mix
            )
            self._save_hrda_debug_images(seg_debug, batch_size, means, stds, mix_seg_weight)
        if self._is_master_process() and self.stage2_sadg_debug_data is not None:
            save_debug_sadg_images(
                os.path.join(self.cfg.respth, 'debug', 'sadg_stage2'),
                self.local_iter,
                batch_size,
                src_img_base.detach(),
                self.stage2_sadg_debug_data['aug_img'],
                src_seg_lbl_base.detach(),
                self.stage2_sadg_debug_data['src_logits'],
                self.stage2_sadg_debug_data['aug_logits'],
                means,
                stds,
                self.stage2_sadg_debug_data['aug_info'],
            )
        weather_debug_interval = int(self.weather_aligned_mix_debug_img_interval or 0)
        if self._is_master_process() and self.weather_mix_debug_data is not None and \
                weather_debug_interval > 0 and \
                (self.local_iter + 1) % weather_debug_interval == 0:
            save_weather_aligned_mix_debug_images(
                os.path.join(self.cfg.respth, 'debug', 'weather_mix'),
                self.local_iter,
                batch_size,
                self.weather_mix_debug_data['src_original'],
                self.weather_mix_debug_data['src_weather'],
                tar_img.detach(),
                self.weather_mix_debug_data['mix_img'],
                self.weather_mix_debug_data['mix_masks'],
                means,
                stds,
                weather_types=self.weather_mix_debug_data['weather_types'],
                aug_types=self.weather_mix_debug_data['aug_types'],
                severities=self.weather_mix_debug_data['severities'],
            )
        if self._is_master_process() and semi_enabled and \
                self.vfm_pl_filter_enabled and self.vplf_debug_data is not None:
            save_vplf_debug_images(
                os.path.join(self.cfg.respth, 'debug', 'vplf'),
                self.local_iter + 1,
                tar_img,
                means,
                stds,
                self.vplf_debug_data,
                max_samples=self.vplf_cfg['debug_max_samples'],
            )
            
        self.local_iter += 1
        
        return log_vars
    
    def _get_prediction_for_debug(self, pred, target_shape):
        """Normalize prediction tensor shape for debug visualization.

        统一调试可视化中预测结果的格式和空间尺寸。
        """
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.shape[-2:] != target_shape:
            pred = F.interpolate(pred, size=target_shape, mode='bilinear', align_corners=False)
        return pred
    
    def _save_debug_visualization(self, batch_size, means, stds, dataset_class,
                                src_img, tar_img, mix_img, 
                                src_seg_lbl, tar_seg_lbl, pseudo_label, pseudo_weight, pseudo_mask, mix_seg_lbl,
                                num_class_choice, mix_masks, mix_seg_weight,
                                src_seg_pred, mix_seg_pred, dep_tar, dep_mix):
        """Dispatch debug visualization to the matching save routine.

        根据当前训练分支选择对应的调试可视化保存逻辑。
        """
        
        # 处理预测结果格式
        src_seg_pred = self._get_prediction_for_debug(src_seg_pred, src_seg_lbl.shape[-2:])
        mix_seg_pred = self._get_prediction_for_debug(mix_seg_pred, mix_seg_lbl.shape[-2:])
        
        # 根据不同情况选择可视化方式
        if self.enable_masking and self.mask_mode.startswith('separate'):
            self._save_masked_debug_images(
                batch_size, means, stds, dataset_class,
                src_img, tar_img, mix_img, 
                src_seg_lbl, tar_seg_lbl, pseudo_label, pseudo_weight, pseudo_mask, mix_seg_lbl,
                num_class_choice, mix_masks, mix_seg_weight,
                src_seg_pred, mix_seg_pred, dep_tar, dep_mix
            )
        elif self.enable_token_masking:
            self._save_token_masked_debug_images(
                batch_size, means, stds, dataset_class,
                src_img, tar_img, mix_img,
                src_seg_lbl, tar_seg_lbl, pseudo_label, pseudo_weight, pseudo_mask, mix_seg_lbl,
                num_class_choice, mix_masks, mix_seg_weight,
                src_seg_pred, mix_seg_pred, dep_tar, dep_mix
            )
        else:
            self._save_standard_debug_images(
                batch_size, means, stds, dataset_class,
                src_img, tar_img, mix_img,
                src_seg_lbl, tar_seg_lbl, pseudo_label, pseudo_weight, pseudo_mask, mix_seg_lbl,
                num_class_choice, mix_masks, mix_seg_weight,
                src_seg_pred, mix_seg_pred, dep_tar, dep_mix
            )
    
    def _save_masked_debug_images(self, batch_size, means, stds, dataset_class,
                                src_img, tar_img, mix_img,
                                src_seg_lbl, tar_seg_lbl, pseudo_label, pseudo_weight, pseudo_mask, mix_seg_lbl,
                                num_class_choice, mix_masks, mix_seg_weight,
                                src_seg_pred, mix_seg_pred, dep_tar, dep_mix):
        """Save debug images for masked image consistency.

        保存 masked image consistency 分支的调试图像。
        """
        masked_img = self.mic.debug_output['Masked'].pop('Image')
        masked_pred = self.mic.debug_output['Masked'].pop('Seg Pred')
        masked_gt = self.mic.debug_output['Masked'].pop('Seg GT')
        masked_pseudo_weight = self.mic.debug_output['Masked']['PL Weight']
        
        # 处理掩码预测格式
        masked_pred = self._get_prediction_for_debug(masked_pred, masked_gt.shape[-2:])
        
        save_debug_mic_images(self, 
                        batch_size, means, stds, dataset_class,
                        src_img, tar_img, mix_img, masked_img,
                        src_seg_lbl, tar_seg_lbl, pseudo_label, pseudo_weight, pseudo_mask, mix_seg_lbl, masked_gt, masked_pseudo_weight,
                        num_class_choice, mix_masks, mix_seg_weight, self.debug_fdist_mask, self.debug_gt_rescale,
                        torch.argmax(src_seg_pred, dim=1), torch.argmax(mix_seg_pred, dim=1), torch.argmax(masked_pred, dim=1), dep_tar, dep_mix
                        )
    
    def _save_token_masked_debug_images(self, batch_size, means, stds, dataset_class,
                                      src_img, tar_img, mix_img,
                                      src_seg_lbl, tar_seg_lbl, pseudo_label, pseudo_weight, pseudo_mask, mix_seg_lbl,
                                      num_class_choice, mix_masks, mix_seg_weight,
                                      src_seg_pred, mix_seg_pred, dep_tar, dep_mix):
        """Save debug images for token masking consistency.

        保存 token masking consistency 分支的调试图像。
        """
        patch_size = self.get_model().backbone.patch_size
        token_masked_pred = getattr(self, '_last_token_masked_pred', None)  # 需要在forward中保存
        token_mask = self.get_model().token_masking.token_mask
        
        if token_masked_pred is not None:
            token_masked_pred = self._get_prediction_for_debug(token_masked_pred, mix_seg_lbl.shape[-2:])
            
            save_debug_tkm_images(self, 
                            batch_size, means, stds, dataset_class, patch_size,
                            src_img, tar_img, mix_img, token_mask,
                            src_seg_lbl, tar_seg_lbl, pseudo_label, pseudo_weight, pseudo_mask, mix_seg_lbl,
                            num_class_choice, mix_masks, mix_seg_weight, self.debug_fdist_mask, self.debug_gt_rescale,
                            torch.argmax(src_seg_pred, dim=1), torch.argmax(mix_seg_pred, dim=1), torch.argmax(token_masked_pred, dim=1), dep_tar, dep_mix)
    
    def _save_standard_debug_images(self, batch_size, means, stds, dataset_class,
                                  src_img, tar_img, mix_img,
                                  src_seg_lbl, tar_seg_lbl, pseudo_label, pseudo_weight, pseudo_mask, mix_seg_lbl,
                                  num_class_choice, mix_masks, mix_seg_weight,
                                  src_seg_pred, mix_seg_pred, dep_tar, dep_mix):
        """Save debug images for the standard DACS path.

        保存标准 DACS 训练路径的调试图像。
        """
        save_debug_images(self,
                          batch_size, means, stds, dataset_class,
                          src_img, tar_img, mix_img,
                          src_seg_lbl, tar_seg_lbl, pseudo_label, pseudo_weight, pseudo_mask, mix_seg_lbl,
                          num_class_choice, mix_masks, mix_seg_weight, self.debug_fdist_mask, self.debug_gt_rescale,
                          torch.argmax(src_seg_pred, dim=1), torch.argmax(mix_seg_pred, dim=1), dep_tar, dep_mix)
    
    def _save_hrda_debug_images(self, seg_debug, batch_size, means, stds, mix_seg_weight):
        """Save HRDA-specific debug images.

        保存 HRDA 结构专用的调试图像。
        """
        if isinstance(self.get_model(), HRDAEncoderDecoder):
            if 'Target' in seg_debug:
                seg_debug['Target']['Pseudo W.'] = mix_seg_weight.cpu().numpy()
            save_debug_hrda_images(
                    out_dir=os.path.join(self.cfg.respth, 'debug'),
                    local_iter=self.local_iter,
                    seg_debug=seg_debug,
                    batch_size=batch_size,
                    means=means,
                    stds=stds,
                    palette=self.debug_palette,
                )
    
