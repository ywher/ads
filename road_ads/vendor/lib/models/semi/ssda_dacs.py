# ---------------------------------------------------------------
# SSDA extension built on top of the DACS semi-training wrapper.
# ---------------------------------------------------------------
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lib.loss.losses import parse_losses
from lib.models.model_utils.dacs_transforms import (
    get_class_masks,
    get_context_class_masks,
    strong_transform,
    strong_transform_wo_mix,
)
from lib.models.model_utils.funcs import add_prefix

from .dacs import DACS, dataset_class
from .adaptive_mix_weighting import (
    compute_adaptive_target_mix_share,
    compute_bounded_residual_target_mix_share,
    compute_reliable_ratio_target_mix_share,
    rebalance_mix_weights,
)
from .feature_prototype import (
    class_score_weight_map,
    combine_target_prototypes,
    compute_class_feature_prototypes,
    feature_prototype_contrastive_loss,
    protect_structure_class_scores,
    scatter_class_prototypes,
    source_target_class_scores,
    target_rare_class_scores,
    update_prototype_bank,
)
from .feature_prototype_diagnostics import FeaturePrototypeDiagnosticExporter
from .effective_budget_target_sup import (
    resolve_effective_budget_target_sup_config,
)
from .relative_target_supervision import (
    resolve_relative_target_supervision_config,
)
from .prototype_classmix import get_incompatibility_veto_class_masks
from .prototype_classmix import get_prototype_guided_class_masks
from .prototype_classmix import get_target_deficit_quota_class_masks
from .prototype_classmix import get_target_need_mask_routing_class_masks
from .source_assist_diagnostics import SourceAssistDiagnosticExporter
from .target_patch_memory import (
    TargetPatchMemoryBank,
    apply_target_patch_memory_mix,
)
from .target_class_memory import (
    TargetClassMemoryBank,
    apply_target_class_memory_mix,
)
from .target_labeled_reliability import (
    apply_target_labeled_reliability,
    target_labeled_class_reliability,
)
from .target_need_source_assist import (
    class_gradient_conflict_route_scores,
    class_score_pixel_weights,
    class_conditional_source_route_scores,
    combine_target_need_source_scores,
    per_class_gradient_cosines,
    self_calibrated_class_route_scores,
    target_class_reliability_scores,
    target_need_class_scores,
)
from .target_anchor_replay import build_class_conditional_anchor_weight_map
from .tri_prototype import histogram_js_divergence, prediction_histograms


class SSDADACS(DACS):
    """DACS-style trainer for source-supervised domain adaptation.

    One training iteration contains three data sources:
    `S_l` source labeled data, `T_l` target labeled data, and `T_u` target
    unlabeled data. The default objective is:

    - supervised loss on `S_l`;
    - supervised loss on `T_l`;
    - semi ClassMix loss on `T_l + T_u`;
    - source-transfer ClassMix loss where half of `S_l` is mixed with `T_l`
      labels and half is mixed with `T_u` pseudo labels.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        ssda_cfg = self.semi_cfg.get('ssda', {})
        self.source_sup_weight = float(ssda_cfg.get('source_sup_weight', 1.0))
        self.source_sup_weight_final = float(
            ssda_cfg.get('source_sup_weight_final', self.source_sup_weight))
        self.source_sup_weight_schedule = ssda_cfg.get(
            'source_sup_weight_schedule', 'constant')
        self.source_sup_weight_begin_iter = int(
            ssda_cfg.get('source_sup_weight_begin_iter', 0))

        self.target_sup_weight = float(ssda_cfg.get('target_sup_weight', 1.0))
        self.target_sup_weight_final = float(
            ssda_cfg.get('target_sup_weight_final', self.target_sup_weight))
        self.target_sup_weight_schedule = ssda_cfg.get(
            'target_sup_weight_schedule', 'constant')
        self.target_sup_weight_begin_iter = int(
            ssda_cfg.get('target_sup_weight_begin_iter', 0))
        self.target_sup_weight_end_iter = int(
            ssda_cfg.get('target_sup_weight_end_iter', self.max_iters))

        self.source_mix_weight = float(ssda_cfg.get('source_mix_weight', 1.0))
        self.source_mix_weight_final = float(
            ssda_cfg.get('source_mix_weight_final', self.source_mix_weight))
        self.source_mix_weight_schedule = ssda_cfg.get(
            'source_mix_weight_schedule', 'constant')
        self.source_mix_weight_begin_iter = int(
            ssda_cfg.get('source_mix_weight_begin_iter', 0))

        self.target_mix_weight = float(ssda_cfg.get('target_mix_weight', 1.0))
        self.target_mix_weight_final = float(
            ssda_cfg.get('target_mix_weight_final', self.target_mix_weight))
        self.target_mix_weight_schedule = ssda_cfg.get(
            'target_mix_weight_schedule', 'constant')
        self.target_mix_weight_begin_iter = int(
            ssda_cfg.get('target_mix_weight_begin_iter', 0))
        self.redistribute_target_sup_from_target_mix = bool(
            ssda_cfg.get(
                'redistribute_target_sup_from_target_mix',
                False,
            ))
        self.effective_budget_target_sup = bool(
            ssda_cfg.get('effective_budget_target_sup', False))
        self.target_sup_budget_sample_count = None
        self.target_sup_budget_effective_count = None
        self.target_sup_budget_gate = None
        self.target_sup_budget_labeled_ratio = None
        if self.effective_budget_target_sup:
            self._configure_effective_budget_target_sup(ssda_cfg)
        self.relative_target_supervision_calibration = bool(
            ssda_cfg.get('relative_target_supervision_calibration', False))
        if (self.relative_target_supervision_calibration
                and not ssda_cfg.get(
                    'relative_target_sup_runtime_resolved', False)):
            resolved_relative_tsup = (
                resolve_relative_target_supervision_config(self.cfg))
            self.target_sup_weight = resolved_relative_tsup['weight']
            self.target_sup_weight_final = resolved_relative_tsup['weight']
            self.target_sup_weight_schedule = 'constant'
        self.relative_target_sup_target_count = ssda_cfg.get(
            'relative_target_sup_target_count', None)
        self.relative_target_sup_source_count = ssda_cfg.get(
            'relative_target_sup_source_count', None)
        self.relative_target_sup_coverage = ssda_cfg.get(
            'relative_target_sup_coverage', None)

        self.enable_source_target_mix = bool(
            ssda_cfg.get('enable_source_target_mix', True))
        self.enable_target_semi_mix = bool(
            ssda_cfg.get('enable_target_semi_mix', True))
        self.source_labeled_mix_ratio = float(
            ssda_cfg.get('source_labeled_mix_ratio', 0.5))
        self.source_mix_target_mode = str(
            ssda_cfg.get('source_mix_target_mode', 'hybrid')).lower()
        valid_source_mix_target_modes = {
            'unlabeled_only', 'hybrid', 'labeled_only',
        }
        if self.source_mix_target_mode not in valid_source_mix_target_modes:
            raise ValueError(
                'source_mix_target_mode must be one of '
                f'{sorted(valid_source_mix_target_modes)}, got '
                f'{self.source_mix_target_mode!r}')
        self.enable_source_labeled_aux_mix = bool(
            ssda_cfg.get('enable_source_labeled_aux_mix', False))
        self.source_labeled_aux_mix_weight = float(
            ssda_cfg.get('source_labeled_aux_mix_weight', 1.0))
        self.target_dominant_mix = bool(
            ssda_cfg.get('target_dominant_mix', False))
        self.target_mix_share = float(
            ssda_cfg.get('target_mix_share', 0.5))
        self.target_mix_share_final = float(
            ssda_cfg.get('target_mix_share_final', self.target_mix_share))
        self.target_mix_share_schedule = ssda_cfg.get(
            'target_mix_share_schedule', 'constant')
        self.effective_budget_source_calibration = bool(
            ssda_cfg.get('effective_budget_source_calibration', False))
        self.source_calibration_sample_count = int(
            ssda_cfg.get('source_calibration_sample_count', 0))
        self.source_calibration_effective_count = float(
            ssda_cfg.get('source_calibration_effective_count', 0.0))
        self.source_calibration_budget_gate = float(
            ssda_cfg.get('source_calibration_budget_gate', 0.0))
        total_weight = ssda_cfg.get('target_dominant_mix_total_weight', None)
        self.target_dominant_mix_total_weight = (
            None if total_weight is None else float(total_weight))
        self.adaptive_target_dominant_mix = bool(
            ssda_cfg.get('adaptive_target_dominant_mix', False))
        self.adaptive_mix_mode = str(
            ssda_cfg.get(
                'adaptive_mix_mode',
                'reliability_affinity',
            )).lower()
        valid_adaptive_mix_modes = {
            'class_balanced_consistency_residual',
            'reliability_affinity',
            'reliable_pixel_ratio',
        }
        if self.adaptive_mix_mode not in valid_adaptive_mix_modes:
            raise ValueError(
                'adaptive_mix_mode must be one of '
                f'{sorted(valid_adaptive_mix_modes)}, got '
                f'{self.adaptive_mix_mode!r}')
        self.adaptive_mix_warmup_iter = int(
            ssda_cfg.get('adaptive_mix_warmup_iter', self.semi_begin_iter))
        self.adaptive_mix_momentum = float(
            ssda_cfg.get('adaptive_mix_momentum', 0.95))
        self.adaptive_mix_schedule_floor = bool(
            ssda_cfg.get('adaptive_mix_schedule_floor', False))
        self.adaptive_mix_progress_gate = bool(
            ssda_cfg.get('adaptive_mix_progress_gate', False))
        self.adaptive_mix_default_reliability = float(
            ssda_cfg.get('adaptive_mix_default_reliability', 0.5))
        self.adaptive_mix_default_source_affinity = float(
            ssda_cfg.get('adaptive_mix_default_source_affinity', 0.5))
        self.adaptive_mix_reliability_weight = float(
            ssda_cfg.get('adaptive_mix_reliability_weight', 1.0))
        self.adaptive_mix_affinity_weight = float(
            ssda_cfg.get('adaptive_mix_affinity_weight', 1.0))
        self.adaptive_mix_conf_threshold = float(
            ssda_cfg.get('adaptive_mix_conf_threshold', self.pseudo_threshold))
        self.adaptive_mix_unlabeled_weight = float(
            ssda_cfg.get('adaptive_mix_unlabeled_weight', 0.5))
        self.adaptive_mix_residual_bound = float(
            ssda_cfg.get('adaptive_mix_residual_bound', 0.05))
        self.target_consistency_conf_threshold = float(
            ssda_cfg.get(
                'target_consistency_conf_threshold',
                self.pseudo_threshold,
            ))
        self.target_consistency_min_class_pixels = int(
            ssda_cfg.get('target_consistency_min_class_pixels', 32))
        self.consistency_aware_target_mix = bool(
            ssda_cfg.get('consistency_aware_target_mix', False))
        self.consistency_target_mix_disagreement_weight = float(
            ssda_cfg.get(
                'consistency_target_mix_disagreement_weight',
                0.25,
            ))
        if self.adaptive_mix_residual_bound < 0:
            raise ValueError('adaptive_mix_residual_bound must be nonnegative')
        if not 0.0 <= self.target_consistency_conf_threshold <= 1.0:
            raise ValueError(
                'target_consistency_conf_threshold must be in [0, 1]')
        if self.target_consistency_min_class_pixels < 1:
            raise ValueError(
                'target_consistency_min_class_pixels must be positive')
        if not 0.0 <= self.consistency_target_mix_disagreement_weight <= 1.0:
            raise ValueError(
                'consistency_target_mix_disagreement_weight must be in [0, 1]')
        self._adaptive_mix_target_reliability_ema = None
        self._adaptive_mix_source_affinity_ema = None
        self._adaptive_mix_reliability_reference = None
        self._last_adaptive_mix_log_vars = {}

        self.gradient_aligned_source_assistance = bool(
            ssda_cfg.get('gradient_aligned_source_assistance', False))
        self.gradient_aligned_source_begin_iter = int(
            ssda_cfg.get('gradient_aligned_source_begin_iter',
                         self.semi_begin_iter))
        self.gradient_aligned_source_min_scale = float(
            ssda_cfg.get('gradient_aligned_source_min_scale', 0.35))
        self.gradient_aligned_source_max_scale = float(
            ssda_cfg.get('gradient_aligned_source_max_scale', 1.0))
        self.gradient_aligned_source_conflict_threshold = float(
            ssda_cfg.get('gradient_aligned_source_conflict_threshold', 0.0))
        self.gradient_aligned_source_apply_to_mix = bool(
            ssda_cfg.get('gradient_aligned_source_apply_to_mix', True))
        self.gradient_aligned_source_param_keywords = tuple(
            ssda_cfg.get(
                'gradient_aligned_source_param_keywords',
                ['conv_seg', 'classifier', 'cls_seg']))
        self._gradient_aligned_source_param_cache = None

        self.conflict_aware_class_routing = bool(
            ssda_cfg.get('conflict_aware_class_routing', False))
        self.conflict_aware_class_route_begin_iter = int(
            ssda_cfg.get(
                'conflict_aware_class_route_begin_iter', self.semi_begin_iter))
        self.conflict_aware_class_route_update_interval = max(
            1, int(ssda_cfg.get(
                'conflict_aware_class_route_update_interval', 500)))
        self.conflict_aware_class_route_momentum = float(
            ssda_cfg.get('conflict_aware_class_route_momentum', 0.9))
        self.conflict_aware_class_route_max_classes = max(
            1, int(ssda_cfg.get(
                'conflict_aware_class_route_max_classes', 6)))
        self.conflict_aware_class_route_min_source_pixels = max(
            1, int(ssda_cfg.get(
                'conflict_aware_class_route_min_source_pixels', 16)))
        self.conflict_aware_class_route_min_target_pixels = max(
            1, int(ssda_cfg.get(
                'conflict_aware_class_route_min_target_pixels', 16)))
        self.conflict_aware_class_route_min_abs_cosine = float(
            ssda_cfg.get('conflict_aware_class_route_min_abs_cosine', 0.05))
        self.conflict_aware_class_route_assist_strength = float(
            ssda_cfg.get('conflict_aware_class_route_assist_strength', 0.5))
        self.conflict_aware_class_route_reject_strength = float(
            ssda_cfg.get('conflict_aware_class_route_reject_strength', 0.5))
        self.conflict_aware_class_route_min_score = float(
            ssda_cfg.get('conflict_aware_class_route_min_score', 0.25))
        self.conflict_aware_class_route_max_score = float(
            ssda_cfg.get('conflict_aware_class_route_max_score', 1.75))
        self.conflict_aware_class_route_param_keywords = tuple(
            ssda_cfg.get(
                'conflict_aware_class_route_param_keywords',
                ['conv_seg', 'classifier', 'cls_seg']))
        self.conflict_aware_class_route_random_prob = float(
            ssda_cfg.get(
                'conflict_aware_class_route_random_prob',
                ssda_cfg.get('target_need_source_mix_random_prob', 0.2)))
        self._conflict_aware_class_route_param_cache = None
        self._conflict_aware_class_route_cosine_ema = None
        self._conflict_aware_class_route_valid = None
        self._target_need_current_scores = None

        self.conf_aware_target_mix = bool(
            ssda_cfg.get('conf_aware_target_mix', False))
        self.conf_aware_target_mix_mode = str(
            ssda_cfg.get('conf_aware_target_mix_mode', 'conf_entropy')).lower()
        self.conf_aware_target_mix_threshold = float(
            ssda_cfg.get('conf_aware_target_mix_threshold', self.pseudo_threshold))
        self.conf_aware_target_mix_conf_gamma = float(
            ssda_cfg.get('conf_aware_target_mix_conf_gamma', 1.0))
        self.conf_aware_target_mix_entropy_gamma = float(
            ssda_cfg.get('conf_aware_target_mix_entropy_gamma', 1.0))
        self.conf_aware_target_mix_min_weight = float(
            ssda_cfg.get('conf_aware_target_mix_min_weight', 0.25))
        self.conf_aware_target_mix_max_weight = float(
            ssda_cfg.get('conf_aware_target_mix_max_weight', 1.0))
        self.conf_aware_target_mix_blend = float(
            ssda_cfg.get('conf_aware_target_mix_blend', 0.0))
        self.conf_aware_target_mix_blend_final = float(
            ssda_cfg.get(
                'conf_aware_target_mix_blend_final',
                self.conf_aware_target_mix_blend))
        self.conf_aware_target_mix_blend_schedule = str(
            ssda_cfg.get('conf_aware_target_mix_blend_schedule', 'constant'))

        self.target_labeled_reliability_calibration = bool(
            ssda_cfg.get('target_labeled_reliability_calibration', False))
        self.target_labeled_reliability_begin_iter = int(
            ssda_cfg.get(
                'target_labeled_reliability_begin_iter',
                self.semi_begin_iter))
        self.target_labeled_reliability_min = float(
            ssda_cfg.get('target_labeled_reliability_min', 0.2))
        self.target_labeled_reliability_max = float(
            ssda_cfg.get('target_labeled_reliability_max', 1.0))
        self.target_labeled_reliability_default = float(
            ssda_cfg.get('target_labeled_reliability_default', 1.0))
        self.target_labeled_reliability_affine_floor = ssda_cfg.get(
            'target_labeled_reliability_affine_floor', None)
        if self.target_labeled_reliability_affine_floor is not None:
            self.target_labeled_reliability_affine_floor = float(
                self.target_labeled_reliability_affine_floor)
        self.target_labeled_reliability_momentum = float(
            ssda_cfg.get('target_labeled_reliability_momentum', 0.9))
        self.target_labeled_reliability_blend = float(
            ssda_cfg.get('target_labeled_reliability_blend', 0.0))
        self.target_labeled_reliability_blend_final = float(
            ssda_cfg.get(
                'target_labeled_reliability_blend_final',
                self.target_labeled_reliability_blend))
        self.target_labeled_reliability_blend_schedule = str(
            ssda_cfg.get(
                'target_labeled_reliability_blend_schedule',
                'constant'))
        self.target_labeled_reliability_target_mix_only = bool(
            ssda_cfg.get(
                'target_labeled_reliability_target_mix_only', False))
        self._target_labeled_class_reliability = None
        self._target_labeled_class_reliability_valid = None

        self.target_guided_source_filter = bool(
            ssda_cfg.get('target_guided_source_filter', False))
        self.target_guided_source_filter_mode = str(
            ssda_cfg.get('target_guided_source_filter_mode', 'class_hist')).lower()
        self.target_guided_source_filter_conf_threshold = float(
            ssda_cfg.get('target_guided_source_filter_conf_threshold',
                         self.pseudo_threshold))
        self.target_guided_source_filter_min_weight = float(
            ssda_cfg.get('target_guided_source_filter_min_weight', 0.5))
        self.target_guided_source_filter_max_weight = float(
            ssda_cfg.get('target_guided_source_filter_max_weight', 1.0))
        self.target_guided_source_filter_gamma = float(
            ssda_cfg.get('target_guided_source_filter_gamma', 1.0))
        self.target_guided_source_filter_blend = float(
            ssda_cfg.get('target_guided_source_filter_blend', 0.0))
        self.target_guided_source_filter_blend_final = float(
            ssda_cfg.get(
                'target_guided_source_filter_blend_final',
                self.target_guided_source_filter_blend))
        self.target_guided_source_filter_blend_schedule = str(
            ssda_cfg.get('target_guided_source_filter_blend_schedule', 'constant'))
        self.target_guided_source_filter_unlabeled_weight = float(
            ssda_cfg.get('target_guided_source_filter_unlabeled_weight', 0.5))
        self.target_guided_source_filter_labeled_weight = float(
            ssda_cfg.get('target_guided_source_filter_labeled_weight', 1.0))
        self.target_guided_source_filter_proto_update = str(
            ssda_cfg.get('target_guided_source_filter_proto_update', 'batch')).lower()
        self.target_guided_source_filter_proto_momentum = float(
            ssda_cfg.get('target_guided_source_filter_proto_momentum', 0.9))
        self._source_hist_proto = None
        self._target_labeled_hist_proto = None
        self._target_unlabeled_hist_proto = None

        self.feature_prototype_source_calibration = bool(
            ssda_cfg.get('feature_prototype_source_calibration', False))
        self.feature_prototype_source_weight = bool(
            ssda_cfg.get('feature_prototype_source_weight', True))
        self.feature_prototype_source_mix = bool(
            ssda_cfg.get('feature_prototype_source_mix', False))
        self.feature_prototype_begin_iter = int(
            ssda_cfg.get('feature_prototype_begin_iter', self.semi_begin_iter))
        self.feature_prototype_feature_level = ssda_cfg.get(
            'feature_prototype_feature_level', 0)
        self.feature_prototype_num_prototypes_per_class = int(
            ssda_cfg.get('feature_prototype_num_prototypes_per_class', 1))
        self.feature_prototype_min_pixels = int(
            ssda_cfg.get('feature_prototype_min_pixels', 8))
        self.feature_prototype_conf_threshold = float(
            ssda_cfg.get('feature_prototype_conf_threshold',
                         self.pseudo_threshold))
        self.feature_prototype_min_score = float(
            ssda_cfg.get('feature_prototype_min_score', 0.1))
        self.feature_prototype_default_score = float(
            ssda_cfg.get('feature_prototype_default_score', 1.0))
        self.feature_prototype_score_norm = str(
            ssda_cfg.get('feature_prototype_score_norm', 'none')).lower()
        self.feature_prototype_score_temperature = float(
            ssda_cfg.get('feature_prototype_score_temperature', 1.0))
        self.feature_prototype_quantile_low = float(
            ssda_cfg.get('feature_prototype_quantile_low', 0.05))
        self.feature_prototype_quantile_high = float(
            ssda_cfg.get('feature_prototype_quantile_high', 0.95))
        self.feature_prototype_min_weight = float(
            ssda_cfg.get('feature_prototype_min_weight', 0.5))
        self.feature_prototype_max_weight = float(
            ssda_cfg.get('feature_prototype_max_weight', 1.0))
        self.feature_prototype_gamma = float(
            ssda_cfg.get('feature_prototype_gamma', 1.0))
        self.feature_prototype_target_labeled_weight = float(
            ssda_cfg.get('feature_prototype_target_labeled_weight', 1.0))
        self.feature_prototype_target_unlabeled_weight = float(
            ssda_cfg.get('feature_prototype_target_unlabeled_weight', 0.5))
        self.feature_prototype_momentum = float(
            ssda_cfg.get('feature_prototype_momentum', 0.95))
        self.feature_prototype_unlabeled_forward = bool(
            ssda_cfg.get('feature_prototype_unlabeled_forward', True))
        self.feature_prototype_mix_random_prob = float(
            ssda_cfg.get('feature_prototype_mix_random_prob', 0.2))
        self.feature_prototype_source_mix_balance = bool(
            ssda_cfg.get('feature_prototype_source_mix_balance', False))
        self.feature_prototype_source_mix_rare_gamma = float(
            ssda_cfg.get('feature_prototype_source_mix_rare_gamma', 0.5))
        self.feature_prototype_source_mix_rare_min = float(
            ssda_cfg.get('feature_prototype_source_mix_rare_min', 0.25))
        self.feature_prototype_source_mix_rare_max = float(
            ssda_cfg.get('feature_prototype_source_mix_rare_max', 4.0))
        self.feature_prototype_source_mix_structure_protection = bool(
            ssda_cfg.get(
                'feature_prototype_source_mix_structure_protection', False))
        self.feature_prototype_source_mix_structure_classes = (
            self._parse_feature_prototype_class_ids(
                ssda_cfg.get(
                    'feature_prototype_source_mix_structure_classes',
                    None)))
        self.feature_prototype_source_mix_structure_mode = str(
            ssda_cfg.get(
                'feature_prototype_source_mix_structure_mode',
                'floor')).lower()
        self.feature_prototype_source_mix_structure_min_score = float(
            ssda_cfg.get(
                'feature_prototype_source_mix_structure_min_score', 1.0))
        self.feature_prototype_source_mix_structure_default_score = float(
            ssda_cfg.get(
                'feature_prototype_source_mix_structure_default_score', 1.0))
        self.feature_prototype_diagnostics = bool(
            ssda_cfg.get('feature_prototype_diagnostics', False))
        self.feature_prototype_diagnostic_interval = int(
            ssda_cfg.get('feature_prototype_diagnostic_interval',
                         self.debug_img_interval or 2000))
        self.feature_prototype_diagnostic_dir = ssda_cfg.get(
            'feature_prototype_diagnostic_dir', None)
        self.prototype_incompatibility_veto = bool(
            ssda_cfg.get('prototype_incompatibility_veto', False))
        self.prototype_incompatibility_veto_begin_iter = int(
            ssda_cfg.get(
                'prototype_incompatibility_veto_begin_iter',
                self.semi_begin_iter))
        self._feature_proto_source = None
        self._feature_proto_source_valid = None
        self._feature_proto_source_counts = None
        self._feature_proto_target_labeled = None
        self._feature_proto_target_labeled_valid = None
        self._feature_proto_target_labeled_counts = None
        self._feature_proto_target_unlabeled = None
        self._feature_proto_target_unlabeled_valid = None
        self._feature_proto_target_unlabeled_counts = None
        self._feature_proto_target_unlabeled_confidence = None
        self._feature_proto_source_mix_selected_counts = None
        self._feature_proto_source_mix_total_count = 0.0
        self._feature_proto_target_mix_selected_counts = None
        self._feature_proto_target_mix_total_count = 0.0
        self._feature_proto_loss_contributions = {}
        self._feature_proto_diagnostic_exporter = None

        self.target_need_source_mix = bool(
            ssda_cfg.get('target_need_source_mix', False))
        self.target_need_source_mix_begin_iter = int(
            ssda_cfg.get('target_need_source_mix_begin_iter', self.semi_begin_iter))
        self.target_need_source_mix_labeled_weight = float(
            ssda_cfg.get('target_need_source_mix_labeled_weight', 1.0))
        self.target_need_source_mix_unlabeled_weight = float(
            ssda_cfg.get('target_need_source_mix_unlabeled_weight', 0.5))
        self.target_need_source_mix_coverage_gamma = float(
            ssda_cfg.get('target_need_source_mix_coverage_gamma', 0.5))
        self.target_need_source_mix_uncertainty_weight = float(
            ssda_cfg.get('target_need_source_mix_uncertainty_weight', 0.5))
        self.target_need_source_mix_uncertainty_gamma = float(
            ssda_cfg.get('target_need_source_mix_uncertainty_gamma', 1.0))
        self.target_need_source_mix_min_score = float(
            ssda_cfg.get('target_need_source_mix_min_score', 0.25))
        self.target_need_source_mix_max_score = float(
            ssda_cfg.get('target_need_source_mix_max_score', 4.0))
        self.target_need_source_mix_random_prob = float(
            ssda_cfg.get('target_need_source_mix_random_prob', 0.2))
        self.target_need_source_mix_use_source_transfer = bool(
            ssda_cfg.get(
                'target_need_source_mix_use_source_transfer', False))
        self.target_need_source_mix_source_transfer_weight = float(
            ssda_cfg.get(
                'target_need_source_mix_source_transfer_weight', 1.0))
        self.target_need_source_mix_use_target_loss = bool(
            ssda_cfg.get('target_need_source_mix_use_target_loss', False))
        self.target_need_source_mix_target_loss_weight = float(
            ssda_cfg.get('target_need_source_mix_target_loss_weight', 0.0))
        self.target_need_source_mix_target_loss_momentum = float(
            ssda_cfg.get('target_need_source_mix_target_loss_momentum', 0.9))
        self.target_need_source_mix_apply_to_classmix = bool(
            ssda_cfg.get('target_need_source_mix_apply_to_classmix', True))
        self.target_need_source_mix_soft_stuff_classes = (
            self._parse_feature_prototype_class_ids(
                ssda_cfg.get(
                    'target_need_source_mix_soft_stuff_classes', [])))
        self.target_need_source_mix_soft_stuff_max = int(
            ssda_cfg.get('target_need_source_mix_soft_stuff_max', 1))
        self.target_need_target_mix = bool(
            ssda_cfg.get('target_need_target_mix', False))
        self.target_need_target_mix_random_prob = float(
            ssda_cfg.get(
                'target_need_target_mix_random_prob',
                self.target_need_source_mix_random_prob))
        self.target_need_target_mix_quota = bool(
            ssda_cfg.get('target_need_target_mix_quota', False))
        self._target_need_loss_ema = None
        self.target_deficit_quota_source_mix = bool(
            ssda_cfg.get('target_deficit_quota_source_mix', False))
        self.target_deficit_quota_min_classes = int(
            ssda_cfg.get('target_deficit_quota_min_classes', 1))
        self.target_deficit_quota_topk = int(
            ssda_cfg.get('target_deficit_quota_topk', 6))
        self.target_deficit_quota_stuff_classes = (
            self._parse_feature_prototype_class_ids(
                ssda_cfg.get(
                    'target_deficit_quota_stuff_classes',
                    'cityscapes_large_stuff')))
        self.target_deficit_quota_stuff_max = int(
            ssda_cfg.get('target_deficit_quota_stuff_max', 1))
        self.target_deficit_quota_random_prob = float(
            ssda_cfg.get(
                'target_deficit_quota_random_prob',
                self.target_need_source_mix_random_prob))
        self.target_deficit_quota_random_tie_break = bool(
            ssda_cfg.get(
                'target_deficit_quota_random_tie_break', False))
        self.target_need_mask_routing_v2 = bool(
            ssda_cfg.get('target_need_mask_routing_v2', False))
        self.target_need_mask_routing_use_tdef = bool(
            ssda_cfg.get('target_need_mask_routing_use_tdef', True))
        self.target_need_mask_routing_need_min_classes = int(
            ssda_cfg.get('target_need_mask_routing_need_min_classes', 2))
        self.target_need_mask_routing_need_topk = int(
            ssda_cfg.get('target_need_mask_routing_need_topk', 6))
        self.target_need_mask_routing_structure_classes = (
            self._parse_feature_prototype_class_ids(
                ssda_cfg.get(
                    'target_need_mask_routing_structure_classes',
                    'cityscapes_stuff')))
        self.target_need_mask_routing_structure_min_classes = int(
            ssda_cfg.get(
                'target_need_mask_routing_structure_min_classes', 1))
        self.target_need_mask_routing_structure_max_classes = int(
            ssda_cfg.get(
                'target_need_mask_routing_structure_max_classes', 1))
        self.target_need_mask_routing_dynamic_classes = (
            self._parse_feature_prototype_class_ids(
                ssda_cfg.get(
                    'target_need_mask_routing_dynamic_classes',
                    'cityscapes_dynamic')))
        self.target_need_mask_routing_dynamic_min_classes = int(
            ssda_cfg.get(
                'target_need_mask_routing_dynamic_min_classes', 1))
        self.target_need_mask_routing_random_prob = float(
            ssda_cfg.get(
                'target_need_mask_routing_random_prob',
                self.target_need_source_mix_random_prob))
        self.target_deficit_source_pixel_reweight = bool(
            ssda_cfg.get('target_deficit_source_pixel_reweight', False))
        self.target_deficit_source_pixel_min_weight = float(
            ssda_cfg.get('target_deficit_source_pixel_min_weight', 0.25))
        self.target_deficit_source_pixel_max_weight = float(
            ssda_cfg.get('target_deficit_source_pixel_max_weight', 4.0))
        self.target_deficit_source_pixel_gamma = float(
            ssda_cfg.get('target_deficit_source_pixel_gamma', 1.0))
        self.source_assist_diagnostics = bool(
            ssda_cfg.get('source_assist_diagnostics', False))
        self.source_assist_diagnostic_interval = int(
            ssda_cfg.get('source_assist_diagnostic_interval',
                         self.debug_img_interval or 2000))
        self.source_assist_diagnostic_dir = ssda_cfg.get(
            'source_assist_diagnostic_dir', None)
        self._source_assist_target_labeled_counts = None
        self._source_assist_target_unlabeled_counts = None
        self._source_assist_target_unlabeled_confidence = None
        self._source_assist_target_deficit_scores = None
        self._source_assist_route_scores = None
        self._source_assist_source_mix_scores = None
        self._source_assist_source_mix_selected_counts = None
        self._source_assist_source_mix_total_count = 0.0
        self._source_assist_loss_contributions = {}
        self._source_assist_diagnostic_exporter = None

        self.class_conditional_source_routing = bool(
            ssda_cfg.get('class_conditional_source_routing', False))
        self.class_conditional_source_route_enhance_classes = (
            self._parse_feature_prototype_class_ids(
                ssda_cfg.get(
                    'class_conditional_source_route_enhance_classes',
                    None)))
        self.class_conditional_source_route_suppress_classes = (
            self._parse_feature_prototype_class_ids(
                ssda_cfg.get(
                    'class_conditional_source_route_suppress_classes',
                    None)))
        self.class_conditional_source_route_enhance_score = float(
            ssda_cfg.get('class_conditional_source_route_enhance_score', 1.5))
        self.class_conditional_source_route_suppress_score = float(
            ssda_cfg.get('class_conditional_source_route_suppress_score', 0.5))
        self.class_conditional_source_route_min_score = float(
            ssda_cfg.get('class_conditional_source_route_min_score', 0.25))
        self.class_conditional_source_route_max_score = float(
            ssda_cfg.get('class_conditional_source_route_max_score', 4.0))
        self.class_conditional_source_route_random_prob = float(
            ssda_cfg.get('class_conditional_source_route_random_prob', 0.2))
        self.self_calibrated_class_routing = bool(
            ssda_cfg.get('self_calibrated_class_routing', False))
        self.self_calibrated_class_route_begin_iter = int(
            ssda_cfg.get(
                'self_calibrated_class_route_begin_iter',
                self.target_need_source_mix_begin_iter))
        self.self_calibrated_class_route_momentum = float(
            ssda_cfg.get('self_calibrated_class_route_momentum', 0.9))
        self.self_calibrated_class_route_reliability_min = float(
            ssda_cfg.get(
                'self_calibrated_class_route_reliability_min', 0.25))
        self.self_calibrated_class_route_intervention_quantile = float(
            ssda_cfg.get(
                'self_calibrated_class_route_intervention_quantile', 0.25))
        self.self_calibrated_class_route_max_delta = float(
            ssda_cfg.get('self_calibrated_class_route_max_delta', 0.5))
        self.self_calibrated_class_route_min_need_percentile = float(
            ssda_cfg.get(
                'self_calibrated_class_route_min_need_percentile', 0.5))
        self.self_calibrated_class_route_random_prob = float(
            ssda_cfg.get(
                'self_calibrated_class_route_random_prob',
                self.target_need_source_mix_random_prob))
        self._self_calibrated_class_route_deficit_ema = None
        self._self_calibrated_class_route_transfer_ema = None
        self._self_calibrated_class_route_reliability_ema = None
        self._self_calibrated_class_route_deficit_ema_valid = None
        self._self_calibrated_class_route_transfer_ema_valid = None
        self._self_calibrated_class_route_reliability_ema_valid = None
        self._self_calibrated_class_route_current_scores = None

        self.feature_tri_prototype_enabled = bool(
            ssda_cfg.get('feature_tri_prototype_enabled', False))
        self.feature_tri_prototype_weight = float(
            ssda_cfg.get('feature_tri_prototype_weight', 0.0))
        self.feature_tri_prototype_weight_final = float(
            ssda_cfg.get(
                'feature_tri_prototype_weight_final',
                self.feature_tri_prototype_weight))
        self.feature_tri_prototype_weight_schedule = str(
            ssda_cfg.get('feature_tri_prototype_weight_schedule', 'constant'))
        self.feature_tri_prototype_begin_iter = int(
            ssda_cfg.get('feature_tri_prototype_begin_iter',
                         self.semi_begin_iter))
        self.feature_tri_prototype_temperature = float(
            ssda_cfg.get('feature_tri_prototype_temperature', 0.1))
        self.feature_tri_prototype_conf_threshold = float(
            ssda_cfg.get('feature_tri_prototype_conf_threshold',
                         self.feature_prototype_conf_threshold))
        self.feature_tri_prototype_target_labeled_weight = float(
            ssda_cfg.get('feature_tri_prototype_target_labeled_weight', 1.0))
        self.feature_tri_prototype_target_unlabeled_weight = float(
            ssda_cfg.get('feature_tri_prototype_target_unlabeled_weight', 0.5))
        self.feature_tri_prototype_confidence_weight = bool(
            ssda_cfg.get('feature_tri_prototype_confidence_weight', True))
        self.feature_tri_prototype_unlabeled_forward = bool(
            ssda_cfg.get('feature_tri_prototype_unlabeled_forward', True))
        self.feature_tri_prototype_min_valid_pixels = int(
            ssda_cfg.get('feature_tri_prototype_min_valid_pixels', 16))

        self.tri_prototype_enabled = bool(
            ssda_cfg.get('tri_prototype_enabled', False))
        self.tri_prototype_weight = float(
            ssda_cfg.get('tri_prototype_weight', 0.0))
        self.tri_prototype_weight_final = float(
            ssda_cfg.get('tri_prototype_weight_final',
                         self.tri_prototype_weight))
        self.tri_prototype_weight_schedule = str(
            ssda_cfg.get('tri_prototype_weight_schedule', 'constant'))
        self.tri_prototype_begin_iter = int(
            ssda_cfg.get('tri_prototype_begin_iter', self.semi_begin_iter))
        self.tri_prototype_temperature = float(
            ssda_cfg.get('tri_prototype_temperature', 1.0))
        self.tri_prototype_conf_threshold = float(
            ssda_cfg.get('tri_prototype_conf_threshold', self.pseudo_threshold))
        self.tri_prototype_source_weight = float(
            ssda_cfg.get('tri_prototype_source_weight', 0.5))
        self.tri_prototype_target_weight = float(
            ssda_cfg.get('tri_prototype_target_weight', 1.0))
        self.tri_prototype_target_labeled_weight = float(
            ssda_cfg.get('tri_prototype_target_labeled_weight', 1.0))
        self.tri_prototype_target_unlabeled_weight = float(
            ssda_cfg.get('tri_prototype_target_unlabeled_weight', 0.5))
        self.tri_prototype_source_target_beta = float(
            ssda_cfg.get('tri_prototype_source_target_beta', 0.15))
        self.tri_prototype_proto_update = str(
            ssda_cfg.get('tri_prototype_proto_update', 'ema')).lower()
        self.tri_prototype_proto_momentum = float(
            ssda_cfg.get('tri_prototype_proto_momentum', 0.95))
        self._tri_source_hist_proto = None
        self._tri_target_labeled_hist_proto = None
        self._tri_target_unlabeled_hist_proto = None

        self.prototype_classmix_enabled = bool(
            ssda_cfg.get('prototype_classmix_enabled', False))
        self.prototype_classmix_begin_iter = int(
            ssda_cfg.get('prototype_classmix_begin_iter', self.semi_begin_iter))
        self.prototype_classmix_random_prob = float(
            ssda_cfg.get('prototype_classmix_random_prob', 0.2))
        self.prototype_classmix_need_gamma = float(
            ssda_cfg.get('prototype_classmix_need_gamma', 1.0))
        self.prototype_classmix_min_score = float(
            ssda_cfg.get('prototype_classmix_min_score', 0.05))
        self.prototype_classmix_conf_threshold = float(
            ssda_cfg.get('prototype_classmix_conf_threshold', self.pseudo_threshold))
        self.prototype_classmix_target_labeled_weight = float(
            ssda_cfg.get('prototype_classmix_target_labeled_weight', 1.0))
        self.prototype_classmix_target_unlabeled_weight = float(
            ssda_cfg.get('prototype_classmix_target_unlabeled_weight', 0.5))
        self.prototype_classmix_proto_update = str(
            ssda_cfg.get('prototype_classmix_proto_update', 'ema')).lower()
        self.prototype_classmix_proto_momentum = float(
            ssda_cfg.get('prototype_classmix_proto_momentum', 0.95))
        self._prototype_classmix_target_labeled_hist_proto = None
        self._prototype_classmix_target_unlabeled_hist_proto = None

        self.target_patch_memory_mix_enabled = bool(
            ssda_cfg.get('target_patch_memory_mix_enabled', False))
        self.target_patch_memory_mix_begin_iter = int(
            ssda_cfg.get('target_patch_memory_mix_begin_iter', self.semi_begin_iter))
        self.target_patch_memory_mix_grid_size = int(
            ssda_cfg.get('target_patch_memory_mix_grid_size', 8))
        self.target_patch_memory_mix_replace_ratio = float(
            ssda_cfg.get('target_patch_memory_mix_replace_ratio', 0.125))
        self.target_patch_memory_mix_capacity = int(
            ssda_cfg.get('target_patch_memory_mix_capacity', 256))
        self.target_patch_memory_mix_min_class_ratio = float(
            ssda_cfg.get('target_patch_memory_mix_min_class_ratio', 0.05))
        self.target_patch_memory_bank = (
            TargetPatchMemoryBank(self.target_patch_memory_mix_capacity)
            if self.target_patch_memory_mix_enabled else None)
        self.target_class_memory_mix_enabled = bool(
            ssda_cfg.get('target_class_memory_mix_enabled', False))
        self.target_class_memory_mix_begin_iter = int(
            ssda_cfg.get(
                'target_class_memory_mix_begin_iter',
                self.target_need_source_mix_begin_iter))
        self.target_class_memory_capacity_per_class = int(
            ssda_cfg.get('target_class_memory_capacity_per_class', 4))
        self.target_class_memory_min_pixels = int(
            ssda_cfg.get('target_class_memory_min_pixels', 32))
        self.target_class_memory_max_area_ratio = float(
            ssda_cfg.get('target_class_memory_max_area_ratio', 0.35))
        self.target_class_memory_max_classes = int(
            ssda_cfg.get('target_class_memory_max_classes', 2))
        self.target_class_memory_random_prob = float(
            ssda_cfg.get(
                'target_class_memory_random_prob',
                self.target_need_target_mix_random_prob))
        self.target_class_memory_min_score = float(
            ssda_cfg.get('target_class_memory_min_score', 0.0))
        self.target_class_memory_allowed_classes = (
            self._parse_feature_prototype_class_ids(
                ssda_cfg.get('target_class_memory_allowed_classes', None)))
        self.target_class_memory_blocked_classes = (
            self._parse_feature_prototype_class_ids(
                ssda_cfg.get('target_class_memory_blocked_classes', None)))
        self.target_class_memory_min_pseudo_conf = float(
            ssda_cfg.get('target_class_memory_min_pseudo_conf', 0.0))
        self.target_class_memory_offline_bank_path = str(
            ssda_cfg.get('target_class_memory_offline_bank_path', '') or '')
        self.target_class_memory_offline_strict = bool(
            ssda_cfg.get('target_class_memory_offline_strict', True))
        self.target_class_memory_offline_complete_only = bool(
            ssda_cfg.get('target_class_memory_offline_complete_only', False))
        self.target_class_memory_update_online = bool(
            ssda_cfg.get('target_class_memory_update_online', True))
        self.target_class_memory_sample_strategy = str(
            ssda_cfg.get('target_class_memory_sample_strategy', 'latest')).lower()
        self.target_class_memory_context_paste = bool(
            ssda_cfg.get('target_class_memory_context_paste', False))
        self.target_class_memory_context_candidates = int(
            ssda_cfg.get('target_class_memory_context_candidates', 9))
        self.target_class_memory_context_y_jitter = float(
            ssda_cfg.get('target_class_memory_context_y_jitter', 0.08))
        self.target_class_memory_max_paste_area_ratio = float(
            ssda_cfg.get('target_class_memory_max_paste_area_ratio', 0.08))
        self.target_class_memory_aux_enabled = bool(
            ssda_cfg.get('target_class_memory_aux_enabled', False))
        self.target_class_memory_aux_begin_iter = int(
            ssda_cfg.get(
                'target_class_memory_aux_begin_iter',
                self.target_class_memory_mix_begin_iter))
        self.target_class_memory_aux_weight = float(
            ssda_cfg.get('target_class_memory_aux_weight', 0.2))
        self.target_class_memory_aux_mask_only = bool(
            ssda_cfg.get('target_class_memory_aux_mask_only', False))
        self.target_class_memory_diagnostics = bool(
            ssda_cfg.get('target_class_memory_diagnostics', False))
        self.target_class_memory_diagnostic_per_class = bool(
            ssda_cfg.get('target_class_memory_diagnostic_per_class', True))
        self.target_class_memory_bank = (
            TargetClassMemoryBank(
                self.target_class_memory_capacity_per_class,
                sample_strategy=self.target_class_memory_sample_strategy)
            if (
                self.target_class_memory_mix_enabled
                or self.target_class_memory_aux_enabled
            ) else None)
        self.target_class_memory_offline_loaded = 0
        if (
            self.target_class_memory_bank is not None
            and self.target_class_memory_offline_bank_path
        ):
            self.target_class_memory_offline_bank_path = (
                self._format_target_class_memory_bank_path(
                    self.target_class_memory_offline_bank_path))
            self.target_class_memory_offline_loaded = (
                self.target_class_memory_bank.load_offline_jsonl(
                    self.target_class_memory_offline_bank_path,
                    rgb_mean=self.img_mean,
                    rgb_std=self.img_std,
                    allowed_classes=self.target_class_memory_allowed_classes,
                    min_pixels=self.target_class_memory_min_pixels,
                    max_area_ratio=self.target_class_memory_max_area_ratio,
                    require_object_complete=(
                        self.target_class_memory_offline_complete_only),
                    strict=self.target_class_memory_offline_strict,
                ))

        self.target_anchor_replay_enabled = bool(
            ssda_cfg.get('target_anchor_replay_enabled', False))
        self.target_anchor_replay_weight = float(
            ssda_cfg.get('target_anchor_replay_weight', 0.0))
        self.target_anchor_replay_weight_final = float(
            ssda_cfg.get(
                'target_anchor_replay_weight_final',
                self.target_anchor_replay_weight))
        self.target_anchor_replay_weight_schedule = str(
            ssda_cfg.get('target_anchor_replay_weight_schedule', 'constant'))
        self.target_anchor_replay_begin_iter = int(
            ssda_cfg.get('target_anchor_replay_begin_iter', self.semi_begin_iter))
        self.target_anchor_replay_rare_class_weight = float(
            ssda_cfg.get('target_anchor_replay_rare_class_weight', 1.0))
        self.target_anchor_replay_rare_class_ids = (
            self._parse_feature_prototype_class_ids(
                ssda_cfg.get('target_anchor_replay_rare_class_ids', None)))
        self.target_anchor_replay_class_conditional = bool(
            ssda_cfg.get('target_anchor_replay_class_conditional', False))
        self.target_anchor_replay_enhance_class_weight = float(
            ssda_cfg.get(
                'target_anchor_replay_enhance_class_weight',
                self.target_anchor_replay_rare_class_weight))
        enhance_ids = ssda_cfg.get('target_anchor_replay_enhance_class_ids', None)
        self.target_anchor_replay_enhance_class_ids = (
            self._parse_feature_prototype_class_ids(enhance_ids)
            if enhance_ids is not None else self.target_anchor_replay_rare_class_ids)
        self.target_anchor_replay_protect_class_weight = float(
            ssda_cfg.get('target_anchor_replay_protect_class_weight', 1.0))
        self.target_anchor_replay_protect_class_ids = (
            self._parse_feature_prototype_class_ids(
                ssda_cfg.get('target_anchor_replay_protect_class_ids', None)))
        self.target_anchor_replay_default_class_weight = float(
            ssda_cfg.get('target_anchor_replay_default_class_weight', 1.0))
        self.target_anchor_replay_normalize_mean = bool(
            ssda_cfg.get('target_anchor_replay_normalize_mean', False))

        self.unlabeled_consistency_enabled = bool(
            ssda_cfg.get('unlabeled_consistency_enabled', False))
        self.unlabeled_consistency_views = int(
            ssda_cfg.get('unlabeled_consistency_views', 1))
        self.unlabeled_consistency_views = min(2, max(1, self.unlabeled_consistency_views))
        self.unlabeled_consistency_weight = float(
            ssda_cfg.get('unlabeled_consistency_weight', 0.0))
        self.unlabeled_consistency_weight_final = float(
            ssda_cfg.get(
                'unlabeled_consistency_weight_final',
                self.unlabeled_consistency_weight))
        self.unlabeled_consistency_weight_schedule = str(
            ssda_cfg.get('unlabeled_consistency_weight_schedule', 'constant'))
        self.unlabeled_consistency_begin_iter = int(
            ssda_cfg.get('unlabeled_consistency_begin_iter', self.semi_begin_iter))
        self.unlabeled_consistency_total_divisor = float(
            ssda_cfg.get(
                'unlabeled_consistency_total_divisor',
                self.unlabeled_consistency_views))
        self.unlabeled_consistency_conf_thresh = float(
            ssda_cfg.get('unlabeled_consistency_conf_thresh', self.pseudo_threshold))
        self.unlabeled_consistency_confidence_weight = bool(
            ssda_cfg.get('unlabeled_consistency_confidence_weight', False))
        self.unlabeled_consistency_use_dacs_pseudo_weight = bool(
            ssda_cfg.get('unlabeled_consistency_use_dacs_pseudo_weight', True))
        self.unlabeled_consistency_cutmix_prob = float(
            ssda_cfg.get('unlabeled_consistency_cutmix_prob', 0.5))
        self.unlabeled_consistency_cutmix_min = float(
            ssda_cfg.get('unlabeled_consistency_cutmix_size_min', 0.02))
        self.unlabeled_consistency_cutmix_max = float(
            ssda_cfg.get('unlabeled_consistency_cutmix_size_max', 0.4))
        self.unlabeled_consistency_cutmix_ratio_min = float(
            ssda_cfg.get('unlabeled_consistency_cutmix_ratio_min', 0.3))
        self.unlabeled_consistency_cutmix_ratio_max = float(
            ssda_cfg.get('unlabeled_consistency_cutmix_ratio_max', 1 / 0.3))
        self.logger.info(
            '[SSDA] source_sup %.3f->%.3f (%s), target_sup %.3f->%.3f (%s), '
            'source_mix %.3f->%.3f (%s), target_mix %.3f->%.3f (%s), '
            'source_mix_target_mode=%s, source_labeled_mix_ratio=%.2f, '
            'target_dominant_mix=%s, '
            'target_mix_share %.3f->%.3f (%s), '
            'target_sup_redistribution=%s',
            self.source_sup_weight,
            self.source_sup_weight_final,
            self.source_sup_weight_schedule,
            self.target_sup_weight,
            self.target_sup_weight_final,
            self.target_sup_weight_schedule,
            self.source_mix_weight,
            self.source_mix_weight_final,
            self.source_mix_weight_schedule,
            self.target_mix_weight,
            self.target_mix_weight_final,
            self.target_mix_weight_schedule,
            self.source_mix_target_mode,
            self.source_labeled_mix_ratio,
            self.target_dominant_mix,
            self.target_mix_share,
            self.target_mix_share_final,
            self.target_mix_share_schedule,
            self.redistribute_target_sup_from_target_mix,
        )
        if self.effective_budget_target_sup:
            ratio_text = (
                'unknown'
                if self.target_sup_budget_labeled_ratio is None
                else f'{self.target_sup_budget_labeled_ratio:.6f}'
            )
            self.logger.info(
                '[SSDA] effective-budget T-Sup: labeled=%d, ratio=%s, '
                'effective=%.3f, gate=%.4f, weight=0->%.4f, ramp=%d->%d',
                self.target_sup_budget_sample_count,
                ratio_text,
                self.target_sup_budget_effective_count,
                self.target_sup_budget_gate,
                self.target_sup_weight_final,
                self.target_sup_weight_begin_iter,
                self.target_sup_weight_end_iter,
            )
        if self.relative_target_supervision_calibration:
            self.logger.info(
                '[SSDA] relative T-Sup: target=%d, source=%d, '
                'coverage=%.4f, weight=%.4f; target-mix unchanged',
                self.relative_target_sup_target_count,
                self.relative_target_sup_source_count,
                self.relative_target_sup_coverage,
                self.target_sup_weight,
            )
        if self.effective_budget_source_calibration:
            self.logger.info(
                '[SSDA] effective-budget source calibration: labeled=%d, '
                'effective=%.3f, gate=%.4f, source_sup=1->%.4f, '
                'target_mix_share=0.5->%.4f',
                self.source_calibration_sample_count,
                self.source_calibration_effective_count,
                self.source_calibration_budget_gate,
                self.source_sup_weight_final,
                self.target_mix_share_final,
            )
        if self.enable_source_labeled_aux_mix:
            self.logger.info(
                '[SSDA] auxiliary Mix(S,T_l) enabled with weight %.3f',
                self.source_labeled_aux_mix_weight,
            )
        if self.adaptive_target_dominant_mix:
            if self.adaptive_mix_mode == 'class_balanced_consistency_residual':
                self.logger.info(
                    '[SSDA] adaptive target-dominant mix: mode=%s, '
                    'schedule %.3f->%.3f (%s), warmup_iter=%d, '
                    'momentum=%.3f, residual_bound=%.3f, '
                    'consistency_threshold=%.3f, min_class_pixels=%d',
                    self.adaptive_mix_mode,
                    self.target_mix_share,
                    self.target_mix_share_final,
                    self.target_mix_share_schedule,
                    self.adaptive_mix_warmup_iter,
                    self.adaptive_mix_momentum,
                    self.adaptive_mix_residual_bound,
                    self.target_consistency_conf_threshold,
                    self.target_consistency_min_class_pixels,
                )
            elif self.adaptive_mix_mode == 'reliable_pixel_ratio':
                self.logger.info(
                    '[SSDA] adaptive target-dominant mix: mode=%s, '
                    'base_share=%.3f, warmup_iter=%d, momentum=%.3f, '
                    'progress_gate=%s',
                    self.adaptive_mix_mode,
                    self.target_mix_share,
                    self.adaptive_mix_warmup_iter,
                    self.adaptive_mix_momentum,
                    self.adaptive_mix_progress_gate,
                )
            else:
                self.logger.info(
                    '[SSDA] adaptive target-dominant mix: mode=%s, '
                    'share range %.3f->%.3f, warmup_iter=%d, momentum=%.3f, '
                    'conf_threshold=%.3f, unlabeled_weight=%.2f, '
                    'rel_w=%.2f, affinity_w=%.2f',
                    self.adaptive_mix_mode,
                    self.target_mix_share,
                    self.target_mix_share_final,
                    self.adaptive_mix_warmup_iter,
                    self.adaptive_mix_momentum,
                    self.adaptive_mix_conf_threshold,
                    self.adaptive_mix_unlabeled_weight,
                    self.adaptive_mix_reliability_weight,
                    self.adaptive_mix_affinity_weight,
                )
        if self.consistency_aware_target_mix:
            self.logger.info(
                '[SSDA] consistency-aware target mix: threshold=%.3f, '
                'min_class_pixels=%d, disagreement_weight=%.3f',
                self.target_consistency_conf_threshold,
                self.target_consistency_min_class_pixels,
                self.consistency_target_mix_disagreement_weight,
            )
        if self.conf_aware_target_mix:
            self.logger.info(
                '[SSDA] confidence-aware target mix: mode=%s, threshold=%.3f, '
                'conf_gamma=%.2f, entropy_gamma=%.2f, weight=[%.2f, %.2f], '
                'blend %.2f->%.2f (%s)',
                self.conf_aware_target_mix_mode,
                self.conf_aware_target_mix_threshold,
                self.conf_aware_target_mix_conf_gamma,
                self.conf_aware_target_mix_entropy_gamma,
                self.conf_aware_target_mix_min_weight,
                self.conf_aware_target_mix_max_weight,
                self.conf_aware_target_mix_blend,
                self.conf_aware_target_mix_blend_final,
                self.conf_aware_target_mix_blend_schedule,
            )
        if self.target_labeled_reliability_calibration:
            self.logger.info(
                '[SSDA] target-labeled reliability calibration: begin_iter=%d, '
                'reliability=[%.2f, %.2f], default=%.2f, momentum=%.3f, '
                'blend %.2f->%.2f (%s)',
                self.target_labeled_reliability_begin_iter,
                self.target_labeled_reliability_min,
                self.target_labeled_reliability_max,
                self.target_labeled_reliability_default,
                self.target_labeled_reliability_momentum,
                self.target_labeled_reliability_blend,
                self.target_labeled_reliability_blend_final,
                self.target_labeled_reliability_blend_schedule,
            )
        if self.target_guided_source_filter:
            self.logger.info(
                '[SSDA] target-guided source filtering: mode=%s, threshold=%.3f, '
                'weight=[%.2f, %.2f], gamma=%.2f, blend %.2f->%.2f (%s), '
                'labeled_weight=%.2f, unlabeled_weight=%.2f, proto_update=%s, '
                'proto_momentum=%.3f',
                self.target_guided_source_filter_mode,
                self.target_guided_source_filter_conf_threshold,
                self.target_guided_source_filter_min_weight,
                self.target_guided_source_filter_max_weight,
                self.target_guided_source_filter_gamma,
                self.target_guided_source_filter_blend,
                self.target_guided_source_filter_blend_final,
                self.target_guided_source_filter_blend_schedule,
                self.target_guided_source_filter_labeled_weight,
                self.target_guided_source_filter_unlabeled_weight,
                self.target_guided_source_filter_proto_update,
                self.target_guided_source_filter_proto_momentum,
            )
        if self.feature_prototype_source_calibration:
            self.logger.info(
                '[SSDA] feature-prototype source calibration: source_weight=%s, '
                'source_mix=%s, begin_iter=%d, feature_level=%s, min_pixels=%d, '
                'num_proto_per_class=%d, conf_threshold=%.3f, '
                'score=[%.3f, %.3f], weight=[%.2f, %.2f], '
                'target_labeled_weight=%.2f, target_unlabeled_weight=%.2f, '
                'momentum=%.3f, unlabeled_forward=%s, diagnostics=%s/%d',
                self.feature_prototype_source_weight,
                self.feature_prototype_source_mix,
                self.feature_prototype_begin_iter,
                self.feature_prototype_feature_level,
                self.feature_prototype_min_pixels,
                self.feature_prototype_num_prototypes_per_class,
                self.feature_prototype_conf_threshold,
                self.feature_prototype_min_score,
                self.feature_prototype_default_score,
                self.feature_prototype_min_weight,
                self.feature_prototype_max_weight,
                self.feature_prototype_target_labeled_weight,
                self.feature_prototype_target_unlabeled_weight,
                self.feature_prototype_momentum,
                self.feature_prototype_unlabeled_forward,
                self.feature_prototype_diagnostics,
                self.feature_prototype_diagnostic_interval,
            )
            if self.feature_prototype_source_mix_structure_protection:
                self.logger.info(
                    '[SSDA] feature-prototype source-mix structure protection: '
                    'classes=%s, mode=%s, min_score=%.3f, default_score=%.3f',
                    self.feature_prototype_source_mix_structure_classes,
                    self.feature_prototype_source_mix_structure_mode,
                    self.feature_prototype_source_mix_structure_min_score,
                    self.feature_prototype_source_mix_structure_default_score,
                )
        if self.feature_tri_prototype_enabled:
            self.logger.info(
                '[SSDA] feature tri-prototype target-only v2: weight %.3f->%.3f '
                '(%s), begin_iter=%d, temp=%.3f, conf_threshold=%.3f, '
                'target_labeled_weight=%.2f, target_unlabeled_weight=%.2f, '
                'confidence_weight=%s, unlabeled_forward=%s, '
                'num_proto_per_class=%d (current implementation uses 1)',
                self.feature_tri_prototype_weight,
                self.feature_tri_prototype_weight_final,
                self.feature_tri_prototype_weight_schedule,
                self.feature_tri_prototype_begin_iter,
                self.feature_tri_prototype_temperature,
                self.feature_tri_prototype_conf_threshold,
                self.feature_tri_prototype_target_labeled_weight,
                self.feature_tri_prototype_target_unlabeled_weight,
                self.feature_tri_prototype_confidence_weight,
                self.feature_tri_prototype_unlabeled_forward,
                self.feature_prototype_num_prototypes_per_class,
            )
        if self.tri_prototype_enabled:
            self.logger.info(
                '[SSDA] tri-prototype structured supervision: weight %.3f->%.3f '
                '(%s), begin_iter=%d, temp=%.2f, conf_threshold=%.3f, '
                'source_weight=%.2f, target_weight=%.2f, target_labeled_weight=%.2f, '
                'target_unlabeled_weight=%.2f, source_target_beta=%.2f, '
                'proto_update=%s, proto_momentum=%.3f',
                self.tri_prototype_weight,
                self.tri_prototype_weight_final,
                self.tri_prototype_weight_schedule,
                self.tri_prototype_begin_iter,
                self.tri_prototype_temperature,
                self.tri_prototype_conf_threshold,
                self.tri_prototype_source_weight,
                self.tri_prototype_target_weight,
                self.tri_prototype_target_labeled_weight,
                self.tri_prototype_target_unlabeled_weight,
                self.tri_prototype_source_target_beta,
                self.tri_prototype_proto_update,
                self.tri_prototype_proto_momentum,
            )
        if self.prototype_classmix_enabled:
            self.logger.info(
                '[SSDA] prototype-guided ClassMix: begin_iter=%d, '
                'random_prob=%.2f, need_gamma=%.2f, min_score=%.3f, '
                'conf_threshold=%.3f, target_labeled_weight=%.2f, '
                'target_unlabeled_weight=%.2f, proto_update=%s, '
                'proto_momentum=%.3f',
                self.prototype_classmix_begin_iter,
                self.prototype_classmix_random_prob,
                self.prototype_classmix_need_gamma,
                self.prototype_classmix_min_score,
                self.prototype_classmix_conf_threshold,
                self.prototype_classmix_target_labeled_weight,
                self.prototype_classmix_target_unlabeled_weight,
                self.prototype_classmix_proto_update,
                self.prototype_classmix_proto_momentum,
            )
        if self.target_need_source_mix:
            self.logger.info(
                '[SSDA] target-need source assistance: begin_iter=%d, '
                'score=[%.2f, %.2f], random_prob=%.2f, '
                'target_loss=%s/%.2f, source_transfer=%s/%.2f',
                self.target_need_source_mix_begin_iter,
                self.target_need_source_mix_min_score,
                self.target_need_source_mix_max_score,
                self.target_need_source_mix_random_prob,
                self.target_need_source_mix_use_target_loss,
                self.target_need_source_mix_target_loss_weight,
                self.target_need_source_mix_use_source_transfer,
                self.target_need_source_mix_source_transfer_weight,
            )
        if self.target_need_target_mix:
            self.logger.info(
                '[SSDA] target-need target mix: begin_iter=%d, '
                'random_prob=%.2f, quota=%s',
                self.target_need_source_mix_begin_iter,
                self.target_need_target_mix_random_prob,
                self.target_need_target_mix_quota,
            )
        if self.class_conditional_source_routing:
            self.logger.info(
                '[SSDA] class-conditional source routing: enhance=%s x%.2f, '
                'suppress=%s x%.2f, score=[%.2f, %.2f], random_prob=%.2f',
                self.class_conditional_source_route_enhance_classes,
                self.class_conditional_source_route_enhance_score,
                self.class_conditional_source_route_suppress_classes,
                self.class_conditional_source_route_suppress_score,
                self.class_conditional_source_route_min_score,
                self.class_conditional_source_route_max_score,
                self.class_conditional_source_route_random_prob,
            )
        if self.conflict_aware_class_routing:
            self.logger.info(
                '[SSDA] conflict-aware class routing: begin_iter=%d, '
                'interval=%d, momentum=%.3f, max_classes=%d, '
                'min_pixels=(%d,%d), min_abs_cos=%.3f, '
                'strength=(%.2f,%.2f), score=[%.2f,%.2f]',
                self.conflict_aware_class_route_begin_iter,
                self.conflict_aware_class_route_update_interval,
                self.conflict_aware_class_route_momentum,
                self.conflict_aware_class_route_max_classes,
                self.conflict_aware_class_route_min_source_pixels,
                self.conflict_aware_class_route_min_target_pixels,
                self.conflict_aware_class_route_min_abs_cosine,
                self.conflict_aware_class_route_assist_strength,
                self.conflict_aware_class_route_reject_strength,
                self.conflict_aware_class_route_min_score,
                self.conflict_aware_class_route_max_score,
            )
        if self.self_calibrated_class_routing:
            self.logger.info(
                '[SSDA] self-calibrated class-route v3: begin_iter=%d, '
                'momentum=%.3f, reliability_min=%.2f, quantile=%.2f, '
                'max_delta=%.2f, random_prob=%.2f',
                self.self_calibrated_class_route_begin_iter,
                self.self_calibrated_class_route_momentum,
                self.self_calibrated_class_route_reliability_min,
                self.self_calibrated_class_route_intervention_quantile,
                self.self_calibrated_class_route_max_delta,
                self.self_calibrated_class_route_random_prob,
            )
        if self.target_patch_memory_mix_enabled:
            self.logger.info(
                '[SSDA] target patch memory mix: begin_iter=%d, grid=%d, '
                'replace_ratio=%.3f, capacity=%d, min_class_ratio=%.3f',
                self.target_patch_memory_mix_begin_iter,
                self.target_patch_memory_mix_grid_size,
                self.target_patch_memory_mix_replace_ratio,
                self.target_patch_memory_mix_capacity,
                self.target_patch_memory_mix_min_class_ratio,
            )
        if self.target_class_memory_mix_enabled:
            self.logger.info(
                '[SSDA] target class memory mix: begin_iter=%d, '
                'capacity_per_class=%d, min_pixels=%d, max_area=%.2f, '
                'max_classes=%d, random_prob=%.2f, offline_loaded=%d',
                self.target_class_memory_mix_begin_iter,
                self.target_class_memory_capacity_per_class,
                self.target_class_memory_min_pixels,
                self.target_class_memory_max_area_ratio,
                self.target_class_memory_max_classes,
                self.target_class_memory_random_prob,
                self.target_class_memory_offline_loaded,
            )
        if self.target_class_memory_aux_enabled:
            self.logger.info(
                '[SSDA] auxiliary target class memory paste: '
                'begin_iter=%d, weight=%.3f, capacity_per_class=%d, '
                'min_pixels=%d, max_area=%.2f, max_classes=%d, '
                'random_prob=%.2f, min_pseudo_conf=%.2f, allowed=%s, '
                'blocked=%s, update_online=%s, sample=%s, context=%s, '
                'offline=%s (%d), complete_only=%s, mask_only=%s, '
                'diagnostics=%s',
                self.target_class_memory_aux_begin_iter,
                self.target_class_memory_aux_weight,
                self.target_class_memory_capacity_per_class,
                self.target_class_memory_min_pixels,
                self.target_class_memory_max_area_ratio,
                self.target_class_memory_max_classes,
                self.target_class_memory_random_prob,
                self.target_class_memory_min_pseudo_conf,
                self.target_class_memory_allowed_classes,
                self.target_class_memory_blocked_classes,
                self.target_class_memory_update_online,
                self.target_class_memory_sample_strategy,
                self.target_class_memory_context_paste,
                self.target_class_memory_offline_bank_path or 'None',
                self.target_class_memory_offline_loaded,
                self.target_class_memory_offline_complete_only,
                self.target_class_memory_aux_mask_only,
                self.target_class_memory_diagnostics,
            )
        if self.target_anchor_replay_enabled:
            self.logger.info(
                '[SSDA] target anchor replay: weight %.3f->%.3f (%s), '
                'begin_iter=%d, rare_class_weight=%.2f, rare_class_ids=%s, '
                'class_conditional=%s, enhance_weight=%.2f, enhance_ids=%s, '
                'protect_weight=%.2f, protect_ids=%s, normalize_mean=%s',
                self.target_anchor_replay_weight,
                self.target_anchor_replay_weight_final,
                self.target_anchor_replay_weight_schedule,
                self.target_anchor_replay_begin_iter,
                self.target_anchor_replay_rare_class_weight,
                self.target_anchor_replay_rare_class_ids,
                self.target_anchor_replay_class_conditional,
                self.target_anchor_replay_enhance_class_weight,
                self.target_anchor_replay_enhance_class_ids,
                self.target_anchor_replay_protect_class_weight,
                self.target_anchor_replay_protect_class_ids,
                self.target_anchor_replay_normalize_mean,
            )
        if self.unlabeled_consistency_enabled:
            self.logger.info(
                '[SSDA] unlabeled consistency branch: views=%d, weight %.3f->%.3f '
                '(%s), begin_iter=%d, divisor=%.2f, conf_thresh=%.3f, '
                'cutmix_prob=%.2f, confidence_weight=%s',
                self.unlabeled_consistency_views,
                self.unlabeled_consistency_weight,
                self.unlabeled_consistency_weight_final,
                self.unlabeled_consistency_weight_schedule,
                self.unlabeled_consistency_begin_iter,
                self.unlabeled_consistency_total_divisor,
                self.unlabeled_consistency_conf_thresh,
                self.unlabeled_consistency_cutmix_prob,
                self.unlabeled_consistency_confidence_weight,
            )

    def _configure_effective_budget_target_sup(self, ssda_cfg):
        """Resolve T-Sup strength/timing from the final labeled split."""
        resolved = resolve_effective_budget_target_sup_config(self.cfg)
        self.target_sup_weight = resolved['initial_weight']
        self.target_sup_weight_final = resolved['final_weight']
        self.target_sup_weight_schedule = 'linear_between'
        self.target_sup_weight_begin_iter = resolved['begin_iter']
        self.target_sup_weight_end_iter = resolved['end_iter']
        self.target_sup_budget_sample_count = resolved['sample_count']
        self.target_sup_budget_effective_count = resolved['effective_count']
        self.target_sup_budget_gate = resolved['gate']
        self.target_sup_budget_labeled_ratio = ssda_cfg.get(
            'target_sup_budget_labeled_ratio', None)

    def _scheduled_weight(self, initial, final, schedule):
        schedule = schedule.lower()
        if schedule == 'constant':
            return initial

        progress = min(1.0, max(0.0, float(self.local_iter) / max(1, self.max_iters)))
        if schedule in ('linear', 'linear_decay'):
            factor = progress
        elif schedule == 'cosine':
            factor = 0.5 - 0.5 * math.cos(math.pi * progress)
        elif schedule == 'exp1':
            factor = 1.0 - math.exp(-4.0 * progress)
        elif schedule == 'exp2':
            factor = (math.exp(4.0 * progress) - 1.0) / (math.exp(4.0) - 1.0)
        else:
            raise ValueError(
                "Invalid SSDA weight schedule. Choose from 'constant', "
                "'linear', 'linear_decay', 'cosine', 'exp1', or 'exp2'.")
        return initial + (final - initial) * factor

    def _format_target_class_memory_bank_path(self, path):
        data_cfg = self.cfg.data
        try:
            return str(path).format(
                dataset=data_cfg.get('dataset_name', ''),
                source_dataset=data_cfg.get('source_dataset_name', ''),
                split_method=data_cfg.get('split_method', ''),
                ratio=data_cfg.get('split_ratio', ''),
                class_set=data_cfg.get(
                    'class_set',
                    data_cfg.get('dataset_name', ''),
                ),
            )
        except (KeyError, IndexError):
            return str(path)

    def _branch_weight(self, name):
        begin_iter = int(getattr(self, f'{name}_weight_begin_iter', 0))
        if self.local_iter < begin_iter:
            return 0.0
        schedule = str(getattr(self, f'{name}_weight_schedule'))
        if schedule.lower() == 'linear_between':
            initial = float(getattr(self, f'{name}_weight'))
            final = float(getattr(self, f'{name}_weight_final'))
            end_iter = int(getattr(
                self,
                f'{name}_weight_end_iter',
                self.max_iters,
            ))
            if end_iter <= begin_iter:
                raise ValueError(
                    f'{name} weight end_iter must be greater than begin_iter')
            progress = min(
                1.0,
                max(
                    0.0,
                    float(self.local_iter - begin_iter)
                    / (end_iter - begin_iter),
                ),
            )
            return initial + (final - initial) * progress
        if schedule.lower() == 'linear_after_begin':
            initial = float(getattr(self, f'{name}_weight'))
            final = float(getattr(self, f'{name}_weight_final'))
            progress = min(
                1.0,
                max(
                    0.0,
                    float(self.local_iter - begin_iter)
                    / max(1, self.max_iters - begin_iter),
                ),
            )
            return initial + (final - initial) * progress
        return float(self._scheduled_weight(
            getattr(self, f'{name}_weight'),
            getattr(self, f'{name}_weight_final'),
            schedule,
        ))

    def _gradient_aligned_source_active(self):
        return (
            self.gradient_aligned_source_assistance
            and self.local_iter >= self.gradient_aligned_source_begin_iter)

    def _gradient_alignment_params(self):
        if self._gradient_aligned_source_param_cache is not None:
            return self._gradient_aligned_source_param_cache

        params = []
        keywords = tuple(
            str(k).lower()
            for k in self.gradient_aligned_source_param_keywords
            if str(k))
        for name, param in self.get_model().named_parameters():
            if not param.requires_grad:
                continue
            lname = name.lower()
            if keywords and not any(keyword in lname for keyword in keywords):
                continue
            params.append(param)

        if not params:
            # Fallback to the last few trainable tensors so the feature remains
            # usable for decoder variants whose classifier names differ.
            trainable = [
                param for _, param in self.get_model().named_parameters()
                if param.requires_grad
            ]
            params = trainable[-8:]
        self._gradient_aligned_source_param_cache = tuple(params)
        return self._gradient_aligned_source_param_cache

    def _flatten_grad_pair(self, source_grads, target_grads, params):
        source_parts = []
        target_parts = []
        for source_grad, target_grad, param in zip(source_grads, target_grads, params):
            if source_grad is None and target_grad is None:
                continue
            if source_grad is None:
                source_grad = torch.zeros_like(param)
            if target_grad is None:
                target_grad = torch.zeros_like(param)
            source_parts.append(source_grad.detach().float().flatten())
            target_parts.append(target_grad.detach().float().flatten())
        if not source_parts:
            return None, None
        return torch.cat(source_parts), torch.cat(target_parts)

    def _gradient_aligned_source_scale(self, source_loss, target_loss):
        if not self._gradient_aligned_source_active():
            return 1.0, {}
        if source_loss is None or target_loss is None:
            return 1.0, {'ssda_gradalign_valid': 0.0}
        if not source_loss.requires_grad or not target_loss.requires_grad:
            return 1.0, {'ssda_gradalign_valid': 0.0}

        params = self._gradient_alignment_params()
        if not params:
            return 1.0, {'ssda_gradalign_valid': 0.0}

        source_grads = torch.autograd.grad(
            source_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        target_grads = torch.autograd.grad(
            target_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        source_vec, target_vec = self._flatten_grad_pair(
            source_grads,
            target_grads,
            params,
        )
        if source_vec is None or target_vec is None:
            return 1.0, {'ssda_gradalign_valid': 0.0}

        source_norm = source_vec.norm()
        target_norm = target_vec.norm()
        if source_norm.item() <= 0 or target_norm.item() <= 0:
            return 1.0, {
                'ssda_gradalign_valid': 0.0,
                'ssda_gradalign_source_norm': float(source_norm.detach().item()),
                'ssda_gradalign_target_norm': float(target_norm.detach().item()),
            }

        cosine = F.cosine_similarity(source_vec, target_vec, dim=0).clamp(-1.0, 1.0)
        threshold = float(self.gradient_aligned_source_conflict_threshold)
        min_scale = float(self.gradient_aligned_source_min_scale)
        max_scale = float(self.gradient_aligned_source_max_scale)
        if cosine.item() >= threshold:
            scale = max_scale
        else:
            denom = max(1e-6, threshold + 1.0)
            alpha = max(0.0, min(1.0, (float(cosine.item()) + 1.0) / denom))
            scale = min_scale + (max_scale - min_scale) * alpha
        scale = max(min_scale, min(max_scale, float(scale)))
        return scale, {
            'ssda_gradalign_valid': 1.0,
            'ssda_gradalign_cosine': float(cosine.detach().item()),
            'ssda_gradalign_source_scale': float(scale),
            'ssda_gradalign_param_count': float(len(params)),
            'ssda_gradalign_source_norm': float(source_norm.detach().item()),
            'ssda_gradalign_target_norm': float(target_norm.detach().item()),
            'ssda_gradalign_apply_to_mix': float(
                self.gradient_aligned_source_apply_to_mix),
        }

    def _conflict_route_update_due(self):
        if not self.conflict_aware_class_routing:
            return False
        offset = self.local_iter - self.conflict_aware_class_route_begin_iter
        return (
            offset >= 0
            and offset % self.conflict_aware_class_route_update_interval == 0
        )

    def _conflict_route_params(self):
        if self._conflict_aware_class_route_param_cache is not None:
            return self._conflict_aware_class_route_param_cache
        keywords = tuple(
            str(keyword).lower()
            for keyword in self.conflict_aware_class_route_param_keywords
            if str(keyword)
        )
        params = [
            param
            for name, param in self.get_model().named_parameters()
            if param.requires_grad
            and (not keywords or any(key in name.lower() for key in keywords))
        ]
        if not params:
            trainable = [
                param for _, param in self.get_model().named_parameters()
                if param.requires_grad
            ]
            params = trainable[-8:]
        self._conflict_aware_class_route_param_cache = tuple(params)
        return self._conflict_aware_class_route_param_cache

    def _per_class_gradient_losses(self, logits, labels, min_pixels):
        logits = self._select_feature_proto_diag_logits(logits)
        if logits is None or labels is None:
            return {}, {}
        labels = self._resize_diag_map(
            labels.detach().long(),
            logits.shape[-2:],
            logits.device,
        )
        if labels is None:
            return {}, {}
        ignore_index = getattr(self, 'ignore_index', 255)
        valid = labels.ne(ignore_index) & labels.ge(0) & labels.lt(self.num_classes)
        safe_labels = labels.clone()
        safe_labels[~valid] = ignore_index
        pixel_loss = F.cross_entropy(
            logits.float(),
            safe_labels,
            reduction='none',
            ignore_index=ignore_index,
        )
        losses = {}
        counts = {}
        for class_id in torch.unique(safe_labels[valid]).tolist():
            class_id = int(class_id)
            class_mask = valid & safe_labels.eq(class_id)
            count = int(class_mask.sum().item())
            if count < int(min_pixels):
                continue
            losses[class_id] = pixel_loss[class_mask].mean()
            counts[class_id] = count
        return losses, counts

    def _update_conflict_route_ema(self, cosines, valid):
        cosines = cosines.detach().float().flatten()
        valid = valid.detach().bool().flatten()
        if self._conflict_aware_class_route_cosine_ema is None:
            ema = torch.zeros_like(cosines)
            previous_valid = torch.zeros_like(valid)
        else:
            ema = self._conflict_aware_class_route_cosine_ema.to(
                device=cosines.device).clone()
            previous_valid = self._conflict_aware_class_route_valid.to(
                device=cosines.device).bool().clone()
        newly_valid = valid & ~previous_valid
        continuing = valid & previous_valid
        ema[newly_valid] = cosines[newly_valid]
        momentum = float(self.conflict_aware_class_route_momentum)
        ema[continuing] = (
            momentum * ema[continuing]
            + (1.0 - momentum) * cosines[continuing]
        )
        previous_valid |= valid
        self._conflict_aware_class_route_cosine_ema = ema.detach()
        self._conflict_aware_class_route_valid = previous_valid.detach()
        return {
            'ssda_conflict_route_update_valid': float(valid.any()),
            'ssda_conflict_route_updated_classes': float(valid.sum().item()),
            'ssda_conflict_route_cosine_ema_mean': float(
                ema[previous_valid].mean().item()
                if previous_valid.any() else 0.0),
        }

    def _update_conflict_route_gradients(
        self,
        source_logits,
        source_labels,
        target_logits,
        target_labels,
    ):
        source_losses, source_counts = self._per_class_gradient_losses(
            source_logits,
            source_labels,
            self.conflict_aware_class_route_min_source_pixels,
        )
        target_losses, target_counts = self._per_class_gradient_losses(
            target_logits,
            target_labels,
            self.conflict_aware_class_route_min_target_pixels,
        )
        shared = sorted(set(source_losses).intersection(target_losses))
        if not shared:
            return {
                'ssda_conflict_route_update_valid': 0.0,
                'ssda_conflict_route_shared_classes': 0.0,
            }
        if len(shared) > self.conflict_aware_class_route_max_classes:
            if self._target_need_current_scores is None:
                shared = sorted(
                    shared,
                    key=lambda class_id: target_counts[class_id],
                )[:self.conflict_aware_class_route_max_classes]
            else:
                need = self._target_need_current_scores
                shared = sorted(
                    shared,
                    key=lambda class_id: float(need[class_id]),
                    reverse=True,
                )[:self.conflict_aware_class_route_max_classes]
        cosines, valid = per_class_gradient_cosines(
            source_losses,
            target_losses,
            self._conflict_route_params(),
            num_classes=self.num_classes,
            class_ids=shared,
        )
        logs = self._update_conflict_route_ema(cosines, valid)
        logs.update({
            'ssda_conflict_route_shared_classes': float(len(shared)),
            'ssda_conflict_route_source_pixels': float(
                sum(source_counts[class_id] for class_id in shared)),
            'ssda_conflict_route_target_pixels': float(
                sum(target_counts[class_id] for class_id in shared)),
        })
        return logs

    def _conflict_aware_route_scores(self, target_deficit_scores, device):
        if (
            not self.conflict_aware_class_routing
            or self._conflict_aware_class_route_cosine_ema is None
            or self._conflict_aware_class_route_valid is None
        ):
            return None, {'ssda_conflict_route_active': 0.0}
        route, stats = class_gradient_conflict_route_scores(
            target_deficit_scores.to(device=device),
            self._conflict_aware_class_route_cosine_ema.to(device=device),
            self._conflict_aware_class_route_valid.to(device=device),
            assist_strength=self.conflict_aware_class_route_assist_strength,
            reject_strength=self.conflict_aware_class_route_reject_strength,
            min_abs_cosine=self.conflict_aware_class_route_min_abs_cosine,
            min_score=self.conflict_aware_class_route_min_score,
            max_score=self.conflict_aware_class_route_max_score,
        )
        logs = {'ssda_conflict_route_active': 1.0}
        logs.update({
            f'ssda_conflict_route_{key}': float(value)
            for key, value in stats.items()
        })
        return route, logs

    def _tri_prototype_branch_weight(self):
        if not self.tri_prototype_enabled:
            return 0.0
        if self.local_iter < self.tri_prototype_begin_iter:
            return 0.0
        return float(self._scheduled_weight(
            self.tri_prototype_weight,
            self.tri_prototype_weight_final,
            self.tri_prototype_weight_schedule,
        ))

    def _feature_tri_prototype_branch_weight(self):
        if not self.feature_tri_prototype_enabled:
            return 0.0
        if self.local_iter < self.feature_tri_prototype_begin_iter:
            return 0.0
        return float(self._scheduled_weight(
            self.feature_tri_prototype_weight,
            self.feature_tri_prototype_weight_final,
            self.feature_tri_prototype_weight_schedule,
        ))

    def _adaptive_target_mix_share(self, scheduled_share):
        self._last_adaptive_mix_log_vars = {}
        if not self.adaptive_target_dominant_mix:
            return scheduled_share

        target_reliability = (
            self.adaptive_mix_default_reliability
            if self._adaptive_mix_target_reliability_ema is None
            else self._adaptive_mix_target_reliability_ema)
        active = self.local_iter >= self.adaptive_mix_warmup_iter
        if self.adaptive_mix_mode == 'class_balanced_consistency_residual':
            reference = self._adaptive_mix_reliability_reference
            if active and reference is not None:
                adaptive_share = compute_bounded_residual_target_mix_share(
                    scheduled_share=scheduled_share,
                    reliability=target_reliability,
                    reference_reliability=reference,
                    max_residual=self.adaptive_mix_residual_bound,
                    lower_share=self.target_mix_share,
                    upper_share=self.target_mix_share_final,
                )
            else:
                adaptive_share = scheduled_share
            residual = float(adaptive_share) - float(scheduled_share)
            self._last_adaptive_mix_log_vars = {
                'ssda_adaptive_mix_enabled': 1.0,
                'ssda_adaptive_mix_active': float(
                    active and reference is not None),
                'ssda_adaptive_mix_scheduled_share': float(scheduled_share),
                'ssda_adaptive_mix_consistency_ema': float(
                    target_reliability),
                'ssda_adaptive_mix_consistency_reference': float(
                    target_reliability if reference is None else reference),
                'ssda_adaptive_mix_residual': residual,
                'ssda_adaptive_mix_share': float(adaptive_share),
            }
            return adaptive_share

        if self.adaptive_mix_mode == 'reliable_pixel_ratio':
            progress = (
                min(
                    1.0,
                    max(
                        0.0,
                        float(self.local_iter) / max(1, self.max_iters),
                    ),
                )
                if self.adaptive_mix_progress_gate
                else 1.0
            )
            adaptive_share = compute_reliable_ratio_target_mix_share(
                self.target_mix_share,
                target_reliability,
                progress=progress,
            )
            target_mix_share = adaptive_share if active else scheduled_share
            self._last_adaptive_mix_log_vars = {
                'ssda_adaptive_mix_enabled': 1.0,
                'ssda_adaptive_mix_active': float(active),
                'ssda_adaptive_mix_progress_gate': float(
                    self.adaptive_mix_progress_gate),
                'ssda_adaptive_mix_progress': float(progress),
                'ssda_adaptive_mix_scheduled_share': float(scheduled_share),
                'ssda_adaptive_mix_reliable_ratio_ema': float(
                    target_reliability),
                'ssda_adaptive_mix_raw_share': float(adaptive_share),
                'ssda_adaptive_mix_share': float(target_mix_share),
            }
            return target_mix_share

        source_affinity = (
            self.adaptive_mix_default_source_affinity
            if self._adaptive_mix_source_affinity_ema is None
            else self._adaptive_mix_source_affinity_ema)
        denom = self.adaptive_mix_reliability_weight + self.adaptive_mix_affinity_weight
        if denom <= 0:
            adaptive_signal = 0.5
        else:
            adaptive_signal = (
                self.adaptive_mix_reliability_weight * target_reliability
                + self.adaptive_mix_affinity_weight * (1.0 - source_affinity)
            ) / denom

        adaptive_share = compute_adaptive_target_mix_share(
            lower_share=self.target_mix_share,
            upper_share=self.target_mix_share_final,
            target_reliability=target_reliability,
            source_affinity=source_affinity,
            reliability_weight=self.adaptive_mix_reliability_weight,
            affinity_weight=self.adaptive_mix_affinity_weight,
        )
        raw_adaptive_share = adaptive_share
        if active and self.adaptive_mix_schedule_floor:
            adaptive_share = max(scheduled_share, adaptive_share)
        target_mix_share = adaptive_share if active else scheduled_share
        self._last_adaptive_mix_log_vars = {
            'ssda_adaptive_mix_enabled': 1.0,
            'ssda_adaptive_mix_active': float(active),
            'ssda_adaptive_mix_schedule_floor': float(
                self.adaptive_mix_schedule_floor),
            'ssda_adaptive_mix_scheduled_share': float(scheduled_share),
            'ssda_adaptive_mix_signal': float(
                min(1.0, max(0.0, adaptive_signal))),
            'ssda_adaptive_mix_target_reliability_ema': float(target_reliability),
            'ssda_adaptive_mix_source_affinity_ema': float(source_affinity),
            'ssda_adaptive_mix_raw_share': float(raw_adaptive_share),
            'ssda_adaptive_mix_share': float(target_mix_share),
        }
        return target_mix_share

    def _mix_branch_weights(self):
        source_mix_weight = self._branch_weight('source_mix')
        target_mix_weight = self._branch_weight('target_mix')
        target_mix_share = None
        mix_total_weight = source_mix_weight + target_mix_weight

        if self.target_dominant_mix:
            target_mix_share = float(self._scheduled_weight(
                self.target_mix_share,
                self.target_mix_share_final,
                self.target_mix_share_schedule,
            ))
            target_mix_share = min(1.0, max(0.0, target_mix_share))
            target_mix_share = self._adaptive_target_mix_share(target_mix_share)
            if self.target_dominant_mix_total_weight is not None:
                mix_total_weight = self.target_dominant_mix_total_weight
            source_mix_weight, target_mix_weight = rebalance_mix_weights(
                source_mix_weight,
                target_mix_weight,
                target_mix_share,
                total_weight=mix_total_weight,
            )

        return source_mix_weight, target_mix_weight, target_mix_share, mix_total_weight

    def _mix_branch_log_vars(self, source_mix_weight, target_mix_weight,
                             target_mix_share, mix_total_weight):
        log_vars = {
            'ssda_source_mix_weight': source_mix_weight,
            'ssda_target_mix_weight': target_mix_weight,
            'ssda_mix_total_weight': mix_total_weight,
        }
        if target_mix_share is not None:
            log_vars.update({
                'ssda_target_mix_share': target_mix_share,
                'ssda_source_mix_share': 1.0 - target_mix_share,
            })
        log_vars.update(self._last_adaptive_mix_log_vars)
        return log_vars

    def _redistribute_target_supervision(
            self,
            target_sup_weight,
            source_mix_weight,
            target_mix_weight,
            target_mix_share,
            mix_total_weight):
        """Transfer target-mix weight to clean target supervision.

        The operation keeps the combined source-mix, target-mix, and direct
        target-supervision scale unchanged. It is disabled by default and is
        used only by the budget-aware diagnostic configs.
        """
        if not self.redistribute_target_sup_from_target_mix:
            return (
                target_sup_weight,
                source_mix_weight,
                target_mix_weight,
                target_mix_share,
                mix_total_weight,
                {},
            )

        requested_weight = max(0.0, float(target_sup_weight))
        transferred_weight = min(
            requested_weight,
            max(0.0, float(target_mix_weight)),
        )
        original_mix_total = float(mix_total_weight)
        original_target_mix_share = target_mix_share
        target_sup_weight = transferred_weight
        target_mix_weight = max(
            0.0,
            float(target_mix_weight) - transferred_weight,
        )
        mix_total_weight = float(source_mix_weight) + target_mix_weight
        target_mix_share = (
            target_mix_weight / mix_total_weight
            if mix_total_weight > 0
            else 0.0
        )
        log_vars = {
            'ssda_target_sup_requested_weight': requested_weight,
            'ssda_target_sup_redistributed_weight': transferred_weight,
            'ssda_target_assistance_total_weight': (
                mix_total_weight + target_sup_weight
            ),
        }
        if original_target_mix_share is not None:
            log_vars['ssda_target_mix_share_pre_redistribution'] = float(
                original_target_mix_share)
        if abs(
                mix_total_weight + target_sup_weight - original_mix_total
        ) > 1e-6:
            raise RuntimeError(
                'Target-supervision redistribution changed the combined '
                'objective scale')
        return (
            target_sup_weight,
            source_mix_weight,
            target_mix_weight,
            target_mix_share,
            mix_total_weight,
            log_vars,
        )

    @staticmethod
    def _update_scalar_ema(old_value, new_value, momentum):
        new_value = min(1.0, max(0.0, float(new_value)))
        if old_value is None:
            return new_value
        momentum = min(0.9999, max(0.0, float(momentum)))
        return momentum * float(old_value) + (1.0 - momentum) * new_value

    def _compute_adaptive_mix_target_reliability(self, pseudo_weight, pseudo_conf):
        if pseudo_weight is None:
            return None, None
        pseudo_weight = pseudo_weight.detach().float()
        if pseudo_weight.dim() == 4 and pseudo_weight.shape[1] == 1:
            pseudo_weight = pseudo_weight.squeeze(1)
        valid = pseudo_weight > 0
        mask_ratio = float(valid.float().mean().item())
        if pseudo_conf is None:
            return float(pseudo_weight.mean().item()), mask_ratio
        pseudo_conf = pseudo_conf.detach().float()
        if pseudo_conf.dim() == 4 and pseudo_conf.shape[1] == 1:
            pseudo_conf = pseudo_conf.squeeze(1)
        if valid.any():
            reliability = pseudo_conf[valid].mean()
        else:
            reliability = pseudo_conf.new_tensor(0.0)
        return float(reliability.clamp(0.0, 1.0).item()), mask_ratio

    @staticmethod
    def _compute_adaptive_mix_reliable_pixel_ratio(
        pseudo_mask,
        pseudo_weight=None,
    ):
        if pseudo_mask is None:
            return None
        reliable = pseudo_mask.detach().bool()
        if reliable.dim() == 4 and reliable.shape[1] == 1:
            reliable = reliable.squeeze(1)

        if pseudo_weight is None:
            return float(reliable.float().mean().item())

        valid = pseudo_weight.detach().float()
        if valid.dim() == 4 and valid.shape[1] == 1:
            valid = valid.squeeze(1)
        valid = valid > 0
        if valid.any():
            return float(reliable[valid].float().mean().item())
        return 0.0

    @staticmethod
    def _class_balanced_consistency_reliability(
        pseudo_label,
        pseudo_conf,
        view_label,
        view_conf,
        valid_weight=None,
        conf_threshold=0.968,
        min_class_pixels=32,
    ):
        """Measure flip-view agreement with equal weight for present classes."""
        pseudo_label = pseudo_label.detach().long()
        view_label = view_label.detach().long()
        pseudo_conf = pseudo_conf.detach().float()
        view_conf = view_conf.detach().float()
        if valid_weight is None:
            valid = torch.ones_like(pseudo_label, dtype=torch.bool)
        else:
            valid = valid_weight.detach().float().gt(0)
        reliable = (
            valid
            & pseudo_conf.ge(float(conf_threshold))
            & view_conf.ge(float(conf_threshold))
            & pseudo_label.eq(view_label)
        )

        class_scores = []
        min_class_pixels = max(1, int(min_class_pixels))
        for class_id in torch.unique(pseudo_label[valid]):
            class_mask = valid & pseudo_label.eq(class_id)
            class_count = int(class_mask.sum().item())
            if class_count < min_class_pixels:
                continue
            class_scores.append(reliable[class_mask].float().mean())

        if class_scores:
            reliability = torch.stack(class_scores).mean()
        elif valid.any():
            reliability = reliable[valid].float().mean()
        else:
            reliability = pseudo_conf.new_tensor(0.0)
        return (
            float(reliability.clamp(0.0, 1.0).item()),
            reliable,
            len(class_scores),
        )

    def _target_flip_consistency_state(
        self,
        tgt_u_img,
        pseudo_label,
        pseudo_weight,
        pseudo_conf,
    ):
        needs_consistency = (
            self.consistency_aware_target_mix
            or (
                self.adaptive_target_dominant_mix
                and self.adaptive_mix_mode
                == 'class_balanced_consistency_residual'
            )
        )
        if (
            not needs_consistency
            or tgt_u_img is None
            or pseudo_label is None
            or pseudo_conf is None
        ):
            return None, {}

        with torch.no_grad():
            flip_logits = self.get_ema_model().generate_pseudo_label(
                tgt_u_img.flip(-1))
            if isinstance(flip_logits, (tuple, list)):
                flip_logits = flip_logits[0]
            flip_logits = flip_logits.flip(-1)
            if flip_logits.shape[-2:] != pseudo_label.shape[-2:]:
                flip_logits = F.interpolate(
                    flip_logits,
                    size=pseudo_label.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )
            flip_prob = torch.softmax(flip_logits.detach(), dim=1)
            flip_conf, flip_label = torch.max(flip_prob, dim=1)

        reliability, reliable_mask, class_count = (
            self._class_balanced_consistency_reliability(
                pseudo_label,
                pseudo_conf,
                flip_label,
                flip_conf,
                valid_weight=pseudo_weight,
                conf_threshold=self.target_consistency_conf_threshold,
                min_class_pixels=self.target_consistency_min_class_pixels,
            ))
        valid = (
            torch.ones_like(reliable_mask)
            if pseudo_weight is None
            else pseudo_weight.detach().float().gt(0)
        )
        valid_count = valid.float().sum().clamp_min(1.0)
        pixel_ratio = reliable_mask.float().sum() / valid_count
        state = {
            'class_balanced_reliability': reliability,
            'reliable_mask': reliable_mask,
        }
        logs = {
            'ssda_consistency_class_balanced_reliability': reliability,
            'ssda_consistency_reliable_pixel_ratio': float(pixel_ratio.item()),
            'ssda_consistency_valid_class_count': float(class_count),
            'ssda_consistency_view_conf_mean': float(
                flip_conf[valid].mean().item()) if valid.any() else 0.0,
        }
        return state, logs

    def _consistency_aware_target_mix_weight(
        self,
        pseudo_weight,
        consistency_state,
    ):
        if (
            not self.consistency_aware_target_mix
            or consistency_state is None
        ):
            return pseudo_weight, {}
        reliable = consistency_state['reliable_mask'].to(
            device=pseudo_weight.device,
            dtype=pseudo_weight.dtype,
        )
        min_weight = self.consistency_target_mix_disagreement_weight
        factor = min_weight + (1.0 - min_weight) * reliable
        weighted = pseudo_weight * factor
        return weighted, {
            'ssda_target_mix_consistency_aware': 1.0,
            'ssda_target_mix_consistency_factor_mean': float(
                factor.mean().item()),
            'ssda_target_mix_consistency_weight_before': float(
                pseudo_weight.detach().float().mean().item()),
            'ssda_target_mix_consistency_weight_after': float(
                weighted.detach().float().mean().item()),
        }

    def _adaptive_mix_source_target_affinity(
        self,
        src_seg_lbl,
        tgt_l_seg_lbl,
        pseudo_label=None,
        pseudo_weight=None,
        pseudo_conf=None,
    ):
        src_hist = self._normalize_hist(
            self._label_histograms(src_seg_lbl).sum(dim=0))
        tgt_l_hist = self._normalize_hist(
            self._label_histograms(tgt_l_seg_lbl).sum(dim=0))
        target_hist = tgt_l_hist
        used_unlabeled = 0.0

        if pseudo_label is not None and pseudo_weight is not None:
            pseudo_pixel_weight = pseudo_weight.detach().float()
            if pseudo_pixel_weight.dim() == 4 and pseudo_pixel_weight.shape[1] == 1:
                pseudo_pixel_weight = pseudo_pixel_weight.squeeze(1)
            if pseudo_conf is not None:
                pseudo_conf_map = pseudo_conf.detach().float()
                if pseudo_conf_map.dim() == 4 and pseudo_conf_map.shape[1] == 1:
                    pseudo_conf_map = pseudo_conf_map.squeeze(1)
                pseudo_pixel_weight = pseudo_pixel_weight * pseudo_conf_map.ge(
                    self.adaptive_mix_conf_threshold).float()
            pseudo_hist = self._normalize_hist(
                self._label_histograms(pseudo_label, pseudo_pixel_weight).sum(dim=0))
            if float(pseudo_hist.sum().detach().item()) > 0:
                target_hist = self._normalize_hist(
                    tgt_l_hist + self.adaptive_mix_unlabeled_weight * pseudo_hist)
                used_unlabeled = 1.0

        denom = src_hist.norm().clamp_min(1e-6) * target_hist.norm().clamp_min(1e-6)
        affinity = (src_hist * target_hist).sum() / denom
        return float(affinity.clamp(0.0, 1.0).detach().item()), used_unlabeled

    def _update_adaptive_mix_state(
        self,
        src_seg_lbl,
        tgt_l_seg_lbl,
        pseudo_label=None,
        pseudo_weight=None,
        pseudo_mask=None,
        pseudo_conf=None,
        consistency_reliability=None,
    ):
        if not self.adaptive_target_dominant_mix:
            return {}

        if self.adaptive_mix_mode == 'class_balanced_consistency_residual':
            reliability = (
                self.adaptive_mix_default_reliability
                if consistency_reliability is None
                else float(consistency_reliability)
            )
            self._adaptive_mix_target_reliability_ema = (
                self._update_scalar_ema(
                    self._adaptive_mix_target_reliability_ema,
                    reliability,
                    self.adaptive_mix_momentum,
                ))
            if (
                self.local_iter >= self.adaptive_mix_warmup_iter
                and self._adaptive_mix_reliability_reference is None
            ):
                self._adaptive_mix_reliability_reference = float(
                    self._adaptive_mix_target_reliability_ema)
            return {
                'ssda_adaptive_mix_consistency_cur': reliability,
                'ssda_adaptive_mix_consistency_ema': float(
                    self._adaptive_mix_target_reliability_ema),
                'ssda_adaptive_mix_consistency_reference': float(
                    self._adaptive_mix_target_reliability_ema
                    if self._adaptive_mix_reliability_reference is None
                    else self._adaptive_mix_reliability_reference),
            }

        if self.adaptive_mix_mode == 'reliable_pixel_ratio':
            reliable_ratio = self._compute_adaptive_mix_reliable_pixel_ratio(
                pseudo_mask,
                pseudo_weight,
            )
            if reliable_ratio is None:
                reliable_ratio = self.adaptive_mix_default_reliability
            self._adaptive_mix_target_reliability_ema = self._update_scalar_ema(
                self._adaptive_mix_target_reliability_ema,
                reliable_ratio,
                self.adaptive_mix_momentum,
            )
            return {
                'ssda_adaptive_mix_reliable_ratio_cur': float(reliable_ratio),
                'ssda_adaptive_mix_reliable_ratio_ema': float(
                    self._adaptive_mix_target_reliability_ema),
            }

        reliability, mask_ratio = self._compute_adaptive_mix_target_reliability(
            pseudo_weight,
            pseudo_conf,
        )
        if reliability is None:
            reliability = self.adaptive_mix_default_reliability
            mask_ratio = 0.0
        affinity, used_unlabeled = self._adaptive_mix_source_target_affinity(
            src_seg_lbl,
            tgt_l_seg_lbl,
            pseudo_label,
            pseudo_weight,
            pseudo_conf,
        )
        self._adaptive_mix_target_reliability_ema = self._update_scalar_ema(
            self._adaptive_mix_target_reliability_ema,
            reliability,
            self.adaptive_mix_momentum,
        )
        self._adaptive_mix_source_affinity_ema = self._update_scalar_ema(
            self._adaptive_mix_source_affinity_ema,
            affinity,
            self.adaptive_mix_momentum,
        )

        return {
            'ssda_adaptive_mix_target_reliability_cur': float(reliability),
            'ssda_adaptive_mix_source_affinity_cur': float(affinity),
            'ssda_adaptive_mix_pseudo_mask_ratio': float(mask_ratio),
            'ssda_adaptive_mix_target_unlabeled_used': float(used_unlabeled),
            'ssda_adaptive_mix_target_reliability_ema': float(
                self._adaptive_mix_target_reliability_ema),
            'ssda_adaptive_mix_source_affinity_ema': float(
                self._adaptive_mix_source_affinity_ema),
        }

    def _confidence_aware_target_mix_weight(self, pseudo_weight, pseudo_conf):
        """Weight target-mix pseudo pixels by confidence and entropy proxy."""
        if not self.conf_aware_target_mix or pseudo_conf is None:
            return pseudo_weight, {}

        conf = pseudo_conf.detach().float().clamp(0.0, 1.0)
        if self.conf_aware_target_mix_mode == 'power':
            reliability = conf.pow(max(1e-6, self.conf_aware_target_mix_conf_gamma))
        elif self.conf_aware_target_mix_mode == 'threshold_linear':
            denom = max(1e-6, 1.0 - self.conf_aware_target_mix_threshold)
            reliability = ((conf - self.conf_aware_target_mix_threshold) / denom)
            reliability = reliability.clamp(0.0, 1.0).pow(
                max(1e-6, self.conf_aware_target_mix_conf_gamma))
        elif self.conf_aware_target_mix_mode == 'hard':
            reliability = conf.ge(self.conf_aware_target_mix_threshold).float()
        elif self.conf_aware_target_mix_mode in ('conf_entropy', 'confidence_entropy'):
            denom = max(1e-6, 1.0 - self.conf_aware_target_mix_threshold)
            conf_factor = ((conf - self.conf_aware_target_mix_threshold) / denom)
            conf_factor = conf_factor.clamp(0.0, 1.0).pow(
                max(1e-6, self.conf_aware_target_mix_conf_gamma))

            # The teacher currently exposes max probability only. This binary
            # entropy proxy still separates uncertain pixels from confident ones
            # without storing full logits/probability maps.
            eps = 1e-6
            p = conf.clamp(eps, 1.0 - eps)
            entropy_proxy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
            entropy_proxy = entropy_proxy / math.log(2.0)
            entropy_factor = (1.0 - entropy_proxy).clamp(0.0, 1.0).pow(
                max(1e-6, self.conf_aware_target_mix_entropy_gamma))
            reliability = conf_factor * entropy_factor
        else:
            raise ValueError(
                "Invalid conf_aware_target_mix_mode. Choose from 'power', "
                "'threshold_linear', 'hard', or 'conf_entropy'.")

        reliability = reliability.clamp(
            self.conf_aware_target_mix_min_weight,
            self.conf_aware_target_mix_max_weight,
        )
        blend = float(self._scheduled_weight(
            self.conf_aware_target_mix_blend,
            self.conf_aware_target_mix_blend_final,
            self.conf_aware_target_mix_blend_schedule,
        ))
        blend = min(1.0, max(0.0, blend))
        final_factor = (1.0 - blend) + blend * reliability
        weighted_pseudo = pseudo_weight * final_factor.to(
            device=pseudo_weight.device,
            dtype=pseudo_weight.dtype,
        )

        entropy_log = 1.0 - conf
        if self.conf_aware_target_mix_mode in ('conf_entropy', 'confidence_entropy'):
            entropy_log = entropy_proxy
        return weighted_pseudo, {
            'ssda_target_mix_conf_aware': 1.0,
            'ssda_target_mix_conf_blend': blend,
            'ssda_target_mix_conf_mean': float(conf.mean().item()),
            'ssda_target_mix_entropy_proxy_mean': float(entropy_log.mean().item()),
            'ssda_target_mix_reliability_mean': float(reliability.mean().item()),
            'ssda_target_mix_reliability_min': float(reliability.min().item()),
            'ssda_target_mix_reliability_max': float(reliability.max().item()),
            'ssda_target_mix_pseudo_weight_mean_before': float(
                pseudo_weight.detach().float().mean().item()),
            'ssda_target_mix_pseudo_weight_mean_after': float(
                weighted_pseudo.detach().float().mean().item()),
        }

    def _target_labeled_reliability_active(self):
        return (
            self.target_labeled_reliability_calibration
            and self.local_iter >= self.target_labeled_reliability_begin_iter)

    def _update_target_labeled_reliability(self, logits, labels):
        if (
            not self._target_labeled_reliability_active()
            or logits is None
            or labels is None
        ):
            return {}

        with torch.no_grad():
            reliability, valid_class, stats = target_labeled_class_reliability(
                logits,
                labels,
                self.num_classes,
                min_reliability=self.target_labeled_reliability_min,
                max_reliability=self.target_labeled_reliability_max,
                default_reliability=self.target_labeled_reliability_default,
                affine_floor=self.target_labeled_reliability_affine_floor,
                ignore_index=getattr(self, 'ignore_index', 255),
            )
            if self._target_labeled_class_reliability is None:
                self._target_labeled_class_reliability = reliability.detach()
                self._target_labeled_class_reliability_valid = valid_class.detach()
            elif valid_class.any():
                old = self._target_labeled_class_reliability.to(
                    device=reliability.device,
                    dtype=reliability.dtype,
                )
                momentum = min(
                    0.9999,
                    max(0.0, float(self.target_labeled_reliability_momentum)))
                old[valid_class] = (
                    momentum * old[valid_class]
                    + (1.0 - momentum) * reliability[valid_class])
                self._target_labeled_class_reliability = old.detach()
                if self._target_labeled_class_reliability_valid is None:
                    self._target_labeled_class_reliability_valid = valid_class.detach()
                else:
                    self._target_labeled_class_reliability_valid = (
                        self._target_labeled_class_reliability_valid.to(
                            device=valid_class.device)
                        | valid_class.detach())

            ema = self._target_labeled_class_reliability.detach()
            if self._target_labeled_class_reliability_valid is not None:
                ema_valid = self._target_labeled_class_reliability_valid.to(
                    device=ema.device)
                valid_values = ema[ema_valid] if ema_valid.any() else ema
            else:
                valid_values = ema
            return {
                'ssda_tlrc_valid_class_count': stats['valid_class_count'],
                'ssda_tlrc_batch_reliability_mean': stats['reliability_mean'],
                'ssda_tlrc_batch_reliability_min': stats['reliability_min'],
                'ssda_tlrc_batch_reliability_max': stats['reliability_max'],
                'ssda_tlrc_ema_reliability_mean': float(
                    valid_values.mean().detach().item()),
                'ssda_tlrc_ema_reliability_min': float(
                    valid_values.min().detach().item()),
                'ssda_tlrc_ema_reliability_max': float(
                    valid_values.max().detach().item()),
            }

    def _target_labeled_reliability_weight(
        self,
        pseudo_label,
        pseudo_weight,
        prefix='target_mix',
    ):
        if (
            not self._target_labeled_reliability_active()
            or self._target_labeled_class_reliability is None
            or pseudo_label is None
            or pseudo_weight is None
        ):
            return pseudo_weight, {}

        blend = float(self._scheduled_weight(
            self.target_labeled_reliability_blend,
            self.target_labeled_reliability_blend_final,
            self.target_labeled_reliability_blend_schedule,
        ))
        weighted, stats = apply_target_labeled_reliability(
            pseudo_label,
            pseudo_weight,
            self._target_labeled_class_reliability,
            blend=blend,
            min_weight=self.target_labeled_reliability_min,
            max_weight=self.target_labeled_reliability_max,
            ignore_index=getattr(self, 'ignore_index', 255),
        )
        return weighted, {
            f'ssda_{prefix}_{key}': value
            for key, value in stats.items()
        }

    def _target_anchor_replay_branch_weight(self):
        if not self.target_anchor_replay_enabled:
            return 0.0
        if self.local_iter < self.target_anchor_replay_begin_iter:
            return 0.0
        return float(self._scheduled_weight(
            self.target_anchor_replay_weight,
            self.target_anchor_replay_weight_final,
            self.target_anchor_replay_weight_schedule,
        ))

    def _target_anchor_replay_weight_map(self, labels):
        if self.target_anchor_replay_class_conditional:
            return build_class_conditional_anchor_weight_map(
                labels,
                enhance_class_ids=self.target_anchor_replay_enhance_class_ids,
                enhance_weight=self.target_anchor_replay_enhance_class_weight,
                protect_class_ids=self.target_anchor_replay_protect_class_ids,
                protect_weight=self.target_anchor_replay_protect_class_weight,
                default_weight=self.target_anchor_replay_default_class_weight,
                ignore_index=getattr(self, 'ignore_index', 255),
                normalize_mean=self.target_anchor_replay_normalize_mean,
            )

        if (
            self.target_anchor_replay_rare_class_weight <= 1.0
            or not self.target_anchor_replay_rare_class_ids
        ):
            return None, {}

        if labels.dim() == 4 and labels.shape[1] == 1:
            label_map = labels[:, 0]
        else:
            label_map = labels
        weight_map = torch.ones_like(label_map, dtype=torch.float32)
        for class_id in self.target_anchor_replay_rare_class_ids:
            weight_map[label_map.eq(int(class_id))] = float(
                self.target_anchor_replay_rare_class_weight)
        return weight_map, {
            'enhance_pixel_ratio': float(
                torch.zeros_like(weight_map).float().mean().item()),
            'protect_pixel_ratio': 0.0,
            'valid_pixel_ratio': 1.0,
            'weight_mean_before': float(weight_map.detach().mean().item()),
            'weight_mean_after': float(weight_map.detach().mean().item()),
            'weight_min': float(weight_map.detach().min().item()),
            'weight_max': float(weight_map.detach().max().item()),
        }

    def _forward_target_anchor_replay_branch(
        self,
        tgt_l_img,
        tgt_l_seg_lbl,
        means,
        stds,
        seg_debug,
        branch_weight,
    ):
        if branch_weight <= 0:
            return {'log_vars': {}, 'loss_value': 0.0, 'raw_loss_value': 0.0}

        strong_img = self._make_unlabeled_strong_view(tgt_l_img, means, stds)
        replay_weight, replay_weight_stats = self._target_anchor_replay_weight_map(
            tgt_l_seg_lbl)
        state = self._forward_labeled_loss(
            strong_img,
            tgt_l_seg_lbl,
            seg_debug,
            'Target Anchor Replay',
            'tgt_anchor',
            branch_weight,
            backward=True,
            seg_weight=replay_weight,
            return_feat=False,
            loss_key='target_labeled',
        )
        state['log_vars'].update({
            'ssda_target_anchor_replay_weight': float(branch_weight),
            'ssda_target_anchor_replay_rare_class_weight': float(
                self.target_anchor_replay_rare_class_weight),
            'ssda_target_anchor_replay_rare_class_count': float(
                len(self.target_anchor_replay_rare_class_ids)),
            'ssda_target_anchor_replay_class_conditional': float(
                self.target_anchor_replay_class_conditional),
            'ssda_target_anchor_replay_enhance_class_weight': float(
                self.target_anchor_replay_enhance_class_weight),
            'ssda_target_anchor_replay_protect_class_weight': float(
                self.target_anchor_replay_protect_class_weight),
            'ssda_target_anchor_replay_enhance_class_count': float(
                len(self.target_anchor_replay_enhance_class_ids)),
            'ssda_target_anchor_replay_protect_class_count': float(
                len(self.target_anchor_replay_protect_class_ids)),
        })
        state['log_vars'].update({
            f'ssda_target_anchor_replay_{key}': value
            for key, value in replay_weight_stats.items()
        })
        return state

    def _label_histograms(self, labels, pixel_weights=None):
        """Return per-image class histograms normalized with L1 norm."""
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        labels = labels.detach().long()
        batch_size = labels.shape[0]
        histograms = []
        if pixel_weights is not None:
            if pixel_weights.dim() == 4 and pixel_weights.shape[1] == 1:
                pixel_weights = pixel_weights.squeeze(1)
            pixel_weights = pixel_weights.detach().float()

        for idx in range(batch_size):
            label = labels[idx]
            valid = (label >= 0) & (label < self.num_classes)
            if pixel_weights is None:
                weights = None
            else:
                weights = pixel_weights[idx].to(device=label.device)
                valid = valid & (weights > 0)
            if not torch.any(valid):
                hist = torch.zeros(self.num_classes, device=label.device)
            else:
                flat_label = label[valid]
                if pixel_weights is None:
                    hist = torch.bincount(
                        flat_label,
                        minlength=self.num_classes,
                    ).float()
                else:
                    hist = torch.bincount(
                        flat_label,
                        weights=weights[valid],
                        minlength=self.num_classes,
                    ).float()
            hist = hist / hist.sum().clamp_min(1e-6)
            histograms.append(hist)

        return torch.stack(histograms, dim=0)

    @staticmethod
    def _normalize_hist(hist):
        return hist / hist.sum().clamp_min(1e-6)

    def _update_hist_prototype(self, attr_name, hist, update=True,
                               update_mode=None, momentum=None):
        """Update or read one class-histogram prototype."""
        hist = self._normalize_hist(hist).detach()
        update_mode = update_mode or self.target_guided_source_filter_proto_update
        momentum = (
            self.target_guided_source_filter_proto_momentum
            if momentum is None else momentum)
        if update_mode == 'batch':
            return hist
        if update_mode != 'ema':
            raise ValueError(
                "Invalid histogram prototype update mode. Choose "
                "from 'batch' or 'ema'.")

        proto = getattr(self, attr_name, None)
        if update or proto is None:
            if proto is None:
                proto = hist
            else:
                proto = proto.to(device=hist.device, dtype=hist.dtype)
                momentum = min(0.9999, max(0.0, momentum))
                proto = momentum * proto + (1.0 - momentum) * hist
                proto = self._normalize_hist(proto)
            setattr(self, attr_name, proto.detach())
        else:
            proto = proto.to(device=hist.device, dtype=hist.dtype)
        return proto

    def _tri_update_hist_prototype(self, attr_name, hist, update=True):
        return self._update_hist_prototype(
            attr_name,
            hist,
            update=update,
            update_mode=self.tri_prototype_proto_update,
            momentum=self.tri_prototype_proto_momentum,
        )

    def _prototype_classmix_update_hist_prototype(self, attr_name, hist,
                                                  update=True):
        return self._update_hist_prototype(
            attr_name,
            hist,
            update=update,
            update_mode=self.prototype_classmix_proto_update,
            momentum=self.prototype_classmix_proto_momentum,
        )

    def _prototype_classmix_active(self):
        return (
            self.prototype_classmix_enabled
            and self.local_iter >= self.prototype_classmix_begin_iter)

    def _update_prototype_classmix_target_state(
        self,
        tgt_l_seg_lbl,
        pseudo_label=None,
        pseudo_weight=None,
        pseudo_conf=None,
    ):
        """Update target class-distribution prototypes for guided ClassMix."""
        if not self.prototype_classmix_enabled:
            return {}

        tgt_l_hist = self._normalize_hist(
            self._label_histograms(tgt_l_seg_lbl).sum(dim=0))
        tgt_l_proto = self._prototype_classmix_update_hist_prototype(
            '_prototype_classmix_target_labeled_hist_proto',
            tgt_l_hist,
            update=True,
        )
        used_unlabeled = 0.0
        tgt_u_proto = None
        if pseudo_label is not None and pseudo_weight is not None:
            pseudo_pixel_weight = pseudo_weight.detach().float()
            if pseudo_conf is not None:
                pseudo_pixel_weight = pseudo_pixel_weight * pseudo_conf.detach().float().ge(
                    self.prototype_classmix_conf_threshold).float()
            pseudo_hist = self._normalize_hist(
                self._label_histograms(pseudo_label, pseudo_pixel_weight).sum(dim=0))
            if float(pseudo_hist.sum().detach().item()) > 0:
                tgt_u_proto = self._prototype_classmix_update_hist_prototype(
                    '_prototype_classmix_target_unlabeled_hist_proto',
                    pseudo_hist,
                    update=True,
                )
                used_unlabeled = 1.0

        target_proto = (
            self.prototype_classmix_target_labeled_weight * tgt_l_proto)
        if tgt_u_proto is not None:
            target_proto = (
                target_proto
                + self.prototype_classmix_target_unlabeled_weight * tgt_u_proto)
        target_proto = self._normalize_hist(target_proto)

        return {
            'ssda_proto_classmix_target_labeled_coverage': float(
                tgt_l_proto.gt(0).float().mean().detach().item()),
            'ssda_proto_classmix_target_unlabeled_used': used_unlabeled,
            'ssda_proto_classmix_target_unlabeled_coverage': (
                float(tgt_u_proto.gt(0).float().mean().detach().item())
                if tgt_u_proto is not None else 0.0),
            'ssda_proto_classmix_target_entropy': float(
                (-(target_proto * target_proto.clamp_min(1e-6).log()).sum())
                .detach().item()),
        }

    def _prototype_classmix_target_proto(self, device):
        tgt_l_proto = self._prototype_classmix_target_labeled_hist_proto
        tgt_u_proto = self._prototype_classmix_target_unlabeled_hist_proto
        if tgt_l_proto is None and tgt_u_proto is None:
            return None
        target_proto = None
        if tgt_l_proto is not None:
            target_proto = (
                self.prototype_classmix_target_labeled_weight
                * tgt_l_proto.to(device=device))
        if tgt_u_proto is not None:
            weighted = (
                self.prototype_classmix_target_unlabeled_weight
                * tgt_u_proto.to(device=device))
            target_proto = weighted if target_proto is None else target_proto + weighted
        return self._normalize_hist(target_proto)

    def _prototype_classmix_scores(self, device):
        target_proto = self._prototype_classmix_target_proto(device)
        if target_proto is None:
            return None
        scores = (1.0 - target_proto).clamp_min(
            self.prototype_classmix_min_score)
        scores = scores.pow(max(1e-6, self.prototype_classmix_need_gamma))
        return scores / scores.mean().clamp_min(1e-6)

    def _parse_feature_prototype_class_ids(self, value):
        if value is None:
            return []
        aliases = {
            'cityscapes_structure': [0, 1, 2, 3, 4, 5, 8, 9, 10],
            'cityscapes_layout': [0, 1, 2, 3, 4, 5, 8, 9, 10],
            'cityscapes_static': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'cityscapes_dynamic': [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
            'cityscapes_large_stuff': [0, 1, 2, 8, 10],
            'cityscapes_stuff': [0, 1, 2, 3, 4, 8, 9, 10],
            'syn_city_structure': [0, 1, 2, 3, 4, 5, 8, 9],
            'syn_city_dynamic': [5, 6, 7, 10, 11, 12, 13, 14, 15],
            'syn_city_large_stuff': [0, 1, 2, 8],
            'syn_city_stuff': [0, 1, 2, 3, 4, 8, 9],
        }
        if isinstance(value, str):
            key = value.strip().lower()
            if key in aliases:
                return aliases[key]
            if not key:
                return []
            raw_items = key.replace(';', ',').split(',')
        else:
            raw_items = value

        class_ids = []
        for item in raw_items:
            try:
                class_id = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= class_id < int(self.num_classes) and class_id not in class_ids:
                class_ids.append(class_id)
        return class_ids

    def _build_prototype_or_random_mix_masks(self, labels, class_ratio,
                                             branch_prefix,
                                             class_scores_override=None,
                                             random_prob_override=None):
        if (
            self.prototype_incompatibility_veto
            and class_scores_override is not None
        ):
            class_scores = class_scores_override.to(labels.device)
            mix_masks, num_class_choice, vetoed_counts = \
                get_incompatibility_veto_class_masks(
                    labels,
                    class_ratio=class_ratio,
                    class_scores=class_scores,
                    apply_context=self.get_context_class_mask,
                    num_classes=self.num_classes,
                    ignore_index=getattr(self, 'ignore_index', 255),
                )
            sample_count = max(1, len(vetoed_counts))
            return mix_masks, num_class_choice, {
                f'{branch_prefix}_proto_veto_active': 1.0,
                f'{branch_prefix}_proto_veto_class_count': float(
                    sum(vetoed_counts)),
                f'{branch_prefix}_proto_veto_class_mean': float(
                    sum(vetoed_counts) / sample_count),
                f'{branch_prefix}_proto_veto_score_min': float(
                    class_scores.min().detach().item()),
                f'{branch_prefix}_proto_veto_score_max': float(
                    class_scores.max().detach().item()),
                f'{branch_prefix}_proto_veto_score_mean': float(
                    class_scores.mean().detach().item()),
            }

        if class_scores_override is not None:
            class_scores = class_scores_override.to(labels.device)
            random_prob = (
                self.feature_prototype_mix_random_prob
                if random_prob_override is None
                else float(random_prob_override))
            mix_masks, num_class_choice = get_prototype_guided_class_masks(
                labels,
                class_ratio=class_ratio,
                class_scores=class_scores,
                random_prob=random_prob,
                stuff_classes=(
                    self.target_need_source_mix_soft_stuff_classes),
                stuff_max=self.target_need_source_mix_soft_stuff_max,
                apply_context=self.get_context_class_mask,
                num_classes=self.num_classes,
            )
            return mix_masks, num_class_choice, {
                f'{branch_prefix}_featproto_classmix_active': 1.0,
                f'{branch_prefix}_featproto_classmix_random_prob': float(
                    random_prob),
                f'{branch_prefix}_featproto_classmix_score_min': float(
                    class_scores.min().detach().item()),
                f'{branch_prefix}_featproto_classmix_score_max': float(
                    class_scores.max().detach().item()),
                f'{branch_prefix}_featproto_classmix_score_mean': float(
                    class_scores.mean().detach().item()),
                f'{branch_prefix}_featproto_classmix_stuff_cap_active': float(
                    bool(self.target_need_source_mix_soft_stuff_classes)),
                f'{branch_prefix}_featproto_classmix_stuff_max': float(
                    self.target_need_source_mix_soft_stuff_max),
            }

        if self._prototype_classmix_active():
            class_scores = self._prototype_classmix_scores(labels.device)
            if class_scores is not None:
                mix_masks, num_class_choice = get_prototype_guided_class_masks(
                    labels,
                    class_ratio=class_ratio,
                    class_scores=class_scores,
                    random_prob=self.prototype_classmix_random_prob,
                    apply_context=self.get_context_class_mask,
                    num_classes=self.num_classes,
                )
                return mix_masks, num_class_choice, {
                    f'{branch_prefix}_proto_classmix_active': 1.0,
                    f'{branch_prefix}_proto_classmix_random_prob': float(
                        self.prototype_classmix_random_prob),
                    f'{branch_prefix}_proto_classmix_score_min': float(
                        class_scores.min().detach().item()),
                    f'{branch_prefix}_proto_classmix_score_max': float(
                        class_scores.max().detach().item()),
                    f'{branch_prefix}_proto_classmix_score_mean': float(
                        class_scores.mean().detach().item()),
                }

        if self.get_context_class_mask:
            mix_masks, num_class_choice = get_context_class_masks(
                labels,
                class_ratio=class_ratio,
                num_classes=self.num_classes,
            )
        else:
            mix_masks, num_class_choice = get_class_masks(
                labels,
                class_ratio=class_ratio,
            )
        return mix_masks, num_class_choice, {
            f'{branch_prefix}_proto_classmix_active': 0.0,
        }

    def _build_target_deficit_quota_mix_masks(
        self,
        labels,
        class_ratio,
        branch_prefix,
        class_scores,
    ):
        if class_scores is None:
            return self._build_prototype_or_random_mix_masks(
                labels,
                class_ratio,
                branch_prefix,
            )
        scores = class_scores.to(labels.device)
        mix_masks, num_class_choice = get_target_deficit_quota_class_masks(
            labels,
            class_ratio=class_ratio,
            class_scores=scores,
            quota=self.target_deficit_quota_min_classes,
            topk=self.target_deficit_quota_topk,
            stuff_classes=self.target_deficit_quota_stuff_classes,
            stuff_max=self.target_deficit_quota_stuff_max,
            random_prob=self.target_deficit_quota_random_prob,
            random_tie_break=(
                self.target_deficit_quota_random_tie_break),
            apply_context=self.get_context_class_mask,
            num_classes=self.num_classes,
            ignore_index=getattr(self, 'ignore_index', 255),
        )
        return mix_masks, num_class_choice, {
            f'{branch_prefix}_tdef_quota_classmix_active': 1.0,
            f'{branch_prefix}_tdef_quota_min_classes': float(
                self.target_deficit_quota_min_classes),
            f'{branch_prefix}_tdef_quota_topk': float(
                self.target_deficit_quota_topk),
            f'{branch_prefix}_tdef_quota_stuff_max': float(
                self.target_deficit_quota_stuff_max),
            f'{branch_prefix}_tdef_quota_random_prob': float(
                self.target_deficit_quota_random_prob),
            f'{branch_prefix}_tdef_quota_score_min': float(
                scores.min().detach().item()),
            f'{branch_prefix}_tdef_quota_score_max': float(
                scores.max().detach().item()),
            f'{branch_prefix}_tdef_quota_score_mean': float(
                scores.mean().detach().item()),
        }

    def _build_target_need_mask_routing_mix_masks(
        self,
        labels,
        class_ratio,
        branch_prefix,
        class_scores,
    ):
        if class_scores is None:
            return self._build_prototype_or_random_mix_masks(
                labels,
                class_ratio,
                branch_prefix,
            )
        scores = class_scores.to(labels.device)
        mix_masks, num_class_choice = get_target_need_mask_routing_class_masks(
            labels,
            class_ratio=class_ratio,
            class_scores=scores,
            need_topk=self.target_need_mask_routing_need_topk,
            need_min_classes=(
                self.target_need_mask_routing_need_min_classes),
            structure_classes=(
                self.target_need_mask_routing_structure_classes),
            structure_min_classes=(
                self.target_need_mask_routing_structure_min_classes),
            structure_max_classes=(
                self.target_need_mask_routing_structure_max_classes),
            dynamic_classes=self.target_need_mask_routing_dynamic_classes,
            dynamic_min_classes=(
                self.target_need_mask_routing_dynamic_min_classes),
            random_prob=self.target_need_mask_routing_random_prob,
            apply_context=self.get_context_class_mask,
            num_classes=self.num_classes,
            ignore_index=getattr(self, 'ignore_index', 255),
        )
        return mix_masks, num_class_choice, {
            f'{branch_prefix}_tnmr_v2_classmix_active': 1.0,
            f'{branch_prefix}_tnmr_v2_need_topk': float(
                self.target_need_mask_routing_need_topk),
            f'{branch_prefix}_tnmr_v2_need_min_classes': float(
                self.target_need_mask_routing_need_min_classes),
            f'{branch_prefix}_tnmr_v2_structure_min_classes': float(
                self.target_need_mask_routing_structure_min_classes),
            f'{branch_prefix}_tnmr_v2_structure_max_classes': float(
                self.target_need_mask_routing_structure_max_classes),
            f'{branch_prefix}_tnmr_v2_dynamic_min_classes': float(
                self.target_need_mask_routing_dynamic_min_classes),
            f'{branch_prefix}_tnmr_v2_random_prob': float(
                self.target_need_mask_routing_random_prob),
            f'{branch_prefix}_tnmr_v2_score_min': float(
                scores.min().detach().item()),
            f'{branch_prefix}_tnmr_v2_score_max': float(
                scores.max().detach().item()),
            f'{branch_prefix}_tnmr_v2_score_mean': float(
                scores.mean().detach().item()),
        }

    def _source_deficit_pixel_weight_map(self, labels, class_scores):
        if (
            not self.target_deficit_source_pixel_reweight
            or class_scores is None
        ):
            return None
        return class_score_pixel_weights(
            labels,
            class_scores,
            min_weight=self.target_deficit_source_pixel_min_weight,
            max_weight=self.target_deficit_source_pixel_max_weight,
            gamma=self.target_deficit_source_pixel_gamma,
            ignore_index=getattr(self, 'ignore_index', 255),
            default_weight=1.0,
        )

    def _self_calibrated_class_route_scores(
        self,
        target_deficit_scores,
        source_transfer_scores,
        reliability_scores,
        valid_classes,
    ):
        """Update v3 class-signal EMAs and build conservative route scores."""
        if not self.self_calibrated_class_routing:
            return None, {}

        def update_ema(attr_name, current, valid):
            current = current.detach().float().flatten()
            valid = valid.to(device=current.device).bool().flatten()
            previous = getattr(self, attr_name)
            valid_attr_name = f'{attr_name}_valid'
            previous_valid = getattr(self, valid_attr_name, None)
            if previous is None:
                previous = torch.zeros_like(current)
                previous[valid] = current[valid]
                previous_valid = valid.clone()
            else:
                previous = previous.to(device=current.device).clone()
                if previous_valid is None:
                    previous_valid = torch.zeros_like(valid)
                else:
                    previous_valid = previous_valid.to(
                        device=current.device).bool().clone()
                momentum = float(self.self_calibrated_class_route_momentum)
                newly_valid = valid & ~previous_valid
                continuing = valid & previous_valid
                previous[newly_valid] = current[newly_valid]
                previous[continuing] = (
                    momentum * previous[continuing]
                    + (1.0 - momentum) * current[continuing]
                )
                previous_valid |= valid
            setattr(self, attr_name, previous.detach())
            setattr(self, valid_attr_name, previous_valid.detach())
            return previous

        valid = torch.as_tensor(
            valid_classes,
            device=target_deficit_scores.device,
            dtype=torch.bool,
        ).flatten()
        deficit_ema = update_ema(
            '_self_calibrated_class_route_deficit_ema',
            target_deficit_scores,
            valid,
        )
        reliability_ema = update_ema(
            '_self_calibrated_class_route_reliability_ema',
            reliability_scores,
            valid,
        )
        if source_transfer_scores is None:
            self._self_calibrated_class_route_current_scores = None
            return None, {
                'ssda_class_route_v3_active': 0.0,
                'ssda_class_route_v3_transfer_valid': 0.0,
            }
        transfer_ema = update_ema(
            '_self_calibrated_class_route_transfer_ema',
            source_transfer_scores,
            valid,
        )
        if self.local_iter < self.self_calibrated_class_route_begin_iter:
            self._self_calibrated_class_route_current_scores = None
            return None, {
                'ssda_class_route_v3_active': 0.0,
                'ssda_class_route_v3_warmup': 1.0,
                'ssda_class_route_v3_reliable_class_count': float(
                    valid.sum().item()),
            }

        route, stats = self_calibrated_class_route_scores(
            deficit_ema,
            transfer_ema,
            reliability_ema,
            reliability_min=(
                self.self_calibrated_class_route_reliability_min),
            intervention_quantile=(
                self.self_calibrated_class_route_intervention_quantile),
            max_delta=self.self_calibrated_class_route_max_delta,
            min_need_percentile=(
                self.self_calibrated_class_route_min_need_percentile),
        )
        self._self_calibrated_class_route_current_scores = route.detach()
        logs = {
            'ssda_class_route_v3_active': 1.0,
            'ssda_class_route_v3_warmup': 0.0,
            'ssda_class_route_v3_transfer_valid': 1.0,
        }
        logs.update({
            f'ssda_class_route_v3_{key}': float(value)
            for key, value in stats.items()
        })
        return route, logs

    def _target_need_source_mix_scores(
        self,
        tgt_l_seg_lbl,
        pseudo_label,
        pseudo_weight,
        pseudo_conf,
        device,
    ):
        if not self.target_need_source_mix:
            return None, {}
        target_need_active = (
            self.local_iter >= self.target_need_source_mix_begin_iter)
        if not target_need_active and not self.self_calibrated_class_routing:
            return None, {}
        scores, stats = target_need_class_scores(
            tgt_l_seg_lbl,
            pseudo_label,
            pseudo_weight,
            pseudo_conf,
            num_classes=self.num_classes,
            labeled_weight=self.target_need_source_mix_labeled_weight,
            unlabeled_weight=self.target_need_source_mix_unlabeled_weight,
            coverage_gamma=self.target_need_source_mix_coverage_gamma,
            uncertainty_weight=self.target_need_source_mix_uncertainty_weight,
            uncertainty_gamma=self.target_need_source_mix_uncertainty_gamma,
            min_score=self.target_need_source_mix_min_score,
            max_score=self.target_need_source_mix_max_score,
            ignore_index=getattr(self, 'ignore_index', 255),
        )
        route_deficit_scores = scores
        self._source_assist_set_target_deficit_scores(scores)
        source_transfer_scores = None
        if self.target_need_source_mix_use_source_transfer:
            source_transfer_scores = self._feature_proto_source_scores(device)
            if source_transfer_scores is None:
                stats['source_transfer_valid'] = 0.0
            else:
                stats['source_transfer_valid'] = 1.0
                scores, combined_stats = combine_target_need_source_scores(
                    scores,
                    source_transfer_scores=source_transfer_scores,
                    source_transfer_weight=(
                        self.target_need_source_mix_source_transfer_weight),
                    min_score=self.target_need_source_mix_min_score,
                    max_score=self.target_need_source_mix_max_score,
                )
                stats.update({
                    f'combined_{key}': value
                    for key, value in combined_stats.items()
                })
        if (
            self.target_need_source_mix_use_target_loss
            and self._target_need_loss_ema is not None
            and self.target_need_source_mix_target_loss_weight > 0
        ):
            scores, combined_stats = combine_target_need_source_scores(
                scores,
                target_loss_scores=self._target_need_loss_ema.to(device=device),
                target_loss_weight=(
                    self.target_need_source_mix_target_loss_weight),
                min_score=self.target_need_source_mix_min_score,
                max_score=self.target_need_source_mix_max_score,
            )
            stats.update({
                f'loss_{key}': value
                for key, value in combined_stats.items()
            })
            stats['target_loss_feedback_valid'] = 1.0
        elif self.target_need_source_mix_use_target_loss:
            stats['target_loss_feedback_valid'] = 0.0

        route_log_vars = {}
        if self.self_calibrated_class_routing:
            route_deficit_scores = route_deficit_scores.to(device=device)
            if (
                self.target_need_source_mix_use_target_loss
                and self._target_need_loss_ema is not None
                and self.target_need_source_mix_target_loss_weight > 0
            ):
                route_deficit_scores, _ = combine_target_need_source_scores(
                    route_deficit_scores,
                    target_loss_scores=self._target_need_loss_ema.to(
                        device=device),
                    target_loss_weight=(
                        self.target_need_source_mix_target_loss_weight),
                    min_score=self.target_need_source_mix_min_score,
                    max_score=self.target_need_source_mix_max_score,
                )
            reliability, reliability_valid, reliability_stats = \
                target_class_reliability_scores(
                    tgt_l_seg_lbl,
                    pseudo_label,
                    pseudo_weight,
                    pseudo_conf,
                    num_classes=self.num_classes,
                    labeled_weight=(
                        self.target_need_source_mix_labeled_weight),
                    unlabeled_weight=(
                        self.target_need_source_mix_unlabeled_weight),
                    ignore_index=getattr(self, 'ignore_index', 255),
                )
            reliability = reliability.to(device=device)
            reliability_valid = reliability_valid.to(device=device)
            if (
                self._feature_proto_source_valid is not None
                and source_transfer_scores is not None
            ):
                _, target_proto_valid = self._feature_proto_target_state(device)
                if target_proto_valid is not None:
                    reliability_valid = (
                        reliability_valid
                        & self._feature_proto_source_valid.to(
                            device=device).bool()
                        & target_proto_valid.to(device=device).bool()
                    )
            _, route_log_vars = self._self_calibrated_class_route_scores(
                route_deficit_scores,
                source_transfer_scores,
                reliability,
                reliability_valid,
            )
            route_log_vars.update({
                f'ssda_class_route_v3_current_{key}': float(value)
                for key, value in reliability_stats.items()
            })

        if not target_need_active:
            return None, route_log_vars
        scores = scores.to(device=device)
        self._target_need_current_scores = scores.detach()
        log_vars = {
            'ssda_tnsa_mix_active': 1.0,
            'ssda_tnsa_mix_random_prob': float(
                self.target_need_source_mix_random_prob),
        }
        log_vars.update({
            f'ssda_tnsa_mix_{key}': float(value)
            for key, value in stats.items()
        })
        log_vars.update(route_log_vars)
        return scores, log_vars

    def _class_conditional_source_route_scores(self, device):
        if self.conflict_aware_class_routing:
            if self._target_need_current_scores is None:
                return None, {'ssda_conflict_route_active': 0.0}
            return self._conflict_aware_route_scores(
                self._target_need_current_scores,
                device,
            )
        if self.self_calibrated_class_routing:
            scores = self._self_calibrated_class_route_current_scores
            if scores is None:
                return None, {}
            return scores.to(device=device), {
                'ssda_class_route_v3_applied': 1.0,
                'ssda_class_route_v3_random_prob': float(
                    self.self_calibrated_class_route_random_prob),
            }
        if not self.class_conditional_source_routing:
            return None, {}
        scores, stats = class_conditional_source_route_scores(
            self.num_classes,
            enhance_classes=self.class_conditional_source_route_enhance_classes,
            suppress_classes=self.class_conditional_source_route_suppress_classes,
            enhance_score=self.class_conditional_source_route_enhance_score,
            suppress_score=self.class_conditional_source_route_suppress_score,
            min_score=self.class_conditional_source_route_min_score,
            max_score=self.class_conditional_source_route_max_score,
            device=device,
        )
        log_vars = {
            'ssda_class_route_active': 1.0,
            'ssda_class_route_random_prob': float(
                self.class_conditional_source_route_random_prob),
        }
        log_vars.update({
            f'ssda_class_route_{key}': float(value)
            for key, value in stats.items()
        })
        return scores.to(device=device), log_vars

    def _tri_batch_proto(self, labels, pixel_weights=None):
        return self._normalize_hist(
            self._label_histograms(labels, pixel_weights).sum(dim=0))

    @staticmethod
    def _safe_jsd_log_value(proto_a, proto_b):
        if proto_a is None or proto_b is None:
            return 0.0
        return float(histogram_js_divergence(
            proto_a.detach().unsqueeze(0),
            proto_b.detach(),
        ).detach().item())

    def _forward_tri_prototype_loss(
        self,
        src_seg_pred,
        src_seg_lbl,
        tgt_l_seg_pred,
        tgt_l_seg_lbl,
        pseudo_label=None,
        pseudo_weight=None,
        pseudo_conf=None,
        branch_weight=0.0,
    ):
        """Align student class distributions to source/target prototypes."""
        if not self.tri_prototype_enabled or branch_weight <= 0:
            return {
                'loss_tensor': None,
                'loss_value': 0.0,
                'raw_loss_value': 0.0,
                'log_vars': {},
            }

        src_proto = self._tri_update_hist_prototype(
            '_tri_source_hist_proto',
            self._tri_batch_proto(src_seg_lbl),
            update=True,
        )
        tgt_l_proto = self._tri_update_hist_prototype(
            '_tri_target_labeled_hist_proto',
            self._tri_batch_proto(tgt_l_seg_lbl),
            update=True,
        )

        used_unlabeled = 0.0
        tgt_u_proto = None
        if pseudo_label is not None and pseudo_weight is not None:
            pseudo_pixel_weight = pseudo_weight.detach().float()
            if pseudo_conf is not None:
                pseudo_pixel_weight = pseudo_pixel_weight * pseudo_conf.detach().float().ge(
                    self.tri_prototype_conf_threshold).float()
            pseudo_hist = self._tri_batch_proto(pseudo_label, pseudo_pixel_weight)
            if float(pseudo_hist.sum().detach().item()) > 0:
                tgt_u_proto = self._tri_update_hist_prototype(
                    '_tri_target_unlabeled_hist_proto',
                    pseudo_hist,
                    update=True,
                )
                used_unlabeled = 1.0
        elif self._tri_target_unlabeled_hist_proto is not None:
            tgt_u_proto = self._tri_target_unlabeled_hist_proto.to(
                device=tgt_l_proto.device,
                dtype=tgt_l_proto.dtype,
            )
            used_unlabeled = 1.0

        target_proto = self.tri_prototype_target_labeled_weight * tgt_l_proto
        if tgt_u_proto is not None:
            target_proto = (
                target_proto
                + self.tri_prototype_target_unlabeled_weight * tgt_u_proto)
        target_proto = self._normalize_hist(target_proto)

        beta = min(1.0, max(0.0, self.tri_prototype_source_target_beta))
        calibrated_source_proto = self._normalize_hist(
            (1.0 - beta) * src_proto + beta * target_proto)

        src_pred_hist = prediction_histograms(
            src_seg_pred,
            self.num_classes,
            temperature=self.tri_prototype_temperature,
        )
        tgt_l_pred_hist = prediction_histograms(
            tgt_l_seg_pred,
            self.num_classes,
            temperature=self.tri_prototype_temperature,
        )

        src_loss = histogram_js_divergence(src_pred_hist, calibrated_source_proto)
        tgt_loss = histogram_js_divergence(tgt_l_pred_hist, target_proto)
        raw_loss = (
            self.tri_prototype_source_weight * src_loss
            + self.tri_prototype_target_weight * tgt_loss)
        weighted_loss = raw_loss * float(branch_weight)

        log_vars = {
            'tri_proto_loss': float(raw_loss.detach().item()),
            'tri_proto_weighted_loss': float(weighted_loss.detach().item()),
            'tri_proto_weight': float(branch_weight),
            'tri_proto_source_loss': float(src_loss.detach().item()),
            'tri_proto_target_loss': float(tgt_loss.detach().item()),
            'tri_proto_used_unlabeled': used_unlabeled,
            'tri_proto_source_target_beta': beta,
            'tri_proto_source_class_coverage': float(
                src_proto.gt(0).float().mean().detach().item()),
            'tri_proto_target_labeled_class_coverage': float(
                tgt_l_proto.gt(0).float().mean().detach().item()),
            'tri_proto_target_unlabeled_class_coverage': (
                float(tgt_u_proto.gt(0).float().mean().detach().item())
                if tgt_u_proto is not None else 0.0),
            'tri_proto_source_target_jsd': self._safe_jsd_log_value(
                src_proto, target_proto),
            'tri_proto_tl_tu_jsd': self._safe_jsd_log_value(
                tgt_l_proto, tgt_u_proto),
        }
        return {
            'loss_tensor': weighted_loss,
            'loss_value': float(weighted_loss.detach().item()),
            'raw_loss_value': float(raw_loss.detach().item()),
            'log_vars': log_vars,
        }

    def _feature_tri_pseudo_weight(self, pseudo_weight, pseudo_conf):
        if pseudo_weight is None:
            return None
        pixel_weight = pseudo_weight.detach().float()
        if pseudo_conf is not None:
            conf = pseudo_conf.detach().float()
            pixel_weight = pixel_weight * conf.ge(
                self.feature_tri_prototype_conf_threshold).float()
            if self.feature_tri_prototype_confidence_weight:
                pixel_weight = pixel_weight * conf.clamp(0.0, 1.0)
        return pixel_weight

    def _forward_feature_tri_prototype_loss(
        self,
        tgt_l_features,
        tgt_l_seg_lbl,
        tgt_u_img=None,
        pseudo_label=None,
        pseudo_weight=None,
        pseudo_conf=None,
        branch_weight=0.0,
    ):
        """Target-only feature tri-prototype v2 supervision."""
        if not self.feature_tri_prototype_enabled or branch_weight <= 0:
            return {
                'loss_tensor': None,
                'loss_value': 0.0,
                'raw_loss_value': 0.0,
                'log_vars': {},
            }
        if tgt_l_features is None:
            return {
                'loss_tensor': None,
                'loss_value': 0.0,
                'raw_loss_value': 0.0,
                'log_vars': {'feat_tri_proto_valid': 0.0},
            }

        target_proto, target_valid = self._feature_proto_target_state(
            tgt_l_seg_lbl.device)
        if target_proto is None or target_valid is None or not target_valid.any():
            return {
                'loss_tensor': None,
                'loss_value': 0.0,
                'raw_loss_value': 0.0,
                'log_vars': {'feat_tri_proto_valid': 0.0},
            }

        labeled_loss, labeled_stats = feature_prototype_contrastive_loss(
            tgt_l_features,
            tgt_l_seg_lbl,
            target_proto.detach(),
            target_valid.detach(),
            feature_level=self.feature_prototype_feature_level,
            temperature=self.feature_tri_prototype_temperature,
            ignore_index=getattr(self, 'ignore_index', 255),
            min_valid_pixels=self.feature_tri_prototype_min_valid_pixels,
        )

        unlabeled_loss = labeled_loss.new_zeros(())
        unlabeled_stats = {
            'valid_pixel_count': 0,
            'valid_class_count': 0,
            'mean_weight': 0.0,
        }
        used_unlabeled = 0.0
        if (
            self.feature_tri_prototype_unlabeled_forward
            and self.feature_tri_prototype_target_unlabeled_weight > 0
            and tgt_u_img is not None
            and pseudo_label is not None
            and pseudo_weight is not None
        ):
            pixel_weight = self._feature_tri_pseudo_weight(
                pseudo_weight,
                pseudo_conf,
            )
            if pixel_weight is not None and float(pixel_weight.sum().detach().item()) > 0:
                tgt_u_features = self.get_model().extract_feat(tgt_u_img)
                unlabeled_loss, unlabeled_stats = feature_prototype_contrastive_loss(
                    tgt_u_features,
                    pseudo_label,
                    target_proto.detach(),
                    target_valid.detach(),
                    feature_level=self.feature_prototype_feature_level,
                    temperature=self.feature_tri_prototype_temperature,
                    weights=pixel_weight,
                    ignore_index=getattr(self, 'ignore_index', 255),
                    min_valid_pixels=self.feature_tri_prototype_min_valid_pixels,
                )
                used_unlabeled = 1.0

        raw_loss = (
            self.feature_tri_prototype_target_labeled_weight * labeled_loss
            + self.feature_tri_prototype_target_unlabeled_weight * unlabeled_loss)
        weighted_loss = raw_loss * float(branch_weight)
        log_vars = {
            'feat_tri_proto_valid': 1.0,
            'feat_tri_proto_loss': float(raw_loss.detach().item()),
            'feat_tri_proto_weighted_loss': float(weighted_loss.detach().item()),
            'feat_tri_proto_weight': float(branch_weight),
            'feat_tri_proto_target_labeled_loss': float(
                labeled_loss.detach().item()),
            'feat_tri_proto_target_unlabeled_loss': float(
                unlabeled_loss.detach().item()),
            'feat_tri_proto_used_unlabeled': used_unlabeled,
            'feat_tri_proto_target_class_coverage': float(
                target_valid.float().mean().detach().item()),
            'feat_tri_proto_tl_valid_pixels': float(
                labeled_stats['valid_pixel_count']),
            'feat_tri_proto_tl_valid_classes': float(
                labeled_stats['valid_class_count']),
            'feat_tri_proto_tu_valid_pixels': float(
                unlabeled_stats['valid_pixel_count']),
            'feat_tri_proto_tu_valid_classes': float(
                unlabeled_stats['valid_class_count']),
            'feat_tri_proto_tu_mean_weight': float(
                unlabeled_stats['mean_weight']),
        }
        return {
            'loss_tensor': weighted_loss,
            'loss_value': float(weighted_loss.detach().item()),
            'raw_loss_value': float(raw_loss.detach().item()),
            'log_vars': log_vars,
        }

    def _target_guided_source_filter_weight(
        self,
        src_seg_lbl,
        tgt_l_seg_lbl,
        pseudo_label=None,
        pseudo_weight=None,
        pseudo_conf=None,
        prefix='source',
        update_labeled_prototypes=True,
        update_unlabeled_prototype=True,
    ):
        """Calibrate source assistance by target-domain class prototypes."""
        if not self.target_guided_source_filter:
            return 1.0, {}
        if self.target_guided_source_filter_mode != 'class_hist':
            raise ValueError(
                "Only target_guided_source_filter_mode='class_hist' is "
                'currently implemented.')

        src_hist = self._label_histograms(src_seg_lbl)
        src_batch_proto = self._normalize_hist(src_hist.sum(dim=0))
        tgt_l_batch_proto = self._normalize_hist(
            self._label_histograms(tgt_l_seg_lbl).sum(dim=0))
        src_proto = self._update_hist_prototype(
            '_source_hist_proto',
            src_batch_proto,
            update_labeled_prototypes,
        )
        tgt_l_proto = self._update_hist_prototype(
            '_target_labeled_hist_proto',
            tgt_l_batch_proto,
            update_labeled_prototypes,
        )
        tgt_hist = self.target_guided_source_filter_labeled_weight * tgt_l_proto
        used_unlabeled = 0.0

        if pseudo_label is not None and pseudo_weight is not None:
            pseudo_pixel_weight = pseudo_weight.detach().float()
            if pseudo_conf is not None:
                pseudo_pixel_weight = pseudo_pixel_weight * pseudo_conf.detach().float().ge(
                    self.target_guided_source_filter_conf_threshold).float()
            pseudo_hist = self._label_histograms(
                pseudo_label,
                pseudo_pixel_weight,
            ).sum(dim=0)
            if float(pseudo_hist.sum().detach().item()) > 0:
                tgt_u_proto = self._update_hist_prototype(
                    '_target_unlabeled_hist_proto',
                    pseudo_hist,
                    update_unlabeled_prototype,
                )
                tgt_hist = tgt_hist + self.target_guided_source_filter_unlabeled_weight * tgt_u_proto
                used_unlabeled = 1.0
        elif self._target_unlabeled_hist_proto is not None:
            tgt_u_proto = self._target_unlabeled_hist_proto.to(
                device=tgt_hist.device,
                dtype=tgt_hist.dtype,
            )
            tgt_hist = tgt_hist + self.target_guided_source_filter_unlabeled_weight * tgt_u_proto
            used_unlabeled = 1.0

        if float(tgt_hist.sum().detach().item()) <= 0:
            return 1.0, {f'ssda_{prefix}_src_filter_valid': 0.0}

        tgt_hist = self._normalize_hist(tgt_hist)
        eps = 1e-6
        affinity = (src_hist * tgt_hist.unsqueeze(0)).sum(dim=1)
        affinity = affinity / (
            src_hist.norm(dim=1).clamp_min(eps) * tgt_hist.norm().clamp_min(eps))
        affinity = affinity.clamp(0.0, 1.0)
        proto_affinity = (src_proto * tgt_hist).sum() / (
            src_proto.norm().clamp_min(eps) * tgt_hist.norm().clamp_min(eps))
        proto_affinity = proto_affinity.clamp(0.0, 1.0)
        raw_weight = self.target_guided_source_filter_min_weight + (
            self.target_guided_source_filter_max_weight
            - self.target_guided_source_filter_min_weight
        ) * affinity.pow(max(1e-6, self.target_guided_source_filter_gamma))

        blend = float(self._scheduled_weight(
            self.target_guided_source_filter_blend,
            self.target_guided_source_filter_blend_final,
            self.target_guided_source_filter_blend_schedule,
        ))
        blend = min(1.0, max(0.0, blend))
        final_weight = (1.0 - blend) + blend * raw_weight
        scalar = float(final_weight.mean().detach().item())

        return scalar, {
            f'ssda_{prefix}_src_filter_valid': 1.0,
            f'ssda_{prefix}_src_filter_used_unlabeled': used_unlabeled,
            f'ssda_{prefix}_src_filter_blend': blend,
            f'ssda_{prefix}_src_filter_weight': scalar,
            f'ssda_{prefix}_src_filter_proto_update': 1.0 if
            self.target_guided_source_filter_proto_update == 'ema' else 0.0,
            f'ssda_{prefix}_src_filter_proto_affinity': float(
                proto_affinity.detach().item()),
            f'ssda_{prefix}_src_filter_target_class_coverage': float(
                tgt_hist.gt(0).float().mean().detach().item()),
            f'ssda_{prefix}_src_filter_weight_min': float(
                final_weight.min().detach().item()),
            f'ssda_{prefix}_src_filter_weight_max': float(
                final_weight.max().detach().item()),
            f'ssda_{prefix}_src_filter_affinity_mean': float(
                affinity.mean().detach().item()),
            f'ssda_{prefix}_src_filter_affinity_min': float(
                affinity.min().detach().item()),
            f'ssda_{prefix}_src_filter_affinity_max': float(
                affinity.max().detach().item()),
        }

    def _feature_proto_active(self):
        return (
            (
                self.feature_prototype_source_calibration
                or self.feature_tri_prototype_enabled
                or self.feature_prototype_diagnostics
            )
            and self.local_iter >= self.feature_prototype_begin_iter)

    def _feature_proto_compute_dense(self, features, labels, weights=None):
        dense, valid, _ = self._feature_proto_compute_dense_with_counts(
            features,
            labels,
            weights=weights,
        )
        return dense, valid

    def _feature_proto_compute_dense_with_counts(self, features, labels, weights=None):
        if features is None:
            return None, None, None
        try:
            result = compute_class_feature_prototypes(
                features,
                labels,
                self.num_classes,
                feature_level=self.feature_prototype_feature_level,
                weights=weights,
                ignore_index=getattr(self, 'ignore_index', 255),
                min_pixels=self.feature_prototype_min_pixels,
                detach=True,
            )
        except ValueError:
            return None, None, None
        dense, valid = scatter_class_prototypes(
            result.class_ids,
            result.prototypes,
            self.num_classes,
        )
        counts = dense.new_zeros((self.num_classes,))
        if result.class_ids.numel() > 0:
            counts[result.class_ids.long()] = result.counts.to(
                device=counts.device,
                dtype=counts.dtype,
            )
        return dense, valid, counts

    def _feature_proto_update_bank(self, proto_attr, valid_attr, dense, valid):
        if dense is None or valid is None:
            return False
        proto, proto_valid = update_prototype_bank(
            getattr(self, proto_attr),
            getattr(self, valid_attr),
            dense,
            valid,
            momentum=self.feature_prototype_momentum,
        )
        setattr(self, proto_attr, proto.detach())
        setattr(self, valid_attr, proto_valid.detach())
        return bool(proto_valid.any().detach().item())

    def _feature_proto_update_confidence_bank(self, conf_values, valid):
        if conf_values is None or valid is None:
            return
        conf_values = conf_values.detach().float()
        valid = valid.detach().bool().to(device=conf_values.device)
        if self._feature_proto_target_unlabeled_confidence is None:
            self._feature_proto_target_unlabeled_confidence = conf_values.clone()
            return
        old = self._feature_proto_target_unlabeled_confidence.to(
            device=conf_values.device,
            dtype=conf_values.dtype,
        ).clone()
        momentum = min(0.9999, max(0.0, self.feature_prototype_momentum))
        old[valid] = momentum * old[valid] + (1.0 - momentum) * conf_values[valid]
        self._feature_proto_target_unlabeled_confidence = old.detach()

    def _feature_proto_pseudo_confidence_by_class(
        self,
        pseudo_label,
        pseudo_weight,
        pseudo_conf,
    ):
        if pseudo_label is None or pseudo_weight is None or pseudo_conf is None:
            return None, None
        labels = pseudo_label.detach().long()
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        weight = pseudo_weight.detach().float()
        if weight.dim() == 4 and weight.shape[1] == 1:
            weight = weight[:, 0]
        conf = pseudo_conf.detach().float()
        if conf.dim() == 4 and conf.shape[1] == 1:
            conf = conf[:, 0]
        weight = weight * conf.ge(self.feature_prototype_conf_threshold).float()

        values = torch.zeros(
            self.num_classes,
            device=labels.device,
            dtype=torch.float32,
        )
        valid = torch.zeros(
            self.num_classes,
            device=labels.device,
            dtype=torch.bool,
        )
        for class_id in range(self.num_classes):
            mask = labels.eq(class_id) & weight.gt(0)
            if not mask.any():
                continue
            values[class_id] = conf[mask].mean()
            valid[class_id] = True
        return values, valid

    def _feature_proto_target_state(self, device):
        target_proto, target_valid = combine_target_prototypes(
            self._feature_proto_target_labeled,
            self._feature_proto_target_labeled_valid,
            self._feature_proto_target_unlabeled,
            self._feature_proto_target_unlabeled_valid,
            labeled_weight=self.feature_prototype_target_labeled_weight,
            unlabeled_weight=self.feature_prototype_target_unlabeled_weight,
        )
        if target_proto is None:
            return None, None
        return (
            target_proto.to(device=device),
            target_valid.to(device=device),
        )

    def _feature_proto_target_counts(self, device):
        ref = None
        for counts in (
            self._feature_proto_target_labeled_counts,
            self._feature_proto_target_unlabeled_counts,
        ):
            if counts is not None:
                ref = counts
                break
        if ref is None:
            return None, None

        target_counts = torch.zeros_like(ref, device=device, dtype=torch.float32)
        target_valid = torch.zeros(
            ref.shape[0],
            device=device,
            dtype=torch.bool,
        )
        if (
            self._feature_proto_target_labeled_counts is not None
            and self._feature_proto_target_labeled_valid is not None
        ):
            valid = self._feature_proto_target_labeled_valid.to(device).bool()
            counts = self._feature_proto_target_labeled_counts.to(
                device=device,
                dtype=target_counts.dtype,
            )
            target_counts[valid] += (
                self.feature_prototype_target_labeled_weight * counts[valid])
            target_valid |= valid
        if (
            self._feature_proto_target_unlabeled_counts is not None
            and self._feature_proto_target_unlabeled_valid is not None
        ):
            valid = self._feature_proto_target_unlabeled_valid.to(device).bool()
            counts = self._feature_proto_target_unlabeled_counts.to(
                device=device,
                dtype=target_counts.dtype,
            )
            target_counts[valid] += (
                self.feature_prototype_target_unlabeled_weight * counts[valid])
            target_valid |= valid
        return target_counts, target_valid

    def _feature_proto_source_scores(self, device):
        target_proto, target_valid = self._feature_proto_target_state(device)
        source_proto = self._feature_proto_source
        source_valid = self._feature_proto_source_valid
        if source_proto is None or source_valid is None or target_proto is None:
            return None
        return source_target_class_scores(
            source_proto.to(device=device),
            source_valid.to(device=device),
            target_proto,
            target_valid,
            min_score=self.feature_prototype_min_score,
            default_score=self.feature_prototype_default_score,
            score_norm=self.feature_prototype_score_norm,
            score_temperature=self.feature_prototype_score_temperature,
            quantile_low=self.feature_prototype_quantile_low,
            quantile_high=self.feature_prototype_quantile_high,
        )

    def _feature_proto_source_weight_map(self, src_seg_lbl):
        if not self._feature_proto_active() or not self.feature_prototype_source_weight:
            return None, {}
        scores = self._feature_proto_source_scores(src_seg_lbl.device)
        if scores is None:
            return None, {'ssda_featproto_src_weight_valid': 0.0}
        weight_map = class_score_weight_map(
            src_seg_lbl,
            scores,
            min_weight=self.feature_prototype_min_weight,
            max_weight=self.feature_prototype_max_weight,
            gamma=self.feature_prototype_gamma,
            default_weight=1.0,
            ignore_index=getattr(self, 'ignore_index', 255),
        )
        return weight_map, {
            'ssda_featproto_src_weight_valid': 1.0,
            'ssda_featproto_src_weight_mean': float(
                weight_map.detach().mean().item()),
            'ssda_featproto_src_weight_min': float(
                weight_map.detach().min().item()),
            'ssda_featproto_src_weight_max': float(
                weight_map.detach().max().item()),
            'ssda_featproto_class_score_mean': float(
                scores.detach().mean().item()),
            'ssda_featproto_class_score_min': float(
                scores.detach().min().item()),
            'ssda_featproto_class_score_max': float(
                scores.detach().max().item()),
        }

    def _feature_proto_class_weight_scores(self, scores):
        if scores is None:
            return None
        scores = scores.detach().float().clamp(0.0, 1.0)
        scaled = scores.pow(max(1e-6, self.feature_prototype_gamma))
        return (
            float(self.feature_prototype_min_weight)
            + (float(self.feature_prototype_max_weight)
               - float(self.feature_prototype_min_weight)) * scaled
        )

    def _feature_proto_source_mix_scores(self, device):
        if not self._feature_proto_active() or not self.feature_prototype_source_mix:
            return None, {}
        scores = self._feature_proto_source_scores(device)
        if scores is None:
            return None, {'ssda_featproto_src_mix_valid': 0.0}
        log_vars = {
            'ssda_featproto_src_mix_valid': 1.0,
            'ssda_featproto_src_mix_score_mean': float(
                scores.detach().mean().item()),
            'ssda_featproto_src_mix_score_min': float(
                scores.detach().min().item()),
            'ssda_featproto_src_mix_score_max': float(
                scores.detach().max().item()),
        }
        if self.feature_prototype_source_mix_balance:
            target_counts, target_valid = self._feature_proto_target_counts(device)
            if target_counts is not None and target_valid is not None:
                rarity = target_rare_class_scores(
                    target_counts,
                    target_valid,
                    gamma=self.feature_prototype_source_mix_rare_gamma,
                    min_score=self.feature_prototype_source_mix_rare_min,
                    max_score=self.feature_prototype_source_mix_rare_max,
                    normalize_mean=True,
                )
                scores = (scores * rarity).clamp_min(0.0)
                log_vars.update({
                    'ssda_featproto_src_mix_balance': 1.0,
                    'ssda_featproto_src_mix_rare_mean': float(
                        rarity.detach().mean().item()),
                    'ssda_featproto_src_mix_rare_min': float(
                        rarity.detach().min().item()),
                    'ssda_featproto_src_mix_rare_max': float(
                        rarity.detach().max().item()),
                    'ssda_featproto_src_mix_balanced_score_mean': float(
                        scores.detach().mean().item()),
                    'ssda_featproto_src_mix_balanced_score_min': float(
                        scores.detach().min().item()),
                    'ssda_featproto_src_mix_balanced_score_max': float(
                        scores.detach().max().item()),
                })
            else:
                log_vars['ssda_featproto_src_mix_balance'] = 0.0
        if (
            self.feature_prototype_source_mix_structure_protection
            and self.feature_prototype_source_mix_structure_classes
        ):
            before = scores.detach()
            scores = protect_structure_class_scores(
                scores,
                structure_classes=(
                    self.feature_prototype_source_mix_structure_classes),
                mode=self.feature_prototype_source_mix_structure_mode,
                min_score=(
                    self.feature_prototype_source_mix_structure_min_score),
                default_score=(
                    self.feature_prototype_source_mix_structure_default_score),
            )
            class_idx = torch.as_tensor(
                self.feature_prototype_source_mix_structure_classes,
                device=scores.device,
                dtype=torch.long,
            )
            log_vars.update({
                'ssda_featproto_src_mix_structure_protection': 1.0,
                'ssda_featproto_src_mix_structure_count': float(
                    class_idx.numel()),
                'ssda_featproto_src_mix_structure_score_before_mean': float(
                    before[class_idx].mean().detach().item()),
                'ssda_featproto_src_mix_structure_score_after_mean': float(
                    scores[class_idx].mean().detach().item()),
                'ssda_featproto_src_mix_structure_score_min': float(
                    scores[class_idx].min().detach().item()),
                'ssda_featproto_src_mix_structure_score_max': float(
                    scores[class_idx].max().detach().item()),
            })
        else:
            log_vars['ssda_featproto_src_mix_structure_protection'] = 0.0
        return scores, {
            **log_vars,
        }

    def _feature_proto_update_labeled_state(
        self,
        src_features,
        src_seg_lbl,
        tgt_l_features,
        tgt_l_seg_lbl,
    ):
        if not self._feature_proto_active():
            return {}

        src_dense, src_valid, src_counts = self._feature_proto_compute_dense_with_counts(
            src_features,
            src_seg_lbl,
        )
        tgt_dense, tgt_valid, tgt_counts = self._feature_proto_compute_dense_with_counts(
            tgt_l_features,
            tgt_l_seg_lbl,
        )
        source_updated = self._feature_proto_update_bank(
            '_feature_proto_source',
            '_feature_proto_source_valid',
            src_dense,
            src_valid,
        )
        target_l_updated = self._feature_proto_update_bank(
            '_feature_proto_target_labeled',
            '_feature_proto_target_labeled_valid',
            tgt_dense,
            tgt_valid,
        )
        if src_counts is not None:
            self._feature_proto_source_counts = src_counts.detach()
        if tgt_counts is not None:
            self._feature_proto_target_labeled_counts = tgt_counts.detach()

        log_vars = {
            'ssda_featproto_active': 1.0,
            'ssda_featproto_source_updated': 1.0 if source_updated else 0.0,
            'ssda_featproto_target_labeled_updated': (
                1.0 if target_l_updated else 0.0),
        }
        if self._feature_proto_source_valid is not None:
            log_vars['ssda_featproto_source_coverage'] = float(
                self._feature_proto_source_valid.float().mean().item())
        if self._feature_proto_target_labeled_valid is not None:
            log_vars['ssda_featproto_target_labeled_coverage'] = float(
                self._feature_proto_target_labeled_valid.float().mean().item())
        return log_vars

    def _feature_proto_update_unlabeled_state(
        self,
        tgt_u_img,
        pseudo_label,
        pseudo_weight,
        pseudo_conf,
    ):
        if (
            not self._feature_proto_active()
            or not self.feature_prototype_unlabeled_forward
            or self.feature_prototype_target_unlabeled_weight <= 0
            or pseudo_label is None
            or pseudo_weight is None
        ):
            return {}

        pixel_weight = pseudo_weight.detach().float()
        if pseudo_conf is not None:
            pixel_weight = pixel_weight * pseudo_conf.detach().float().ge(
                self.feature_prototype_conf_threshold).float()
        if float(pixel_weight.sum().detach().item()) <= 0:
            return {'ssda_featproto_target_unlabeled_updated': 0.0}
        conf_values, conf_valid = self._feature_proto_pseudo_confidence_by_class(
            pseudo_label,
            pseudo_weight,
            pseudo_conf,
        )
        self._feature_proto_update_confidence_bank(conf_values, conf_valid)

        was_training = self.get_model().training
        with torch.no_grad():
            features = self.get_model().extract_feat(tgt_u_img)
        if was_training:
            self.get_model().train()

        dense, valid, counts = self._feature_proto_compute_dense_with_counts(
            features,
            pseudo_label,
            weights=pixel_weight,
        )
        target_u_updated = self._feature_proto_update_bank(
            '_feature_proto_target_unlabeled',
            '_feature_proto_target_unlabeled_valid',
            dense,
            valid,
        )
        if counts is not None:
            self._feature_proto_target_unlabeled_counts = counts.detach()
        log_vars = {
            'ssda_featproto_target_unlabeled_updated': (
                1.0 if target_u_updated else 0.0),
        }
        if self._feature_proto_target_unlabeled_valid is not None:
            log_vars['ssda_featproto_target_unlabeled_coverage'] = float(
                self._feature_proto_target_unlabeled_valid.float().mean().item())
        if conf_values is not None and conf_valid is not None and conf_valid.any():
            log_vars['ssda_featproto_target_unlabeled_conf_mean'] = float(
                conf_values[conf_valid].mean().detach().item())
        return log_vars

    def _feature_proto_update_source_mix_counts(self, src_seg_lbl, mix_masks):
        if (
            not self._feature_proto_active()
            or not self.feature_prototype_diagnostics
            or src_seg_lbl is None
            or not mix_masks
        ):
            return
        counts = torch.zeros(
            self.num_classes,
            dtype=torch.float32,
            device=src_seg_lbl.device,
        )
        total = 0.0
        labels = src_seg_lbl.long()
        for idx, mix_mask in enumerate(mix_masks):
            if idx >= labels.shape[0]:
                break
            mask = mix_mask
            if isinstance(mask, (list, tuple)):
                mask = mask[0]
            if mask.dim() == 4:
                mask = mask[0, 0]
            elif mask.dim() == 3:
                mask = mask[0]
            mask = mask.to(device=labels.device).bool()
            label = labels[idx]
            if mask.shape != label.shape:
                mask = torch.nn.functional.interpolate(
                    mask.float().view(1, 1, *mask.shape),
                    size=label.shape,
                    mode='nearest',
                ).view(*label.shape).bool()
            selected = label[mask]
            valid = (
                selected.ne(getattr(self, 'ignore_index', 255))
                & selected.ge(0)
                & selected.lt(self.num_classes)
            )
            selected = selected[valid]
            if selected.numel() == 0:
                continue
            counts += torch.bincount(
                selected,
                minlength=self.num_classes,
            ).float()
            total += float(selected.numel())
        self._feature_proto_source_mix_selected_counts = counts.detach()
        self._feature_proto_source_mix_total_count = float(total)

    def _feature_proto_update_target_mix_counts(self, tgt_l_seg_lbl, mix_masks):
        if (
            not self._feature_proto_active()
            or not self.feature_prototype_diagnostics
            or tgt_l_seg_lbl is None
            or not mix_masks
        ):
            return
        counts = torch.zeros(
            self.num_classes,
            dtype=torch.float32,
            device=tgt_l_seg_lbl.device,
        )
        total = 0.0
        labels = tgt_l_seg_lbl.long()
        for idx, mix_mask in enumerate(mix_masks):
            if idx >= labels.shape[0]:
                break
            mask = mix_mask
            if isinstance(mask, (list, tuple)):
                mask = mask[0]
            if mask.dim() == 4:
                mask = mask[0, 0]
            elif mask.dim() == 3:
                mask = mask[0]
            mask = mask.to(device=labels.device).bool()
            label = labels[idx]
            if mask.shape != label.shape:
                mask = F.interpolate(
                    mask.float().view(1, 1, *mask.shape),
                    size=label.shape,
                    mode='nearest',
                ).view(*label.shape).bool()
            selected = label[mask]
            valid = (
                selected.ne(getattr(self, 'ignore_index', 255))
                & selected.ge(0)
                & selected.lt(self.num_classes)
            )
            selected = selected[valid]
            if selected.numel() == 0:
                continue
            counts += torch.bincount(
                selected,
                minlength=self.num_classes,
            ).float()
            total += float(selected.numel())
        self._feature_proto_target_mix_selected_counts = counts.detach()
        self._feature_proto_target_mix_total_count = float(total)

    def _source_assist_reset_diagnostic_iteration_state(self):
        if not self.source_assist_diagnostics:
            return
        self._source_assist_target_labeled_counts = None
        self._source_assist_target_unlabeled_counts = None
        self._source_assist_target_unlabeled_confidence = None
        self._source_assist_target_deficit_scores = None
        self._source_assist_route_scores = None
        self._source_assist_source_mix_scores = None
        self._source_assist_source_mix_selected_counts = None
        self._source_assist_source_mix_total_count = 0.0
        self._source_assist_loss_contributions = {}

    def _source_assist_class_counts(self, labels, weights=None):
        labels = labels.long()
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        ignore_index = getattr(self, 'ignore_index', 255)
        valid = labels.ne(ignore_index) & labels.ge(0) & labels.lt(self.num_classes)
        if weights is not None:
            weights = weights.to(device=labels.device).float().clamp_min(0.0)
            if weights.dim() == 4 and weights.shape[1] == 1:
                weights = weights[:, 0]
            if weights.shape != labels.shape:
                weights = self._resize_diag_map(
                    weights,
                    labels.shape[-2:],
                    labels.device,
                    dtype=torch.float32,
                )
            if weights is not None:
                valid = valid & weights.gt(0)
        if not valid.any():
            return torch.zeros(
                self.num_classes,
                dtype=torch.float32,
                device=labels.device,
            )
        flat_labels = labels[valid].reshape(-1)
        if weights is None:
            return torch.bincount(
                flat_labels,
                minlength=self.num_classes,
            ).float()
        flat_weights = weights[valid].reshape(-1)
        return torch.bincount(
            flat_labels,
            weights=flat_weights,
            minlength=self.num_classes,
        ).float()

    def _source_assist_record_target_state(
        self,
        tgt_l_seg_lbl,
        pseudo_label,
        pseudo_weight,
        pseudo_conf,
    ):
        if not self.source_assist_diagnostics:
            return
        with torch.no_grad():
            if tgt_l_seg_lbl is not None:
                self._source_assist_target_labeled_counts = (
                    self._source_assist_class_counts(tgt_l_seg_lbl).detach())
            if pseudo_label is None:
                return
            self._source_assist_target_unlabeled_counts = (
                self._source_assist_class_counts(
                    pseudo_label,
                    pseudo_weight,
                ).detach())
            if pseudo_conf is None:
                return
            weights = pseudo_weight
            if weights is None:
                weights = torch.ones_like(pseudo_label, dtype=torch.float32)
            labels = pseudo_label.long()
            if labels.dim() == 4 and labels.shape[1] == 1:
                labels = labels[:, 0]
            weights = weights.to(device=labels.device).float().clamp_min(0.0)
            if weights.dim() == 4 and weights.shape[1] == 1:
                weights = weights[:, 0]
            conf = pseudo_conf.to(device=labels.device).float()
            if conf.dim() == 4 and conf.shape[1] == 1:
                conf = conf[:, 0]
            ignore_index = getattr(self, 'ignore_index', 255)
            valid = (
                labels.ne(ignore_index)
                & labels.ge(0)
                & labels.lt(self.num_classes)
                & weights.gt(0)
            )
            if not valid.any():
                return
            flat_labels = labels[valid].reshape(-1)
            flat_weights = weights[valid].reshape(-1)
            conf_sum = torch.bincount(
                flat_labels,
                weights=(conf[valid].reshape(-1) * flat_weights),
                minlength=self.num_classes,
            ).float()
            weight_sum = torch.bincount(
                flat_labels,
                weights=flat_weights,
                minlength=self.num_classes,
            ).float()
            conf_mean = torch.full(
                (self.num_classes,),
                float('nan'),
                device=labels.device,
                dtype=torch.float32,
            )
            valid_classes = weight_sum.gt(0)
            conf_mean[valid_classes] = (
                conf_sum[valid_classes] / weight_sum[valid_classes].clamp_min(1e-6)
            )
            self._source_assist_target_unlabeled_confidence = (
                conf_mean.detach())

    def _source_assist_update_source_mix_counts(self, src_seg_lbl, mix_masks):
        if (
            not self.source_assist_diagnostics
            or src_seg_lbl is None
            or not mix_masks
        ):
            return
        counts = torch.zeros(
            self.num_classes,
            dtype=torch.float32,
            device=src_seg_lbl.device,
        )
        total = 0.0
        labels = src_seg_lbl.long()
        for idx, mix_mask in enumerate(mix_masks):
            if idx >= labels.shape[0]:
                break
            mask = mix_mask
            if isinstance(mask, (list, tuple)):
                mask = mask[0]
            if mask.dim() == 4:
                mask = mask[0, 0]
            elif mask.dim() == 3:
                mask = mask[0]
            mask = mask.to(device=labels.device).bool()
            label = labels[idx]
            if mask.shape != label.shape:
                mask = F.interpolate(
                    mask.float().view(1, 1, *mask.shape),
                    size=label.shape,
                    mode='nearest',
                ).view(*label.shape).bool()
            selected = label[mask]
            valid = (
                selected.ne(getattr(self, 'ignore_index', 255))
                & selected.ge(0)
                & selected.lt(self.num_classes)
            )
            selected = selected[valid]
            if selected.numel() == 0:
                continue
            counts += torch.bincount(
                selected,
                minlength=self.num_classes,
            ).float()
            total += float(selected.numel())
        self._source_assist_source_mix_selected_counts = counts.detach()
        self._source_assist_source_mix_total_count = float(total)

    def _source_assist_set_target_deficit_scores(self, scores):
        if self.source_assist_diagnostics and scores is not None:
            self._source_assist_target_deficit_scores = scores.detach()

    def _source_assist_set_route_scores(self, scores):
        if self.source_assist_diagnostics and scores is not None:
            self._source_assist_route_scores = scores.detach()

    def _source_assist_set_source_mix_scores(self, scores):
        if self.source_assist_diagnostics and scores is not None:
            self._source_assist_source_mix_scores = scores.detach()

    def _feature_proto_reset_diagnostic_iteration_state(self):
        if not self.feature_prototype_diagnostics:
            return
        self._feature_proto_source_mix_selected_counts = None
        self._feature_proto_source_mix_total_count = 0.0
        self._feature_proto_target_mix_selected_counts = None
        self._feature_proto_target_mix_total_count = 0.0
        self._feature_proto_loss_contributions = {}

    def _select_feature_proto_diag_logits(self, logits):
        """Pick a segmentation logits tensor from HRDA-style nested outputs."""
        candidates = []

        def collect(item):
            if item is None:
                return
            if torch.is_tensor(item):
                if item.dim() == 4:
                    candidates.append(item)
                return
            if isinstance(item, dict):
                for key in ('seg_logits', 'logits', 'S'):
                    if key in item:
                        collect(item[key])
                for value in item.values():
                    collect(value)
                return
            if isinstance(item, (list, tuple)):
                for value in item:
                    collect(value)

        collect(logits)
        if not candidates:
            return None
        for candidate in candidates:
            if candidate.shape[1] == self.num_classes:
                return candidate
        return candidates[0]

    @staticmethod
    def _resize_diag_map(tensor, size, device, dtype=None, mode='nearest'):
        if tensor is None:
            return None
        if tensor.dim() == 4 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        if tensor.dim() != 3:
            return None
        tensor = tensor.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if tuple(tensor.shape[-2:]) == tuple(size):
            return tensor
        target_dtype = tensor.dtype
        interp_tensor = tensor
        if not interp_tensor.is_floating_point():
            interp_tensor = interp_tensor.float()
        resized = F.interpolate(
            interp_tensor.unsqueeze(1),
            size=size,
            mode=mode,
        ).squeeze(1)
        if resized.dtype != target_dtype:
            resized = resized.to(dtype=target_dtype)
        return resized

    def _diagnostic_per_class_loss(self, logits, labels, weights=None):
        if logits is None or labels is None:
            return None
        with torch.no_grad():
            logits = self._select_feature_proto_diag_logits(logits)
            if logits is None:
                return None
            logits = logits.detach().float()
            labels = self._resize_diag_map(
                labels.detach().long(),
                logits.shape[-2:],
                logits.device,
            )
            if labels is None:
                return None
            ignore_index = getattr(self, 'ignore_index', 255)
            valid = labels.ne(ignore_index) & labels.ge(0) & labels.lt(self.num_classes)
            safe_labels = labels.clone()
            safe_labels[~valid] = ignore_index
            pixel_loss = F.cross_entropy(
                logits,
                safe_labels,
                reduction='none',
                ignore_index=ignore_index,
            )
            if weights is not None:
                weights = self._resize_diag_map(
                    weights.detach().float(),
                    logits.shape[-2:],
                    logits.device,
                    dtype=torch.float32,
                )
                if weights is not None:
                    weights = weights.clamp_min(0.0)
                    pixel_loss = pixel_loss * weights
                    count_weight = weights
                    valid = valid & weights.gt(0)
                else:
                    count_weight = torch.ones_like(pixel_loss)
            else:
                count_weight = torch.ones_like(pixel_loss)

            if not valid.any():
                return None
            flat_labels = safe_labels[valid].reshape(-1)
            flat_loss = pixel_loss[valid].reshape(-1)
            flat_count = count_weight[valid].reshape(-1)
            loss_sum = torch.bincount(
                flat_labels,
                weights=flat_loss,
                minlength=self.num_classes,
            ).float()
            count_sum = torch.bincount(
                flat_labels,
                weights=flat_count,
                minlength=self.num_classes,
            ).float()
            return {
                'sum': loss_sum.detach(),
                'count': count_sum.detach(),
            }

    def _feature_proto_record_per_class_loss(
        self,
        branch,
        logits,
        labels,
        weights=None,
    ):
        if (
            not self._feature_proto_active()
            or not self.feature_prototype_diagnostics
        ):
            return
        stats = self._diagnostic_per_class_loss(logits, labels, weights)
        if stats is not None:
            self._feature_proto_loss_contributions[str(branch)] = {
                'sum': stats['sum'],
                'count': stats['count'],
            }

    def _source_assist_record_per_class_loss(
        self,
        branch,
        logits,
        labels,
        weights=None,
    ):
        if not self.source_assist_diagnostics:
            return
        stats = self._diagnostic_per_class_loss(logits, labels, weights)
        if stats is not None:
            self._source_assist_loss_contributions[str(branch)] = {
                'sum': stats['sum'],
                'count': stats['count'],
            }

    def _target_need_update_loss_feedback(self, logits, labels, weights=None):
        if (
            not self.target_need_source_mix
            or not self.target_need_source_mix_use_target_loss
            or self.target_need_source_mix_target_loss_weight <= 0
            or logits is None
            or labels is None
        ):
            return {}
        with torch.no_grad():
            logits = self._select_feature_proto_diag_logits(logits)
            if logits is None:
                return {'ssda_tnsa_loss_feedback_valid': 0.0}
            logits = logits.detach().float()
            labels = self._resize_diag_map(
                labels.detach().long(),
                logits.shape[-2:],
                logits.device,
            )
            if labels is None:
                return {'ssda_tnsa_loss_feedback_valid': 0.0}
            ignore_index = getattr(self, 'ignore_index', 255)
            valid = labels.ne(ignore_index) & labels.ge(0) & labels.lt(self.num_classes)
            safe_labels = labels.clone()
            safe_labels[~valid] = ignore_index
            pixel_loss = F.cross_entropy(
                logits,
                safe_labels,
                reduction='none',
                ignore_index=ignore_index,
            )
            if weights is not None:
                weights = self._resize_diag_map(
                    weights.detach().float(),
                    logits.shape[-2:],
                    logits.device,
                    dtype=torch.float32,
                )
                if weights is not None:
                    weights = weights.clamp_min(0.0)
                    valid = valid & weights.gt(0)
                    pixel_loss = pixel_loss * weights
                    count_weight = weights
                else:
                    count_weight = torch.ones_like(pixel_loss)
            else:
                count_weight = torch.ones_like(pixel_loss)
            if not valid.any():
                return {'ssda_tnsa_loss_feedback_valid': 0.0}
            flat_labels = safe_labels[valid].reshape(-1)
            loss_sum = torch.bincount(
                flat_labels,
                weights=pixel_loss[valid].reshape(-1),
                minlength=self.num_classes,
            ).float()
            count_sum = torch.bincount(
                flat_labels,
                weights=count_weight[valid].reshape(-1),
                minlength=self.num_classes,
            ).float()
            valid_classes = count_sum.gt(0)
            if not valid_classes.any():
                return {'ssda_tnsa_loss_feedback_valid': 0.0}
            class_loss = torch.ones(
                self.num_classes,
                device=logits.device,
                dtype=torch.float32,
            )
            class_loss[valid_classes] = (
                loss_sum[valid_classes] / count_sum[valid_classes].clamp_min(1e-6)
            )
            if self._target_need_loss_ema is None:
                self._target_need_loss_ema = class_loss.detach()
            else:
                old = self._target_need_loss_ema.to(
                    device=class_loss.device,
                    dtype=class_loss.dtype,
                )
                momentum = min(
                    0.9999,
                    max(0.0, self.target_need_source_mix_target_loss_momentum),
                )
                updated = old.clone()
                updated[valid_classes] = (
                    momentum * old[valid_classes]
                    + (1.0 - momentum) * class_loss[valid_classes])
                self._target_need_loss_ema = updated.detach()
            valid_loss = self._target_need_loss_ema[valid_classes]
            return {
                'ssda_tnsa_loss_feedback_valid': 1.0,
                'ssda_tnsa_loss_feedback_class_count': float(
                    valid_classes.float().sum().item()),
                'ssda_tnsa_loss_feedback_mean': float(
                    valid_loss.mean().detach().item()),
                'ssda_tnsa_loss_feedback_max': float(
                    valid_loss.max().detach().item()),
            }

    def _feature_proto_get_diagnostic_exporter(self):
        if self._feature_proto_diagnostic_exporter is not None:
            return self._feature_proto_diagnostic_exporter
        work_dir = self.cfg.model.get('train_cfg', {}).get('work_dir', None)
        if self.feature_prototype_diagnostic_dir is not None:
            out_dir = Path(self.feature_prototype_diagnostic_dir)
        elif work_dir is not None:
            out_dir = Path(work_dir) / 'diagnostics' / 'feature_prototype'
        else:
            out_dir = (
                Path('work_dirs') / 'reports' / 'diagnostics' /
                'feature_prototype')
        self._feature_proto_diagnostic_exporter = FeaturePrototypeDiagnosticExporter(
            out_dir,
            class_names=self.dataset_class,
        )
        return self._feature_proto_diagnostic_exporter

    def _feature_proto_export_diagnostics(self):
        if (
            not self.feature_prototype_diagnostics
            or not self._feature_proto_active()
            or not self._is_master_process()
        ):
            return {}
        interval = max(1, int(self.feature_prototype_diagnostic_interval))
        iteration = int(self.local_iter + 1)
        if iteration % interval != 0:
            return {}
        device = None
        for proto in (
            self._feature_proto_source,
            self._feature_proto_target_labeled,
            self._feature_proto_target_unlabeled,
        ):
            if proto is not None:
                device = proto.device
                break
        if device is None:
            return {'ssda_featproto_diag_exported': 0.0}
        target_proto, target_valid = self._feature_proto_target_state(device)
        scores = self._feature_proto_source_scores(device)
        class_weights = self._feature_proto_class_weight_scores(scores)
        exporter = self._feature_proto_get_diagnostic_exporter()
        exporter.export(
            iteration=iteration,
            source_proto=self._feature_proto_source,
            source_valid=self._feature_proto_source_valid,
            target_labeled_proto=self._feature_proto_target_labeled,
            target_labeled_valid=self._feature_proto_target_labeled_valid,
            target_unlabeled_proto=self._feature_proto_target_unlabeled,
            target_unlabeled_valid=self._feature_proto_target_unlabeled_valid,
            target_proto=target_proto,
            target_valid=target_valid,
            class_scores=scores,
            class_weights=class_weights,
            source_counts=self._feature_proto_source_counts,
            target_labeled_counts=self._feature_proto_target_labeled_counts,
            target_unlabeled_counts=self._feature_proto_target_unlabeled_counts,
            target_unlabeled_confidence=(
                self._feature_proto_target_unlabeled_confidence),
            source_mix_selected_counts=self._feature_proto_source_mix_selected_counts,
            source_mix_total_count=self._feature_proto_source_mix_total_count,
            target_mix_selected_counts=self._feature_proto_target_mix_selected_counts,
            target_mix_total_count=self._feature_proto_target_mix_total_count,
            loss_contributions=self._feature_proto_loss_contributions,
        )
        return {'ssda_featproto_diag_exported': 1.0}

    def _source_assist_get_diagnostic_exporter(self):
        if self._source_assist_diagnostic_exporter is not None:
            return self._source_assist_diagnostic_exporter
        work_dir = self.cfg.model.get('train_cfg', {}).get('work_dir', None)
        if self.source_assist_diagnostic_dir is not None:
            out_dir = Path(self.source_assist_diagnostic_dir)
        elif work_dir is not None:
            out_dir = Path(work_dir) / 'diagnostics' / 'source_assist'
        else:
            out_dir = (
                Path('work_dirs') / 'reports' / 'diagnostics' /
                'source_assist')
        self._source_assist_diagnostic_exporter = (
            SourceAssistDiagnosticExporter(
                out_dir,
                class_names=self.dataset_class,
            ))
        return self._source_assist_diagnostic_exporter

    def _source_assist_export_diagnostics(self):
        if (
            not self.source_assist_diagnostics
            or not self._is_master_process()
        ):
            return {}
        interval = max(1, int(self.source_assist_diagnostic_interval))
        iteration = int(self.local_iter + 1)
        if iteration % interval != 0:
            return {}
        has_signal = any(value is not None for value in (
            self._source_assist_target_labeled_counts,
            self._source_assist_target_unlabeled_counts,
            self._source_assist_target_deficit_scores,
            self._source_assist_route_scores,
            self._source_assist_source_mix_scores,
            self._source_assist_source_mix_selected_counts,
        ))
        if not has_signal:
            return {'ssda_source_assist_diag_exported': 0.0}
        exporter = self._source_assist_get_diagnostic_exporter()
        exporter.export(
            iteration=iteration,
            target_labeled_counts=self._source_assist_target_labeled_counts,
            target_unlabeled_counts=self._source_assist_target_unlabeled_counts,
            target_unlabeled_confidence=(
                self._source_assist_target_unlabeled_confidence),
            target_deficit_scores=self._source_assist_target_deficit_scores,
            route_scores=self._source_assist_route_scores,
            source_mix_scores=self._source_assist_source_mix_scores,
            source_mix_selected_counts=(
                self._source_assist_source_mix_selected_counts),
            source_mix_total_count=self._source_assist_source_mix_total_count,
            target_loss_feedback=self._target_need_loss_ema,
            loss_contributions=self._source_assist_loss_contributions,
        )
        return {'ssda_source_assist_diag_exported': 1.0}

    def _unlabeled_consistency_branch_weight(self):
        if not self.unlabeled_consistency_enabled:
            return 0.0
        if self.local_iter < self.unlabeled_consistency_begin_iter:
            return 0.0
        return float(self._scheduled_weight(
            self.unlabeled_consistency_weight,
            self.unlabeled_consistency_weight_final,
            self.unlabeled_consistency_weight_schedule,
        ))

    def _make_unlabeled_strong_view(self, img, means, stds):
        batch_size = img.shape[0]
        params = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[:batch_size],
            'std': stds[:batch_size],
        }
        strong_img, _ = strong_transform_wo_mix(params, data=img.clone())
        return strong_img

    def _obtain_unlabeled_cutmix_masks(self, batch_size, height, width, device):
        masks = torch.zeros(
            (batch_size, height, width), device=device, dtype=torch.float32)
        if self.unlabeled_consistency_cutmix_prob <= 0 or batch_size < 2:
            return masks

        area = float(height * width)
        for idx in range(batch_size):
            if random.uniform(0, 1) > self.unlabeled_consistency_cutmix_prob:
                continue
            for _ in range(10):
                target_area = random.uniform(
                    self.unlabeled_consistency_cutmix_min,
                    self.unlabeled_consistency_cutmix_max) * area
                ratio = random.uniform(
                    self.unlabeled_consistency_cutmix_ratio_min,
                    self.unlabeled_consistency_cutmix_ratio_max)
                cutmix_w = int(np.sqrt(target_area / ratio))
                cutmix_h = int(np.sqrt(target_area * ratio))
                if cutmix_w <= 0 or cutmix_h <= 0:
                    continue
                if cutmix_w <= width and cutmix_h <= height:
                    x = random.randint(0, width - cutmix_w)
                    y = random.randint(0, height - cutmix_h)
                    masks[idx, y:y + cutmix_h, x:x + cutmix_w] = 1
                    break
        return masks

    @staticmethod
    def _cutmix_map(value, mask):
        if value is None:
            return None
        mixed = value.clone()
        flipped = value.flip(0)
        mixed[mask.bool()] = flipped[mask.bool()]
        return mixed

    def _apply_unlabeled_cutmix(self, img, pseudo_label, pseudo_weight, mask):
        if float(mask.sum().detach().item()) <= 0:
            return img, pseudo_label, pseudo_weight
        mixed_img = img.clone()
        img_mask = mask.bool().unsqueeze(1).expand_as(mixed_img)
        mixed_img[img_mask] = img.flip(0)[img_mask]
        mixed_label = self._cutmix_map(pseudo_label, mask)
        mixed_weight = self._cutmix_map(pseudo_weight, mask)
        return mixed_img, mixed_label, mixed_weight

    def _build_unlabeled_consistency_weight(self, pseudo_weight, pseudo_conf):
        valid_region = pseudo_weight > 0
        confident = pseudo_conf.ge(self.unlabeled_consistency_conf_thresh)
        if self.unlabeled_consistency_use_dacs_pseudo_weight:
            base_weight = pseudo_weight
        else:
            base_weight = valid_region.to(pseudo_weight.dtype)
        weight = base_weight * confident.to(pseudo_weight.dtype)
        if self.unlabeled_consistency_confidence_weight:
            weight = weight * pseudo_conf.detach()

        valid_den = valid_region.float().sum().clamp_min(1.0)
        mask_ratio = (confident & valid_region).float().sum() / valid_den
        return weight, float(mask_ratio.detach().item())

    def _forward_unlabeled_consistency_branch(
        self,
        tgt_u_img,
        pseudo_label,
        pseudo_weight,
        pseudo_conf,
        means,
        stds,
        batch_size,
        dev,
        seg_debug,
        branch_weight,
    ):
        """UniMatch-style weak-to-strong consistency on unlabeled target data."""
        if branch_weight <= 0:
            return {
                'log_vars': {},
                'loss_value': 0.0,
                'raw_loss_value': 0.0,
            }

        confident_weight, mask_ratio = self._build_unlabeled_consistency_weight(
            pseudo_weight,
            pseudo_conf,
        )
        strong_imgs = []
        strong_labels = []
        strong_weights = []
        cutmix_ratios = []
        for _ in range(self.unlabeled_consistency_views):
            strong_img = self._make_unlabeled_strong_view(tgt_u_img, means, stds)
            _, _, height, width = strong_img.shape
            cutmix_mask = self._obtain_unlabeled_cutmix_masks(
                batch_size,
                height,
                width,
                dev,
            )
            strong_img, strong_label, strong_weight = self._apply_unlabeled_cutmix(
                strong_img,
                pseudo_label,
                confident_weight,
                cutmix_mask,
            )
            strong_imgs.append(strong_img)
            strong_labels.append(strong_label)
            strong_weights.append(strong_weight)
            cutmix_ratios.append(float(cutmix_mask.mean().detach().item()))

        strong_img = torch.cat(strong_imgs, dim=0)
        strong_label = torch.cat(strong_labels, dim=0)
        strong_weight = torch.cat(strong_weights, dim=0)
        pred_results = self.get_model().forward_train(
            (strong_img, strong_label),
            seg_weight=strong_weight,
            return_feat=False,
            loss_key='unsup',
        )
        seg_debug['Unlabeled Strong'] = self.get_model().decode_head.debug_output
        pred_results.pop('seg_logits', None)
        raw_loss, log_vars = parse_losses(pred_results)
        total_weight = float(branch_weight) / max(
            self.unlabeled_consistency_total_divisor,
            1e-6,
        )
        weighted_loss = raw_loss * total_weight
        weighted_loss.backward()

        prefixed = add_prefix(log_vars, 'ulb')
        prefixed.update({
            'ulb_loss_weight': float(branch_weight),
            'ulb_total_divisor': float(self.unlabeled_consistency_total_divisor),
            'ulb_weighted_loss': float(weighted_loss.detach().item()),
            'ssda_unlabeled_branch_weight': float(branch_weight),
            'ssda_unlabeled_branch_views': float(self.unlabeled_consistency_views),
            'ssda_unlabeled_mask_ratio': mask_ratio,
            'ssda_unlabeled_cutmix_ratio': float(np.mean(cutmix_ratios)),
            'ssda_unlabeled_conf_thresh': self.unlabeled_consistency_conf_thresh,
        })
        return {
            'log_vars': prefixed,
            'loss_value': float(weighted_loss.detach().item()),
            'raw_loss_value': float(raw_loss.detach().item()),
        }

    def _forward_labeled_loss(self, img, seg_lbl, seg_debug, debug_key,
                              branch_prefix, loss_weight=1.0, backward=True,
                              seg_weight=None, return_feat=False,
                              detach_features=True, loss_key=None):
        """Run one labeled supervised branch and backpropagate it."""
        pred_results = self.get_model().forward_train(
            (img, seg_lbl),
            seg_weight=seg_weight,
            return_feat=return_feat,
            loss_key=loss_key,
        )
        seg_debug[debug_key] = self.get_model().decode_head.debug_output
        features = pred_results.pop('features', None)
        if detach_features:
            features = self._detach_feature_payload(features)
        seg_pred = pred_results.pop('seg_logits', None)
        seg_loss, log_vars = parse_losses(pred_results)
        weighted_loss = seg_loss * float(loss_weight)
        if backward and float(loss_weight) > 0:
            weighted_loss.backward()

        prefixed = add_prefix(log_vars, branch_prefix)
        prefixed[f'{branch_prefix}_loss_weight'] = float(loss_weight)
        prefixed[f'{branch_prefix}_weighted_loss'] = float(
            weighted_loss.detach().item())
        loss_tensor = weighted_loss.detach() if backward else weighted_loss
        raw_loss_tensor = seg_loss.detach() if backward else seg_loss
        return {
            'seg_pred': seg_pred,
            'features': features,
            'log_vars': prefixed,
            'loss_value': float(weighted_loss.detach().item()),
            'raw_loss_value': float(seg_loss.detach().item()),
            'loss_tensor': loss_tensor,
            'raw_loss_tensor': raw_loss_tensor,
        }

    def _build_gt_mix_batch(self, src_img, src_seg_lbl, tgt_img, tgt_seg_lbl,
                            strong_parameters, batch_size, dev,
                            class_scores=None,
                            class_score_random_prob=None,
                            quota_classmix=False,
                            mask_routing_classmix=False,
                            source_pixel_class_scores=None):
        """Build ClassMix samples when both sides have ground-truth labels."""
        mix_img, mix_seg_lbl = [None] * batch_size, [None] * batch_size
        mix_seg_weight = torch.ones(tgt_seg_lbl.shape, device=dev)
        src_pixel_weight = self._source_deficit_pixel_weight_map(
            src_seg_lbl,
            source_pixel_class_scores,
        )
        if src_pixel_weight is None:
            src_pixel_weight = torch.ones(src_seg_lbl.shape, device=dev)
        tgt_pixel_weight = torch.ones(tgt_seg_lbl.shape, device=dev)

        src_mix_ratio = self.get_src_cls_mix_ratio()
        if mask_routing_classmix:
            mix_masks, num_class_choice, mix_log_vars = \
                self._build_target_need_mask_routing_mix_masks(
                    src_seg_lbl.unsqueeze(1),
                    src_mix_ratio,
                    'gt_mix',
                    class_scores,
                )
        elif quota_classmix:
            mix_masks, num_class_choice, mix_log_vars = \
                self._build_target_deficit_quota_mix_masks(
                    src_seg_lbl.unsqueeze(1),
                    src_mix_ratio,
                    'gt_mix',
                    class_scores,
                )
        else:
            mix_masks, num_class_choice, mix_log_vars = \
                self._build_prototype_or_random_mix_masks(
                    src_seg_lbl.unsqueeze(1),
                    src_mix_ratio,
                    'gt_mix',
                    class_scores_override=class_scores,
                    random_prob_override=class_score_random_prob,
                )
        if source_pixel_class_scores is not None:
            mix_log_vars.update({
                'gt_mix_tdef_source_pixel_reweight_active': 1.0,
                'gt_mix_tdef_source_pixel_weight_min': float(
                    src_pixel_weight.min().detach().item()),
                'gt_mix_tdef_source_pixel_weight_max': float(
                    src_pixel_weight.max().detach().item()),
                'gt_mix_tdef_source_pixel_weight_mean': float(
                    src_pixel_weight.mean().detach().item()),
            })

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mix_img[i], mix_seg_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((src_img[i], tgt_img[i])),
                target=torch.stack((src_seg_lbl[i], tgt_seg_lbl[i])),
            )
            _, mix_seg_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((src_pixel_weight[i], tgt_pixel_weight[i])),
            )

        return (
            torch.cat(mix_img),
            torch.cat(mix_seg_lbl),
            mix_seg_weight,
            mix_masks,
            num_class_choice,
            src_mix_ratio,
            mix_log_vars,
        )

    def _build_class_mix_batch(self, src_img, src_seg_lbl, tar_img, pseudo_label,
                               pseudo_weight, pseudo_conf, strong_parameters,
                               batch_size, dev, target_img_paths=None,
                               class_scores=None,
                               class_score_random_prob=None,
                               quota_classmix=False,
                               mask_routing_classmix=False,
                               source_pixel_class_scores=None):
        """Create ClassMix samples with optional prototype-guided masks."""
        mix_img, mix_seg_lbl = [None] * batch_size, [None] * batch_size
        target_mix_weight, mix_conf_log_vars = self._apply_mix_confidence_weight(
            pseudo_weight,
            pseudo_conf,
        )
        mix_seg_weight = target_mix_weight.clone()
        gt_pixel_weight = self._source_deficit_pixel_weight_map(
            src_seg_lbl,
            source_pixel_class_scores,
        )
        if gt_pixel_weight is None:
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)
        src_img_for_mix, weather_mix_log_vars = self._prepare_weather_aligned_source_mix(
            src_img,
            target_img_paths,
            batch_size,
            target_img=tar_img,
            pseudo_weight=pseudo_weight)

        src_mix_ratio = self.get_src_cls_mix_ratio()
        if mask_routing_classmix:
            mix_masks, num_class_choice, proto_mix_log_vars = \
                self._build_target_need_mask_routing_mix_masks(
                    src_seg_lbl.unsqueeze(1),
                    src_mix_ratio,
                    'class_mix',
                    class_scores,
                )
        elif quota_classmix:
            mix_masks, num_class_choice, proto_mix_log_vars = \
                self._build_target_deficit_quota_mix_masks(
                    src_seg_lbl.unsqueeze(1),
                    src_mix_ratio,
                    'class_mix',
                    class_scores,
                )
        else:
            mix_masks, num_class_choice, proto_mix_log_vars = \
                self._build_prototype_or_random_mix_masks(
                    src_seg_lbl.unsqueeze(1),
                    src_mix_ratio,
                    'class_mix',
                    class_scores_override=class_scores,
                    random_prob_override=class_score_random_prob,
                )
        if source_pixel_class_scores is not None:
            proto_mix_log_vars.update({
                'class_mix_tdef_source_pixel_reweight_active': 1.0,
                'class_mix_tdef_source_pixel_weight_min': float(
                    gt_pixel_weight.min().detach().item()),
                'class_mix_tdef_source_pixel_weight_max': float(
                    gt_pixel_weight.max().detach().item()),
                'class_mix_tdef_source_pixel_weight_mean': float(
                    gt_pixel_weight.mean().detach().item()),
            })

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
        mix_build_log_vars.update(proto_mix_log_vars)
        return (
            mix_img,
            mix_seg_lbl,
            mix_seg_weight,
            mix_masks,
            num_class_choice,
            src_mix_ratio,
            mix_build_log_vars,
        )

    def _target_patch_memory_active(self):
        return (
            self.target_patch_memory_mix_enabled
            and self.local_iter >= self.target_patch_memory_mix_begin_iter
            and self.target_patch_memory_bank is not None)

    def _target_need_target_mix_active(self):
        return (
            self.target_need_target_mix
            and self.local_iter >= self.target_need_source_mix_begin_iter)

    def _target_class_memory_active(self):
        return (
            self.target_class_memory_mix_enabled
            and self.local_iter >= self.target_class_memory_mix_begin_iter
            and self.target_class_memory_bank is not None)

    def _target_class_memory_aux_active(self):
        return (
            self.target_class_memory_aux_enabled
            and self.local_iter >= self.target_class_memory_aux_begin_iter
            and self.target_class_memory_bank is not None
            and self.target_class_memory_aux_weight > 0)

    def _target_class_memory_diagnostic_log_vars(self, result, class_scores):
        """Return scalar diagnostics for target class-memory selection.

        The aggregate values are useful in text logs, while the per-class
        values are mainly intended for TensorBoard / log parsing.
        """
        if not self.target_class_memory_diagnostics:
            return {}
        log_vars = {
            'target_class_memory_diag_active': 1.0,
            'target_class_memory_diag_selected_class_count': float(
                len(result.selected_classes)),
        }
        if result.selected_classes:
            for rank, class_id in enumerate(result.selected_classes[:5]):
                log_vars[
                    f'target_class_memory_selected_rank{rank}_class'
                ] = float(class_id)

        if not self.target_class_memory_diagnostic_per_class:
            return log_vars

        scores = None
        if class_scores is not None:
            scores = class_scores.detach().float().cpu().flatten()
        deficit_scores = self._source_assist_target_deficit_scores
        if deficit_scores is not None:
            deficit_scores = deficit_scores.detach().float().cpu().flatten()

        for class_id in range(self.num_classes):
            prefix = f'target_class_memory_cls_{class_id:02d}'
            log_vars[f'{prefix}_select_count'] = float(
                result.selected_class_counts.get(class_id, 0))
            log_vars[f'{prefix}_pixel_count'] = float(
                result.selected_class_pixels.get(class_id, 0))
            if scores is not None and class_id < int(scores.numel()):
                log_vars[f'{prefix}_score'] = float(scores[class_id].item())
            if (
                deficit_scores is not None
                and class_id < int(deficit_scores.numel())
            ):
                log_vars[f'{prefix}_deficit_score'] = float(
                    deficit_scores[class_id].item())
        return log_vars

    def _target_class_memory_aux_loss_log_vars(self, logits, labels, weights):
        """Return per-class loss diagnostics for the auxiliary memory branch."""
        if not self.target_class_memory_diagnostics:
            return {}
        stats = self._diagnostic_per_class_loss(logits, labels, weights)
        if stats is None:
            return {'target_mem_aux_per_class_loss_valid': 0.0}
        loss_sum = stats['sum'].detach().float().cpu()
        count_sum = stats['count'].detach().float().cpu()
        log_vars = {'target_mem_aux_per_class_loss_valid': 1.0}
        for class_id in range(self.num_classes):
            prefix = f'target_mem_aux_cls_{class_id:02d}'
            count = float(count_sum[class_id].item())
            loss = (
                float(loss_sum[class_id].item()) / max(count, 1e-6)
                if count > 0 else 0.0
            )
            log_vars[f'{prefix}_loss'] = loss
            log_vars[f'{prefix}_loss_count'] = count
        return log_vars

    def _build_target_class_memory_mix_batch(
        self,
        tgt_l_img,
        tgt_l_seg_lbl,
        tgt_u_img,
        pseudo_label,
        pseudo_weight,
        pseudo_conf,
        strong_parameters,
        class_scores,
        mask_only_weight=False,
    ):
        """Build target-only class memory mixed samples.

        Target-labeled class masks fill a small class-wise memory. High
        target-deficit classes are pasted into target-unlabeled samples and
        supervised with GT labels from the memory.
        """
        if class_scores is None:
            return None
        if self.target_class_memory_update_online:
            self.target_class_memory_bank.update_from_labeled_batch(
                tgt_l_img,
                tgt_l_seg_lbl,
                num_classes=self.num_classes,
                min_pixels=self.target_class_memory_min_pixels,
                max_area_ratio=self.target_class_memory_max_area_ratio,
                ignore_index=getattr(self, 'ignore_index', 255),
            )
        result = apply_target_class_memory_mix(
            tgt_u_img,
            pseudo_label,
            pseudo_weight,
            self.target_class_memory_bank,
            class_scores,
            num_classes=self.num_classes,
            max_classes=self.target_class_memory_max_classes,
            random_prob=self.target_class_memory_random_prob,
            min_score=self.target_class_memory_min_score,
            pseudo_conf=pseudo_conf,
            allowed_classes=self.target_class_memory_allowed_classes,
            blocked_classes=self.target_class_memory_blocked_classes,
            min_pseudo_conf=self.target_class_memory_min_pseudo_conf,
            context_paste=self.target_class_memory_context_paste,
            context_candidates=self.target_class_memory_context_candidates,
            context_y_jitter=self.target_class_memory_context_y_jitter,
            max_paste_area_ratio=(
                self.target_class_memory_max_paste_area_ratio),
            sample_strategy=self.target_class_memory_sample_strategy,
            mask_only_weight=mask_only_weight,
            ignore_index=getattr(self, 'ignore_index', 255),
        )
        if result.num_replaced <= 0:
            return None

        mix_img, mix_seg_lbl = strong_transform_wo_mix(
            strong_parameters,
            data=result.images,
            target=result.labels,
        )
        log_vars = dict(result.log_vars)
        log_vars.update({
            'target_class_memory_active': 1.0,
            'target_class_memory_bank_size': float(
                len(self.target_class_memory_bank)),
            'target_class_memory_capacity_per_class': float(
                self.target_class_memory_capacity_per_class),
            'target_class_memory_max_classes': float(
                self.target_class_memory_max_classes),
            'target_class_memory_score_min': float(
                class_scores.min().detach().item()),
            'target_class_memory_score_max': float(
                class_scores.max().detach().item()),
            'target_class_memory_score_mean': float(
                class_scores.mean().detach().item()),
            'target_class_memory_allowed_count': float(
                len(self.target_class_memory_allowed_classes)),
            'target_class_memory_blocked_count': float(
                len(self.target_class_memory_blocked_classes)),
            'target_class_memory_min_pseudo_conf': float(
                self.target_class_memory_min_pseudo_conf),
            'target_class_memory_offline_loaded': float(
                self.target_class_memory_offline_loaded),
            'target_class_memory_update_online': float(
                self.target_class_memory_update_online),
            'target_class_memory_context_paste': float(
                self.target_class_memory_context_paste),
            'target_class_memory_max_paste_area_ratio': float(
                self.target_class_memory_max_paste_area_ratio),
        })
        log_vars.update(self._target_class_memory_diagnostic_log_vars(
            result,
            class_scores,
        ))
        return (
            mix_img,
            mix_seg_lbl,
            result.weights,
            result.mix_masks,
            result.num_class_choice,
            result.mix_ratio,
            log_vars,
        )

    def _build_target_patch_memory_mix_batch(
        self,
        tgt_l_img,
        tgt_l_seg_lbl,
        tgt_u_img,
        pseudo_label,
        pseudo_weight,
        pseudo_conf,
        strong_parameters,
    ):
        """Build target-only patch memory mixed samples.

        The target-unlabeled image is the base. Low-confidence patches are
        replaced by semantically matched target-labeled memory patches. The
        returned tuple matches `_build_class_mix_batch`.
        """
        result = apply_target_patch_memory_mix(
            tgt_l_img,
            tgt_l_seg_lbl,
            tgt_u_img,
            pseudo_label,
            pseudo_weight,
            pseudo_conf,
            self.target_patch_memory_bank,
            num_classes=self.num_classes,
            grid_size=self.target_patch_memory_mix_grid_size,
            replace_ratio=self.target_patch_memory_mix_replace_ratio,
            min_labeled_class_ratio=self.target_patch_memory_mix_min_class_ratio,
        )
        if result.num_replaced <= 0:
            return None

        mix_img, mix_seg_lbl = strong_transform_wo_mix(
            strong_parameters,
            data=result.images,
            target=result.labels,
        )
        log_vars = dict(result.log_vars)
        log_vars.update({
            'target_patch_memory_active': 1.0,
            'target_patch_memory_bank_size': float(len(self.target_patch_memory_bank)),
            'target_patch_memory_grid_size': float(self.target_patch_memory_mix_grid_size),
            'target_patch_memory_replace_ratio_cfg': float(
                self.target_patch_memory_mix_replace_ratio),
        })
        return (
            mix_img,
            mix_seg_lbl,
            result.weights,
            result.mix_masks,
            result.num_class_choice,
            result.mix_ratio,
            log_vars,
        )

    @staticmethod
    def _detach_feature_payload(features):
        if torch.is_tensor(features):
            return features.detach()
        if isinstance(features, dict):
            return {
                key: SSDADACS._detach_feature_payload(value)
                for key, value in features.items()
            }
        if isinstance(features, tuple):
            return tuple(
                SSDADACS._detach_feature_payload(value)
                for value in features
            )
        if isinstance(features, list):
            return [
                SSDADACS._detach_feature_payload(value)
                for value in features
            ]
        return features

    @staticmethod
    def _slice_paths(paths, indices):
        if paths is None:
            return None
        if isinstance(paths, (list, tuple)):
            return [paths[i] for i in indices]
        return paths

    def _build_source_target_mix_batch(
        self,
        src_img,
        src_seg_lbl,
        tgt_l_img,
        tgt_l_seg_lbl,
        tgt_u_img,
        pseudo_label,
        pseudo_weight,
        pseudo_conf,
        strong_parameters,
        batch_size,
        dev,
        target_img_paths=None,
        source_class_scores=None,
        source_class_score_random_prob=None,
        source_quota_classmix=False,
        source_mask_routing_classmix=False,
        source_pixel_class_scores=None,
    ):
        """Mix source images with both labeled and unlabeled target images."""
        if self.source_mix_target_mode == 'unlabeled_only':
            labeled_indices = []
            unlabeled_indices = list(range(batch_size))
        elif self.source_mix_target_mode == 'labeled_only':
            labeled_indices = list(range(batch_size))
            unlabeled_indices = []
        elif batch_size == 1:
            labeled_indices = [0] if self.local_iter % 2 == 0 else []
            unlabeled_indices = [] if labeled_indices else [0]
        else:
            labeled_count = int(round(batch_size * self.source_labeled_mix_ratio))
            labeled_count = min(max(1, labeled_count), batch_size - 1)
            labeled_indices = list(range(labeled_count))
            unlabeled_indices = list(range(labeled_count, batch_size))

        mix_imgs = []
        mix_lbls = []
        mix_weights = []
        mix_masks = []
        num_choices = []
        log_vars = {}
        src_mix_ratio = self.get_src_cls_mix_ratio()

        if labeled_indices:
            idx = torch.as_tensor(labeled_indices, device=dev, dtype=torch.long)
            labeled_mix = self._build_gt_mix_batch(
                src_img.index_select(0, idx),
                src_seg_lbl.index_select(0, idx),
                tgt_l_img.index_select(0, idx),
                tgt_l_seg_lbl.index_select(0, idx),
                dict(strong_parameters),
                len(labeled_indices),
                dev,
                class_scores=source_class_scores,
                class_score_random_prob=source_class_score_random_prob,
                quota_classmix=source_quota_classmix,
                mask_routing_classmix=source_mask_routing_classmix,
                source_pixel_class_scores=source_pixel_class_scores,
            )
            mix_imgs.append(labeled_mix[0])
            mix_lbls.append(labeled_mix[1])
            mix_weights.append(labeled_mix[2])
            mix_masks.extend(labeled_mix[3])
            num_choices.extend(labeled_mix[4])
            src_mix_ratio = labeled_mix[5]
            log_vars.update(labeled_mix[6])
        selected_src_labels = []
        if labeled_indices:
            idx = torch.as_tensor(labeled_indices, device=dev, dtype=torch.long)
            selected_src_labels.append(src_seg_lbl.index_select(0, idx))

        if unlabeled_indices:
            idx = torch.as_tensor(unlabeled_indices, device=dev, dtype=torch.long)
            unlabeled_mix = self._build_class_mix_batch(
                src_img.index_select(0, idx),
                src_seg_lbl.index_select(0, idx),
                tgt_u_img.index_select(0, idx),
                pseudo_label.index_select(0, idx),
                pseudo_weight.index_select(0, idx),
                pseudo_conf.index_select(0, idx) if pseudo_conf is not None else None,
                dict(strong_parameters),
                len(unlabeled_indices),
                dev,
                self._slice_paths(target_img_paths, unlabeled_indices),
                class_scores=source_class_scores,
                class_score_random_prob=source_class_score_random_prob,
                quota_classmix=source_quota_classmix,
                mask_routing_classmix=source_mask_routing_classmix,
                source_pixel_class_scores=source_pixel_class_scores,
            )
            mix_imgs.append(unlabeled_mix[0])
            mix_lbls.append(unlabeled_mix[1])
            mix_weights.append(unlabeled_mix[2])
            mix_masks.extend(unlabeled_mix[3])
            num_choices.extend(unlabeled_mix[4])
            src_mix_ratio = unlabeled_mix[5]
            log_vars.update(unlabeled_mix[6])
            selected_src_labels.append(src_seg_lbl.index_select(0, idx))

        if selected_src_labels:
            selected_src_labels = torch.cat(selected_src_labels, dim=0)
            self._feature_proto_update_source_mix_counts(
                selected_src_labels,
                mix_masks,
            )
            self._source_assist_update_source_mix_counts(
                selected_src_labels,
                mix_masks,
            )

        log_vars.update({
            'ssda_source_mix_labeled_ratio': float(len(labeled_indices)) / max(1, batch_size),
            'ssda_source_mix_unlabeled_ratio': float(len(unlabeled_indices)) / max(1, batch_size),
        })
        return (
            torch.cat(mix_imgs, dim=0),
            torch.cat(mix_lbls, dim=0),
            torch.cat(mix_weights, dim=0),
            mix_masks,
            num_choices,
            src_mix_ratio,
            log_vars,
        )

    def _forward_weighted_mix_loss(self, mix_img, mix_seg_lbl, mix_seg_weight,
                                   seg_debug, debug_key, branch_prefix,
                                   branch_weight=1.0):
        """Run a mixed branch with the unsupervised loss context."""
        mix_pred_results = self.get_model().forward_train(
            (mix_img, mix_seg_lbl.squeeze(1)),
            seg_weight=mix_seg_weight,
            return_feat=False,
            loss_key='unsup',
        )
        seg_debug[debug_key] = self.get_model().decode_head.debug_output
        mix_seg_pred = mix_pred_results.pop('seg_logits', None)
        mix_seg_loss, mix_log_vars = parse_losses(mix_pred_results)

        dacs_mix_weight = float(self._get_mix_loss_weight())
        total_weight = dacs_mix_weight * float(branch_weight)
        weighted_mix_loss = mix_seg_loss * total_weight
        if total_weight > 0:
            weighted_mix_loss.backward()

        prefixed = add_prefix(mix_log_vars, branch_prefix)
        prefixed[f'{branch_prefix}_loss_weight'] = float(branch_weight)
        prefixed[f'{branch_prefix}_dacs_mix_weight'] = dacs_mix_weight
        prefixed[f'{branch_prefix}_weighted_loss'] = float(
            weighted_mix_loss.detach().item())
        return {
            'seg_pred': mix_seg_pred,
            'log_vars': prefixed,
            'loss_value': float(weighted_mix_loss.detach().item()),
            'raw_loss_value': float(mix_seg_loss.detach().item()),
        }

    def forward_train_step(self, data_batch, valid_pseudo_mask=None):
        """Run one SSDA iteration and return scalar log values."""
        offline_teacher_confidence = None
        if len(data_batch) == 8:
            (
                src_img,
                src_seg_lbl,
                tgt_l_img,
                tgt_l_seg_lbl,
                tgt_u_img,
                tgt_u_seg_lbl,
                target_img_paths,
                offline_teacher_confidence,
            ) = data_batch
        elif len(data_batch) == 7:
            (
                src_img,
                src_seg_lbl,
                tgt_l_img,
                tgt_l_seg_lbl,
                tgt_u_img,
                tgt_u_seg_lbl,
                target_img_paths,
            ) = data_batch
        else:
            (
                src_img,
                src_seg_lbl,
                tgt_l_img,
                tgt_l_seg_lbl,
                tgt_u_img,
                tgt_u_seg_lbl,
            ) = data_batch
            target_img_paths = None

        log_vars = {}
        batch_size = src_img.shape[0]
        dev = src_img.device
        self._grad_conflict_source_grads = None
        self._adapter_grad_conflict_source_grads = None
        self._feature_proto_reset_diagnostic_iteration_state()
        self._source_assist_reset_diagnostic_iteration_state()

        self._update_teacher_and_mic_state()
        self.update_debug_state()
        seg_debug = {}
        semi_enabled = not self.source_only and self.local_iter >= self.semi_begin_iter
        means, stds, strong_parameters = self._build_strong_parameters(batch_size, dev)

        source_sup_weight = self._branch_weight('source_sup')
        target_sup_weight = self._branch_weight('target_sup')
        target_anchor_weight = self._target_anchor_replay_branch_weight()
        tri_prototype_weight = self._tri_prototype_branch_weight()
        feature_tri_prototype_weight = self._feature_tri_prototype_branch_weight()
        gradient_alignment_active = (
            semi_enabled and self._gradient_aligned_source_active())
        conflict_route_update_due = (
            semi_enabled and self._conflict_route_update_due())
        delay_labeled_backward = (
            tri_prototype_weight > 0
            or feature_tri_prototype_weight > 0
            or gradient_alignment_active
            or conflict_route_update_due)
        source_mix_weight, target_mix_weight, target_mix_share, mix_total_weight = \
            self._mix_branch_weights()
        (
            target_sup_weight,
            source_mix_weight,
            target_mix_weight,
            target_mix_share,
            mix_total_weight,
            target_sup_redistribution_log_vars,
        ) = self._redistribute_target_supervision(
            target_sup_weight,
            source_mix_weight,
            target_mix_weight,
            target_mix_share,
            mix_total_weight,
        )
        log_vars.update({
            'ssda_source_sup_weight': source_sup_weight,
            'ssda_target_sup_weight': target_sup_weight,
            'ssda_target_anchor_weight': target_anchor_weight,
            'ssda_tri_prototype_weight': tri_prototype_weight,
            'ssda_feature_tri_prototype_weight': feature_tri_prototype_weight,
            'ssda_gradient_alignment_active': float(gradient_alignment_active),
            'ssda_conflict_route_update_due': float(conflict_route_update_due),
        })
        log_vars.update(self._mix_branch_log_vars(
            source_mix_weight,
            target_mix_weight,
            target_mix_share,
            mix_total_weight,
        ))
        log_vars.update(target_sup_redistribution_log_vars)

        source_sup_filter_weight, source_sup_filter_log_vars = \
            self._target_guided_source_filter_weight(
                src_seg_lbl,
                tgt_l_seg_lbl,
                prefix='source_sup',
            )
        if source_sup_filter_log_vars:
            log_vars.update(source_sup_filter_log_vars)
            log_vars['ssda_source_sup_effective_weight'] = (
                source_sup_weight * source_sup_filter_weight)
        source_sup_weight = source_sup_weight * source_sup_filter_weight

        feature_proto_return_feat = self._feature_proto_active()
        source_feature_weight_map, source_feature_weight_log_vars = \
            self._feature_proto_source_weight_map(src_seg_lbl)
        log_vars.update(source_feature_weight_log_vars)

        total_loss_value = 0.0
        source_state = self._forward_labeled_loss(
            src_img,
            src_seg_lbl,
            seg_debug,
            'Source',
            'src',
            source_sup_weight,
            backward=not delay_labeled_backward,
            seg_weight=source_feature_weight_map,
            return_feat=feature_proto_return_feat,
        )
        log_vars.update(source_state['log_vars'])
        self._feature_proto_record_per_class_loss(
            'src',
            source_state.get('seg_pred'),
            src_seg_lbl,
            source_feature_weight_map,
        )
        self._source_assist_record_per_class_loss(
            'src',
            source_state.get('seg_pred'),
            src_seg_lbl,
            source_feature_weight_map,
        )
        total_loss_value += source_state['loss_value']

        target_state = self._forward_labeled_loss(
            tgt_l_img,
            tgt_l_seg_lbl,
            seg_debug,
            'Target Labeled',
            'tgt',
            target_sup_weight,
            backward=not delay_labeled_backward,
            return_feat=feature_proto_return_feat,
            detach_features=feature_tri_prototype_weight <= 0,
            loss_key='target_labeled',
        )
        log_vars.update(target_state['log_vars'])
        self._feature_proto_record_per_class_loss(
            'tgt',
            target_state.get('seg_pred'),
            tgt_l_seg_lbl,
        )
        self._source_assist_record_per_class_loss(
            'tgt',
            target_state.get('seg_pred'),
            tgt_l_seg_lbl,
        )
        log_vars.update(self._target_need_update_loss_feedback(
            target_state.get('seg_pred'),
            tgt_l_seg_lbl,
        ))
        log_vars.update(self._update_target_labeled_reliability(
            target_state.get('seg_pred'),
            tgt_l_seg_lbl,
        ))
        total_loss_value += target_state['loss_value']

        if conflict_route_update_due:
            log_vars.update(self._update_conflict_route_gradients(
                source_state.get('seg_pred'),
                src_seg_lbl,
                target_state.get('seg_pred'),
                tgt_l_seg_lbl,
            ))

        gradient_source_scale = 1.0
        if gradient_alignment_active:
            gradient_source_scale, gradient_log_vars = (
                self._gradient_aligned_source_scale(
                    source_state.get('loss_tensor'),
                    target_state.get('loss_tensor'),
                ))
            log_vars.update(gradient_log_vars)
            total_loss_value += (
                source_state['loss_value'] * (gradient_source_scale - 1.0))
            log_vars['ssda_source_sup_gradalign_effective_weight'] = (
                source_sup_weight * gradient_source_scale)
            if self.gradient_aligned_source_apply_to_mix:
                source_mix_weight *= gradient_source_scale
                log_vars['ssda_source_mix_gradalign_effective_weight'] = (
                    source_mix_weight)

        target_anchor_raw = 0.0
        target_anchor_weighted = 0.0
        if target_anchor_weight > 0:
            target_anchor_state = self._forward_target_anchor_replay_branch(
                tgt_l_img,
                tgt_l_seg_lbl,
                means,
                stds,
                seg_debug,
                target_anchor_weight,
            )
            log_vars.update(target_anchor_state['log_vars'])
            total_loss_value += target_anchor_state['loss_value']
            target_anchor_raw = target_anchor_state['raw_loss_value']
            target_anchor_weighted = target_anchor_state['loss_value']

        log_vars.update(self._feature_proto_update_labeled_state(
            source_state.get('features'),
            src_seg_lbl,
            target_state.get('features'),
            tgt_l_seg_lbl,
        ))

        total_mix_raw = 0.0
        total_mix_weighted = 0.0
        pseudo_label, pseudo_weight, pseudo_mask, pseudo_conf = None, None, None, None
        dep_tar = None
        source_mix = None
        target_mix = None
        source_mix_pred = None
        target_mix_pred = None
        target_consistency_state = None
        unlabeled_branch_raw = 0.0
        unlabeled_branch_weighted = 0.0
        tri_prototype_raw = 0.0
        tri_prototype_weighted = 0.0
        feature_tri_prototype_raw = 0.0
        feature_tri_prototype_weighted = 0.0

        if semi_enabled:
            pseudo_label, pseudo_weight, pseudo_mask, pseudo_conf, dep_tar, vplf_log_vars = \
                self._generate_target_pseudo_state(
                    tgt_u_img,
                    valid_pseudo_mask,
                    seg_debug,
                    offline_teacher_label=tgt_u_seg_lbl,
                    offline_teacher_confidence=offline_teacher_confidence,
                )
            log_vars.update(vplf_log_vars)
            target_consistency_state, consistency_log_vars = (
                self._target_flip_consistency_state(
                    tgt_u_img,
                    pseudo_label,
                    pseudo_weight,
                    pseudo_conf,
                ))
            log_vars.update(consistency_log_vars)
            self._source_assist_record_target_state(
                tgt_l_seg_lbl,
                pseudo_label,
                pseudo_weight,
                pseudo_conf,
            )
            proto_classmix_log_vars = self._update_prototype_classmix_target_state(
                tgt_l_seg_lbl,
                pseudo_label,
                pseudo_weight,
                pseudo_conf,
            )
            log_vars.update(proto_classmix_log_vars)
            log_vars.update(self._feature_proto_update_unlabeled_state(
                tgt_u_img,
                pseudo_label,
                pseudo_weight,
                pseudo_conf,
            ))
            adaptive_mix_log_vars = self._update_adaptive_mix_state(
                src_seg_lbl,
                tgt_l_seg_lbl,
                pseudo_label,
                pseudo_weight,
                pseudo_mask,
                pseudo_conf,
                consistency_reliability=(
                    None
                    if target_consistency_state is None
                    else target_consistency_state[
                        'class_balanced_reliability']
                ),
            )
            if adaptive_mix_log_vars:
                log_vars.update(adaptive_mix_log_vars)
                source_mix_weight, target_mix_weight, target_mix_share, mix_total_weight = \
                    self._mix_branch_weights()
                (
                    target_sup_weight,
                    source_mix_weight,
                    target_mix_weight,
                    target_mix_share,
                    mix_total_weight,
                    target_sup_redistribution_log_vars,
                ) = self._redistribute_target_supervision(
                    target_sup_weight,
                    source_mix_weight,
                    target_mix_weight,
                    target_mix_share,
                    mix_total_weight,
                )
                log_vars.update(self._mix_branch_log_vars(
                    source_mix_weight,
                    target_mix_weight,
                    target_mix_share,
                    mix_total_weight,
                ))
                log_vars.update({
                    'ssda_target_sup_weight': target_sup_weight,
                })
                log_vars.update(target_sup_redistribution_log_vars)

            if delay_labeled_backward:
                tri_state = self._forward_tri_prototype_loss(
                    source_state['seg_pred'],
                    src_seg_lbl,
                    target_state['seg_pred'],
                    tgt_l_seg_lbl,
                    pseudo_label,
                    pseudo_weight,
                    pseudo_conf,
                    branch_weight=tri_prototype_weight,
                )
                log_vars.update(tri_state['log_vars'])
                feature_tri_state = self._forward_feature_tri_prototype_loss(
                    target_state.get('features'),
                    tgt_l_seg_lbl,
                    tgt_u_img=tgt_u_img,
                    pseudo_label=pseudo_label,
                    pseudo_weight=pseudo_weight,
                    pseudo_conf=pseudo_conf,
                    branch_weight=feature_tri_prototype_weight,
                )
                log_vars.update(feature_tri_state['log_vars'])
                supervised_loss = (
                    source_state['loss_tensor'] * gradient_source_scale
                    + target_state['loss_tensor'])
                if tri_state['loss_tensor'] is not None:
                    supervised_loss = supervised_loss + tri_state['loss_tensor']
                    total_loss_value += tri_state['loss_value']
                    tri_prototype_raw = tri_state['raw_loss_value']
                    tri_prototype_weighted = tri_state['loss_value']
                if feature_tri_state['loss_tensor'] is not None:
                    supervised_loss = (
                        supervised_loss + feature_tri_state['loss_tensor'])
                    total_loss_value += feature_tri_state['loss_value']
                    feature_tri_prototype_raw = feature_tri_state['raw_loss_value']
                    feature_tri_prototype_weighted = feature_tri_state['loss_value']
                supervised_loss.backward()

            target_mix_pseudo_weight = pseudo_weight
            target_mix_conf_log_vars = {}
            if self.enable_target_semi_mix and target_mix_weight > 0:
                target_mix_pseudo_weight, target_mix_conf_log_vars = \
                    self._confidence_aware_target_mix_weight(
                        pseudo_weight,
                        pseudo_conf,
                    )
                log_vars.update(target_mix_conf_log_vars)
                (
                    target_mix_pseudo_weight,
                    target_mix_consistency_log_vars,
                ) = self._consistency_aware_target_mix_weight(
                    target_mix_pseudo_weight,
                    target_consistency_state,
                )
                log_vars.update(target_mix_consistency_log_vars)
                target_mix_pseudo_weight, target_mix_tlrc_log_vars = \
                    self._target_labeled_reliability_weight(
                        pseudo_label,
                        target_mix_pseudo_weight,
                        prefix='target_mix',
                    )
                log_vars.update(target_mix_tlrc_log_vars)

            source_mix_pseudo_weight = pseudo_weight
            if (
                self.enable_source_target_mix
                and source_mix_weight > 0
                and not self.target_labeled_reliability_target_mix_only
            ):
                source_mix_pseudo_weight, source_mix_tlrc_log_vars = \
                    self._target_labeled_reliability_weight(
                        pseudo_label,
                        source_mix_pseudo_weight,
                        prefix='source_mix',
                    )
                log_vars.update(source_mix_tlrc_log_vars)

            source_mix_filter_weight, source_mix_filter_log_vars = \
                self._target_guided_source_filter_weight(
                    src_seg_lbl,
                    tgt_l_seg_lbl,
                    pseudo_label,
                    pseudo_weight,
                    pseudo_conf,
                    prefix='source_mix',
                    update_labeled_prototypes=False,
                    update_unlabeled_prototype=True,
                )
            if source_mix_filter_log_vars:
                log_vars.update(source_mix_filter_log_vars)
                log_vars['ssda_source_mix_effective_weight'] = (
                    source_mix_weight * source_mix_filter_weight)
            source_mix_weight = source_mix_weight * source_mix_filter_weight

            source_mix_class_scores, source_mix_featproto_log_vars = \
                self._feature_proto_source_mix_scores(dev)
            log_vars.update(source_mix_featproto_log_vars)
            source_mix_score_random_prob = None
            source_pixel_class_scores = None
            target_mix_class_scores = None
            target_mix_score_random_prob = None
            target_quota_classmix = False

            if (
                self.prototype_incompatibility_veto
                and self.local_iter
                >= self.prototype_incompatibility_veto_begin_iter
            ):
                veto_scores = self._feature_proto_source_scores(dev)
                if veto_scores is not None:
                    source_mix_class_scores = veto_scores
                    log_vars['ssda_proto_veto_active'] = 1.0
                else:
                    log_vars['ssda_proto_veto_active'] = 0.0

            tnsa_class_scores, tnsa_log_vars = self._target_need_source_mix_scores(
                tgt_l_seg_lbl,
                pseudo_label,
                pseudo_weight,
                pseudo_conf,
                dev,
            )
            if tnsa_log_vars:
                log_vars.update(tnsa_log_vars)
            if tnsa_class_scores is not None:
                if self.target_deficit_source_pixel_reweight:
                    source_pixel_class_scores = tnsa_class_scores
                    log_vars['ssda_tdef_source_pixel_reweight_active'] = 1.0
                if self.target_need_source_mix_apply_to_classmix:
                    if source_mix_class_scores is None:
                        source_mix_class_scores = tnsa_class_scores
                    else:
                        source_mix_class_scores = (
                            source_mix_class_scores.to(device=dev)
                            * tnsa_class_scores.to(device=dev)
                        )
                        source_mix_class_scores = (
                            source_mix_class_scores
                            / source_mix_class_scores.mean().clamp_min(1e-6)
                        )
                        log_vars['ssda_tnsa_mix_combined_with_featproto'] = 1.0
                    source_mix_score_random_prob = (
                        self.target_need_source_mix_random_prob)
                else:
                    log_vars['ssda_tnsa_mix_apply_to_classmix'] = 0.0
                if self._target_need_target_mix_active():
                    target_mix_class_scores = tnsa_class_scores
                    target_mix_score_random_prob = (
                        self.target_need_target_mix_random_prob)
                    target_quota_classmix = self.target_need_target_mix_quota
                    log_vars['ssda_tdef_target_mix_active'] = 1.0
                    log_vars['ssda_tdef_target_mix_random_prob'] = float(
                        self.target_need_target_mix_random_prob)
                    log_vars['ssda_tdef_target_mix_quota'] = float(
                        self.target_need_target_mix_quota)

            route_class_scores, route_log_vars = \
                self._class_conditional_source_route_scores(dev)
            if route_log_vars:
                log_vars.update(route_log_vars)
            self._source_assist_set_route_scores(route_class_scores)
            if route_class_scores is not None:
                if source_mix_class_scores is None:
                    source_mix_class_scores = route_class_scores
                else:
                    source_mix_class_scores = (
                        source_mix_class_scores.to(device=dev)
                        * route_class_scores.to(device=dev)
                    )
                    source_mix_class_scores = (
                        source_mix_class_scores
                        / source_mix_class_scores.mean().clamp_min(1e-6)
                    )
                    log_vars['ssda_class_route_combined_with_existing'] = 1.0
                source_mix_score_random_prob = (
                    self.self_calibrated_class_route_random_prob
                    if self.self_calibrated_class_routing
                    else self.class_conditional_source_route_random_prob)

            self._source_assist_set_source_mix_scores(source_mix_class_scores)
            source_mask_routing_classmix = (
                self.target_need_mask_routing_v2
                and source_mix_class_scores is not None)
            source_quota_classmix = (
                self.target_deficit_quota_source_mix
                and source_mix_class_scores is not None
                and not source_mask_routing_classmix)
            if source_mask_routing_classmix:
                log_vars['ssda_tnmr_v2_source_mix_active'] = 1.0
            if source_quota_classmix:
                log_vars['ssda_tdef_quota_source_mix_active'] = 1.0

            if self.enable_source_target_mix and source_mix_weight > 0:
                source_mix = self._build_source_target_mix_batch(
                    src_img,
                    src_seg_lbl,
                    tgt_l_img,
                    tgt_l_seg_lbl,
                    tgt_u_img,
                    pseudo_label,
                    source_mix_pseudo_weight,
                    pseudo_conf,
                    strong_parameters,
                    batch_size,
                    dev,
                    target_img_paths,
                    source_class_scores=source_mix_class_scores,
                    source_class_score_random_prob=source_mix_score_random_prob,
                    source_quota_classmix=source_quota_classmix,
                    source_mask_routing_classmix=source_mask_routing_classmix,
                    source_pixel_class_scores=source_pixel_class_scores,
                )
                (
                    source_mix_img,
                    source_mix_lbl,
                    source_mix_weight_map,
                    source_mix_masks,
                    source_num_choice,
                    source_src_mix_ratio,
                    source_mix_log_vars,
                ) = source_mix
                log_vars.update(source_mix_log_vars)
                log_vars['src_mix_ratio'] = source_src_mix_ratio
                source_mix_state = self._forward_weighted_mix_loss(
                    source_mix_img,
                    source_mix_lbl,
                    source_mix_weight_map,
                    seg_debug,
                    'Source Mix',
                    'src_mix',
                    source_mix_weight,
                )
                source_mix_pred = source_mix_state['seg_pred']
                self._feature_proto_record_per_class_loss(
                    'src_mix',
                    source_mix_state.get('seg_pred'),
                    source_mix_lbl,
                    source_mix_weight_map,
                )
                self._source_assist_record_per_class_loss(
                    'src_mix',
                    source_mix_state.get('seg_pred'),
                    source_mix_lbl,
                    source_mix_weight_map,
                )
                log_vars.update(source_mix_state['log_vars'])
                total_loss_value += source_mix_state['loss_value']
                total_mix_raw += source_mix_state['raw_loss_value']
                total_mix_weighted += source_mix_state['loss_value']

            if (
                self.enable_source_labeled_aux_mix
                and self.source_labeled_aux_mix_weight > 0
            ):
                source_labeled_mix = self._build_gt_mix_batch(
                    src_img,
                    src_seg_lbl,
                    tgt_l_img,
                    tgt_l_seg_lbl,
                    dict(strong_parameters),
                    batch_size,
                    dev,
                )
                log_vars.update(add_prefix(
                    source_labeled_mix[6],
                    'src_labeled_mix_build',
                ))
                log_vars['src_labeled_mix_ratio'] = source_labeled_mix[5]
                source_labeled_mix_state = self._forward_weighted_mix_loss(
                    source_labeled_mix[0],
                    source_labeled_mix[1],
                    source_labeled_mix[2],
                    seg_debug,
                    'Source Labeled Mix',
                    'src_labeled_mix',
                    self.source_labeled_aux_mix_weight,
                )
                log_vars.update(source_labeled_mix_state['log_vars'])
                total_loss_value += source_labeled_mix_state['loss_value']
                total_mix_raw += source_labeled_mix_state['raw_loss_value']
                total_mix_weighted += source_labeled_mix_state['loss_value']

            if self.enable_target_semi_mix and target_mix_weight > 0:
                if (
                    self._target_class_memory_active()
                    and target_mix_class_scores is not None
                ):
                    target_mix = self._build_target_class_memory_mix_batch(
                        tgt_l_img,
                        tgt_l_seg_lbl,
                        tgt_u_img,
                        pseudo_label,
                        target_mix_pseudo_weight,
                        pseudo_conf,
                        strong_parameters,
                        target_mix_class_scores,
                    )
                if target_mix is None and self._target_patch_memory_active():
                    target_mix = self._build_target_patch_memory_mix_batch(
                        tgt_l_img,
                        tgt_l_seg_lbl,
                        tgt_u_img,
                        pseudo_label,
                        target_mix_pseudo_weight,
                        pseudo_conf,
                        strong_parameters,
                    )
                if target_mix is None:
                    target_mix = self._build_class_mix_batch(
                        tgt_l_img,
                        tgt_l_seg_lbl,
                        tgt_u_img,
                        pseudo_label,
                        target_mix_pseudo_weight,
                        pseudo_conf,
                        strong_parameters,
                        batch_size,
                        dev,
                        target_img_paths,
                        class_scores=target_mix_class_scores,
                        class_score_random_prob=target_mix_score_random_prob,
                        quota_classmix=target_quota_classmix,
                    )
                    self._feature_proto_update_target_mix_counts(
                        tgt_l_seg_lbl,
                        target_mix[3],
                    )
                (
                    target_mix_img,
                    target_mix_lbl,
                    target_mix_weight_map,
                    target_mix_masks,
                    target_num_choice,
                    target_src_mix_ratio,
                    target_mix_log_vars,
                ) = target_mix
                log_vars.update(add_prefix(target_mix_log_vars, 'target_mix_build'))
                log_vars['tgt_mix_ratio'] = target_src_mix_ratio
                target_mix_state = self._forward_weighted_mix_loss(
                    target_mix_img,
                    target_mix_lbl,
                    target_mix_weight_map,
                    seg_debug,
                    'Target Mix',
                    'tgt_mix',
                    target_mix_weight,
                )
                target_mix_pred = target_mix_state['seg_pred']
                self._feature_proto_record_per_class_loss(
                    'tgt_mix',
                    target_mix_state.get('seg_pred'),
                    target_mix_lbl,
                    target_mix_weight_map,
                )
                self._source_assist_record_per_class_loss(
                    'tgt_mix',
                    target_mix_state.get('seg_pred'),
                    target_mix_lbl,
                    target_mix_weight_map,
                )
                log_vars.update(add_prefix(
                    self._target_need_update_loss_feedback(
                        target_mix_state.get('seg_pred'),
                        target_mix_lbl,
                        target_mix_weight_map,
                    ),
                    'target_mix',
                ))
                log_vars.update(target_mix_state['log_vars'])
                total_loss_value += target_mix_state['loss_value']
                total_mix_raw += target_mix_state['raw_loss_value']
                total_mix_weighted += target_mix_state['loss_value']

                if (
                    self._target_class_memory_aux_active()
                    and tnsa_class_scores is not None
                ):
                    target_memory_aux = self._build_target_class_memory_mix_batch(
                        tgt_l_img,
                        tgt_l_seg_lbl,
                        tgt_u_img,
                        pseudo_label,
                        target_mix_pseudo_weight,
                        pseudo_conf,
                        strong_parameters,
                        tnsa_class_scores,
                        mask_only_weight=self.target_class_memory_aux_mask_only,
                    )
                    if target_memory_aux is not None:
                        (
                            target_mem_img,
                            target_mem_lbl,
                            target_mem_weight_map,
                            target_mem_masks,
                            target_mem_num_choice,
                            target_mem_mix_ratio,
                            target_mem_log_vars,
                        ) = target_memory_aux
                        log_vars.update(add_prefix(
                            target_mem_log_vars,
                            'target_mem_aux_build',
                        ))
                        log_vars['target_mem_aux_mix_ratio'] = target_mem_mix_ratio
                        target_mem_aux_weight = (
                            target_mix_weight
                            * self.target_class_memory_aux_weight)
                        log_vars['target_mem_aux_relative_weight'] = float(
                            self.target_class_memory_aux_weight)
                        log_vars['target_mem_aux_effective_weight'] = float(
                            target_mem_aux_weight)
                        target_mem_state = self._forward_weighted_mix_loss(
                            target_mem_img,
                            target_mem_lbl,
                            target_mem_weight_map,
                            seg_debug,
                            'Target Memory Aux',
                            'tgt_mem_aux',
                            target_mem_aux_weight,
                        )
                        log_vars['target_mem_aux_loss'] = float(
                            target_mem_state['raw_loss_value'])
                        log_vars['target_mem_aux_weighted_loss'] = float(
                            target_mem_state['loss_value'])
                        self._feature_proto_record_per_class_loss(
                            'tgt_mem_aux',
                            target_mem_state.get('seg_pred'),
                            target_mem_lbl,
                            target_mem_weight_map,
                        )
                        self._source_assist_record_per_class_loss(
                            'tgt_mem_aux',
                            target_mem_state.get('seg_pred'),
                            target_mem_lbl,
                            target_mem_weight_map,
                        )
                        log_vars.update(
                            self._target_class_memory_aux_loss_log_vars(
                                target_mem_state.get('seg_pred'),
                                target_mem_lbl,
                                target_mem_weight_map,
                            ))
                        log_vars.update(target_mem_state['log_vars'])
                        total_loss_value += target_mem_state['loss_value']
                        total_mix_raw += target_mem_state['raw_loss_value']
                        total_mix_weighted += target_mem_state['loss_value']

            unlabeled_branch_weight = self._unlabeled_consistency_branch_weight()
            if unlabeled_branch_weight > 0:
                unlabeled_state = self._forward_unlabeled_consistency_branch(
                    tgt_u_img,
                    pseudo_label,
                    pseudo_weight,
                    pseudo_conf,
                    means,
                    stds,
                    batch_size,
                    dev,
                    seg_debug,
                    unlabeled_branch_weight,
                )
                log_vars.update(unlabeled_state['log_vars'])
                total_loss_value += unlabeled_state['loss_value']
                unlabeled_branch_raw = unlabeled_state['raw_loss_value']
                unlabeled_branch_weighted = unlabeled_state['loss_value']
            elif self.unlabeled_consistency_enabled:
                log_vars['ssda_unlabeled_branch_weight'] = 0.0
        elif delay_labeled_backward:
            tri_state = self._forward_tri_prototype_loss(
                source_state['seg_pred'],
                src_seg_lbl,
                target_state['seg_pred'],
                tgt_l_seg_lbl,
                branch_weight=tri_prototype_weight,
            )
            log_vars.update(tri_state['log_vars'])
            feature_tri_state = self._forward_feature_tri_prototype_loss(
                target_state.get('features'),
                tgt_l_seg_lbl,
                branch_weight=feature_tri_prototype_weight,
            )
            log_vars.update(feature_tri_state['log_vars'])
            supervised_loss = (
                source_state['loss_tensor']
                + target_state['loss_tensor'])
            if tri_state['loss_tensor'] is not None:
                supervised_loss = supervised_loss + tri_state['loss_tensor']
                total_loss_value += tri_state['loss_value']
                tri_prototype_raw = tri_state['raw_loss_value']
                tri_prototype_weighted = tri_state['loss_value']
            if feature_tri_state['loss_tensor'] is not None:
                supervised_loss = supervised_loss + feature_tri_state['loss_tensor']
                total_loss_value += feature_tri_state['loss_value']
                feature_tri_prototype_raw = feature_tri_state['raw_loss_value']
                feature_tri_prototype_weighted = feature_tri_state['loss_value']
            supervised_loss.backward()

        log_vars['mix_loss'] = total_mix_raw
        log_vars['mix_seg_loss'] = total_mix_raw
        log_vars['mix_weighted_loss'] = total_mix_weighted
        log_vars['target_anchor_loss'] = target_anchor_weighted
        log_vars['target_anchor_seg_loss'] = target_anchor_raw
        log_vars['unlabeled_branch_loss'] = unlabeled_branch_weighted
        log_vars['unlabeled_branch_seg_loss'] = unlabeled_branch_raw
        log_vars['tri_prototype_loss'] = tri_prototype_weighted
        log_vars['tri_prototype_seg_loss'] = tri_prototype_raw
        log_vars['feature_tri_prototype_loss'] = feature_tri_prototype_weighted
        log_vars['feature_tri_prototype_seg_loss'] = feature_tri_prototype_raw
        log_vars['total_loss'] = total_loss_value
        log_vars.update(self._feature_proto_export_diagnostics())
        log_vars.update(self._source_assist_export_diagnostics())

        if self._is_master_process() and semi_enabled and self.debug_img_interval > 0 and \
                (self.local_iter + 1) % self.debug_img_interval == 0:
            debug_weight = None
            if target_mix is not None:
                self._save_debug_visualization(
                    batch_size,
                    means,
                    stds,
                    dataset_class,
                    tgt_l_img,
                    tgt_u_img,
                    target_mix[0],
                    tgt_l_seg_lbl,
                    tgt_u_seg_lbl,
                    pseudo_label,
                    pseudo_weight,
                    pseudo_mask,
                    target_mix[1],
                    target_mix[4],
                    target_mix[3],
                    target_mix[2],
                    target_state['seg_pred'],
                    target_mix_pred,
                    dep_tar,
                    None,
                )
                debug_weight = target_mix[2]
            elif source_mix is not None:
                debug_weight = source_mix[2]
            if debug_weight is not None:
                self._save_hrda_debug_images(
                    seg_debug,
                    batch_size,
                    means,
                    stds,
                    debug_weight,
                )

        del source_mix_pred, target_mix_pred, pseudo_mask
        self.local_iter += 1
        return log_vars
