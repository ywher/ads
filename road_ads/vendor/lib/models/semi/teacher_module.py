# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

"""EMA teacher used by the semi-supervised masking consistency module.

This is the semi-supervised counterpart of DAFormer's UDA teacher module.  It
keeps the behavior needed by MIC, but builds the teacher through the current
semi code path instead of depending on the removed UDA decorator.
"""

from copy import deepcopy

import numpy as np
import torch
from timm.layers import DropPath
from torch.nn import Module
from torch.nn.modules.dropout import _DropoutNd

from .segmentor import SemiSegmentor, get_module


class EMATeacher(Module):
    """A lightweight EMA teacher for pseudo-label generation."""

    def __init__(self, use_mask_params, cfg):
        super().__init__()
        semi_cfg = cfg.semi
        prefix = 'mask_' if use_mask_params else ''

        self.alpha = semi_cfg[f'{prefix}alpha']
        if self.alpha == 'same':
            self.alpha = semi_cfg['alpha']

        self.pseudo_threshold = semi_cfg[f'{prefix}pseudo_threshold']
        if self.pseudo_threshold == 'same':
            self.pseudo_threshold = semi_cfg['pseudo_threshold']

        self.psweight_ignore_top = semi_cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = semi_cfg['pseudo_weight_ignore_bottom']

        ema_cfg = deepcopy(cfg.model)
        if 'ema_backbone_pretrained' in ema_cfg:
            ema_cfg['backbone_pretrained'] = ema_cfg.pop('ema_backbone_pretrained')
        if 'ema_decoder_pretrained' in ema_cfg:
            ema_cfg['decoder_pretrained'] = ema_cfg.pop('ema_decoder_pretrained')

        builder = SemiSegmentor(ema_cfg)
        self.ema_model = builder.get_model()

        self.debug = False
        self.debug_output = {}

    def get_ema_model(self):
        return get_module(self.ema_model)

    @torch.no_grad()
    def _init_ema_weights(self, model):
        model = get_module(model)
        self.get_ema_model().load_state_dict(model.state_dict(), strict=True)
        for param in self.get_ema_model().parameters():
            param.detach_()

    @torch.no_grad()
    def _update_ema(self, model, iter):
        model = get_module(model)
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)

        ema_state = self.get_ema_model().state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key]
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(alpha_teacher).add_(model_value, alpha=1 - alpha_teacher)
            else:
                ema_value.copy_(model_value)

    def update_debug_state(self):
        self.get_ema_model().automatic_debug = False
        self.get_ema_model().debug = self.debug

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        if self.pseudo_threshold is not None:
            ps_large_p = pseudo_prob.ge(self.pseudo_threshold)
            ps_size = np.size(np.array(pseudo_label.cpu()))
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = pseudo_weight * torch.ones(
                pseudo_prob.shape, device=logits.device)
        else:
            pseudo_weight = torch.ones(pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            if valid_pseudo_mask.dim() == 4 and valid_pseudo_mask.shape[1] == 1:
                valid_pseudo_mask = valid_pseudo_mask.squeeze(1)
            pseudo_weight *= valid_pseudo_mask.to(
                device=pseudo_weight.device, dtype=pseudo_weight.dtype)
        return pseudo_weight

    def update_weights(self, model, iter):
        if iter == 0:
            self._init_ema_weights(model)
        else:
            self._update_ema(model, iter)

    def __call__(self, target_img, valid_pseudo_mask):
        self.update_debug_state()

        for module in self.get_ema_model().modules():
            if isinstance(module, _DropoutNd):
                module.training = False
            if isinstance(module, DropPath):
                module.training = False

        with torch.no_grad():
            ema_logits = self.get_ema_model().generate_pseudo_label(target_img)
        if isinstance(ema_logits, (tuple, list)):
            ema_logits = ema_logits[0]

        pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(ema_logits)
        del ema_logits

        pseudo_weight = self.filter_valid_pseudo_region(
            pseudo_weight, valid_pseudo_mask)
        self.debug_output = getattr(self.ema_model, 'debug_output', {})

        return pseudo_label, pseudo_weight
