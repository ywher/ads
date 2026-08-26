# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import random

import torch
from torch.nn import Module

from .teacher_module import EMATeacher  # checked
from lib.models.model_utils.dacs_transforms import strong_transform, get_mean_std_self
from lib.models.model_utils.masking_transforms import build_mask_generator  # checked


class MaskingConsistencyModule(Module):

    def __init__(self, require_teacher, cfg):
        super(MaskingConsistencyModule, self).__init__()
        
        self.cfg = cfg
        self.img_mean = self.cfg.data['rgb_mean']
        self.img_std = self.cfg.data['rgb_std']

        self.source_only = cfg.get('source_only', False)  # False
        self.max_iters = cfg.max_iters  # 40000
        self.color_jitter_s = cfg.semi['color_jitter_strength']  # 0.2
        self.color_jitter_p = cfg.semi['color_jitter_probability']  # 0.2

        self.mask_mode = cfg.semi['mask_mode']  # separatetrgaug
        self.mask_alpha = cfg.semi['mask_alpha']  # same
        self.mask_pseudo_threshold = cfg.semi['mask_pseudo_threshold']  # same
        self.mask_lambda = cfg.semi['mask_lambda']  # 1
        self.mask_gen = build_mask_generator(cfg.semi['mask_generator'])

        assert self.mask_mode in [
            'separate', 'separatesrc', 
            'separatetrg', 'separateaug',
            'separatesrcaug', 'separatetrgaug'
        ]

        self.teacher = None
        # no use in this case
        if require_teacher or self.mask_alpha != 'same' or self.mask_pseudo_threshold != 'same':
            self.teacher = EMATeacher(use_mask_params=True, cfg=cfg)

        self.debug = False
        self.debug_output = {}

    def update_weights(self, model, iter):
        if self.teacher is not None:
            self.teacher.update_weights(model, iter)

    def update_debug_state(self):
        if self.teacher is not None:
            self.teacher.debug = self.debug

    def __call__(self,
                 model,
                 img,
                 gt_semantic_seg,
                 target_img,
                 valid_pseudo_mask,
                 pseudo_label=None,
                 pseudo_weight=None,
                 loss_key=None):
        self.update_debug_state()
        self.debug_output = {}
        model.debug_output = {}
        dev = img.device
        batch_size = img.shape[0]
        means, stds = get_mean_std_self(self.img_mean, self.img_std, batch_size, dev)

        if not self.source_only:
            # Share the pseudo labels with the host semi method
            if self.teacher is None:
                assert self.mask_alpha == 'same'
                assert self.mask_pseudo_threshold == 'same'
                assert pseudo_label is not None
                assert pseudo_weight is not None
                masked_plabel = pseudo_label  # (2, 512, 512)
                masked_pweight = pseudo_weight  # (2, 512, 512)
            # Use a separate EMA teacher for MIC
            else:
                masked_plabel, masked_pweight = \
                    self.teacher(target_img, valid_pseudo_mask)
                if self.debug:
                    self.debug_output['Mask Teacher'] = {
                        'image': target_img.detach(),
                        'pseudo label': masked_plabel.cpu().numpy(),
                        'pseudo weight': masked_pweight.cpu().numpy(),
                    }
        # Don't use target images at all
        if self.source_only:
            masked_img = img
            masked_lbl = gt_semantic_seg
            b, _, h, w = gt_semantic_seg.shape
            masked_seg_weight = None
        # Use 1x source image and 1x target image for MIC
        elif self.mask_mode in ['separate', 'separateaug']:
            assert img.shape[0] == 2
            masked_img = torch.stack([img[0], target_img[0]])
            masked_lbl = torch.stack(
                [gt_semantic_seg[0], masked_plabel[0]])
            gt_pixel_weight = torch.ones(masked_pweight[0].shape, device=dev)
            masked_seg_weight = torch.stack(
                [gt_pixel_weight, masked_pweight[0]])
        # Use only source images for MIC
        elif self.mask_mode in ['separatesrc', 'separatesrcaug']:
            masked_img = img
            masked_lbl = gt_semantic_seg
            masked_seg_weight = None
        # Use only target images for MIC
        elif self.mask_mode in ['separatetrg', 'separatetrgaug']:
            masked_img = target_img  # [2, 3, 512, 512]
            masked_lbl = masked_plabel  # [2, 512, 512] .unsqueeze(1)
            masked_seg_weight = masked_pweight  # [2, 512, 512]
        else:
            raise NotImplementedError(self.mask_mode)

        # Apply color augmentation
        if 'aug' in self.mask_mode:  # True
            strong_parameters = {
                'mix': None,
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': self.color_jitter_s,
                'color_jitter_p': self.color_jitter_p,
                'blur': random.uniform(0, 1),
                'mean': means[0].unsqueeze(0),
                'std': stds[0].unsqueeze(0)
            }
            masked_img, _ = strong_transform(strong_parameters, data=masked_img.clone())

        # Apply masking to image
        masked_img = self.mask_gen.mask_image(masked_img)  # [2, 3, 512, 512]

        # Train on masked images
        masked_results = model.forward_train(
            (masked_img, masked_lbl),
            masked_seg_weight,
            loss_key=loss_key,
        )
        masked_logits = masked_results['seg_logits']
        masked_loss_dict = {}
        masked_loss_dict['seg_loss'] = masked_results['seg_loss']
        if self.mask_lambda != 1:
            masked_loss_dict['seg_loss'] *= self.mask_lambda

        if self.debug:
            debug_content = {
                'Image': masked_img,
                'Seg Pred': masked_logits,
                'Seg GT': masked_lbl,
            }
            self.debug_output['Masked'] = debug_content
            self.debug_output['Masked'].update(model.decode_head.debug_output)
            if masked_seg_weight is not None:
                # unique_weights = torch.unique(masked_seg_weight)
                # pl_weight_str = f"PL Weight"
                # for unique_weight in unique_weights:
                #     if unique_weight > 0:
                #         pl_weight_str += f" {unique_weight:.2f}"
                #     else:
                #         pl_weight_str += f" {unique_weight:.0f}"
                self.debug_output['Masked']['PL Weight'] = \
                    masked_seg_weight.cpu().numpy()

        return masked_loss_dict
