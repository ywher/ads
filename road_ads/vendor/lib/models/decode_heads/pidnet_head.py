import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.loss.losses import CrossEntropyLoss, DyCELoss, OhemCELoss


class PIDNetHead(nn.Module):
    """Loss wrapper for PIDNet outputs.

    PIDNet predicts logits inside the backbone. This head keeps the repository's
    decode-head contract: forward returns the main segmentation logits, while
    cal_loss can also supervise PIDNet's auxiliary P-branch and D-branch.
    """

    def __init__(self, decoder_config, pretrained=None):
        super().__init__()
        self.logger = logging.getLogger()
        self.num_classes = decoder_config['num_classes']
        self.align_corners = decoder_config.get('align_corners', False)
        self.loss_config = decoder_config.get(
            'loss_decode',
            dict(type='CrossEntropyLoss', loss_weight=1.0),
        )
        self.unsup_loss_config = decoder_config.get('unsup_loss_decode', None)
        self.target_loss_config = decoder_config.get('target_loss_decode', None)
        self.aux_loss_weight = float(decoder_config.get('aux_loss_weight', 0.4))
        self.boundary_loss_weight = float(
            decoder_config.get('boundary_loss_weight', 0.0))
        self.boundary_ignore_index = int(
            decoder_config.get('boundary_ignore_index', 255))
        self._last_aux_logits = None
        self._last_boundary_logits = None
        self.debug = True
        self.debug_output = {}
        self._init_loss_functions()

        if pretrained is not None:
            self.load_pretrained(pretrained)

    def _build_loss_context(self, loss_config):
        loss_type = loss_config.get('type', 'CrossEntropyLoss')
        ignore_index = loss_config.get('ignore_index', 255)

        if loss_type == 'CrossEntropyLoss':
            return {
                'mode': 'single',
                'loss_name': 'CrossEntropyLoss',
                'decode_loss': CrossEntropyLoss(ignore_index=ignore_index),
                'total_loss_weight': loss_config.get('loss_weight', 1.0),
            }
        if loss_type == 'OhemCELoss':
            return {
                'mode': 'single',
                'loss_name': 'OhemCELoss',
                'decode_loss': OhemCELoss(
                    thresh=loss_config.get('thresh', 0.9),
                    min_kept=loss_config.get('min_kept', 131072),
                    ignore_index=ignore_index,
                ),
                'total_loss_weight': loss_config.get('loss_weight', 1.0),
            }
        if loss_type == 'DyCELoss':
            return {
                'mode': 'single',
                'loss_name': 'DyCELoss',
                'decode_loss': DyCELoss(
                    ignore_index=ignore_index,
                    top_k_percent=loss_config.get('top_k_percent', 0.2),
                    omega=loss_config.get('omega', 0.5),
                    min_kept=loss_config.get('min_kept', 1),
                ),
                'total_loss_weight': loss_config.get('loss_weight', 1.0),
            }
        if loss_type == 'CombinedLoss':
            # Keep PIDNet support intentionally compact: reuse the first
            # configured CE/OHEM/DyCE-like loss as the branch loss.
            losses = loss_config.get('losses', {})
            for key in ('OhemCELoss', 'CrossEntropyLoss', 'DyCELoss'):
                if key in losses:
                    merged = dict(losses[key])
                    merged['type'] = key
                    merged.setdefault('ignore_index', ignore_index)
                    return self._build_loss_context(merged)
        raise ValueError(f'Unsupported PIDNet loss type: {loss_type}')

    def _describe_loss_context(self, context):
        return f"{context['loss_name']}(w={context['total_loss_weight']:.2f})"

    def _init_loss_functions(self):
        source_context = self._build_loss_context(self.loss_config)
        unsup_context = (
            self._build_loss_context(self.unsup_loss_config)
            if self.unsup_loss_config is not None else source_context)
        target_context = (
            self._build_loss_context(self.target_loss_config)
            if self.target_loss_config is not None else source_context)

        self.loss_contexts = {}
        for key in ('default', 'source', 'src', 'supervised', 'labeled'):
            self.loss_contexts[key] = source_context
        for key in ('target_labeled', 'target_sup', 'tgt', 'target_gt'):
            self.loss_contexts[key] = target_context
        for key in ('unsup', 'unlabeled', 'target', 'mix', 'pseudo'):
            self.loss_contexts[key] = unsup_context

        self.logger.info(
            'PIDNetHead: source loss: %s',
            self._describe_loss_context(source_context))
        if self.unsup_loss_config is not None:
            self.logger.info(
                'PIDNetHead: unsup/mix loss: %s',
                self._describe_loss_context(unsup_context))
        if self.target_loss_config is not None:
            self.logger.info(
                'PIDNetHead: target labeled loss: %s',
                self._describe_loss_context(target_context))

    def _get_loss_context(self, loss_key=None):
        key = 'default' if loss_key is None else str(loss_key).lower()
        if key not in self.loss_contexts:
            available = ', '.join(sorted(self.loss_contexts.keys()))
            raise KeyError(
                f'Unknown loss_key={loss_key!r}. Available: {available}')
        return self.loss_contexts[key]

    def load_pretrained(self, pretrained):
        # PIDNet's trainable tensors live in the backbone wrapper. The method is
        # present only for compatibility with the shared builder.
        self.logger.info('PIDNetHead ignores decoder_pretrained=%s', pretrained)

    def forward(self, inputs):
        self._last_aux_logits = None
        self._last_boundary_logits = None
        if isinstance(inputs, (tuple, list)):
            if len(inputs) >= 3:
                self._last_aux_logits = inputs[0]
                self._last_boundary_logits = inputs[2]
                main = inputs[1]
            else:
                main = inputs[0]
        elif isinstance(inputs, dict):
            main = inputs.get('seg_logits', inputs.get('main'))
            self._last_aux_logits = inputs.get('aux_logits', None)
            self._last_boundary_logits = inputs.get('boundary_logits', None)
        else:
            main = inputs

        if self.debug:
            # Standard DACS debug already visualizes image/GT/prediction. Keep
            # the PIDNet head debug output empty to avoid writing bulky logits.
            self.debug_output = {}
        return main

    def _resize_like_label(self, logits, label):
        if logits is None:
            return None
        if logits.shape[-2:] != label.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=label.shape[-2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return logits

    def _make_boundary_target(self, seg_label):
        valid = seg_label.ne(self.boundary_ignore_index)
        safe_label = seg_label.clone()
        safe_label[~valid] = 0
        edge = torch.zeros_like(safe_label, dtype=torch.bool)
        edge[:, :, 1:] |= safe_label[:, :, 1:] != safe_label[:, :, :-1]
        edge[:, :, :-1] |= safe_label[:, :, 1:] != safe_label[:, :, :-1]
        edge[:, 1:, :] |= safe_label[:, 1:, :] != safe_label[:, :-1, :]
        edge[:, :-1, :] |= safe_label[:, 1:, :] != safe_label[:, :-1, :]
        edge &= valid
        return edge.float(), valid.float()

    def _calc_branch_loss(self, logits, seg_label, seg_weight, context):
        loss = context['decode_loss'](logits, seg_label, seg_weight)
        return loss * context['total_loss_weight']

    def cal_loss(self, seg_logits, seg_label, seg_weight=None, loss_key=None):
        if seg_label.dim() == 4 and seg_label.size(1) == 1:
            seg_label = seg_label.squeeze(1)
        seg_logits = self._resize_like_label(seg_logits, seg_label)

        context = self._get_loss_context(loss_key)
        loss = self._calc_branch_loss(
            seg_logits, seg_label, seg_weight, context)
        loss_dict = {'seg_loss': loss}

        aux_logits = self._resize_like_label(self._last_aux_logits, seg_label)
        if aux_logits is not None and self.aux_loss_weight > 0:
            aux_loss = self._calc_branch_loss(
                aux_logits, seg_label, seg_weight, context)
            loss_dict['aux_seg_loss'] = aux_loss * self.aux_loss_weight

        boundary_logits = self._resize_like_label(
            self._last_boundary_logits, seg_label)
        if boundary_logits is not None and self.boundary_loss_weight > 0:
            boundary_target, valid = self._make_boundary_target(seg_label)
            bce = F.binary_cross_entropy_with_logits(
                boundary_logits.squeeze(1),
                boundary_target,
                reduction='none')
            if seg_weight is not None:
                bce = bce * seg_weight.float()
                valid = valid * seg_weight.float().gt(0).float()
            denom = valid.sum().clamp_min(1.0)
            boundary_loss = (bce * valid).sum() / denom
            loss_dict['boundary_loss'] = (
                boundary_loss * self.boundary_loss_weight)

        return loss_dict
