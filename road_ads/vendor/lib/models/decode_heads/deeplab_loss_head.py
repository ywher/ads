"""Loss/output adapter for the standalone DeepLab models."""

import logging

import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.deeplab_resnet import FrozenBatchNorm2d
from lib.loss.losses import (
    CrossEntropyLoss,
    DiceLoss,
    DyCELoss,
    FocalLoss,
    OhemCELoss,
)


class DeepLabLossHead(nn.Module):
    """Expose standalone DeepLab logits through the shared decoder contract."""

    def __init__(self, decoder_config, pretrained=None):
        super().__init__()
        self.logger = logging.getLogger()
        self.num_classes = int(decoder_config['num_classes'])
        self.align_corners = bool(
            decoder_config.get('align_corners', True))
        self.loss_config = decoder_config.get(
            'loss_decode',
            {'type': 'CrossEntropyLoss', 'loss_weight': 1.0},
        )
        self.unsup_loss_config = decoder_config.get(
            'unsup_loss_decode', None)
        self.target_loss_config = decoder_config.get(
            'target_loss_decode', None)
        self.debug = True
        self.debug_output = {}
        self._init_loss_functions()
        if pretrained is not None:
            self.logger.info(
                'DeepLabLossHead has no trainable pretrained state; '
                'ignoring decoder_pretrained=%s',
                pretrained,
            )

    def _build_loss_context(self, loss_config):
        loss_type = loss_config.get('type', 'CrossEntropyLoss')
        ignore_index = int(loss_config.get('ignore_index', 255))
        weight = float(loss_config.get('loss_weight', 1.0))

        if loss_type == 'CrossEntropyLoss':
            return {
                'mode': 'single',
                'decode_loss': CrossEntropyLoss(
                    ignore_index=ignore_index),
                'loss_weight': weight,
            }
        if loss_type == 'OhemCELoss':
            return {
                'mode': 'single',
                'decode_loss': OhemCELoss(
                    thresh=loss_config.get('thresh', 0.7),
                    min_kept=loss_config.get('min_kept', None),
                    ignore_index=ignore_index,
                ),
                'loss_weight': weight,
            }
        if loss_type == 'DyCELoss':
            return {
                'mode': 'single',
                'decode_loss': DyCELoss(
                    ignore_index=ignore_index,
                    top_k_percent=loss_config.get(
                        'top_k_percent', 0.2),
                    omega=loss_config.get('omega', 0.5),
                    min_kept=loss_config.get('min_kept', 1),
                ),
                'loss_weight': weight,
            }
        if loss_type != 'CombinedLoss':
            raise ValueError(f'Unsupported DeepLab loss: {loss_type}')

        functions = {}
        weights = {}
        for name, cfg in loss_config.get('losses', {}).items():
            loss_weight = float(cfg.get('loss_weight', 1.0))
            if name == 'CrossEntropyLoss':
                function = CrossEntropyLoss(ignore_index=ignore_index)
            elif name == 'OhemCELoss':
                function = OhemCELoss(
                    thresh=cfg.get('thresh', 0.7),
                    min_kept=cfg.get('min_kept', None),
                    ignore_index=ignore_index,
                )
            elif name == 'DyCELoss':
                function = DyCELoss(
                    ignore_index=ignore_index,
                    top_k_percent=cfg.get('top_k_percent', 0.2),
                    omega=cfg.get('omega', 0.5),
                    min_kept=cfg.get('min_kept', 1),
                )
            elif name == 'DiceLoss':
                function = DiceLoss(
                    smooth=cfg.get('smooth', 1.0),
                    ignore_index=ignore_index,
                )
            elif name == 'FocalLoss':
                function = FocalLoss(
                    gamma=cfg.get('gamma', 2.0),
                    alpha=cfg.get('alpha', None),
                    ignore_index=ignore_index,
                )
            else:
                raise ValueError(
                    f'Unsupported DeepLab combined loss: {name}')
            functions[name] = function
            weights[name] = loss_weight
        return {
            'mode': 'combined',
            'functions': functions,
            'weights': weights,
        }

    def _init_loss_functions(self):
        source = self._build_loss_context(self.loss_config)
        unsup = (
            self._build_loss_context(self.unsup_loss_config)
            if self.unsup_loss_config is not None else source
        )
        target = (
            self._build_loss_context(self.target_loss_config)
            if self.target_loss_config is not None else source
        )
        self.loss_contexts = {}
        for key in ('default', 'source', 'src', 'supervised', 'labeled'):
            self.loss_contexts[key] = source
        for key in ('target_labeled', 'target_sup', 'tgt', 'target_gt'):
            self.loss_contexts[key] = target
        for key in ('unsup', 'unlabeled', 'target', 'mix', 'pseudo'):
            self.loss_contexts[key] = unsup

    def _get_loss_context(self, loss_key):
        key = 'default' if loss_key is None else str(loss_key).lower()
        if key not in self.loss_contexts:
            raise KeyError(f'Unknown DeepLab loss_key={loss_key!r}')
        return self.loss_contexts[key]

    def forward(self, inputs):
        if not isinstance(inputs, dict) or 'seg_logits' not in inputs:
            raise TypeError(
                'Standalone DeepLab backbone must return a dictionary '
                'containing seg_logits.')
        self.debug_output = {}
        return inputs['seg_logits']

    def cal_loss(self, seg_logits, seg_label, seg_weight=None, loss_key=None):
        if seg_label.dim() == 4 and seg_label.shape[1] == 1:
            seg_label = seg_label[:, 0]
        if seg_logits.shape[-2:] != seg_label.shape[-2:]:
            seg_logits = F.interpolate(
                seg_logits,
                size=seg_label.shape[-2:],
                mode='bilinear',
                align_corners=self.align_corners,
            )

        context = self._get_loss_context(loss_key)
        if context['mode'] == 'single':
            loss = context['decode_loss'](
                seg_logits, seg_label, seg_weight)
            return {'seg_loss': loss * context['loss_weight']}

        total = seg_logits.new_zeros(())
        losses = {}
        for name, function in context['functions'].items():
            if name in ('DiceLoss', 'FocalLoss'):
                value = function(seg_logits, seg_label)
            else:
                value = function(seg_logits, seg_label, seg_weight)
            value = value * context['weights'][name]
            losses[f'{name.lower()}_value'] = value
            total = total + value
        losses['seg_loss'] = total
        return losses


class DeepLabAuxiliaryHead(DeepLabLossHead):
    """FCN deep-supervision head for the standalone DeepLab encoder."""

    def __init__(self, decoder_config, pretrained=None):
        super().__init__(decoder_config, pretrained=None)
        in_channels = int(decoder_config.get('in_channels', 1024))
        channels = int(decoder_config.get('channels', 256))
        self.in_index = int(decoder_config.get('in_index', 2))
        dropout_ratio = float(decoder_config.get('dropout_ratio', 0.1))
        norm_name = str(
            decoder_config.get('norm_type', 'syncbn')
        ).lower().replace('_', '')
        if norm_name in ('frozen', 'frozenbn'):
            norm_layer = FrozenBatchNorm2d
        elif norm_name in ('batchnorm', 'bn'):
            norm_layer = nn.BatchNorm2d
        elif norm_name in ('syncbatchnorm', 'syncbn'):
            norm_layer = nn.SyncBatchNorm
        else:
            raise ValueError(
                f'Unsupported auxiliary norm_type={norm_name!r}')

        self.classifier = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            norm_layer(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(channels, self.num_classes, kernel_size=1),
        )
        for module in self.classifier.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        if pretrained is not None:
            self.logger.info(
                'DeepLabAuxiliaryHead ignores decoder_pretrained=%s',
                pretrained,
            )

    def forward(self, inputs):
        if not isinstance(inputs, dict) or 'features' not in inputs:
            raise TypeError(
                'DeepLabAuxiliaryHead expects a dictionary containing '
                'the encoder feature list.')
        features = inputs['features']
        if self.in_index >= len(features):
            raise IndexError(
                f'auxiliary in_index={self.in_index} for '
                f'{len(features)} feature levels')
        self.debug_output = {}
        return self.classifier(features[self.in_index])
