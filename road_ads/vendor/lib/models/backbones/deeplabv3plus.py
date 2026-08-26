"""Standalone DeepLabV3+-ResNet101 model adapted from RIPU/HALO."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deeplab_resnet import (
    FrozenBatchNorm2d,
    ResNet101Encoder,
    load_deeplab_pretrained,
)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        norm_layer=FrozenBatchNorm2d,
    ):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DepthwiseSeparableASPP(nn.Module):
    """RIPU/HALO DeepLabV3+ ASPP and low-level decoder."""

    def __init__(
        self,
        in_channels,
        num_classes,
        dilation_series=(1, 6, 12, 18),
        norm_layer=FrozenBatchNorm2d,
        align_corners=True,
    ):
        super().__init__()
        self.align_corners = bool(align_corners)
        aspp_channels = 512
        branches = []
        for dilation in dilation_series:
            if dilation == 1:
                branch = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        aspp_channels,
                        kernel_size=1,
                        bias=False,
                    ),
                    norm_layer(aspp_channels),
                    nn.ReLU(inplace=True),
                )
            else:
                branch = DepthwiseSeparableConv2d(
                    in_channels,
                    aspp_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    norm_layer=norm_layer,
                )
            branches.append(branch)
        self.parallel_branches = nn.ModuleList(branches)

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels,
                aspp_channels,
                kernel_size=1,
                bias=False,
            ),
            norm_layer(aspp_channels),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                aspp_channels * (len(dilation_series) + 1),
                aspp_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            norm_layer(aspp_channels),
            nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            norm_layer(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            DepthwiseSeparableConv2d(
                560,
                512,
                kernel_size=3,
                padding=1,
                norm_layer=norm_layer,
            ),
            DepthwiseSeparableConv2d(
                512,
                512,
                kernel_size=3,
                padding=1,
                norm_layer=norm_layer,
            ),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, low_level_feature, high_level_feature):
        aspp_features = [
            branch(high_level_feature)
            for branch in self.parallel_branches
        ]
        global_features = self.global_branch(high_level_feature)
        global_features = F.interpolate(
            global_features,
            size=high_level_feature.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners,
        )
        aspp_features.append(global_features)
        aspp_features = self.bottleneck(
            torch.cat(aspp_features, dim=1))
        aspp_features = F.interpolate(
            aspp_features,
            size=low_level_feature.shape[-2:],
            mode='bilinear',
            align_corners=self.align_corners,
        )
        low_level_feature = self.shortcut(low_level_feature)
        return self.decoder(
            torch.cat([aspp_features, low_level_feature], dim=1))


class DeepLabV3PlusResNet101(nn.Module):
    """DeepLabV3+ with the standard-stem R101b-compatible encoder."""

    def __init__(self, backbone_config, pretrained=None):
        super().__init__()
        cfg = dict(backbone_config)
        variant = str(cfg.get('variant', 'r101b')).lower()
        if variant not in ('r101b', 'resnet101', 'standard'):
            raise ValueError(
                'DeepLabV3PlusResNet101 supports the standard-stem R101b '
                f'variant only, got {variant!r}')
        self.num_classes = int(cfg.get('num_classes', 19))
        freeze_bn = bool(cfg.get('freeze_bn', True))
        norm_name = str(cfg.get(
            'norm_type',
            'frozenbn' if freeze_bn else 'batchnorm',
        )).lower().replace('_', '')
        if norm_name in ('frozen', 'frozenbn'):
            norm_layer = FrozenBatchNorm2d
        elif norm_name in ('batchnorm', 'bn'):
            norm_layer = nn.BatchNorm2d
        elif norm_name in ('syncbatchnorm', 'syncbn'):
            norm_layer = nn.SyncBatchNorm
        else:
            raise ValueError(
                f'Unsupported DeepLabV3+ norm_type={norm_name!r}')
        self.encoder = ResNet101Encoder(
            freeze_bn=freeze_bn,
            output_stride=int(cfg.get('output_stride', 8)),
            norm_type=norm_name,
        )
        self.classifier = DepthwiseSeparableASPP(
            in_channels=2048,
            num_classes=self.num_classes,
            dilation_series=tuple(
                cfg.get('dilations', (1, 6, 12, 18))),
            norm_layer=norm_layer,
            align_corners=bool(cfg.get('align_corners', True)),
        )

        self.patch_size = 1
        self.enable_adapter = False
        self.save_whole_backbone = True
        self.logger = logging.getLogger()
        load_deeplab_pretrained(self, pretrained, logger=self.logger)

    def forward(self, x, *args, **kwargs):
        features = self.encoder(x)
        logits = self.classifier(features[0], features[-1])
        return {
            'features': features,
            'seg_logits': logits,
        }
