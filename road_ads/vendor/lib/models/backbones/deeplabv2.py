"""Standalone DeepLabV2-ResNet101 model adapted from RIPU/HALO."""

import logging

import torch.nn as nn

from .deeplab_resnet import ResNet101Encoder, load_deeplab_pretrained


class ASPPClassifierV2(nn.Module):
    """Four parallel atrous classifiers summed at output stride 8."""

    def __init__(
        self,
        in_channels,
        num_classes,
        dilation_series=(6, 12, 18, 24),
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                num_classes,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=True,
            )
            for dilation in dilation_series
        ])
        for conv in self.convs:
            nn.init.normal_(conv.weight, mean=0.0, std=0.01)
            nn.init.zeros_(conv.bias)

    def forward(self, x):
        out = self.convs[0](x)
        for conv in self.convs[1:]:
            out = out + conv(x)
        return out


class DeepLabV2ResNet101(nn.Module):
    """RIPU/HALO-style DeepLabV2 with a frozen-BN ResNet-101."""

    def __init__(self, backbone_config, pretrained=None):
        super().__init__()
        cfg = dict(backbone_config)
        self.num_classes = int(cfg.get('num_classes', 19))
        self.encoder = ResNet101Encoder(
            freeze_bn=bool(cfg.get('freeze_bn', True)),
            output_stride=int(cfg.get('output_stride', 8)),
        )
        self.classifier = ASPPClassifierV2(
            in_channels=2048,
            num_classes=self.num_classes,
            dilation_series=tuple(
                cfg.get('dilations', (6, 12, 18, 24))),
        )

        self.patch_size = 1
        self.enable_adapter = False
        self.save_whole_backbone = True
        self.logger = logging.getLogger()
        load_deeplab_pretrained(self, pretrained, logger=self.logger)

    def forward(self, x, *args, **kwargs):
        features = self.encoder(x)
        logits = self.classifier(features[-1])
        # Keep features first so prototype utilities select a semantic feature
        # tensor rather than the classifier logits.
        return {
            'features': features,
            'seg_logits': logits,
        }
