"""ResNet-101 encoder shared by the standalone DeepLab models.

The implementation follows the ResNet feature extractors used by RIPU and
HALO. It is kept local so the DeepLab baselines do not depend on mmseg or on a
third-party checkout at runtime.
"""

import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed running statistics and affine parameters."""

    def __init__(self, num_features):
        super().__init__()
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        return (
            x * scale.reshape(1, -1, 1, 1)
            + bias.reshape(1, -1, 1, 1)
        )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(
            width,
            width,
            stride=stride,
            groups=groups,
            dilation=dilation,
        )
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class ResNet101Encoder(nn.Module):
    """Dilated standard-stem ResNet-101 (MMSeg R101b variant)."""

    def __init__(self, freeze_bn=True, output_stride=8, norm_type=None):
        super().__init__()
        if output_stride == 8:
            replace_stride_with_dilation = (False, True, True)
        elif output_stride == 16:
            replace_stride_with_dilation = (False, False, True)
        else:
            raise ValueError(
                f'output_stride must be 8 or 16, got {output_stride}')

        norm_name = (
            'frozen' if norm_type is None and freeze_bn
            else 'batchnorm' if norm_type is None
            else str(norm_type).lower().replace('_', '')
        )
        if norm_name in ('frozen', 'frozenbn'):
            norm_layer = FrozenBatchNorm2d
        elif norm_name in ('batchnorm', 'bn'):
            norm_layer = nn.BatchNorm2d
        elif norm_name in ('syncbatchnorm', 'syncbn'):
            norm_layer = nn.SyncBatchNorm
        else:
            raise ValueError(f'Unsupported ResNet norm_type={norm_type!r}')
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(
            Bottleneck,
            128,
            4,
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            Bottleneck,
            256,
            23,
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            Bottleneck,
            512,
            3,
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.out_channels = (256, 512, 1024, 2048)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


def resolve_pretrained_path(pretrained):
    if not pretrained:
        return None
    if isinstance(pretrained, (str, bytes, os.PathLike)):
        return os.fspath(pretrained)
    if isinstance(pretrained, dict):
        for key in (
            'deeplab',
            'resnet',
            'backbone',
            'imagenet',
            'whole',
        ):
            if pretrained.get(key):
                return os.fspath(pretrained[key])
    return None


def _checkpoint_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ('state_dict', 'model', 'backbone'):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


def _strip_training_prefixes(state_dict):
    prefixes = (
        'module.model.backbone.',
        'model.backbone.',
        'module.backbone.',
        'backbone.',
        'module.model.',
        'model.',
        'module.',
    )
    stripped = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        stripped[new_key] = value
    return stripped


def _compatible_state_dict(module, state_dict):
    module_state = module.state_dict()
    return {
        key: value
        for key, value in state_dict.items()
        if (
            key in module_state
            and tuple(value.shape) == tuple(module_state[key].shape)
        )
    }


def load_deeplab_pretrained(model, pretrained, logger=None):
    """Load ImageNet encoder weights or a whole repository checkpoint."""

    logger = logger or logging.getLogger()
    path = resolve_pretrained_path(pretrained)
    if path is None:
        return
    if not os.path.isfile(path):
        logger.warning('DeepLab pretrained checkpoint not found: %s', path)
        return

    checkpoint = torch.load(path, map_location='cpu')
    state_dict = _strip_training_prefixes(
        _checkpoint_state_dict(checkpoint))

    whole_state = _compatible_state_dict(model, state_dict)
    if whole_state:
        missing, unexpected = model.load_state_dict(
            whole_state, strict=False)
        logger.info(
            '%s loaded %d compatible model tensors from %s',
            model.__class__.__name__,
            len(whole_state),
            path,
        )
    else:
        encoder_state = _compatible_state_dict(model.encoder, state_dict)
        missing, unexpected = model.encoder.load_state_dict(
            encoder_state, strict=False)
        logger.info(
            '%s loaded %d compatible ImageNet encoder tensors from %s',
            model.__class__.__name__,
            len(encoder_state),
            path,
        )

    if missing:
        logger.info('%s missing tensors after load: %d',
                    model.__class__.__name__, len(missing))
    if unexpected:
        logger.info('%s unexpected tensors after load: %d',
                    model.__class__.__name__, len(unexpected))
