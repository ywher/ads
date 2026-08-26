# -------------------------------------------------------------------------
# PIDNet integration for SSDA.
#
# Adapted to this repository's EncoderDecoder interface from the official
# MIT-licensed PIDNet implementation:
# https://github.com/XuJiacong/PIDNet
# -------------------------------------------------------------------------

import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


BN_MOMENTUM = 0.1
ALIGN_CORNERS = False


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 no_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return out if self.no_relu else self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 no_relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(
            planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return out if self.no_relu else self.relu(out)


class SegmentHead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        self.conv1 = nn.Conv2d(
            inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            out = F.interpolate(
                out,
                size=(x.shape[-2] * self.scale_factor,
                      x.shape[-1] * self.scale_factor),
                mode='bilinear',
                align_corners=ALIGN_CORNERS)
        return out


class DAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = self._process(branch_planes)
        self.process2 = self._process(branch_planes)
        self.process3 = self._process(branch_planes)
        self.process4 = self._process(branch_planes)
        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 5, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1,
                      bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    @staticmethod
    def _process(channels):
        return nn.Sequential(
            nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                      bias=False),
        )

    def forward(self, x):
        height, width = x.shape[-2:]
        x0 = self.scale0(x)
        x1 = self.process1(
            F.interpolate(self.scale1(x), size=(height, width),
                          mode='bilinear',
                          align_corners=ALIGN_CORNERS) + x0)
        x2 = self.process2(
            F.interpolate(self.scale2(x), size=(height, width),
                          mode='bilinear',
                          align_corners=ALIGN_CORNERS) + x1)
        x3 = self.process3(
            F.interpolate(self.scale3(x), size=(height, width),
                          mode='bilinear',
                          align_corners=ALIGN_CORNERS) + x2)
        x4 = self.process4(
            F.interpolate(self.scale4(x), size=(height, width),
                          mode='bilinear',
                          align_corners=ALIGN_CORNERS) + x3)
        return self.compression(torch.cat([x0, x1, x2, x3, x4], dim=1)) + \
            self.shortcut(x)


class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale_process = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 4, branch_planes * 4, kernel_size=3,
                      padding=1, groups=4, bias=False),
        )
        self.compression = nn.Sequential(
            nn.BatchNorm2d(branch_planes * 5, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1,
                      bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        height, width = x.shape[-2:]
        x0 = self.scale0(x)
        scales = [
            F.interpolate(self.scale1(x), size=(height, width),
                          mode='bilinear', align_corners=ALIGN_CORNERS) + x0,
            F.interpolate(self.scale2(x), size=(height, width),
                          mode='bilinear', align_corners=ALIGN_CORNERS) + x0,
            F.interpolate(self.scale3(x), size=(height, width),
                          mode='bilinear', align_corners=ALIGN_CORNERS) + x0,
            F.interpolate(self.scale4(x), size=(height, width),
                          mode='bilinear', align_corners=ALIGN_CORNERS) + x0,
        ]
        scale_out = self.scale_process(torch.cat(scales, dim=1))
        return self.compression(torch.cat([x0, scale_out], dim=1)) + \
            self.shortcut(x)


class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False,
                 with_channel=False):
        super().__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(in_channels),
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            x = self.relu(x)
            y = self.relu(y)
        y_q = F.interpolate(
            self.f_y(y),
            size=input_size[2:],
            mode='bilinear',
            align_corners=ALIGN_CORNERS)
        x_k = self.f_x(x)
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1,
                                              keepdim=True))
        y = F.interpolate(
            y,
            size=input_size[2:],
            mode='bilinear',
            align_corners=ALIGN_CORNERS)
        return (1 - sim_map) * x + sim_map * y


class LightBag(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)
        return p_add + i_add


class Bag(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,
                      bias=False),
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1 - edge_att) * i)


class PIDNet(nn.Module):
    def __init__(
        self,
        m=2,
        n=3,
        num_classes=19,
        planes=32,
        ppm_planes=96,
        head_planes=128,
        augment=True,
    ):
        super().__init__()
        self.augment = augment

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(
            BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(
            BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(
            BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(
            Bottleneck, planes * 8, planes * 8, 2, stride=2)

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)
        self.layer3_ = self._make_layer(
            BasicBlock, planes * 2, planes * 2, m)
        self.layer4_ = self._make_layer(
            BasicBlock, planes * 2, planes * 2, m)
        self.layer5_ = self._make_layer(
            Bottleneck, planes * 2, planes * 2, 1)

        if m == 2:
            self.layer3_d = self._make_single_layer(
                BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1,
                          bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1,
                          bias=False),
                nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
            )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = LightBag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(
                BasicBlock, planes * 2, planes * 2)
            self.layer4_d = self._make_single_layer(
                BasicBlock, planes * 2, planes * 2)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1,
                          bias=False),
                nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1,
                          bias=False),
                nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
            )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)
        self.layer5_d = self._make_layer(
            Bottleneck, planes * 2, planes * 2, 1)

        if self.augment:
            self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
            self.seghead_d = SegmentHead(planes * 2, planes, 1)
        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)

        self.init_weights()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,
                               momentum=BN_MOMENTUM),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(
                inplanes, planes, no_relu=(i == blocks - 1)))
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,
                               momentum=BN_MOMENTUM),
            )
        return block(inplanes, planes, stride, downsample, no_relu=True)

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        height_output = x.shape[-2] // 8
        width_output = x.shape[-1] // 8

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))

        x_p = self.layer3_(x)
        x_d = self.layer3_d(x)
        x = self.relu(self.layer3(x))
        x_p = self.pag3(x_p, self.compression3(x))
        x_d = x_d + F.interpolate(
            self.diff3(x),
            size=(height_output, width_output),
            mode='bilinear',
            align_corners=ALIGN_CORNERS)
        temp_p = x_p

        x = self.relu(self.layer4(x))
        x_p = self.layer4_(self.relu(x_p))
        x_d = self.layer4_d(self.relu(x_d))
        x_p = self.pag4(x_p, self.compression4(x))
        x_d = x_d + F.interpolate(
            self.diff4(x),
            size=(height_output, width_output),
            mode='bilinear',
            align_corners=ALIGN_CORNERS)
        temp_d = x_d

        x_p = self.layer5_(self.relu(x_p))
        x_d = self.layer5_d(self.relu(x_d))
        x = F.interpolate(
            self.spp(self.layer5(x)),
            size=(height_output, width_output),
            mode='bilinear',
            align_corners=ALIGN_CORNERS)
        out = self.final_layer(self.dfm(x_p, x, x_d))

        if self.augment:
            return [self.seghead_p(temp_p), out, self.seghead_d(temp_d)]
        return out


class PIDNetS(nn.Module):
    """PIDNet-S wrapper compatible with this repository's model builder."""

    def __init__(self, backbone_config, pretrained=None):
        super().__init__()
        cfg = dict(backbone_config)
        variant = str(cfg.get('variant', 's')).lower()
        num_classes = int(cfg.get('num_classes', 19))
        augment = bool(cfg.get('augment', True))

        if variant in ('s', 'small', 'pidnet_s', 'pidnet-small'):
            params = dict(m=2, n=3, planes=32, ppm_planes=96,
                          head_planes=128)
        elif variant in ('m', 'medium', 'pidnet_m', 'pidnet-medium'):
            params = dict(m=2, n=3, planes=64, ppm_planes=96,
                          head_planes=128)
        elif variant in ('l', 'large', 'pidnet_l', 'pidnet-large'):
            params = dict(m=3, n=4, planes=64, ppm_planes=112,
                          head_planes=256)
        else:
            raise ValueError(f'Unknown PIDNet variant: {variant}')

        self.model = PIDNet(
            num_classes=num_classes,
            augment=augment,
            **params,
        )
        self.variant = variant
        self.num_classes = num_classes
        self.augment = augment
        self.patch_size = 1
        self.enable_adapter = False
        self.save_whole_backbone = True
        self.logger = logging.getLogger()

        self._load_pretrained(pretrained)

    def _resolve_pretrained_path(self, pretrained):
        if not pretrained:
            return None
        if isinstance(pretrained, (str, bytes)):
            return pretrained
        if isinstance(pretrained, dict):
            for key in ('pidnet', 'backbone', 'imagenet', 'cityscapes'):
                if pretrained.get(key):
                    return pretrained[key]
        return None

    def _strip_prefixes(self, state_dict):
        stripped = OrderedDict()
        prefixes = (
            'module.backbone.model.',
            'backbone.model.',
            'module.model.',
            'model.backbone.',
            'module.backbone.',
            'backbone.',
            'model.',
            'module.',
        )
        for key, value in state_dict.items():
            new_key = key
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            stripped[new_key] = value
        return stripped

    def _load_pretrained(self, pretrained):
        path = self._resolve_pretrained_path(pretrained)
        if path is None:
            return
        try:
            checkpoint = torch.load(path, map_location='cpu')
        except FileNotFoundError:
            self.logger.warning('PIDNet pretrained not found: %s', path)
            return

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', None)
            if state_dict is None:
                state_dict = checkpoint.get('model', None)
            if state_dict is None:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        state_dict = self._strip_prefixes(state_dict)
        model_dict = self.model.state_dict()
        loadable = {
            key: value
            for key, value in state_dict.items()
            if key in model_dict and tuple(value.shape) == tuple(model_dict[key].shape)
        }
        model_dict.update(loadable)
        missing, unexpected = self.model.load_state_dict(model_dict, strict=False)
        self.logger.info(
            'PIDNet loaded %d/%d compatible tensors from %s',
            len(loadable), len(state_dict), path)
        if missing:
            self.logger.info('PIDNet missing tensors after load: %d',
                             len(missing))
        if unexpected:
            self.logger.info('PIDNet unexpected tensors after load: %d',
                             len(unexpected))

    def forward(self, x, *args, **kwargs):
        return self.model(x)
