# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
"""DAFormer decode head.

DAFormer 解码头。

The head receives multi-level backbone features, projects each level to a
shared embedding dimension, fuses them with a configurable layer, and predicts
semantic logits at the highest feature resolution.

该解码头接收多层 backbone 特征，将每层投影到统一 embedding 维度，再通过
可配置融合层输出最高特征分辨率下的语义分割 logits。
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.loss.losses import CrossEntropyLoss, DiceLoss, DyCELoss, FocalLoss, OhemCELoss
from lib.models.model_utils.funcs import resize
from lib.models.model_utils.cna import ConvModule, MLP, DepthwiseSeparableConvModule
from .aspp_head import ASPPModule
from .sep_aspp_head import DepthwiseSeparableASPPModule


def build_layer(in_channels, out_channels, type, **kwargs):
    """Build one projection or fusion layer from decoder config.

    根据 decoder 配置构建一个投影层或融合层。
    """
    if type == 'id':
        return nn.Identity()
    elif type == 'mlp':
        return MLP(input_dim=in_channels, embed_dim=out_channels)
    elif type == 'sep_conv':
        return DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'conv':
        return ConvModule(
            in_channels=in_channels,
            out_channels=out_channels,
            padding=kwargs['kernel_size'] // 2,
            **kwargs)
    elif type == 'aspp':
        return ASPPWrapper(
            in_channels=in_channels, channels=out_channels, **kwargs)
    elif type == 'rawconv_and_aspp':
        kernel_size = kwargs.pop('kernel_size')
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2),
            ASPPWrapper(
                in_channels=out_channels, channels=out_channels, **kwargs))
    else:
        raise NotImplementedError(type)


class ASPPWrapper(nn.Module):
    """ASPP fusion wrapper used by DAFormer.

    DAFormer 使用的 ASPP 融合模块包装器。
    """

    def __init__(self,
                 in_channels,
                 channels,
                 sep,
                 dilations,
                 pool,
                 align_corners,
                 norm_cfg=None,
                 act_cfg=None,
                 context_cfg=None):
        super(ASPPWrapper, self).__init__()
        assert isinstance(dilations, (list, tuple))
        self.dilations = dilations
        self.align_corners = align_corners
        if pool:  # False
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        else:
            self.image_pool = None
        if context_cfg is not None:
            self.context_layer = build_layer(in_channels, channels, **context_cfg)
        else:
            self.context_layer = None
        ASPP = {True: DepthwiseSeparableASPPModule, False: ASPPModule}[sep]
        self.aspp_modules = ASPP(
            dilations=dilations,
            in_channels=in_channels,
            channels=channels,
            norm_cfg=norm_cfg,
            conv_cfg=None,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            (len(dilations) + int(pool) + int(bool(context_cfg))) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        """Forward function."""
        aspp_outs = []
        if self.image_pool is not None:
            aspp_outs.append(
                resize(
                    self.image_pool(x),
                    size=x.size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners))
        if self.context_layer is not None:
            aspp_outs.append(self.context_layer(x))
        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)
        return output


class DAFormerHead(nn.Module):
    """Project and fuse multi-level features for semantic segmentation.

    对多层特征进行投影和融合，用于语义分割预测。

    Args:
        decoder_config (dict): Head configuration. `in_channels` and `in_index`
            describe the selected backbone feature levels.
            解码头配置，其中 `in_channels` 和 `in_index` 描述被选中的
            backbone 特征层。
        pretrained (str | None): Optional decoder checkpoint.
            可选的 decoder 预训练权重路径。
    """

    def __init__(self, decoder_config, pretrained=None):
        super(DAFormerHead, self).__init__()
        self.logger = logging.getLogger()
        self.input_transform = 'multiple_select'
        
        self.in_channels = decoder_config['in_channels']  # [64, 128, 320, 512]
        self.in_index = decoder_config['in_index'] # [0, 1, 2, 3]
        self.channels = decoder_config['channels']  # 256
        self.dropout_ratio = decoder_config['dropout_ratio']  # 0.1
        self.num_classes = decoder_config['num_classes']  # 19
        self.align_corners = decoder_config['align_corners']  # False
        self.interpolate = decoder_config.get('interpolate', True)
        self.norm_cfg = decoder_config['norm_cfg']  # dict(type='BN', requires_grad=True)
        
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout = None
            
        self.loss_config = decoder_config['loss_decode']
        self.unsup_loss_config = decoder_config.get('unsup_loss_decode', None)
        self.target_loss_config = decoder_config.get('target_loss_decode', None)
        
        # 初始化损失函数
        self._init_loss_functions()
        
        decoder_params = decoder_config['decoder_params']
        '''
        'embed_dims': 256,
        'embed_cfg': {'type': 'mlp', 'act_cfg': None, 'norm_cfg': None},
        'embed_neck_cfg': {'type': 'mlp', 'act_cfg': None, 'norm_cfg': None},
        'fusion_cfg': {
            'type': 'aspp',
            'sep': True,
            'dilations': (1, 6, 12, 18),
            'pool': False,
            'act_cfg': dict(type='ReLU'),
            'norm_cfg': dict(type='BN', requires_grad=True)
        '''
        embed_dims = decoder_params['embed_dims']  # 256
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * len(self.in_index)  # [256, 256, 256, 256]

        embed_cfg = decoder_params['embed_cfg']  # dict(type='mlp', act_cfg=None, norm_cfg=None)
        embed_neck_cfg = decoder_params['embed_neck_cfg']  # dict(type='mlp', act_cfg=None, norm_cfg=None),
        if embed_neck_cfg == 'same_as_embed_cfg':
            embed_neck_cfg = embed_cfg
        fusion_cfg = decoder_params['fusion_cfg']   # dict(
                                                    # type='aspp',
                                                    # sep=True,
                                                    # dilations=(1, 6, 12, 18),
                                                    # pool=False,
                                                    # act_cfg=dict(type='ReLU'),
                                                    # norm_cfg=dict(type='BN', requires_grad=True))),
        for cfg in [embed_cfg, embed_neck_cfg, fusion_cfg]:
            if cfg is not None and 'aspp' in cfg['type']:
                cfg['align_corners'] = self.align_corners

        self.embed_layers = {}
        for i, in_channels, embed_dim in zip(self.in_index, self.in_channels, embed_dims):
            if i == self.in_index[-1]:
                self.embed_layers[str(i)] = build_layer(in_channels, embed_dim, **embed_neck_cfg)  # mlp
            else:
                self.embed_layers[str(i)] = build_layer(in_channels, embed_dim, **embed_cfg)  # mlp
        self.embed_layers = nn.ModuleDict(self.embed_layers)

        self.fuse_layer = build_layer(sum(embed_dims), self.channels, **fusion_cfg)  # aspp
        self.init_weights()
        self.init_conv_seg()
        
        self.debug = True
        self.debug_output = {}
        
        if pretrained is not None:
            self.pretrained = pretrained
            self.load_pretrained(pretrained)
        
    def load_pretrained(self, pretrained):
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'decoder' in checkpoint:
            state_dict = checkpoint['decoder']
        else:
            state_dict = checkpoint
            
        if len(state_dict) > len(self.state_dict()):
            # extract the dino v2 related params and remove the prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('decode_head.'):
                    new_k = k.replace('decode_head.', '')
                    new_state_dict[new_k] = v
            state_dict = new_state_dict
            
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, False)
        self.logger.info(f'Load decoder checkpoint from pretrained {pretrained}.')
        # print(f'Load decoder checkpoint from pretrained {pretrained}.')
        
        # 统计成功加载的参数
        loaded_keys = len(state_dict) - len(unexpected_keys)
        total_model_keys = len(self.state_dict())
        
        self.logger.info(f'Successfully loaded {loaded_keys}/{len(state_dict)} parameters from checkpoint')
        self.logger.info(f'Model has {total_model_keys} parameters total')
        # print(f'Successfully loaded {loaded_keys}/{len(state_dict)} parameters from checkpoint')
        # print(f'Model has {total_model_keys} parameters total')
        
        if missing_keys:
            self.logger.warning(f'Missing {len(missing_keys)} keys in checkpoint:')
            for key in missing_keys:  # 全部显示缺失的键
                self.logger.warning(f'  - {key}')
        
        if unexpected_keys:
            self.logger.warning(f'Unexpected {len(unexpected_keys)} keys in checkpoint:')
            for key in unexpected_keys:  # 全部显示意外的键
                self.logger.warning(f'  - {key}')
        
        if not missing_keys and not unexpected_keys:
            self.logger.info('Perfect match: all parameters loaded successfully!')

    def _build_loss_context(self, loss_config):
        """Build one named segmentation-loss context."""
        loss_type = loss_config.get('type', 'CrossEntropyLoss')
        ignore_index = loss_config.get('ignore_index', 255)

        if loss_type == 'CrossEntropyLoss':
            return {
                'mode': 'single',
                'loss_name': 'CrossEntropyLoss',
                'decode_loss': CrossEntropyLoss(ignore_index=ignore_index),
                'total_loss_weight': loss_config.get('loss_weight', 1.0),
                'loss_functions': {},
                'loss_weights': {},
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
                'loss_functions': {},
                'loss_weights': {},
            }

        if loss_type == 'CombinedLoss':
            losses_config = loss_config.get('losses', {})
            loss_functions = {}
            loss_weights = {}
            total_weight = 0.0

            if 'CrossEntropyLoss' in losses_config:
                ce_config = losses_config['CrossEntropyLoss']
                weight = ce_config.get('loss_weight', 1.0)
                loss_functions['CrossEntropyLoss'] = CrossEntropyLoss(
                    ignore_index=ignore_index)
                loss_weights['CrossEntropyLoss'] = weight
                total_weight += weight

            if 'DyCELoss' in losses_config:
                dyce_config = losses_config['DyCELoss']
                weight = dyce_config.get('loss_weight', 1.0)
                loss_functions['DyCELoss'] = DyCELoss(
                    ignore_index=ignore_index,
                    top_k_percent=dyce_config.get('top_k_percent', 0.2),
                    omega=dyce_config.get('omega', 0.5),
                    min_kept=dyce_config.get('min_kept', 1))
                loss_weights['DyCELoss'] = weight
                total_weight += weight

            if 'OhemCELoss' in losses_config:
                ohem_config = losses_config['OhemCELoss']
                weight = ohem_config.get('loss_weight', 1.0)
                loss_functions['OhemCELoss'] = OhemCELoss(
                    thresh=ohem_config.get('thresh', 0.7),
                    min_kept=ohem_config.get('min_kept', None),
                    ignore_index=ignore_index)
                loss_weights['OhemCELoss'] = weight
                total_weight += weight

            if 'DiceLoss' in losses_config:
                dice_config = losses_config['DiceLoss']
                weight = dice_config.get('loss_weight', 1.0)
                loss_functions['DiceLoss'] = DiceLoss(
                    smooth=dice_config.get('smooth', 1.0),
                    ignore_index=ignore_index)
                loss_weights['DiceLoss'] = weight
                total_weight += weight

            if 'FocalLoss' in losses_config:
                focal_config = losses_config['FocalLoss']
                weight = focal_config.get('loss_weight', 1.0)
                loss_functions['FocalLoss'] = FocalLoss(
                    gamma=focal_config.get('gamma', 2.0),
                    alpha=focal_config.get('alpha', None),
                    ignore_index=ignore_index)
                loss_weights['FocalLoss'] = weight
                total_weight += weight

            return {
                'mode': 'combined',
                'decode_loss': None,
                'total_loss_weight': total_weight,
                'loss_functions': loss_functions,
                'loss_weights': loss_weights,
            }

        raise ValueError(f"Unsupported loss type: {loss_type}")

    def _describe_loss_context(self, context):
        if context['mode'] == 'single':
            return f"{context.get('loss_name', 'Loss')}(w={context['total_loss_weight']:.2f})"
        loss_info = ', '.join(
            f'{name}(w={weight:.2f})'
            for name, weight in context['loss_weights'].items())
        return f"CombinedLoss: {loss_info}, total_weight={context['total_loss_weight']:.2f}"

    def _use_loss_context(self, context):
        self.loss_mode = context['mode']
        self.decode_loss = context['decode_loss']
        self.loss_functions = context['loss_functions']
        self.loss_weights = context['loss_weights']
        self.total_loss_weight = context['total_loss_weight']

    def _init_loss_functions(self):
        """Initialize source and unsupervised segmentation loss contexts."""
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

        self._use_loss_context(source_context)
        self.logger.info(
            f'{self.__class__.__name__}: source loss: '
            f'{self._describe_loss_context(source_context)}')
        if self.unsup_loss_config is not None:
            self.logger.info(
                f'{self.__class__.__name__}: unsup/mix loss: '
                f'{self._describe_loss_context(unsup_context)}')
        if self.target_loss_config is not None:
            self.logger.info(
                f'{self.__class__.__name__}: target labeled loss: '
                f'{self._describe_loss_context(target_context)}')

    def _get_loss_context(self, loss_key=None):
        key = 'default' if loss_key is None else str(loss_key).lower()
        if key not in self.loss_contexts:
            available = ', '.join(sorted(self.loss_contexts.keys()))
            raise KeyError(
                f'Unknown loss_key={loss_key!r}. Available loss keys: {available}')
        return self.loss_contexts[key]
    
    def cal_loss(self, seg_logits, seg_label, seg_weight=None, loss_key=None):
        """Calculate segmentation loss from logits and labels.

        根据分割 logits 和标签计算损失。
        
        Args:
            seg_logits (Tensor): The output of segmentation head. [B, C, H, W]
            seg_label (Tensor): The ground truth label. [B, H, W]
            seg_weight (Tensor, optional): The weight of each pixel. Defaults to None.
        
        Returns:
            dict: The calculated loss dictionary.
        """
        loss_dict = {}
        
        loss_context = self._get_loss_context(loss_key)
        loss_mode = loss_context['mode']
        
        if loss_mode == 'single':
            # 单一损失模式
            loss = loss_context['decode_loss'](seg_logits, seg_label, seg_weight)
            loss *= loss_context['total_loss_weight']
            loss_dict['seg_loss'] = loss
            
        elif loss_mode == 'combined':
            # 组合损失模式
            total_loss = 0.0
            loss_functions = loss_context['loss_functions']
            loss_weights = loss_context['loss_weights']
            
            for loss_name, loss_fn in loss_functions.items():
                # 计算各个损失
                if loss_name == 'CrossEntropyLoss':
                    individual_loss = loss_fn(seg_logits, seg_label, seg_weight)
                    loss_dict['ce_value'] = individual_loss * loss_weights[loss_name]
                    total_loss += loss_dict['ce_value']

                elif loss_name == 'DyCELoss':
                    individual_loss = loss_fn(seg_logits, seg_label, seg_weight)
                    loss_dict['dyce_value'] = individual_loss * loss_weights[loss_name]
                    total_loss += loss_dict['dyce_value']

                elif loss_name == 'OhemCELoss':
                    individual_loss = loss_fn(seg_logits, seg_label, seg_weight)
                    loss_dict['ohem_value'] = individual_loss * loss_weights[loss_name]
                    total_loss += loss_dict['ohem_value']
                    
                elif loss_name == 'DiceLoss':
                    individual_loss = loss_fn(seg_logits, seg_label)
                    loss_dict['dice_value'] = individual_loss * loss_weights[loss_name]
                    total_loss += loss_dict['dice_value']

                elif loss_name == 'FocalLoss':
                    individual_loss = loss_fn(seg_logits, seg_label)
                    loss_dict['focal_value'] = individual_loss * loss_weights[loss_name]
                    total_loss += loss_dict['focal_value']
            
            # 主要的分割损失（用于向后兼容）
            loss_dict['seg_loss'] = total_loss
            
        else:
            raise ValueError(f"Unknown loss mode: {loss_mode}")
        
        return loss_dict
        
    def init_conv_seg(self):
        """
        Initialize the conv_seg layer.
        """
        if self.conv_seg is not None:
            nn.init.normal_(self.conv_seg.weight, 0, 0.01)
            if self.conv_seg.bias is not None:
                nn.init.constant_(self.conv_seg.bias, 0)
            self.logger.info('Initialize conv_seg layer.')
    
    def init_weights(self):
        """
        Initialize the weights of module.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ModuleList):
                for conv in m:
                    if isinstance(conv, nn.Conv2d):
                        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                        if conv.bias is not None:
                            nn.init.constant_(conv.bias, 0)
                    elif isinstance(conv, (nn.LayerNorm, nn.BatchNorm2d)):
                        nn.init.constant_(conv.weight, 1)
                        nn.init.constant_(conv.bias, 0)
                    elif isinstance(conv, nn.Linear):
                        nn.init.normal_(conv.weight, 0, 0.01)
                        if conv.bias is not None:
                            nn.init.constant_(conv.bias, 0)
            elif isinstance(m, nn.ModuleDict):
                for conv in m.values():
                    if isinstance(conv, nn.Conv2d):
                        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
                        if conv.bias is not None:
                            nn.init.constant_(conv.bias, 0)
                    elif isinstance(conv, (nn.LayerNorm, nn.BatchNorm2d)):
                        nn.init.constant_(conv.weight, 1)
                        nn.init.constant_(conv.bias, 0)
                    elif isinstance(conv, nn.Linear):
                        nn.init.normal_(conv.weight, 0, 0.01)
                        if conv.bias is not None:
                            nn.init.constant_(conv.bias, 0)
            # self.logger.info(f'init ')
        
    def cls_seg(self, feat):
        """Classify each pixel from fused features.

        根据融合特征对每个像素分类。
        """
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        """Forward multi-level features to segmentation logits.

        将多层特征前向计算为分割 logits。

        Args:
            inputs (list[Tensor] | tuple): Either a list of feature maps or a
                tuple whose first item is the feature-map list. Each feature is
                expected in `N x C_i x H_i x W_i` format.
                可以是特征图列表，也可以是第一个元素为特征图列表的 tuple。
                每层特征形状为 `N x C_i x H_i x W_i`。

        Returns:
            Tensor: Segmentation logits with shape `N x num_classes x H x W`.
            返回形状为 `N x num_classes x H x W` 的分割 logits。
        """
        if isinstance(inputs, tuple):
            x = inputs[0]  # 0: multi-scale features, 1: multi-level but same scale features, 2: cls token
        else:
            x = inputs  # multi-level but same scale features
        
            if self.interpolate:
                x[0] = F.interpolate(x[0], scale_factor=4, mode='bilinear', align_corners=self.align_corners)
                x[1] = F.interpolate(x[1], scale_factor=2, mode='bilinear', align_corners=self.align_corners)
                x[3] = F.interpolate(x[3], scale_factor=0.5, mode='bilinear', align_corners=self.align_corners)
        
        n, _, _, _ = x[-1].shape

        os_size = x[0].size()[2:]  # 128 * 128
        
        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}', 'mmseg')
            _c[i] = self.embed_layers[str(i)](x[i])
            if _c[i].dim() == 3:  # (b, hw, c) -> (b, c ,hw)
                _c[i] = _c[i].permute(0, 2, 1).contiguous().reshape(n, -1, x[i].shape[2], x[i].shape[3])
            # mmcv.print_log(f'_c{i}: {_c[i].shape}', 'mmseg')
            if _c[i].size()[2:] != os_size:
                # mmcv.print_log(f'resize {i}', 'mmseg')
                _c[i] = resize(_c[i], size=os_size, mode='bilinear', align_corners=self.align_corners)

        x = self.fuse_layer(torch.cat(list(_c.values()), dim=1))
        x = self.cls_seg(x)

        return x

def save_model_params_summary(model, filename="param_snapshot.txt", show_values=20):
    with open(filename, "w") as f:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            data = param.data.view(-1).cpu().numpy()
            f.write(f"Name: {name}\n")
            f.write(f"  Shape: {list(param.shape)}\n")
            f.write(f"  Mean: {data.mean():.6f}, Std: {data.std():.6f}, Min: {data.min():.6f}, Max: {data.max():.6f}\n")
            shown_values = data[:min(len(data), show_values)]
            f.write(f"  Values (first {len(shown_values)}): {np.array2string(shown_values, precision=4, separator=', ')}\n")
            f.write("-" * 80 + "\n")
