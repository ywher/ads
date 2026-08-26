# Obtained from: https://github.com/NVlabs/SegFormer
# Modifications: Model construction with loop, added loss functions and pretrained loading
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_segformer
"""SegFormer decode head.

SegFormer 解码头。

This head projects each selected backbone feature with an MLP, resizes all
levels to the highest feature resolution, fuses them with a convolution, and
predicts semantic logits.

该解码头用 MLP 投影每个被选中的 backbone 特征层，将所有层 resize 到最高
特征分辨率，通过卷积融合后预测语义分割 logits。
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.loss.losses import CrossEntropyLoss, DiceLoss, DyCELoss, FocalLoss, OhemCELoss
from lib.models.model_utils.funcs import resize
from lib.models.model_utils.cna import ConvModule, MLP


class SegFormerHead(nn.Module):
    """SegFormer MLP decoder for semantic segmentation.

    用于语义分割的 SegFormer MLP decoder。
    """

    def __init__(self, decoder_config, pretrained=None):
        super(SegFormerHead, self).__init__()
        self.logger = logging.getLogger()
        self.input_transform = 'multiple_select'
        
        # 基本配置
        self.in_channels = decoder_config['in_channels']  # e.g., [64, 128, 320, 512] or [1024, 1024, 1024, 1024]
        self.in_index = decoder_config['in_index']  # [0, 1, 2, 3]
        self.channels = decoder_config['channels']  # 256
        self.dropout_ratio = decoder_config['dropout_ratio']  # 0.1
        self.num_classes = decoder_config['num_classes']  # 19
        self.align_corners = decoder_config.get('align_corners', False)
        self.interpolate = decoder_config.get('interpolate', True)
        self.norm_cfg = decoder_config['norm_cfg']  # dict(type='BN', requires_grad=True)
        
        # 损失配置
        self.loss_config = decoder_config['loss_decode']
        self.unsup_loss_config = decoder_config.get('unsup_loss_decode', None)
        self.target_loss_config = decoder_config.get('target_loss_decode', None)
        self._init_loss_functions()
        
        # 解码器参数
        decoder_params = decoder_config['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        conv_kernel_size = decoder_params.get('conv_kernel_size', 1)

        # MLP layers for each scale
        self.linear_c = {}
        for i, in_ch in zip(self.in_index, self.in_channels):
            self.linear_c[str(i)] = MLP(input_dim=in_ch, embed_dim=embedding_dim)
        self.linear_c = nn.ModuleDict(self.linear_c)

        # Fusion layer
        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * len(self.in_index),
            out_channels=embedding_dim,
            kernel_size=conv_kernel_size,
            padding=0 if conv_kernel_size == 1 else conv_kernel_size // 2,
            norm_cfg=self.norm_cfg)

        # Dropout
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout = None

        # Prediction layer
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        
        # Initialize weights
        self.init_weights()
        
        # Debug
        self.debug = True
        self.debug_output = {}
        
        # Load pretrained
        if pretrained is not None:
            self.pretrained = pretrained
            self.load_pretrained(pretrained)

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

    def init_weights(self):
        """Initialize the weights of module."""
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
        self.logger.info('SegFormerHead: Initialize weights.')

    def load_pretrained(self, pretrained):
        """Load pretrained weights."""
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'decoder' in checkpoint:
            state_dict = checkpoint['decoder']
        else:
            state_dict = checkpoint
            
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        self.logger.info(f'SegFormerHead: Load decoder checkpoint from pretrained {pretrained}.')
        
        # 统计成功加载的参数
        loaded_keys = len(state_dict) - len(unexpected_keys)
        total_model_keys = len(self.state_dict())
        
        self.logger.info(f'Successfully loaded {loaded_keys}/{len(state_dict)} parameters from checkpoint')
        self.logger.info(f'Model has {total_model_keys} parameters total')
        
        if missing_keys:
            self.logger.warning(f'Missing {len(missing_keys)} keys in checkpoint:')
            for key in missing_keys:
                self.logger.warning(f'  - {key}')
        
        if unexpected_keys:
            self.logger.warning(f'Unexpected {len(unexpected_keys)} keys in checkpoint:')
            for key in unexpected_keys:
                self.logger.warning(f'  - {key}')
        
        if not missing_keys and not unexpected_keys:
            self.logger.info('Perfect match: all parameters loaded successfully!')

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
            loss = loss_context['decode_loss'](seg_logits, seg_label, seg_weight)
            loss *= loss_context['total_loss_weight']
            loss_dict['seg_loss'] = loss
            
        elif loss_mode == 'combined':
            total_loss = 0.0
            loss_functions = loss_context['loss_functions']
            loss_weights = loss_context['loss_weights']
            
            for loss_name, loss_fn in loss_functions.items():
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
            
            loss_dict['seg_loss'] = total_loss
            
        else:
            raise ValueError(f"Unknown loss mode: {loss_mode}")
        
        return loss_dict

    def forward(self, inputs):
        """Forward selected feature levels to segmentation logits.

        将选中的特征层前向计算为分割 logits。

        Args:
            inputs (list[Tensor] | tuple): Feature maps in
                `N x C_i x H_i x W_i` format, or a tuple whose first item is
                the feature-map list.
                `N x C_i x H_i x W_i` 格式的特征图列表，或第一个元素为
                特征图列表的 tuple。

        Returns:
            Tensor: Segmentation logits at the highest feature resolution.
            返回最高特征分辨率下的分割 logits。
        """
        if isinstance(inputs, tuple):
            x = inputs[0]  # multi-scale features (已经是多尺度)
        else:
            x = inputs
            # 只在非 tuple 输入时（即单尺度输入）才做 interpolate
            # 这样可以将同尺度特征转换为多尺度特征
            if self.interpolate:
                x = list(x)  # 转为列表以便修改
                x[0] = F.interpolate(x[0], scale_factor=4, mode='bilinear', align_corners=self.align_corners)
                x[1] = F.interpolate(x[1], scale_factor=2, mode='bilinear', align_corners=self.align_corners)
                x[3] = F.interpolate(x[3], scale_factor=0.5, mode='bilinear', align_corners=self.align_corners)
        
        n, _, h, w = x[-1].shape

        _c = {}
        for i in self.in_index:
            _c[i] = self.linear_c[str(i)](x[i]).permute(0, 2, 1).contiguous()
            _c[i] = _c[i].reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if i != 0:
                _c[i] = resize(
                    _c[i],
                    size=x[0].size()[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)

        _c = self.linear_fuse(torch.cat(list(_c.values()), dim=1))

        if self.dropout is not None:
            x = self.dropout(_c)
        else:
            x = _c
        x = self.linear_pred(x)

        return x


if __name__ == '__main__':
    # Test SegFormerHead
    # in_channels = [64, 128, 320, 512]  # MiT-B5
    in_channels = [1024, 1024, 1024, 1024]  # DINOv2
    
    config = {
        'in_channels': in_channels,
        'in_index': [0, 1, 2, 3],
        'channels': 256,
        'dropout_ratio': 0.1,
        'num_classes': 19,
        'norm_cfg': dict(type='BN', requires_grad=True),
        'align_corners': False,
        'interpolate': False,
        'decoder_params': {
            'embed_dim': 256,
            'conv_kernel_size': 1,
        },
        'loss_decode': {
            'type': 'CrossEntropyLoss', 
            'use_sigmoid': False, 
            'loss_weight': 1.0
        }
    }
    
    model = SegFormerHead(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"total params: {total_params}")  # 1317139
    print(f"trainable params: {trainable_params}")  # 1317139
    print(f"non-trainable params: {non_trainable_params}")  # 0
    print(f"trainable params ratio: {(trainable_params / total_params)*100:.4f}%")  # 100%
    
    # Test with multi-scale features (interpolate=False, so output size = first feature size)
    x = []
    x.append(torch.randn(2, in_channels[0], 128, 128))  # highest resolution
    x.append(torch.randn(2, in_channels[1], 64, 64))
    x.append(torch.randn(2, in_channels[2], 32, 32))
    x.append(torch.randn(2, in_channels[3], 16, 16))  # lowest resolution
    
    seg_logits = model(x)
    print(f'seg_logits.shape: {seg_logits.shape}')  # [2, 19, 128, 128]
    
    # Test with tuple input
    x2 = []
    x2.append(torch.randn(2, in_channels[0], 128, 128))
    x2.append(torch.randn(2, in_channels[1], 64, 64))
    x2.append(torch.randn(2, in_channels[2], 32, 32))
    x2.append(torch.randn(2, in_channels[3], 16, 16))
    cls_token = torch.randn(2, in_channels[-1])
    inputs = (x2, x2, cls_token)
    
    seg_logits = model(inputs)
    print(f'seg_logits.shape with tuple input: {seg_logits.shape}')
    
    # Test loss calculation (seg_label size should match seg_logits spatial size)
    seg_label = torch.randint(0, 19, (2, 128, 128)).long()
    loss_dict = model.cal_loss(seg_logits, seg_label)
    print(f'loss_dict: {loss_dict}')
