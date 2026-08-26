#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapter + LoRA DINOv3 模块

结合两种参数高效微调(PEFT)方法:
1. Adapter: 通过CNN空间先验与ViT特征的双向交互增强特征表达
2. LoRA: 在注意力层的Q、K、V投影上添加低秩适配

主要特点:
- 模块化设计,易于参数管理和加载
- 支持DINOv3特有的rope编码和storage tokens
- 两种PEFT方法互补,Adapter负责空间增强,LoRA负责注意力微调
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from functools import partial
from torch.nn.init import normal_
from timm.layers import trunc_normal_

from .dino_v3 import DINOv3
from .peft import set_requires_grad, set_train, get_pyramid_feature
from .adapter_modules import SpatialPriorModule, InteractionBlockV3, deform_inputs
from .ops.modules import MSDeformAttn

work_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


class LoRAAdapter(nn.Module):
    """LoRA adapter for DINOv3 attention layers"""
    def __init__(
        self,
        embed_dim: int = 1024,
        num_layers: int = 24,
        lora_rank: int = 16,
        non_adapter_layers: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        target_modules: list = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.lora_rank = lora_rank
        self.non_adapter_layers = non_adapter_layers
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        if target_modules is None:
            target_modules = ['q', 'v']
        self.target_modules = target_modules
        
        valid_modules = ['q', 'k', 'v']
        for module in self.target_modules:
            if module not in valid_modules:
                raise ValueError(f"Invalid target module '{module}'. Valid options: {valid_modules}")
        
        self.valid_layers = num_layers - non_adapter_layers
        
        # 创建LoRA参数
        if 'q' in self.target_modules:
            self.lora_A_q = nn.ModuleList([
                nn.Linear(embed_dim, lora_rank, bias=False) 
                for _ in range(self.valid_layers)
            ])
            self.lora_B_q = nn.ModuleList([
                nn.Linear(lora_rank, embed_dim, bias=False) 
                for _ in range(self.valid_layers)
            ])
        
        if 'k' in self.target_modules:
            self.lora_A_k = nn.ModuleList([
                nn.Linear(embed_dim, lora_rank, bias=False) 
                for _ in range(self.valid_layers)
            ])
            self.lora_B_k = nn.ModuleList([
                nn.Linear(lora_rank, embed_dim, bias=False) 
                for _ in range(self.valid_layers)
            ])
        
        if 'v' in self.target_modules:
            self.lora_A_v = nn.ModuleList([
                nn.Linear(embed_dim, lora_rank, bias=False) 
                for _ in range(self.valid_layers)
            ])
            self.lora_B_v = nn.ModuleList([
                nn.Linear(lora_rank, embed_dim, bias=False) 
                for _ in range(self.valid_layers)
            ])
        
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        
        self.scaling = lora_alpha / lora_rank
        self._reset_parameters()
    
    def _reset_parameters(self):
        """重置LoRA参数"""
        for layer_idx in range(self.valid_layers):
            if hasattr(self, 'lora_A_q'):
                nn.init.kaiming_uniform_(self.lora_A_q[layer_idx].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_q[layer_idx].weight)
            
            if hasattr(self, 'lora_A_k'):
                nn.init.kaiming_uniform_(self.lora_A_k[layer_idx].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_k[layer_idx].weight)
            
            if hasattr(self, 'lora_A_v'):
                nn.init.kaiming_uniform_(self.lora_A_v[layer_idx].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_v[layer_idx].weight)
    
    def forward(self, x, layer_idx):
        """Apply LoRA to input features"""
        if layer_idx < self.non_adapter_layers:
            return None, None, None
        
        adjusted_layer = layer_idx - self.non_adapter_layers
        if adjusted_layer >= self.valid_layers:
            return None, None, None
        
        delta_q = None
        if 'q' in self.target_modules:
            delta_q = self.lora_B_q[adjusted_layer](
                self.lora_dropout(self.lora_A_q[adjusted_layer](x))
            ) * self.scaling
        
        delta_k = None
        if 'k' in self.target_modules:
            delta_k = self.lora_B_k[adjusted_layer](
                self.lora_dropout(self.lora_A_k[adjusted_layer](x))
            ) * self.scaling
        
        delta_v = None
        if 'v' in self.target_modules:
            delta_v = self.lora_B_v[adjusted_layer](
                self.lora_dropout(self.lora_A_v[adjusted_layer](x))
            ) * self.scaling
        
        return delta_q, delta_k, delta_v


class LoRAQKV(nn.Module):
    """LoRA版本的QKV层"""
    def __init__(self, original_qkv, layer_idx, embed_dim, non_adapter_layers):
        super().__init__()
        self.original_qkv = original_qkv
        self.layer_idx = layer_idx
        self.embed_dim = embed_dim
        self.non_adapter_layers = non_adapter_layers
        
        self.in_features = original_qkv.in_features
        self.out_features = original_qkv.out_features
        self.bias = original_qkv.bias
        
        for param in self.original_qkv.parameters():
            param.requires_grad = False
    
    def forward(self, x, adapter_func=None):
        qkv = self.original_qkv(x)
        
        if adapter_func is not None and self.layer_idx >= self.non_adapter_layers:
            delta_q, delta_k, delta_v = adapter_func(x, self.layer_idx)
            
            if delta_q is not None:
                qkv[:, :, :self.embed_dim] += delta_q
            if delta_k is not None:
                qkv[:, :, self.embed_dim:2*self.embed_dim] += delta_k
            if delta_v is not None:
                qkv[:, :, 2*self.embed_dim:] += delta_v
        
        return qkv
    
    @property
    def weight(self):
        return self.original_qkv.weight


class AdapterLoraInteractionBlockV3(InteractionBlockV3):
    """
    增强的交互块,结合Adapter和LoRA两种PEFT方法
    
    工作流程:
    1. Injector: 注入CNN空间先验到ViT特征
    2. ViT blocks + LoRA: 执行ViT块并应用LoRA微调
    3. Extractor: 从ViT提取增强特征回到CNN
    
    与InteractionBlockV3的区别:
    - 在ViT块执行过程中,LoRA自动应用于注意力层的QKV投影
    - 支持DINOv3的rope编码和storage tokens
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W, rope_embed=None):
        """
        前向传播,Adapter和LoRA自动协同工作
        
        Args:
            x: ViT token特征 (B, N_total, dim)
            c: CNN空间先验特征 (B, N_spatial, dim)
            blocks: DINOv3 transformer块(已集成LoRA)
            deform_inputs1: 注入步骤的可变形注意力输入
            deform_inputs2: 提取步骤的可变形注意力输入
            H, W: stride 16的特征图尺寸
            rope_embed: DINOv3的rope位置编码函数
            
        Returns:
            x: 增强的ViT特征
            c: 增强的CNN特征
        """
        # Step 1: Adapter注入 - 注入CNN空间先验到ViT patch特征
        if x.shape[1] > H * W:
            x_patch = x[:, -H*W:, :]
            x_non_patch = x[:, :-H*W, :]
        else:
            x_patch = x
            x_non_patch = None
            
        x_patch = self.injector(query=x_patch, reference_points=deform_inputs1[0],
                               feat=c, spatial_shapes=deform_inputs1[1],
                               level_start_index=deform_inputs1[2])
        
        # Step 2: ViT块处理 + LoRA微调(LoRA自动应用于注意力层)
        if x_non_patch is not None:
            x = torch.cat([x_non_patch, x_patch], dim=1)
        else:
            x = x_patch
            
        # 应用ViT块,LoRA已经集成在块的QKV层中
        for block in blocks:
            if rope_embed is not None:
                rope_sincos = rope_embed(H=H*16//16, W=W*16//16)
                x = block(x, rope_sincos)
            else:
                x = block(x)
        
        # Step 3: Adapter提取 - 提取ViT特征回到CNN
        if x.shape[1] > H * W:
            x_patch = x[:, -H*W:, :]
        else:
            x_patch = x
            
        c = self.extractor(query=c, reference_points=deform_inputs2[0], feat=x_patch, 
                          spatial_shapes=deform_inputs2[1], level_start_index=deform_inputs2[2], 
                          H=H, W=W)
        
        # Step 4: 可选的额外提取层
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0], feat=x_patch, 
                             spatial_shapes=deform_inputs2[1], level_start_index=deform_inputs2[2], 
                             H=H, W=W)
        
        return x, c


class AdapterLoraDINOv3Adapter(nn.Module):
    """
    结合Adapter + LoRA的DINOv3适配器模块
    
    整合:
    - 空间先验模块(SPM): 生成多尺度CNN特征
    - Adapter交互块: 双向特征交换
    - LoRA: 注意力层的低秩微调(通过外部传入,避免参数重复)
    
    与DINOv2版本的区别:
    - 使用InteractionBlockV3,支持rope编码
    - 正确处理DINOv3的storage tokens
    - 兼容DINOv3的token结构和处理流程
    """
    def __init__(self, adapter_config, embed_dim, norm_layer, drop_path_rate):
        super().__init__()
        self.cfg = adapter_config
        self.interaction_indexes = adapter_config['interaction_indexes']
        self.add_vit_feature = adapter_config['add_vit_feature']
        
        # Adapter配置
        conv_inplane = adapter_config['conv_inplane']
        deform_num_heads = adapter_config['deform_num_heads']
        n_points = adapter_config['n_points']
        init_values = adapter_config['init_values']
        with_cffn = adapter_config['with_cffn']
        cffn_ratio = adapter_config['cffn_ratio']
        deform_ratio = adapter_config['deform_ratio']
        use_extra_extractor = adapter_config['use_extra_extractor']
        with_cp = adapter_config['with_cp']

        # Level embedding
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        
        # CNN空间先验模块
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        
        # 增强的交互块(与LoRA协同)
        self.interactions = nn.Sequential(*[
            AdapterLoraInteractionBlockV3(
                dim=embed_dim,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                extra_extractor=((True if i == len(self.interaction_indexes) - 1 else False) and use_extra_extractor),
                with_cp=with_cp
            )
            for i in range(len(self.interaction_indexes))
        ])

        # 上采样和归一化层
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.adapter_norm1 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm2 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm3 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm4 = nn.SyncBatchNorm(embed_dim)

        # 权重初始化
        self._init_all_weights()
        normal_(self.level_embed)

    def _init_all_weights(self):
        """初始化所有权重"""
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
        
        self.apply(_init_weights)
        
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        """添加可学习的层级嵌入"""
        return c2 + self.level_embed[0], c3 + self.level_embed[1], c4 + self.level_embed[2]

    @torch.no_grad()
    def no_weight_decay(self):
        """返回不应用权重衰减的参数"""
        return set()

    def forward(self, x, backbone: DINOv3, masks=None):
        """
        结合Adapter和LoRA的前向传播
        
        Args:
            x: 输入图像 (B, 3, H, W)
            backbone: DINOv3骨干(已集成LoRA)
            masks: 注意力掩码 (B, L)
            
        Returns:
            pyramid_feats: 4层特征金字塔
            raw_feats: 交互阶段的原始ViT特征
            cls_token: 最终CLS token
        """
        # 生成可变形注意力输入
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # 提取多尺度CNN空间先验
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        B, _, h, w = x.shape
        H, W = h // backbone.patch_size, w // backbone.patch_size

        # 准备DINOv3 tokens
        x_tokens, (H, W) = backbone.prepare_tokens_with_masks(x, masks)
        
        vit_cls = x_tokens[:, :1, :]
        if backbone.n_storage_tokens > 0:
            storage_tokens = x_tokens[:, 1:1 + backbone.n_storage_tokens, :]
            x_patch_tokens = x_tokens[:, 1 + backbone.n_storage_tokens:, :]
        else:
            storage_tokens = None
            x_patch_tokens = x_tokens[:, 1:, :]
        
        _, _, dim = x_patch_tokens.shape
        vit_out_scales = []

        # 通过交互块处理(Adapter + LoRA自动协同)
        for i, layer in enumerate(self.interactions):
            blk_range = self.interaction_indexes[i]
            current_blocks = backbone.blocks[blk_range[0]: blk_range[-1] + 1]
            
            if storage_tokens is not None:
                current_tokens = torch.cat([vit_cls, storage_tokens, x_patch_tokens], dim=1)
            else:
                current_tokens = torch.cat([vit_cls, x_patch_tokens], dim=1)
            
            # Adapter交互 + LoRA微调(LoRA已集成在blocks中)
            current_tokens, c = layer(
                current_tokens, c,
                current_blocks,
                deform_inputs1, deform_inputs2, H, W,
                rope_embed=backbone.rope_embed
            )
            
            vit_cls = current_tokens[:, :1, :]
            if storage_tokens is not None:
                storage_tokens = current_tokens[:, 1:1 + backbone.n_storage_tokens, :]
                x_patch_tokens = current_tokens[:, 1 + backbone.n_storage_tokens:, :]
            else:
                x_patch_tokens = current_tokens[:, 1:, :]
            
            vit_out_scales.append(x_patch_tokens.transpose(1, 2).view(B, dim, H, W).contiguous())

        # 重建多尺度CNN特征
        c2_len = c2.size(1)
        c3_len = c3.size(1)
        c4_len = c4.size(1)

        c2_new = c[:, 0:c2_len, :]
        c3_new = c[:, c2_len:c2_len + c3_len, :]
        c4_new = c[:, c2_len + c3_len:c2_len + c3_len + c4_len, :]

        c2_new = c2_new.transpose(1, 2).view(B, dim, H * 2, W * 2).contiguous()
        c3_new = c3_new.transpose(1, 2).view(B, dim, H, W).contiguous()
        c4_new = c4_new.transpose(1, 2).view(B, dim, H // 2, W // 2).contiguous()
        c1_up = self.up(c2_new) + c1

        # 添加ViT特征
        if self.add_vit_feature and len(vit_out_scales) == len(self.interaction_indexes):
            if len(vit_out_scales) >= 4:
                x1, x2, x3, x4 = vit_out_scales[:4]
                x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
                x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
                x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
                c1_up, c2_new, c3_new, c4_new = c1_up + x1, c2_new + x2, c3_new + x3, c4_new + x4

        # 最终归一化
        f1 = self.adapter_norm1(c1_up)
        f2 = self.adapter_norm2(c2_new)
        f3 = self.adapter_norm3(c3_new)
        f4 = self.adapter_norm4(c4_new)
        
        return [f1, f2, f3, f4], vit_out_scales, vit_cls.squeeze(1)


class AdapterLoraDINOv3(DINOv3):
    """
    结合Adapter + LoRA的DINOv3模型
    
    整合两种PEFT方法:
    - Adapter: 通过CNN空间先验增强ViT特征
    - LoRA: 在注意力层的QKV投影上进行低秩微调
    
    模块化设计,易于参数管理:
    - 支持独立加载/保存adapter和lora参数
    - 支持DINOv3特有的rope编码和storage tokens
    """
    def __init__(self, backbone_config=None, pretrained=None):
        super().__init__(**backbone_config['dinov3_config'])
        self.logger = logging.getLogger()
        
        self.enable_adapter = True
        self.save_whole_backbone = False
        
        # 提取配置
        adapter_config = backbone_config['adapter_config']
        lora_config = backbone_config.get('lora_config', None)
        
        # 初始化LoRA适配器
        self.lora = None
        if lora_config is not None:
            self.lora = LoRAAdapter(**lora_config)
            # 替换attention层为LoRA版本
            self._replace_attention_layers()
        
        # 初始化Adapter模块
        self.adapter = AdapterLoraDINOv3Adapter(
            adapter_config=adapter_config,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer_cls,
            drop_path_rate=self.drop_path_rate
        )

        # 加载预训练权重
        if pretrained is not None:
            if isinstance(pretrained, dict):
                if 'dinov3' in pretrained:
                    self.load_dinov3_pretrained(pretrained['dinov3'])
                if 'adapter' in pretrained:
                    self.load_adapter_pretrained(pretrained['adapter'])
                if 'lora' in pretrained and self.lora is not None:
                    self.load_lora_pretrained(pretrained['lora'])
            else:
                self.load_dinov3_pretrained(pretrained)

        # 设置可训练参数
        self.train(True)

    def _replace_attention_layers(self):
        """替换attention层为LoRA版本"""
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx >= self.lora.non_adapter_layers:
                original_qkv = block.attn.qkv
                
                lora_qkv = LoRAQKV(
                    original_qkv=original_qkv,
                    layer_idx=layer_idx,
                    embed_dim=original_qkv.in_features,
                    non_adapter_layers=self.lora.non_adapter_layers
                )
                
                lora_qkv.forward = partial(lora_qkv.forward, adapter_func=self.lora.forward)
                block.attn.qkv = lora_qkv

    def load_dinov3_pretrained(self, pretrained):
        """加载DINOv3骨干预训练参数"""
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        self.logger.info(f'Loading dinov3 checkpoint from {pretrained}')
        
        loaded_keys = len(state_dict) - len(unexpected_keys)
        total_model_keys = len(self.state_dict())
        
        self.logger.info(f'Successfully loaded {loaded_keys}/{len(state_dict)} parameters from checkpoint')
        self.logger.info(f'Model has {total_model_keys} parameters total')
        
        if missing_keys:
            self.logger.warning(f'Missing {len(missing_keys)} keys in checkpoint')
        if unexpected_keys:
            self.logger.warning(f'Unexpected {len(unexpected_keys)} keys in checkpoint')
        if not missing_keys and not unexpected_keys:
            self.logger.info('Perfect match: all parameters loaded successfully!')

    def load_adapter_pretrained(self, pretrained, strict=False):
        """加载Adapter预训练权重"""
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'adapter' in checkpoint:
            state_dict = checkpoint['adapter']
        else:
            state_dict = checkpoint
            
        adapter_state_dict = {}
        for k, v in state_dict.items():
            original_key = k
            if k.startswith('adapter.'):
                k = k.replace('adapter.', '')
            elif k.startswith('backbone.adapter.'):
                k = k.replace('backbone.adapter.', '')
            
            if any(component in original_key for component in ['adapter', 'spm', 'level_embed', 'interactions', 'up', 'adapter_norm']):
                adapter_state_dict[k] = v
        
        if adapter_state_dict:
            missing_keys, unexpected_keys = self.adapter.load_state_dict(adapter_state_dict, strict=strict)
            self.logger.info(f'Loaded adapter checkpoint from {pretrained}')
            loaded_keys = len(state_dict) - len(unexpected_keys)
            
            self.logger.info(f'Successfully loaded {loaded_keys}/{len(state_dict)} parameters')
            
            if not missing_keys and not unexpected_keys:
                self.logger.info('Perfect match: all adapter parameters loaded!')
        else:
            self.logger.warning(f'No adapter parameters found in {pretrained}')

    def load_lora_pretrained(self, pretrained, strict=False):
        """加载LoRA预训练权重"""
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'lora' in checkpoint:
            state_dict = checkpoint['lora']
        else:
            state_dict = checkpoint
            
        lora_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            prefixes_to_remove = ['lora.', 'backbone.lora.', 'model.lora.']
            
            for prefix in prefixes_to_remove:
                if new_key.startswith(prefix):
                    new_key = new_key.replace(prefix, '', 1)
                    break
            
            lora_state_dict[new_key] = v
        
        if lora_state_dict and self.lora is not None:
            missing_keys, unexpected_keys = self.lora.load_state_dict(lora_state_dict, strict=strict)
            self.logger.info(f'Loaded lora checkpoint from {pretrained}')
            loaded_keys = len(state_dict) - len(unexpected_keys)
            
            self.logger.info(f'Successfully loaded {loaded_keys}/{len(state_dict)} parameters')
            
            if not missing_keys and not unexpected_keys:
                self.logger.info('Perfect match: all lora parameters loaded!')
        else:
            self.logger.warning(f'No lora parameters found in {pretrained}')

    def save_adapter(self, path):
        """保存Adapter+LoRA参数"""
        save_dict = {'adapter': self.adapter.state_dict()}
        if self.lora is not None:
            save_dict['lora'] = self.lora.state_dict()
        torch.save(save_dict, path)
        self.logger.info(f'[AdapterLoraDINOv3] Adapter+LoRA saved to {path}')

    def forward(self, x, masks=None):
        """前向传播"""
        if not self.enable_adapter:
            outs = super().forward_features(x, masks)
            return get_pyramid_feature(outs), outs, None
        
        feats, vit_scales, cls_token = self.adapter(x, self, masks)
        return feats, vit_scales, cls_token

    def train(self, mode: bool = True):
        """设置训练模式,冻结骨干,启用adapter和lora"""
        if not mode:
            return super().train(mode)
        
        super().train(mode)
        
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 解冻adapter参数
        for param in self.adapter.parameters():
            param.requires_grad = True
        self.adapter.train(True)
        
        # 解冻lora参数
        if self.lora is not None:
            for param in self.lora.parameters():
                param.requires_grad = True
            self.lora.train(True)
        
        return self


if __name__ == '__main__':
    torch.manual_seed(0)

    backbone_cfg = {
        'adapter_config': {
            'interaction_indexes': [[0, 7], [8, 11], [12, 15], [16, 23]],
            'add_vit_feature': True,
            'conv_inplane': 64,
            'deform_num_heads': 16,
            'n_points': 4,
            'init_values': 0,
            'with_cffn': True,
            'cffn_ratio': 0.25,
            'deform_ratio': 0.5,
            'use_extra_extractor': True,
            'with_cp': False,
        },
        'lora_config': {
            'embed_dim': 1024,
            'num_layers': 24,
            'lora_rank': 16,
            'non_adapter_layers': 0,
            'lora_alpha': 16.0,
            'lora_dropout': 0.0,
            'target_modules': ['q', 'v'],
        },
        'dinov3_config': {
            'img_size': 512,
            'patch_size': 16,
            'pos_embed_rope_rescale_coords': 2.0,
            'pos_embed_rope_dtype': 'fp32',
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'ffn_ratio': 4.0,
            'qkv_bias': True,
            'layerscale_init': 1e-5,
            'ffn_layer': 'mlp',
            'ffn_bias': True,
            'proj_bias': True,
            'n_storage_tokens': 4,
            'mask_k_bias': True,
            'out_indices': [7, 11, 15, 23],
        },
    }

    pretrained_backbone = os.path.join(work_root, 'pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    if not os.path.isfile(pretrained_backbone):
        pretrained_backbone = None
        print('Warn: backbone pretrained not found, using random init.')

    model = AdapterLoraDINOv3(backbone_config=backbone_cfg, 
                              pretrained={'dinov3': pretrained_backbone} if pretrained_backbone else None)
    model.cuda().train(True)

    # 检查可训练参数
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f'Trainable params count: {len(trainable_names)} (should include both adapter and lora)')
    print(f'Sample trainable names (first 10): {trainable_names[:10]}')

    # 前向传播
    x = torch.randn(2, 3, 512, 512).cuda()
    feats, vit_feats, cls_token = model(x)

    print('\nPyramid (adapter+lora output) feats:')
    for i, f in enumerate(feats):
        print(f'  feat[{i}] = {tuple(f.shape)}')

    print('\nViT interaction scale feats:')
    for i, f in enumerate(vit_feats):
        print(f'  vit_scale[{i}] = {tuple(f.shape)}')

    print('\nCLS token shape:', tuple(cls_token.shape))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_m = total_params / 1e6
    trainable_params_m = trainable_params / 1e6
    
    print(f'\nTotal params      : {total_params} ({total_params_m:.2f}M)')  # 327963840 (327.96M)
    print(f'Trainable params  : {trainable_params} ({trainable_params_m:.2f}M)')  # 24809664 (24.81M)
    print(f'Trainable ratio   : {trainable_params / total_params * 100:.4f}%')  # 7.5648%

    # 保存和加载测试
    save_path = 'adapter_lora_v3_test.pth'
    model.save_adapter(save_path)
    print(f'\nSaved adapter+lora -> {save_path}')
    
    # 清理测试文件
    os.remove(save_path)
    print(f'Cleaned up test file: {save_path}')