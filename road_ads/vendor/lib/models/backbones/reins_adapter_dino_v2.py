#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
from timm.layers import trunc_normal_

from .dino_v2 import DINOv2
from .peft import set_requires_grad, set_train, get_pyramid_feature
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs
from .reins import LoRAReins
from .ops.modules import MSDeformAttn


class ReinsAdapterDINOv2Adapter(nn.Module):
    """
    并行 Reins + Adapter 模块，包含两个独立的特征修正分支：
    1. Reins 分支：对 ViT 特征进行修正
    2. VitAdapter 分支：通过空间先验进行特征交互
    
    两个分支独立并行处理，最后分别输出各自的结果。
    """
    
    def __init__(self, adapter_config, reins_config, embed_dim, norm_layer, drop_path_rate):
        super().__init__()
        self.cfg = adapter_config
        self.reins_cfg = reins_config
        self.interaction_indexes = adapter_config['interaction_indexes']
        self.add_vit_feature = adapter_config['add_vit_feature']
        
        # Adapter 分支配置
        conv_inplane = adapter_config['conv_inplane']
        deform_num_heads = adapter_config['deform_num_heads']
        n_points = adapter_config['n_points']
        init_values = adapter_config['init_values']
        with_cffn = adapter_config['with_cffn']
        cffn_ratio = adapter_config['cffn_ratio']
        deform_ratio = adapter_config['deform_ratio']
        use_extra_extractor = adapter_config['use_extra_extractor']
        with_cp = adapter_config['with_cp']

        # === Reins 分支 ===
        self.reins = LoRAReins(**reins_config) if reins_config is not None else None

        # === VitAdapter 分支 ===
        # Level embedding for multi-scale features
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        
        # CNN spatial prior module
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        
        # Interaction blocks
        self.interactions = nn.Sequential(*[
            InteractionBlock(
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

        # Upsample + norms for VitAdapter
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.adapter_norm1 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm2 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm3 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm4 = nn.SyncBatchNorm(embed_dim)

        # 初始化权重
        self._init_all_weights()
        normal_(self.level_embed)

    def _init_all_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
        self.apply(_init_weights)
        # deform attn 重置
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        return c2 + self.level_embed[0], c3 + self.level_embed[1], c4 + self.level_embed[2]

    @torch.no_grad()
    def no_weight_decay(self):
        return set()

    def forward(self, x, backbone: DINOv2, masks=None):
        """
        并行处理 Reins 和 VitAdapter 两个分支
        
        Args:
            x: [B,3,H,W] 输入图像
            backbone: DINOv2 主干网络
            masks: [B, L] 或 None
            
        Returns:
            tuple: (pyramid_feats, vit_scales, cls_tokens) 其中：
                - pyramid_feats: [reins_pyramid_feats, adapter_pyramid_feats] 
                - vit_scales: [reins_vit_feats, adapter_vit_out_scales] (Reins分支返回原始ViT特征)
                - cls_tokens: [reins_cls_token, adapter_cls_token]
        """
        B, _, h, w = x.shape
        H, W = h // backbone.patch_size, w // backbone.patch_size

        # 准备输入 tokens
        x_tokens = backbone.prepare_tokens_with_masks(x, masks)  # [B,1+N,(+reg)?,C]
        
        # === 分支 1: Reins 处理 ===
        reins_x_tokens = x_tokens.clone()  # 复制一份用于 Reins 分支
        reins_vit_feats = []
        
        # Reins 分支：逐层处理 ViT blocks + Reins 修正
        for idx, blk in enumerate(backbone.blocks):
            reins_x_tokens = blk(reins_x_tokens)
            # 应用 Reins 修正（如果在有效层范围内）
            if self.reins is not None and idx >= self.reins.non_adapter_layers:
                reins_x_tokens = self.reins.forward(
                    reins_x_tokens, idx, 
                    batch_first=True, 
                    has_cls_token=True,
                    num_register_token=backbone.num_register_tokens
                )
            
            # 在输出层保存特征
            if idx in backbone.out_indices:
                # 提取 patch tokens (去掉 cls 和 register tokens)
                if backbone.num_register_tokens > 0:
                    patch_tokens = reins_x_tokens[:, 1 + backbone.num_register_tokens:, :]
                else:
                    patch_tokens = reins_x_tokens[:, 1:, :]
                reins_feat = patch_tokens.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                reins_vit_feats.append(reins_feat)
        
        # Reins 分支的金字塔特征
        reins_pyramid_feats = get_pyramid_feature(reins_vit_feats)
        reins_cls_token = reins_x_tokens[:, 0, :]

        # === 分支 2: VitAdapter 处理 ===
        # 准备 deformable attention 输入
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # CNN 空间先验
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)  # [B, (4096+1024+256), C]

        # ViT patch tokens (去掉 cls 用于交互)
        adapter_x_tokens = x_tokens.clone()  # 复制一份用于 Adapter 分支
        vit_cls = adapter_x_tokens[:, :1, :]
        if backbone.num_register_tokens > 0:
            reg_tokens = adapter_x_tokens[:, 1:1 + backbone.num_register_tokens, :]
            x_patch_tokens = adapter_x_tokens[:, 1 + backbone.num_register_tokens:, :]
        else:
            reg_tokens = None
            x_patch_tokens = adapter_x_tokens[:, 1:, :]
        _, _, dim = x_patch_tokens.shape

        adapter_vit_out_scales = []
        
        # 逐 interaction block 处理一段 backbone blocks
        for i, layer in enumerate(self.interactions):
            blk_range = self.interaction_indexes[i]
            current_blocks = backbone.blocks[blk_range[0]: blk_range[-1] + 1]
            
            # 交互：x_patch_tokens 与 c 互相注入/提取
            x_patch_tokens, c, vit_cls, reg_tokens = layer(
                x_patch_tokens, c,
                current_blocks,
                deform_inputs1, deform_inputs2, H, W,
                vit_cls=vit_cls,
                reg_tokens=reg_tokens
            )
            adapter_vit_out_scales.append(x_patch_tokens.transpose(1, 2).view(B, dim, H, W).contiguous())

        # 拆分 c 回到不同尺度
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

        # 可选：添加 ViT 特征
        if self.add_vit_feature and len(adapter_vit_out_scales) == 4:
            x1, x2, x3, x4 = adapter_vit_out_scales
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1_up, c2_new, c3_new, c4_new = c1_up + x1, c2_new + x2, c3_new + x3, c4_new + x4

        # VitAdapter 分支的最终特征
        f1 = self.adapter_norm1(c1_up)
        f2 = self.adapter_norm2(c2_new)
        f3 = self.adapter_norm3(c3_new)
        f4 = self.adapter_norm4(c4_new)
        adapter_pyramid_feats = [f1, f2, f3, f4]
        adapter_cls_token = vit_cls.squeeze(1)

        # 返回两个分支的结果，格式：[pyramid_feats, vit_scales, cls_token]
        # reins_branch: [reins_pyramid_feats, reins_vit_feats, reins_cls_token]  
        # adapter_branch: [adapter_pyramid_feats, adapter_vit_out_scales, adapter_cls_token]
        return [reins_pyramid_feats, adapter_pyramid_feats], [reins_vit_feats, adapter_vit_out_scales], [reins_cls_token, adapter_cls_token]


class ReinsAdapterDINOv2(DINOv2):
    """
    并行 Reins + Adapter DINOv2 模型
    
    该模型将 Reins 和 VitAdapter 作为两个独立的并行分支：
    - Reins 分支：对 ViT 特征进行逐层修正
    - VitAdapter 分支：通过空间先验进行特征交互
    - 两个分支独立处理，最后分别输出结果
    """
    
    def __init__(self, backbone_config=None, pretrained=None):
        super().__init__(**backbone_config['dinov2_config'])
        self.logger = logging.getLogger()
        
        self.enable_adapter = True
        self.save_whole_backbone = False
        
        # 创建并行的 Reins + Adapter 模块
        adapter_config = backbone_config['adapter_config']
        reins_config = backbone_config['reins_config']
        self.adapter = ReinsAdapterDINOv2Adapter(
            adapter_config=adapter_config,
            reins_config=reins_config,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer,
            drop_path_rate=self.drop_path_rate
        )

        # 加载预训练权重
        if pretrained is not None:
            if isinstance(pretrained, dict):
                if 'dinov2' in pretrained:
                    self.load_dinov2_pretrained(pretrained['dinov2'])
                if 'adapter' in pretrained:
                    self.load_adapter_pretrained(pretrained['adapter'])
            else:
                self.load_dinov2_pretrained(pretrained)

        # 设置可训练参数
        self.train(True)

    def load_dinov2_pretrained(self, pretrained):
        """加载 DINOv2 backbone 预训练权重"""
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        self.logger.info(f'Loading dinov2 checkpoint from {pretrained}')
        
        loaded_keys = len(state_dict) - len(unexpected_keys)
        total_model_keys = len(self.state_dict())
        self.logger.info(f'Successfully loaded {loaded_keys}/{len(state_dict)} parameters from checkpoint')
        self.logger.info(f'Model has {total_model_keys} parameters total')
        
        if missing_keys:
            self.logger.info(f'Missing {len(missing_keys)} keys (expected for adapter/reins parameters)')
            for key in missing_keys:
                self.logger.warning(f'  - {key}')
                
        if unexpected_keys:
            self.logger.warning(f'Unexpected {len(unexpected_keys)} keys in checkpoint')
            for key in unexpected_keys:
                self.logger.warning(f'  - {key}')
        
        if not missing_keys and not unexpected_keys:
            self.logger.info('Perfect match: all parameters loaded successfully!')

    def load_adapter_pretrained(self, pretrained, strict=False):
        """加载 adapter 相关的预训练权重，包括 VitAdapter 和 Reins 分支"""
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'adapter' in checkpoint:
            state_dict = checkpoint['adapter']
        else:
            state_dict = checkpoint
            
        # 处理所有 adapter 相关的参数
        adapter_state_dict = {}
        for k, v in state_dict.items():
            # 统一处理各种前缀格式
            original_key = k
            if k.startswith('adapter.'):
                k = k.replace('adapter.', '')
            elif k.startswith('backbone.adapter.'):
                k = k.replace('backbone.adapter.', '')
            elif k.startswith('backbone.'):
                k = k.replace('backbone.', '')
            
            # 保留所有 adapter 相关的参数（包括 reins）
            if any(component in original_key for component in ['adapter', 'reins', 'spm', 'level_embed', 'interactions', 'up', 'adapter_norm']):
                adapter_state_dict[k] = v
        
        if adapter_state_dict:
            missing, unexpected = self.adapter.load_state_dict(adapter_state_dict, strict=strict)
            self.logger.info(f'Loaded adapter parameters (VitAdapter + Reins) from {pretrained}')
            self.logger.info(f'  Missing: {len(missing)} keys, Unexpected: {len(unexpected)} keys')
            
            if missing and len(missing) <= 10:  # 只显示少量缺失的键
                for key in missing:
                    self.logger.debug(f'    Missing: {key}')
            elif missing:
                self.logger.debug(f'    Missing keys (first 5): {missing[:5]}...')
                
            if unexpected and len(unexpected) <= 10:  # 只显示少量意外的键
                for key in unexpected:
                    self.logger.debug(f'    Unexpected: {key}')
            elif unexpected:
                self.logger.debug(f'    Unexpected keys (first 5): {unexpected[:5]}...')

    def save_adapter(self, path):
        """保存所有 adapter 相关的参数"""
        torch.save({
            'adapter': self.adapter.state_dict()
        }, path)
        self.logger.info(f'Adapter (including Reins + VitAdapter) saved to {path}')

    def forward(self, x, masks=None):
        """
        前向传播，返回并行处理的结果
        
        Returns:
            tuple: (pyramid_feats, vit_scales, cls_tokens) 其中：
                - pyramid_feats: [reins_pyramid_feats, adapter_pyramid_feats]
                - vit_scales: [reins_vit_feats, adapter_vit_out_scales] 
                - cls_tokens: [reins_cls_token, adapter_cls_token]
        """
        return self.adapter(x, self, masks)

    def train(self, mode: bool = True):
        """设置训练模式，只有 adapter 参数可训练"""
        if not mode:
            return super().train(mode)
        
        set_requires_grad(self, ["adapter"])
        set_train(self, ["adapter"])
        return self


