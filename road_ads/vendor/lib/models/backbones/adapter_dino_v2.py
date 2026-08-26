import logging
import math
import os
import sys

import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from timm.layers import trunc_normal_

from .dino_v2 import DINOv2
from .peft import set_requires_grad, set_train, get_pyramid_feature
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs
from .ops.modules import MSDeformAttn

class DINOv2Adapter(nn.Module):
    """
    将原先散落在 AdapterDINOv2 里的适配组件(SpatialPrior / InteractionBlocks / Upsample / Norms)
    打包成独立模块，便于单独保存 & 加载(adapter.state_dict())。
    """
    def __init__(self, adapter_config, embed_dim, norm_layer, drop_path_rate):
        super().__init__()
        self.cfg = adapter_config
        self.interaction_indexes = adapter_config['interaction_indexes']  # [[0, 7], [8, 11], [12, 15], [16, 23]]
        self.add_vit_feature = adapter_config['add_vit_feature']  # True
        conv_inplane = adapter_config['conv_inplane']  # 64
        deform_num_heads = adapter_config['deform_num_heads']  # 16
        n_points = adapter_config['n_points']  # 4
        init_values = adapter_config['init_values']  # 0
        with_cffn = adapter_config['with_cffn']  # True
        cffn_ratio = adapter_config['cffn_ratio']  # 0.25
        deform_ratio = adapter_config['deform_ratio']  # 0.5
        use_extra_extractor = adapter_config['use_extra_extractor']  # True
        with_cp = adapter_config['with_cp']  # False

        # level embedding
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))  # 1014 0r 768
        # CNN spatial prior
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        # interaction blocks
        self.interactions = nn.Sequential(*[
            InteractionBlock(
                dim=embed_dim,  # 1024
                num_heads=deform_num_heads,  # 16
                n_points=n_points,  # 4
                init_values=init_values,  # 0
                drop_path=drop_path_rate,  # 0.
                norm_layer=norm_layer,  # LN
                with_cffn=with_cffn,  # True
                cffn_ratio=cffn_ratio,  # 0.25
                deform_ratio=deform_ratio,  # 0.5
                extra_extractor=((True if i == len(self.interaction_indexes) - 1 else False) and use_extra_extractor),  # False False False True
                with_cp=with_cp  # False
            )
            for i in range(len(self.interaction_indexes))
        ])

        # upsample + norms
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.adapter_norm1 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm2 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm3 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm4 = nn.SyncBatchNorm(embed_dim)

        # init
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
        # 若有需要，可排除某些参数的正则
        return set()

    def forward(self, x, backbone: DINOv2, masks=None):
        """
        Args:
            x: [B,3,H,W]
            backbone: 已初始化并加载预训练权重的 DINOv2
            masks: [B, L] 或 None
        Returns:
            pyramid_feats (list[Tensor]): 4 层金字塔特征 (B,C,H/4,W/4 ... )
            raw_feats (list[Tensor]): 交互阶段提取的 ViT 特征 (可选)
            cls_token (Tensor): 最终 cls token (B,C)
        """
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # CNN SpatialPrior
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)  # [B, (4096+1024+256), C] (flatten 后)

        B, _, h, w = x.shape
        H, W = h // backbone.patch_size, w // backbone.patch_size

        # ViT patch tokens (去掉 cls 后参与交互)
        x_tokens = backbone.prepare_tokens_with_masks(x, masks)  # [B,1+N,(+reg)?,C]
        vit_cls = x_tokens[:, :1, :]
        if backbone.num_register_tokens > 0:
            # 兼容 register token
            reg_tokens = x_tokens[:, 1:1 + backbone.num_register_tokens, :]
            x_patch_tokens = x_tokens[:, 1 + backbone.num_register_tokens:, :]
        else:
            reg_tokens = None
            x_patch_tokens = x_tokens[:, 1:, :]
        _, _, dim = x_patch_tokens.shape

        vit_out_scales = []
        # 逐 interaction block 处理一段 backbone blocks
        for i, layer in enumerate(self.interactions):
            blk_range = self.interaction_indexes[i]
            # 获取当前阶段的 blocks，但不在这里执行，交给 InteractionBlock 处理
            current_blocks = backbone.blocks[blk_range[0]: blk_range[-1] + 1]
            
            # 交互：x_seq 与 c 互相注入/提取，并在 InteractionBlock 内部执行 ViT blocks
            # 传递 cls token 和 register tokens 以便正确处理 ViT blocks
            x_patch_tokens, c, vit_cls, reg_tokens = layer(
                x_patch_tokens, c,
                current_blocks,  # 传递当前阶段的 blocks 给 InteractionBlock
                deform_inputs1, deform_inputs2, H, W,
                vit_cls=vit_cls,  # 传递 cls token
                reg_tokens=reg_tokens    # 传递 register tokens
            )
            vit_out_scales.append(x_patch_tokens.transpose(1, 2).view(B, dim, H, W).contiguous())

        # 拆分 c (与原实现一致的比例: 16:4:1 组合 => 16+4+1=21 单位)
        # 原 c2(8s) 长度: (H*W*4) ; c3(16s): (H*W) ; c4(32s): (H*W/4)
        # 直接依据最初保存的尺寸还原:
        c2_len = c2.size(1)
        c3_len = c3.size(1)
        c4_len = c4.size(1)

        c2_new = c[:, 0:c2_len, :]
        c3_new = c[:, c2_len:c2_len + c3_len, :]
        c4_new = c[:, c2_len + c3_len:c2_len + c3_len + c4_len, :]

        c2_new = c2_new.transpose(1, 2).view(B, dim, H * 2, W * 2).contiguous()  # [h/8, w/8]
        c3_new = c3_new.transpose(1, 2).view(B, dim, H, W).contiguous()  # [h/16, w/16]
        c4_new = c4_new.transpose(1, 2).view(B, dim, H // 2, W // 2).contiguous()  # [h/32, w/32]
        c1_up = self.up(c2_new) + c1  # c1: stride4  [h/4, w/4]

        if self.add_vit_feature and len(vit_out_scales) == 4:
            x1, x2, x3, x4 = vit_out_scales
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1_up, c2_new, c3_new, c4_new = c1_up + x1, c2_new + x2, c3_new + x3, c4_new + x4

        f1 = self.adapter_norm1(c1_up)
        f2 = self.adapter_norm2(c2_new)
        f3 = self.adapter_norm3(c3_new)
        f4 = self.adapter_norm4(c4_new)
        return [f1, f2, f3, f4], vit_out_scales, vit_cls.squeeze(1)


class AdapterDINOv2(DINOv2):
    """
    改造版：把所有“Adapter”相关内容集中到 self.adapter 对象中，方便：
      - 单独保存 adapter: torch.save(model.adapter.state_dict(), 'adapter.pth')
      - 单独加载 adapter: model.adapter.load_state_dict(...)
    """
    def __init__(self, backbone_config=None, pretrained=None):
        super().__init__(**backbone_config['dinov2_config'])
        self.logger = logging.getLogger()
        # self.logger.setLevel(logging.INFO)
        
        # # 如果logger没有handler，添加一个console handler
        # if not self.logger.handlers:
        #     console_handler = logging.StreamHandler()
        #     console_handler.setLevel(logging.INFO)
        #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #     console_handler.setFormatter(formatter)
        #     self.logger.addHandler(console_handler)
        
        self.enable_adapter = True
        self.save_whole_backbone = False
        adapter_config = backbone_config['adapter_config']
        self.adapter = DINOv2Adapter(
            adapter_config=adapter_config,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer,
            drop_path_rate=self.drop_path_rate
        )

        # 加载预训练（仅 backbone）
        if pretrained is not None:
            if isinstance(pretrained, dict) and 'dinov2' in pretrained:
                self.load_dinov2_pretrained(pretrained['dinov2'])
                if 'adapter' in pretrained:
                    self.load_adapter_pretrained(pretrained['adapter'])
            else:
                self.load_dinov2_pretrained(pretrained)

        # 设置可训练参数（仅 adapter）
        self.train(True)

    def load_dinov2_pretrained(self, pretrained):
        """加载 DINOv2 backbone 的预训练参数"""
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 使用 strict=False 并获取不匹配信息
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        # 记录加载信息
        self.logger.info(f'Loading dinov2 checkpoint from {pretrained}')
        
        # 统计成功加载的参数
        loaded_keys = len(state_dict) - len(unexpected_keys)
        total_model_keys = len(self.state_dict())
        
        self.logger.info(f'Successfully loaded {loaded_keys}/{len(state_dict)} parameters from checkpoint')
        self.logger.info(f'Model has {total_model_keys} parameters total')
        
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

    def load_adapter_pretrained(self, pretrained, strict=True):
        """加载 adapter 的预训练参数"""
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'adapter' in checkpoint:
            state_dict = checkpoint['adapter']
        else:
            state_dict = checkpoint
            
        # 只加载adapter部分参数, 并将key中adapter或者backbone.adapter前缀去掉
        adapter_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('adapter.'):
                k = k.replace('adapter.', '')
            elif k.startswith('backbone.adapter.'):
                k = k.replace('backbone.adapter.', '')
            adapter_state_dict[k] = v
            
        if adapter_state_dict:
            missing, unexpected = self.adapter.load_state_dict(adapter_state_dict, strict=strict)
            self.logger.info(f'Loaded adapter checkpoint from {pretrained} (missing={len(missing)}, unexpected={len(unexpected)})')
        else:
            self.logger.warning(f'No adapter parameters found in {pretrained}')

    def save_adapter(self, path):
        torch.save({'adapter': self.adapter.state_dict()}, path)
        self.logger.info(f'[AdapterDINOv2] Adapter saved to {path}')

    def forward(self, x, masks=None):
        if not self.enable_adapter:
            # 退化为纯 ViT
            outs = super().forward_features(x, masks)
            return get_pyramid_feature(outs), outs, None
        feats, vit_scales, cls_token = self.adapter(x, self, masks)
        return feats, vit_scales, cls_token

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        # 冻结 backbone，开放 adapter
        super().train(mode)
        set_requires_grad(self, ["adapter"])
        set_train(self, ["adapter"])
        return self


