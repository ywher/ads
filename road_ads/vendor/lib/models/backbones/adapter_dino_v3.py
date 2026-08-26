import logging
import math
import os

import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from timm.layers import trunc_normal_

from .dino_v3 import DINOv3
from .peft import set_requires_grad, set_train, get_pyramid_feature
from .adapter_modules import SpatialPriorModule, InteractionBlockV3, deform_inputs
from .ops.modules import MSDeformAttn

work_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

class DINOv3Adapter(nn.Module):
    """
    DINOv3 Adapter module that packages spatial prior and interaction components
    for bidirectional feature exchange between DINOv3 ViT and CNN features.
    
    Key differences from DINOv2Adapter:
    - DINOv3 blocks require rope position encoding (from backbone.rope_embed)
    - May contain storage tokens (n_storage_tokens > 0) that need to be handled
    - Uses different token preparation and processing flow
    - Optional injector: can skip injection to preserve original DINOv3 features
    """
    def __init__(self, adapter_config, embed_dim, norm_layer, drop_path_rate):
        super().__init__()
        self.cfg = adapter_config
        
        # 验证必要的配置项
        required_keys = ['interaction_indexes', 'add_vit_feature', 'conv_inplane', 
                        'deform_num_heads', 'n_points', 'init_values', 'with_cffn', 
                        'cffn_ratio', 'deform_ratio', 'use_extra_extractor', 'with_cp']
        for key in required_keys:
            if key not in adapter_config:
                raise ValueError(f"Missing required config key: {key}")
        
        self.interaction_indexes = adapter_config['interaction_indexes']
        self.add_vit_feature = adapter_config['add_vit_feature']
        self.use_injector = adapter_config.get('use_injector', True)  # ✅ 新增参数，默认 True
        self.share_interaction_params = adapter_config.get('share_interaction_params', False)  # ✅ 是否共享interaction参数
        self.share_injector_gamma = adapter_config.get('share_injector_gamma', False)  # ✅ 是否共享injector的gamma参数
        
        conv_inplane = adapter_config['conv_inplane']
        deform_num_heads = adapter_config['deform_num_heads']
        n_points = adapter_config['n_points']
        init_values = adapter_config['init_values']
        with_cffn = adapter_config['with_cffn']
        cffn_ratio = adapter_config['cffn_ratio']
        deform_ratio = adapter_config['deform_ratio']
        use_extra_extractor = adapter_config['use_extra_extractor']
        with_cp = adapter_config['with_cp']
        
        # 验证interaction_indexes的合理性
        if not isinstance(self.interaction_indexes, list) or len(self.interaction_indexes) == 0:
            raise ValueError("interaction_indexes must be a non-empty list")
        
        for i, idx_range in enumerate(self.interaction_indexes):
            if not isinstance(idx_range, list) or len(idx_range) != 2:
                raise ValueError(f"interaction_indexes[{i}] must be a list of 2 elements [start, end]")

        # Level embedding for multi-scale CNN features
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        
        # CNN spatial prior module for generating multi-scale features
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        
        # Interaction blocks for bidirectional feature exchange using InteractionBlockV3
        # ✅ 支持参数共享模式
        self.num_interactions = len(self.interaction_indexes)
        
        if self.share_interaction_params:
            # 共享模式：只创建一个InteractionBlockV3，所有阶段复用
            # 添加可学习的level embedding来区分不同阶段
            self.interaction_level_embed = nn.Parameter(torch.zeros(self.num_interactions, embed_dim))
            self.shared_interaction = InteractionBlockV3(
                dim=embed_dim,
                num_heads=deform_num_heads,
                n_points=n_points,
                init_values=init_values,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                deform_ratio=deform_ratio,
                extra_extractor=use_extra_extractor,  # 共享模式下统一使用extra_extractor配置
                with_cp=with_cp,
                use_injector=self.use_injector
            )
            self.interactions = None  # 不使用nn.Sequential
            
            # ✅ 如果不共享gamma且使用了injector，则为每个阶段创建独立的gamma参数
            if not self.share_injector_gamma and self.use_injector:
                self.injector_gammas = nn.ParameterList([
                    nn.Parameter(init_values * torch.ones((embed_dim)), requires_grad=True)
                    for _ in range(self.num_interactions)
                ])
            else:
                self.injector_gammas = None
        else:
            # 非共享模式：每个阶段独立的InteractionBlockV3
            self.interaction_level_embed = None
            self.shared_interaction = None
            self.injector_gammas = None
            self.interactions = nn.Sequential(*[
                InteractionBlockV3(
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
                    with_cp=with_cp,
                    use_injector=self.use_injector
                )
                for i in range(len(self.interaction_indexes))
            ])

        # Upsampling and normalization layers for output feature maps
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.adapter_norm1 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm2 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm3 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm4 = nn.SyncBatchNorm(embed_dim)

        # Initialize all weights
        self._init_all_weights()
        normal_(self.level_embed)
        if self.interaction_level_embed is not None:
            normal_(self.interaction_level_embed)
        
        # Initialize logger for this adapter
        self.logger = logging.getLogger()
        
        # ✅ 记录 injector 使用状态
        if not self.use_injector:
            self.logger.info('[DINOv3Adapter] Injector disabled - using pure DINOv3 features with extractor only')
        else:
            self.logger.info('[DINOv3Adapter] Injector enabled - using bidirectional CNN-ViT feature exchange')
        
        # ✅ 记录参数共享状态
        if self.share_interaction_params:
            self.logger.info(f'[DINOv3Adapter] Interaction params shared across {self.num_interactions} stages - significantly reduced parameters')
        else:
            self.logger.info(f'[DINOv3Adapter] Independent interaction blocks for {self.num_interactions} stages')

    def _init_all_weights(self):
        """Initialize weights for all components"""
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
        # Reset deformable attention weights
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        """Add level embeddings to CNN features for multi-scale representation"""
        return c2 + self.level_embed[0], c3 + self.level_embed[1], c4 + self.level_embed[2]

    @torch.no_grad()
    def no_weight_decay(self):
        """Return parameters that should not have weight decay applied"""
        return set()

    def forward(self, x, backbone: DINOv3, masks=None):
        """
        Forward pass for DINOv3 adapter with bidirectional feature exchange.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            backbone: Initialized DINOv3 model with pretrained weights
            masks: Optional mask tensor [B, L] for attention
            
        Returns:
            pyramid_feats (list[Tensor]): 4-level pyramid features [B, C, H/4, W/4], [B, C, H/8, W/8], [B, C, H/16, W/16], [B, C, H/32, W/32]
            raw_feats (list[Tensor]): Intermediate ViT features from interaction stages (optional)
            cls_token (Tensor): Final classification token [B, C]
        """
        # Prepare deformable attention inputs at different scales
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # Extract CNN spatial prior features (c1: 1/4, c2: 1/8, c3: 1/16, c4: 1/32)
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)  # Concatenate multi-scale features

        B, _, h, w = x.shape
        # DINOv3 token preparation with rope encoding support
        x_tokens, (H, W) = backbone.prepare_tokens_with_masks(x, masks)
        
        # Separate different token types in DINOv3
        vit_cls = x_tokens[:, :1, :]  # Classification token
        if backbone.n_storage_tokens > 0:
            # DINOv3 may contain storage tokens that need special handling
            storage_tokens = x_tokens[:, 1:1 + backbone.n_storage_tokens, :]
            x_patch_tokens = x_tokens[:, 1 + backbone.n_storage_tokens:, :]  # Patch tokens
        else:
            storage_tokens = None
            x_patch_tokens = x_tokens[:, 1:, :]  # Only patch tokens
        _, _, dim = x_patch_tokens.shape

        vit_out_scales = []
        # Process blocks through interaction stages
        for i in range(self.num_interactions):
            blk_range = self.interaction_indexes[i]
            
            # Prepare current tokens for InteractionBlockV3
            if storage_tokens is not None:
                current_tokens = torch.cat([vit_cls, storage_tokens, x_patch_tokens], dim=1)
            else:
                current_tokens = torch.cat([vit_cls, x_patch_tokens], dim=1)
            
            # ✅ 根据是否共享参数选择不同的处理方式
            if self.share_interaction_params:
                # 共享模式：添加level embedding区分不同阶段，使用共享的interaction block
                level_embed = self.interaction_level_embed[i:i+1, :].unsqueeze(0)  # [1, 1, dim]
                # 将level embedding加到patch tokens上
                current_tokens_with_level = current_tokens.clone()
                if storage_tokens is not None:
                    # 只对patch tokens添加level embedding
                    current_tokens_with_level[:, 1 + backbone.n_storage_tokens:, :] = \
                        current_tokens_with_level[:, 1 + backbone.n_storage_tokens:, :] + level_embed
                else:
                    current_tokens_with_level[:, 1:, :] = current_tokens_with_level[:, 1:, :] + level_embed
                
                # 获取当前阶段的gamma参数（如果有）
                current_gamma = self.injector_gammas[i] if self.injector_gammas is not None else None
                
                current_tokens, c = self.shared_interaction(
                    current_tokens_with_level, c,
                    backbone.blocks[blk_range[0]: blk_range[-1] + 1],
                    deform_inputs1, deform_inputs2, H, W,
                    rope_embed=backbone.rope_embed,
                    gamma=current_gamma
                )
            else:
                # 非共享模式：使用独立的interaction block
                layer = self.interactions[i]
                current_tokens, c = layer(
                    current_tokens, c,
                    backbone.blocks[blk_range[0]: blk_range[-1] + 1],
                    deform_inputs1, deform_inputs2, H, W,
                    rope_embed=backbone.rope_embed
                )
            
            # Update token sequences after interaction
            vit_cls = current_tokens[:, :1, :]
            if storage_tokens is not None:
                storage_tokens = current_tokens[:, 1:1 + backbone.n_storage_tokens, :]
                x_patch_tokens = current_tokens[:, 1 + backbone.n_storage_tokens:, :]
            else:
                x_patch_tokens = current_tokens[:, 1:, :]
            
            # Store ViT features for pyramid enhancement
            vit_out_scales.append(x_patch_tokens.transpose(1, 2).view(B, dim, H, W).contiguous())

        # Restore CNN features from concatenated representation
        # Split c back to individual scales (c2: 1/8, c3: 1/16, c4: 1/32)
        c2_len = c2.size(1)
        c3_len = c3.size(1)
        c4_len = c4.size(1)

        c2_new = c[:, 0:c2_len, :]
        c3_new = c[:, c2_len:c2_len + c3_len, :]
        c4_new = c[:, c2_len + c3_len:c2_len + c3_len + c4_len, :]

        # Reshape back to spatial dimensions
        c2_new = c2_new.transpose(1, 2).view(B, dim, H * 2, W * 2).contiguous()  # 1/8 scale
        c3_new = c3_new.transpose(1, 2).view(B, dim, H, W).contiguous()          # 1/16 scale
        c4_new = c4_new.transpose(1, 2).view(B, dim, H // 2, W // 2).contiguous() # 1/32 scale
        
        # Upsample c2 to match c1 scale (1/4)
        c1_up = self.up(c2_new) + c1

        # Add ViT features to CNN features if enabled
        if self.add_vit_feature and len(vit_out_scales) == len(self.interaction_indexes):
            # Ensure we have the expected number of ViT features
            if len(vit_out_scales) >= 4:
                x1, x2, x3, x4 = vit_out_scales[:4]  # Take first 4 scales
                # Interpolate ViT features to match CNN scales
                x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)    # To 1/4
                x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)    # To 1/8
                x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)  # To 1/32
                # Add ViT features to CNN features
                c1_up, c2_new, c3_new, c4_new = c1_up + x1, c2_new + x2, c3_new + x3, c4_new + x4
            else:
                self.logger.warning(f"Expected 4 ViT scales for feature fusion, got {len(vit_out_scales)}")

        # Apply normalization to final features
        f1 = self.adapter_norm1(c1_up)   # 1/4 scale
        f2 = self.adapter_norm2(c2_new)  # 1/8 scale
        f3 = self.adapter_norm3(c3_new)  # 1/16 scale
        f4 = self.adapter_norm4(c4_new)  # 1/32 scale
        
        return [f1, f2, f3, f4], vit_out_scales, vit_cls.squeeze(1)


class AdapterDINOv3(DINOv3):
    """
    DINOv3 with adapter integration for bidirectional CNN-ViT feature exchange.
    
    Features:
    - Modular adapter design for easy saving/loading
    - Support for DINOv3-specific rope encoding and storage tokens
    - Configurable interaction between CNN spatial priors and ViT features
    """
    def __init__(self, backbone_config=None, pretrained=None):
        super().__init__(**backbone_config['dinov3_config'])
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
        
        # Initialize adapter module
        self.adapter = DINOv3Adapter(
            adapter_config=adapter_config,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer_cls,  # DINOv3 uses LayerNorm
            drop_path_rate=self.drop_path_rate  # Default drop_path_rate if not present
        )

        # Load pretrained weights
        if pretrained is not None:
            if isinstance(pretrained, dict) and 'dinov3' in pretrained:
                self.load_dinov3_pretrained(pretrained['dinov3'])
                if 'adapter' in pretrained:
                    self.load_adapter_pretrained(pretrained['adapter'])
            else:
                self.load_dinov3_pretrained(pretrained)

        # Set trainable parameters (adapter only)
        self.train(True)

    def load_dinov3_pretrained(self, pretrained):
        """加载 DINOv3 backbone 的预训练参数"""
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
        self.logger.info(f'Loading dinov3 checkpoint from {pretrained}')
        
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

    def load_adapter_pretrained(self, pretrained, strict=False):
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
            original_key = k
            if k.startswith('adapter.'):
                k = k.replace('adapter.', '')
                adapter_state_dict[k] = v
            elif k.startswith('backbone.adapter.'):
                k = k.replace('backbone.adapter.', '')
                adapter_state_dict[k] = v
            elif 'adapter' in original_key:  # 包含adapter的其他情况
                adapter_state_dict[k] = v
            
        if adapter_state_dict:
            missing_keys, unexpected_keys = self.adapter.load_state_dict(adapter_state_dict, strict=strict)
            self.logger.info(f'Loaded adapter checkpoint from {pretrained}')
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
        else:
            self.logger.warning(f'No adapter parameters found in {pretrained}')

    def save_adapter(self, path):
        """Save adapter weights only"""
        torch.save({'adapter': self.adapter.state_dict()}, path)
        self.logger.info(f'[AdapterDINOv3] Adapter saved to {path}')

    def forward(self, x, masks=None):
        """
        Forward pass with optional adapter functionality.
        
        Args:
            x: Input tensor [B, 3, H, W]
            masks: Optional attention masks
            
        Returns:
            feats: Pyramid features from adapter or backbone
            vit_feats: ViT features (same as feats when adapter enabled)
            cls_token: Classification token from final layer
        """
        if not self.enable_adapter:
            # Fallback to pure ViT without adapter
            outs = super().forward_features(x, masks)
            return get_pyramid_feature(outs), outs, None
        
        # Use adapter for bidirectional CNN-ViT feature exchange
        feats, vit_scales, cls_token = self.adapter(x, self, masks)
        return feats, vit_scales, cls_token

    def train(self, mode: bool = True):
        """
        Set training mode with backbone frozen and adapter trainable.
        
        Args:
            mode: Training mode flag
            
        Returns:
            self
        """
        if not mode:
            return super().train(mode)
        
        # Freeze backbone, enable adapter training
        super().train(mode)
        set_requires_grad(self, ["adapter"])
        set_train(self, ["adapter"])
        return self


if __name__ == '__main__':
    torch.manual_seed(0)

    # ✅ 配置 1: 使用 injector (默认)
    backbone_cfg_with_injector = {
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
            'use_injector': True,  # ✅ 使用 injector
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

    # ✅ 配置 2: 不使用 injector (节省参数，保持 DINOv3 特征不变)
    backbone_cfg_without_injector = {
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
            'use_injector': False,  # ✅ 不使用 injector，节省参数
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

    # ✅ 配置 3: 共享interaction参数 (大幅减少参数量)
    backbone_cfg_shared_interaction = {
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
            'use_injector': True,
            'share_interaction_params': True,  # ✅ 共享interaction参数
            'share_injector_gamma': False,  # ✅ 不共享injector的gamma参数
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

    # ✅ 配置 4: 共享interaction参数 + 不使用injector (最大化参数节省)
    backbone_cfg_shared_no_injector = {
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
            'use_injector': False,  # ✅ 不使用 injector
            'share_interaction_params': True,  # ✅ 共享interaction参数
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
    pretrained_backbone = os.path.join(work_root, 'pretrained/dinov3/dinov3_vitl16.pth')
    # pretrained_backbone = os.path.join(work_root, 'pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    if not os.path.isfile(pretrained_backbone):
        pretrained_backbone = None
        print('Warning: backbone pretrained weights not found.')

    # 测试所有配置
    for i, (name, cfg) in enumerate([
        ("With Injector (default)", backbone_cfg_with_injector),
        ("Without Injector", backbone_cfg_without_injector),
        ("Shared Interaction + Injector", backbone_cfg_shared_interaction),
        ("Shared Interaction + No Injector (最小参数)", backbone_cfg_shared_no_injector),
    ]):
        print(f"\n{'='*80}")
        print(f"Testing Configuration {i+1}: {name}")
        print(f"{'='*80}")
        
        model = AdapterDINOv3(
            backbone_config=cfg, 
            pretrained={'dinov3': pretrained_backbone} if pretrained_backbone else None
        ).cuda().train(True)

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_m = total_params / 1e6
        trainable_params_m = trainable_params / 1e6
        
        # spm和interactions模块参数 (兼容共享模式)
        spm_params = sum(p.numel() for n, p in model.named_parameters() if "spm" in n)
        interactions_params = sum(p.numel() for n, p in model.named_parameters() if "interactions" in n or "shared_interaction" in n)
        interaction_level_embed_params = sum(p.numel() for n, p in model.named_parameters() if "interaction_level_embed" in n)
        spm_params_m = spm_params / 1e6
        interactions_params_m = interactions_params / 1e6
        
        print(f'\nTotal params      : {total_params} ({total_params_m:.2f}M)')
        print(f'Trainable params  : {trainable_params} ({trainable_params_m:.2f}M)')
        print(f'Trainable ratio   : {trainable_params / total_params * 100:.4f}%')

        print(f'\nSPM params        : {spm_params} ({spm_params_m:.2f}M)')
        print(f'Interactions params: {interactions_params} ({interactions_params_m:.2f}M)')
        if interaction_level_embed_params > 0:
            print(f'Level embed params: {interaction_level_embed_params} ({interaction_level_embed_params / 1e6:.4f}M)')
        # 前向测试
        x = torch.randn(2, 3, 512, 512).cuda()
        with torch.no_grad():
            feats, vit_feats, cls_token = model(x)
        
        print(f'\nOutput shapes:')
        for j, f in enumerate(feats):
            print(f'  feat[{j}] = {tuple(f.shape)}')
        
        del model
        torch.cuda.empty_cache()
        
        """
        ================================================================================
        Testing Configuration 1: With Injector (default)
        ================================================================================
        Total params      : 326390976 (326.39M)
        Trainable params  : 23236800 (23.24M)
        Trainable ratio   : 7.1193%
        SPM params        : 1760576 (1.76M)
        Interactions params: 17269632 (17.27M)
        
        ================================================================================
        Testing Configuration 2: Without Injector
        ================================================================================
        Total params      : 319808448 (319.81M)
        Trainable params  : 16654272 (16.65M)
        Trainable ratio   : 5.2076%
        SPM params        : 1760576 (1.76M)
        Interactions params: 10687104 (10.69M)
        
        ================================================================================
        Testing Configuration 3: Shared Interaction + Injector w/wo No Shared Gamma
        ================================================================================
        Total params      : 316114624 (316.11M)
        Trainable params  : 12960448 (12.96M) -> 减少 44% (vs 配置1)
        Trainable ratio   : 4.0999%
        SPM params        : 1760576 (1.76M)
        Interactions params: 6989184 (6.99M) -> 减少 60% (vs 配置1)
        Level embed params: 4096 (0.0041M)
        
        Total params      : 316118720 (316.12M)
        Trainable params  : 12964544 (12.96M)
        Trainable ratio   : 4.1012%

        SPM params        : 1760576 (1.76M)
        Interactions params: 6989184 (6.99M)
        Level embed params: 4096 (0.0041M)
        
        ================================================================================
        Testing Configuration 4: Shared Interaction + No Injector (最小参数)
        ================================================================================
        Total params      : 314468992 (314.47M)
        Trainable params  : 11314816 (11.31M) -> 减少 51% (vs 配置1)
        Trainable ratio   : 3.5981%
        SPM params        : 1760576 (1.76M)
        Interactions params: 5343552 (5.34M) -> 减少 69% (vs 配置1)
        Level embed params: 4096 (0.0041M)
        
        参数对比总结:
        ┌─────────────────────────────────────────────────────────────────┐
        │ 配置                     │ 可训练参数  │ Interaction参数 │ 节省 │
        ├─────────────────────────────────────────────────────────────────┤
        │ 1. Injector (默认)       │ 23.24M     │ 17.27M          │ 0%   │
        │ 2. No Injector          │ 16.65M     │ 10.69M          │ 28%  │
        │ 3. Shared + Injector    │ 12.96M     │ 6.99M           │ 44%  │
        │ 4. Shared + No Injector │ 11.31M     │ 5.34M           │ 51%  │
        └─────────────────────────────────────────────────────────────────┘
        """
