import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
from torch.nn.init import normal_
from timm.layers import trunc_normal_

from .dino_v3 import DINOv3
from .peft import set_requires_grad, set_train, get_pyramid_feature
from .adapter_modules import SpatialPriorModule, InteractionBlockV3, deform_inputs
from .reins import LoRAReins
from .ops.modules import MSDeformAttn

work_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


class AdapterReinsInteractionBlockV3(InteractionBlockV3):
    """
    Enhanced Interaction Block for DINOv3 that combines both Adapter and Reins feature correction methods.
    
    This block extends the original InteractionBlockV3 by adding Reins feature correction
    after each ViT block execution. The process is:
    1. Injector (optional): Inject CNN features into ViT features
    2. ViT blocks + Reins: Execute ViT blocks with Reins feature correction after each block
    3. Extractor: Extract enhanced features from ViT back to CNN features
    
    Key differences from DINOv2 version:
    - Supports rope position encoding for DINOv3 blocks
    - Handles storage tokens properly for DINOv3
    - Modified forward method to work with DINOv3's token structure
    - Optional injector for preserving original DINOv3 features
    
    Args:
        All args same as InteractionBlockV3 (no reins_module needed in __init__)
    """
    def __init__(self, vit_dim=None, adapter_dim=None, **kwargs):
        # If adapter_dim is provided, use it as the dim for InteractionBlockV3
        # Otherwise use dim from kwargs
        dim = kwargs.get('dim')
        if adapter_dim is not None:
            kwargs['dim'] = adapter_dim
        else:
            adapter_dim = dim
            
        if vit_dim is None:
            vit_dim = dim
            
        super().__init__(**kwargs)
        
        self.vit_dim = vit_dim
        self.adapter_dim = adapter_dim
        
        if self.vit_dim != self.adapter_dim:
            self.vit_to_adapter = nn.Linear(vit_dim, adapter_dim)
            self.adapter_to_vit = nn.Linear(adapter_dim, vit_dim)
            
            # Initialize projections
            trunc_normal_(self.vit_to_adapter.weight, std=0.02)
            if self.vit_to_adapter.bias is not None:
                nn.init.zeros_(self.vit_to_adapter.bias)
            trunc_normal_(self.adapter_to_vit.weight, std=0.02)
            if self.adapter_to_vit.bias is not None:
                nn.init.zeros_(self.adapter_to_vit.bias)
        
    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W, rope_embed=None, 
                block_start_idx=0, reins_module=None, gamma=None):
        """
        Enhanced forward pass with Reins feature correction for DINOv3.
        
        Args:
            x (Tensor): ViT token features of shape (B, N_total, dim) including cls and optional storage tokens
            c (Tensor): Concatenated CNN spatial prior features of shape (B, N_spatial, dim)
            blocks (nn.ModuleList): DINOv3 transformer blocks to be applied after injection
            deform_inputs1 (list): Deformable attention inputs for injection step (unused if use_injector=False)
            deform_inputs2 (list): Deformable attention inputs for extraction step  
            H (int): Height of feature map at stride 16
            W (int): Width of feature map at stride 16
            rope_embed (callable, optional): Rope position embedding function for DINOv3 blocks
            block_start_idx (int): Starting global index of the first block in this stage
            reins_module (nn.Module, optional): External Reins module to avoid parameter duplication
            gamma (Tensor, optional): External gamma parameter for injector.
            
        Returns:
            tuple: (x, c) where:
                - x (Tensor): Enhanced ViT features after injection and transformer blocks
                - c (Tensor): Enhanced CNN features after extraction
        """
        # Step 1: Optional Injection - inject CNN spatial priors into ViT patch features
        # ✅ 只在 use_injector=True 时执行注入
        if self.use_injector and self.injector is not None:
            # Extract patch tokens (excluding cls and storage tokens) for injection
            if x.shape[1] > H * W:  # Has cls and/or storage tokens
                x_patch = x[:, -H*W:, :]
                x_non_patch = x[:, :-H*W, :]
            else:
                x_patch = x
                x_non_patch = None
                
            # Project to adapter dim if needed
            if self.vit_dim != self.adapter_dim:
                x_patch_adapter_in = self.vit_to_adapter(x_patch)
            else:
                x_patch_adapter_in = x_patch

            # Inject CNN spatial priors into ViT patch features
            x_patch_adapter_out = self.injector(query=x_patch_adapter_in, reference_points=deform_inputs1[0],
                                   feat=c, spatial_shapes=deform_inputs1[1],
                                   level_start_index=deform_inputs1[2],
                                   gamma=gamma)
            
            # Project back and add residual if needed
            if self.vit_dim != self.adapter_dim:
                # Injector is residual: out = in + attn. 
                # We want x_patch = x_patch + Proj(attn) = x_patch + Proj(out - in)
                delta = x_patch_adapter_out - x_patch_adapter_in
                x_patch = x_patch + self.adapter_to_vit(delta)
            else:
                x_patch = x_patch_adapter_out
            
            # Reconstruct full token sequence
            if x_non_patch is not None:
                x = torch.cat([x_non_patch, x_patch], dim=1)
            else:
                x = x_patch
        # ✅ 如果 use_injector=False, 跳过注入
            
        # Step 2: Apply DINOv3 transformer blocks with Reins feature correction
        for idx, block in enumerate(blocks):
            # Calculate the global block index
            global_block_idx = block_start_idx + idx
            
            # Apply DINOv3 block with rope encoding
            if rope_embed is not None:
                rope_sincos = rope_embed(H=H*16//16, W=W*16//16)
                x = block(x, rope_sincos)
            else:
                x = block(x)
            
            # Apply Reins feature correction if module is available and block index is valid
            if (reins_module is not None and 
                global_block_idx >= reins_module.non_adapter_layers):
                num_storage_tokens = x.shape[1] - H*W - 1
                num_storage_tokens = max(0, num_storage_tokens)
                
                x = reins_module.forward(
                    x, 
                    global_block_idx, 
                    batch_first=True, 
                    has_cls_token=True,
                    num_register_token=num_storage_tokens
                )
        
        # Step 3: Extraction - extract enhanced ViT features back to CNN
        if x.shape[1] > H * W:
            x_patch = x[:, -H*W:, :]
        else:
            x_patch = x
            
        # Project to adapter dim if needed
        if self.vit_dim != self.adapter_dim:
            x_patch_adapter = self.vit_to_adapter(x_patch)
        else:
            x_patch_adapter = x_patch

        c = self.extractor(query=c, reference_points=deform_inputs2[0], feat=x_patch_adapter, 
                          spatial_shapes=deform_inputs2[1], level_start_index=deform_inputs2[2], 
                          H=H, W=W)
        
        # Step 4: Optional additional extraction layers
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0], feat=x_patch_adapter, 
                             spatial_shapes=deform_inputs2[1], level_start_index=deform_inputs2[2], 
                             H=H, W=W)
        
        return x, c


class AdapterReinsDINOv3Adapter(nn.Module):
    """
    Combined Adapter + Reins module for DINOv3 that integrates both spatial adaptation
    and feature correction mechanisms.
    
    This module combines:
    - Spatial Prior Module (SPM) for multi-scale CNN features
    - Adapter Interaction Blocks for bidirectional feature exchange (optional injector)
    - Reins module for feature correction after each ViT block
    
    Key differences from DINOv2 version:
    - Uses InteractionBlockV3 which supports rope encoding
    - Handles DINOv3 storage tokens properly
    - Compatible with DINOv3's token structure and processing flow
    - Optional injector for preserving original DINOv3 features
    """
    def __init__(self, adapter_config, reins_config, embed_dim, norm_layer, drop_path_rate):
        super().__init__()
        self.cfg = adapter_config
        self.reins_cfg = reins_config
        self.interaction_indexes = adapter_config['interaction_indexes']
        self.add_vit_feature = adapter_config['add_vit_feature']
        self.use_injector = adapter_config.get('use_injector', True)  # ✅ 是否使用injector
        self.share_interaction_params = adapter_config.get('share_interaction_params', False)  # ✅ 是否共享interaction参数
        self.share_injector_gamma = adapter_config.get('share_injector_gamma', False)  # ✅ 是否共享injector的gamma参数
        
        # Adapter configurations
        conv_inplane = adapter_config['conv_inplane']
        deform_num_heads = adapter_config['deform_num_heads']
        n_points = adapter_config['n_points']
        init_values = adapter_config['init_values']
        with_cffn = adapter_config['with_cffn']
        cffn_ratio = adapter_config['cffn_ratio']
        deform_ratio = adapter_config['deform_ratio']
        use_extra_extractor = adapter_config['use_extra_extractor']
        with_cp = adapter_config['with_cp']
        
        # Get adapter dimension (default to embed_dim if not specified)
        self.adapter_dim = adapter_config.get('adapter_dim', embed_dim)

        # Initialize Reins module
        self.reins = LoRAReins(**reins_config) if reins_config is not None else None

        # Level embedding for multi-scale features
        self.level_embed = nn.Parameter(torch.zeros(3, self.adapter_dim))
        
        # CNN spatial prior module
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=self.adapter_dim, with_cp=False)
        
        # Enhanced interaction blocks with Reins integration for DINOv3
        # ✅ 支持参数共享模式
        self.num_interactions = len(self.interaction_indexes)
        
        if self.share_interaction_params:
            # 共享模式：只创建一个InteractionBlock，所有阶段复用
            # 添加可学习的level embedding来区分不同阶段
            self.interaction_level_embed = nn.Parameter(torch.zeros(self.num_interactions, self.adapter_dim))
            self.shared_interaction = AdapterReinsInteractionBlockV3(
                dim=self.adapter_dim,
                vit_dim=embed_dim,
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
                    nn.Parameter(init_values * torch.ones((self.adapter_dim)), requires_grad=True)
                    for _ in range(self.num_interactions)
                ])
            else:
                self.injector_gammas = None
        else:
            # 非共享模式：每个阶段独立的InteractionBlock
            self.interaction_level_embed = None
            self.shared_interaction = None
            self.injector_gammas = None
            self.interactions = nn.Sequential(*[
                AdapterReinsInteractionBlockV3(
                    dim=self.adapter_dim,
                    vit_dim=embed_dim,
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

        # Upsampling and normalization layers
        self.up = nn.ConvTranspose2d(self.adapter_dim, self.adapter_dim, 2, 2)
        self.adapter_norm1 = nn.SyncBatchNorm(self.adapter_dim)
        self.adapter_norm2 = nn.SyncBatchNorm(self.adapter_dim)
        self.adapter_norm3 = nn.SyncBatchNorm(self.adapter_dim)
        self.adapter_norm4 = nn.SyncBatchNorm(self.adapter_dim)
        
        # Projection for adding ViT features if dimensions differ
        if self.add_vit_feature and embed_dim != self.adapter_dim:
            self.vit_to_adapter_proj = nn.Linear(embed_dim, self.adapter_dim)
            trunc_normal_(self.vit_to_adapter_proj.weight, std=0.02)
            if self.vit_to_adapter_proj.bias is not None:
                nn.init.zeros_(self.vit_to_adapter_proj.bias)
        else:
            self.vit_to_adapter_proj = None

        # Weight initialization
        self._init_all_weights()
        normal_(self.level_embed)
        if self.interaction_level_embed is not None:
            normal_(self.interaction_level_embed)
        
        # 记录配置状态
        self.logger = logging.getLogger()
        if self.share_interaction_params:
            self.logger.info(f'[AdapterReinsDINOv3Adapter] Interaction params shared across {self.num_interactions} stages')
        if not self.use_injector:
            self.logger.info('[AdapterReinsDINOv3Adapter] Injector disabled')
        if self.adapter_dim != embed_dim:
            self.logger.info(f'[AdapterReinsDINOv3Adapter] Using adapter bottleneck dim: {self.adapter_dim} (ViT dim: {embed_dim})')

    def _init_all_weights(self):
        """Initialize all weights in the module."""
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
        
        # Reset deformable attention parameters
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        """Add learnable level embeddings to multi-scale features."""
        return c2 + self.level_embed[0], c3 + self.level_embed[1], c4 + self.level_embed[2]

    @torch.no_grad()
    def no_weight_decay(self):
        """Return parameters that should not have weight decay."""
        return set()

    def forward(self, x, backbone: DINOv3, masks=None):
        """
        Forward pass combining Adapter and Reins mechanisms for DINOv3.
        
        Args:
            x (Tensor): Input image tensor, shape (B, 3, H, W)
            backbone (DINOv3): DINOv3 backbone model
            masks (Tensor, optional): Attention masks, shape (B, L)
            
        Returns:
            tuple: (pyramid_feats, raw_feats, cls_token) where:
                - pyramid_feats: 4-level feature pyramid
                - raw_feats: Raw ViT features from interaction stages
                - cls_token: Final CLS token
        """
        # Generate deformable attention inputs
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # Extract multi-scale CNN spatial priors
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)  # Concatenate multi-scale features

        B, _, h, w = x.shape
        H, W = h // backbone.patch_size, w // backbone.patch_size

        # Prepare ViT tokens for DINOv3 with storage tokens support
        x_tokens, (H, W) = backbone.prepare_tokens_with_masks(x, masks)
        
        # DINOv3 tokens structure: [cls, storage_tokens (if any), patch_tokens]
        vit_cls = x_tokens[:, :1, :]
        if backbone.n_storage_tokens > 0:
            storage_tokens = x_tokens[:, 1:1 + backbone.n_storage_tokens, :]
            x_patch_tokens = x_tokens[:, 1 + backbone.n_storage_tokens:, :]
        else:
            storage_tokens = None
            x_patch_tokens = x_tokens[:, 1:, :]
        
        _, _, dim = x_patch_tokens.shape
        vit_out_scales = []

        # Process through interaction blocks with Reins feature correction
        for i in range(self.num_interactions):
            blk_range = self.interaction_indexes[i]
            current_blocks = backbone.blocks[blk_range[0]: blk_range[-1] + 1]
            
            # Prepare full token sequence for InteractionBlockV3
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
                    current_blocks,
                    deform_inputs1, deform_inputs2, H, W,
                    rope_embed=backbone.rope_embed,
                    block_start_idx=blk_range[0],
                    reins_module=self.reins,
                    gamma=current_gamma
                )
            else:
                # 非共享模式：使用独立的interaction block
                layer = self.interactions[i]
                current_tokens, c = layer(
                    current_tokens, c,
                    current_blocks,
                    deform_inputs1, deform_inputs2, H, W,
                    rope_embed=backbone.rope_embed,
                    block_start_idx=blk_range[0],
                    reins_module=self.reins
                )
            
            # Update token sequences after interaction
            vit_cls = current_tokens[:, :1, :]
            if storage_tokens is not None:
                storage_tokens = current_tokens[:, 1:1 + backbone.n_storage_tokens, :]
                x_patch_tokens = current_tokens[:, 1 + backbone.n_storage_tokens:, :]
            else:
                x_patch_tokens = current_tokens[:, 1:, :]
            
            # Store intermediate ViT features
            vit_out_scales.append(x_patch_tokens.transpose(1, 2).view(B, dim, H, W).contiguous())

        # Reconstruct multi-scale CNN features
        c2_len = c2.size(1)
        c3_len = c3.size(1)
        c4_len = c4.size(1)

        c2_new = c[:, 0:c2_len, :]
        c3_new = c[:, c2_len:c2_len + c3_len, :]
        c4_new = c[:, c2_len + c3_len:c2_len + c3_len + c4_len, :]

        c2_new = c2_new.transpose(1, 2).view(B, self.adapter_dim, H * 2, W * 2).contiguous()
        c3_new = c3_new.transpose(1, 2).view(B, self.adapter_dim, H, W).contiguous()
        c4_new = c4_new.transpose(1, 2).view(B, self.adapter_dim, H // 2, W // 2).contiguous()
        c1_up = self.up(c2_new) + c1

        # Add ViT features if enabled
        if self.add_vit_feature and len(vit_out_scales) == len(self.interaction_indexes):
            if len(vit_out_scales) >= 4:
                x1, x2, x3, x4 = vit_out_scales[:4]
                
                # Project ViT features to adapter dim if needed
                if self.vit_to_adapter_proj is not None:
                    # x is (B, dim, H, W), linear expects (..., dim)
                    # So permute -> project -> permute
                    x1 = self.vit_to_adapter_proj(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    x2 = self.vit_to_adapter_proj(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    x3 = self.vit_to_adapter_proj(x3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    x4 = self.vit_to_adapter_proj(x4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

                x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
                x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
                x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
                c1_up, c2_new, c3_new, c4_new = c1_up + x1, c2_new + x2, c3_new + x3, c4_new + x4

        # Apply final normalization
        f1 = self.adapter_norm1(c1_up)
        f2 = self.adapter_norm2(c2_new)
        f3 = self.adapter_norm3(c3_new)
        f4 = self.adapter_norm4(c4_new)
        
        return [f1, f2, f3, f4], vit_out_scales, vit_cls.squeeze(1)


class AdapterReinsDINOv3(DINOv3):
    """
    Combined Adapter + Reins DINOv3 model that integrates both spatial adaptation
    and feature correction mechanisms for enhanced performance.
    
    This model combines:
    - DINOv3 backbone with spatial adaptation via Adapter mechanism
    - Feature correction via Reins mechanism after each ViT block
    - Modular design for easy parameter management and loading
    - Support for DINOv3-specific features like rope encoding and storage tokens
    """
    def __init__(self, backbone_config=None, pretrained=None):
        super().__init__(**backbone_config['dinov3_config'])
        self.logger = logging.getLogger()
        
        self.enable_adapter = True
        self.save_whole_backbone = False
        
        # Extract configurations
        adapter_config = backbone_config['adapter_config']
        reins_config = backbone_config.get('reins_config', None)
        
        # Initialize combined adapter + reins module
        self.adapter = AdapterReinsDINOv3Adapter(
            adapter_config=adapter_config,
            reins_config=reins_config,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer_cls,
            drop_path_rate=self.drop_path_rate
        )

        # Load pretrained weights
        if pretrained is not None:
            if isinstance(pretrained, dict):
                if 'dinov3' in pretrained:
                    self.load_dinov3_pretrained(pretrained['dinov3'])
                if 'adapter' in pretrained:
                    self.load_adapter_pretrained(pretrained['adapter'])
            else:
                self.load_dinov3_pretrained(pretrained)

        # Set trainable parameters
        self.train(True)

    def load_dinov3_pretrained(self, pretrained):
        """Load DINOv3 backbone pretrained parameters."""
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
        
        # if missing_keys:
        #     self.logger.warning(f'Missing {len(missing_keys)} keys in checkpoint:')
        #     for key in missing_keys:
        #         self.logger.warning(f'  - {key}')
        
        # if unexpected_keys:
        #     self.logger.warning(f'Unexpected {len(unexpected_keys)} keys in checkpoint:')
        #     for key in unexpected_keys:
        #         self.logger.warning(f'  - {key}')
        
        # if not missing_keys and not unexpected_keys:
        #     self.logger.info('Perfect match: all parameters loaded successfully!')

    def load_adapter_pretrained(self, pretrained, strict=False):
        """加载 adapter 相关的预训练权重，包括 VitAdapter 和 Reins 模块"""
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
        """Save combined adapter + reins parameters."""
        torch.save({'adapter': self.adapter.state_dict()}, path)
        self.logger.info(f'[AdapterReinsDINOv3] Adapter+Reins saved to {path}')

    def forward(self, x, masks=None):
        """Forward pass through the combined model."""
        if not self.enable_adapter:
            # Fallback to pure ViT
            outs = super().forward_features(x, masks)
            return get_pyramid_feature(outs), outs, None
        
        feats, vit_scales, cls_token = self.adapter(x, self, masks)
        return feats, vit_scales, cls_token

    def train(self, mode: bool = True):
        """Set training mode, freezing backbone and enabling adapter+reins."""
        if not mode:
            return super().train(mode)
        
        # Freeze backbone, enable adapter and reins
        super().train(mode)
        set_requires_grad(self, ["adapter"])
        set_train(self, ["adapter"])
        return self


if __name__ == '__main__':
    import sys
    import os
    
    # Add project root to path for imports
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    sys.path.insert(0, project_root)
    
    torch.manual_seed(0)

    # 通用的 dinov3_config 和 reins_config
    dinov3_config = {
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
    }
    
    reins_config = {
        'lora_dim': 16,
        'num_layers': 24,
        'non_adapter_layers': 0,
        'embed_dims': 1024,
        'patch_size': 16,
        'token_length': 100,
        'link_token_to_query': False,
    }

    # =====================================================================
    # 两组不同的 interaction_indexes 设置
    # =====================================================================
    interaction_indexes_configs = {
        'Group A: 4 blocks [[0,7], [8,11], [12,15], [16,23]]': [[0, 7], [8, 11], [12, 15], [16, 23]],
        # 'Group B: 1 block [[20, 23]]': [[20, 23]],
    }

    def create_adapter_configs(interaction_indexes, reins_config, dinov3_config):
        """为指定的 interaction_indexes 创建四种配置"""
        base_adapter_config = {
            'interaction_indexes': interaction_indexes,
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
        }
        
        configs = []
        
        # 配置 1: 使用 injector (默认)
        cfg1 = base_adapter_config.copy()
        cfg1['use_injector'] = True
        cfg1['share_interaction_params'] = False
        configs.append(("With Injector (default)", {
            'adapter_config': cfg1,
            'reins_config': reins_config,
            'dinov3_config': dinov3_config,
        }))
        
        # 配置 2: 不使用 injector
        cfg2 = base_adapter_config.copy()
        cfg2['use_injector'] = False
        cfg2['share_interaction_params'] = False
        configs.append(("Without Injector", {
            'adapter_config': cfg2,
            'reins_config': reins_config,
            'dinov3_config': dinov3_config,
        }))
        
        # 配置 3: 共享interaction参数 + 使用injector
        cfg3 = base_adapter_config.copy()
        cfg3['use_injector'] = True
        cfg3['share_interaction_params'] = True
        configs.append(("Shared Interaction + Injector", {
            'adapter_config': cfg3,
            'reins_config': reins_config,
            'dinov3_config': dinov3_config,
        }))
        
        # 配置 4: 共享interaction参数 + 不使用injector (最小参数)
        cfg4 = base_adapter_config.copy()
        cfg4['use_injector'] = False
        cfg4['share_interaction_params'] = True
        configs.append(("Shared + No Injector (最小参数)", {
            'adapter_config': cfg4,
            'reins_config': reins_config,
            'dinov3_config': dinov3_config,
        }))
        
        # 配置 5: Bottleneck Adapter (adapter_dim=512)
        cfg5 = base_adapter_config.copy()
        cfg5['use_injector'] = True
        cfg5['share_interaction_params'] = False
        cfg5['adapter_dim'] = 128
        configs.append(("Bottleneck Adapter (dim=128)", {
            'adapter_config': cfg5,
            'reins_config': reins_config,
            'dinov3_config': dinov3_config,
        }))
        
        return configs

    # Optional pretrained paths
    pretrained_backbone = os.path.join(work_root, 'pretrained/dinov3/dinov3_vitl16.pth')
    if not os.path.isfile(pretrained_backbone):
        pretrained_backbone = os.path.join(work_root, 'pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    if not os.path.isfile(pretrained_backbone):
        pretrained_backbone = None
        print('Warning: backbone pretrained not found, using random init.')

    # 存储所有测试结果
    all_results = {}

    # 测试两组 interaction_indexes 设置
    for group_name, interaction_indexes in interaction_indexes_configs.items():
        print(f"\n{'#'*80}")
        print(f"# {group_name}")
        print(f"{'#'*80}")
        
        configs = create_adapter_configs(interaction_indexes, reins_config, dinov3_config)
        group_results = []
        
        for i, (name, cfg) in enumerate(configs):
            print(f"\n{'='*80}")
            print(f"Testing Configuration {i+1}: {name}")
            print(f"interaction_indexes = {cfg['adapter_config']['interaction_indexes']}")
            print(f"{'='*80}")
            
            model = AdapterReinsDINOv3(
                backbone_config=cfg, 
                pretrained={'dinov3': pretrained_backbone} if pretrained_backbone else None
            ).cuda().train(True)

            # 统计参数
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params_m = total_params / 1e6
            trainable_params_m = trainable_params / 1e6
            
            # 各模块参数 (兼容共享模式)
            spm_params = sum(p.numel() for n, p in model.named_parameters() if "spm" in n)
            interactions_params = sum(p.numel() for n, p in model.named_parameters() if "interactions" in n or "shared_interaction" in n)
            reins_params = sum(p.numel() for n, p in model.named_parameters() if "reins" in n)
            interaction_level_embed_params = sum(p.numel() for n, p in model.named_parameters() if "interaction_level_embed" in n)
            spm_params_m = spm_params / 1e6
            interactions_params_m = interactions_params / 1e6
            reins_params_m = reins_params / 1e6
            
            print(f'\nTotal params      : {total_params} ({total_params_m:.2f}M)')
            print(f'Trainable params  : {trainable_params} ({trainable_params_m:.2f}M)')
            print(f'Trainable ratio   : {trainable_params / total_params * 100:.4f}%')
            print(f'\nSPM params        : {spm_params} ({spm_params_m:.2f}M)')
            print(f'Interactions params: {interactions_params} ({interactions_params_m:.2f}M)')
            print(f'Reins params      : {reins_params} ({reins_params_m:.2f}M)')
            print(f'num_interactions  : {model.adapter.num_interactions}')
            if interaction_level_embed_params > 0:
                print(f'Level embed params: {interaction_level_embed_params} ({interaction_level_embed_params / 1e6:.4f}M)')
            
            # 前向测试
            x = torch.randn(2, 3, 512, 512).cuda()
            with torch.no_grad():
                feats, vit_feats, cls_token = model(x)
            
            print(f'\nOutput shapes:')
            for j, f in enumerate(feats):
                print(f'  feat[{j}] = {tuple(f.shape)}')
            
            # 记录结果
            group_results.append({
                'name': name,
                'trainable_params_m': trainable_params_m,
                'interactions_params_m': interactions_params_m,
                'reins_params_m': reins_params_m,
                'spm_params_m': spm_params_m,
                'num_interactions': model.adapter.num_interactions,
            })
            
            del model
            torch.cuda.empty_cache()
        
        all_results[group_name] = group_results
    
    # 打印总结
    print(f"\n{'#'*80}")
    print("# 参数对比总结 (AdapterReins)")
    print(f"{'#'*80}")
    
    for group_name, results in all_results.items():
        print(f"\n{'='*80}")
        print(f"{group_name}")
        print(f"{'='*80}")
        print(f"{'配置':<35} | {'可训练参数':>10} | {'Interactions':>12} | {'Reins':>8} | {'SPM':>6}")
        print("-" * 80)
        for r in results:
            print(f"{r['name']:<35} | {r['trainable_params_m']:>8.2f}M | {r['interactions_params_m']:>10.2f}M | {r['reins_params_m']:>6.2f}M | {r['spm_params_m']:>4.2f}M")
    
    print("""
    
================================================================================
测试结果记录 (AdapterReinsDINOv3) - 实际运行结果
================================================================================

Group A: 4 blocks [[0,7], [8,11], [12,15], [16,23]] (num_interactions=4)
--------------------------------------------------------------------------------
配置                                  | 可训练参数  | Interactions | Reins    | SPM
--------------------------------------------------------------------------------
1. With Injector (default)           |   25.77M   |    17.27M    |  2.53M   | 1.76M
2. Without Injector                  |   19.19M   |    10.69M    |  2.53M   | 1.76M
3. Shared Interaction + Injector     |   15.50M   |     6.99M    |  2.53M   | 1.76M
4. Shared + No Injector (min param)  |   13.85M   |     5.34M    |  2.53M   | 1.76M
5. Bottleneck Adapter (dim=512)      |   14.94M   |     9.43M    |  2.53M   | 1.40M
5. Bottleneck Adapter (dim=128)      |    5.58M   |     1.72M    |  2.53M   | 1.13M

================================================================================
关键发现:
================================================================================
1. Reins 参数 (2.53M) 在所有配置中保持不变 (由 num_layers=24 决定，与 interaction_indexes 无关)
2. SPM 参数 (1.76M) 在所有配置中保持不变 (固定结构)
3. Interactions 参数随 num_interactions 和 use_injector 变化:
   - 4 blocks + injector: 17.27M
   - 4 blocks - injector: 10.69M (节省 38%)
   - 1 block + injector:  6.99M  (节省 60% vs 4 blocks)
   - 1 block - injector:  5.34M  (节省 69% vs 4 blocks + injector)
   
4. 共享模式 (share_interaction_params=True) 效果:
   - 对于 4 blocks: 显著减少参数 (17.27M -> 6.99M, 节省 60%)
   - 对于 1 block: 无额外节省 (本来就只有1个block)

5. 最小参数配置: 1 block + no injector = 13.84M 可训练参数
   最大参数配置: 4 blocks + injector = 25.77M 可训练参数
   参数节省: 46.3%
   
6. 1 block vs 4 blocks 对比 (相同配置):
   - With Injector: 15.49M vs 25.77M (节省 39.9%)
   - Without Injector: 13.84M vs 19.19M (节省 27.9%)
   - Interactions 模块: 6.99M vs 17.27M (节省 59.5%)
================================================================================
    """)
