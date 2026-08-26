import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.nn.init import normal_
from timm.layers import trunc_normal_

from .dino_v2 import DINOv2
from .peft import set_requires_grad, set_train, get_pyramid_feature
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs, Injector
from .reins import LoRAReins
from .ops.modules import MSDeformAttn


class AdapterReinsInteractionBlock(InteractionBlock):
    """
    Enhanced Interaction Block that combines both Adapter and Reins feature correction methods.
    
    This block extends the original InteractionBlock by adding Reins feature correction
    after each ViT block execution. The process is:
    1. Injector: Inject CNN features into ViT features
    2. ViT blocks + Reins: Execute ViT blocks with Reins feature correction after each block
    3. Extractor: Extract enhanced features from ViT back to CNN features
    
    Args:
        vit_dim (int, optional): Dimension of ViT features. If different from adapter_dim, projection layers will be added.
        adapter_dim (int, optional): Dimension of adapter features. If None, will use dim from kwargs.
        All args same as InteractionBlock (no reins_module needed in __init__)
    """
    def __init__(self, vit_dim=None, adapter_dim=None, **kwargs):
        # If adapter_dim is provided, use it as the dim for InteractionBlock
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
        
        # Add projection layers if dimensions differ
        if self.vit_dim != self.adapter_dim:
            self.vit_to_adapter = nn.Linear(vit_dim, adapter_dim)
            self.adapter_to_vit = nn.Linear(adapter_dim, vit_dim)
            
            # Initialize projections
            trunc_normal_(self.vit_to_adapter.weight, std=0.02)
            if self.vit_to_adapter.bias is not None:
                nn.init.constant_(self.vit_to_adapter.bias, 0)
            trunc_normal_(self.adapter_to_vit.weight, std=0.02)
            if self.adapter_to_vit.bias is not None:
                nn.init.constant_(self.adapter_to_vit.bias, 0)
        
    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W, vit_cls=None, reg_tokens=None, block_start_idx=0, reins_module=None):
        """
        Enhanced forward pass with Reins feature correction.
        
        Args:
            x (Tensor): ViT patch tokens, shape (B, N_patch, vit_dim)
            c (Tensor): Multi-scale CNN features, shape (B, N_cnn, adapter_dim)
            blocks (list): ViT transformer blocks to execute
            deform_inputs1 (list): Deformable attention inputs for injection
            deform_inputs2 (list): Deformable attention inputs for extraction
            H (int): Height of feature map at stride 16
            W (int): Width of feature map at stride 16
            vit_cls (Tensor, optional): CLS token, shape (B, 1, vit_dim)
            reg_tokens (Tensor, optional): Register tokens, shape (B, N_reg, vit_dim)
            block_start_idx (int): Starting global index of the first block in this stage
            reins_module (nn.Module, optional): External Reins module to avoid parameter duplication
            
        Returns:
            tuple: (x, c, vit_cls, reg_tokens) where:
                - x: Enhanced ViT patch tokens (vit_dim)
                - c: Enhanced CNN features (adapter_dim)
                - vit_cls: Updated CLS token (vit_dim)
                - reg_tokens: Updated register tokens (vit_dim)
        """
        # Step 1: Inject CNN spatial priors into ViT patch features
        # Project ViT features to adapter dimension if needed
        if self.vit_dim != self.adapter_dim:
            x_adapter = self.vit_to_adapter(x)
        else:
            x_adapter = x
        
        # Uses multi-level deformable attention: ViT features attend to CNN features
        x_patch_adapter = self.injector(query=x_adapter, reference_points=deform_inputs1[0],
                               feat=c, spatial_shapes=deform_inputs1[1],
                               level_start_index=deform_inputs1[2])
        
        # Project back to ViT dimension and add residual if needed
        if self.vit_dim != self.adapter_dim:
            x_patch = self.adapter_to_vit(x_patch_adapter) + x
        else:
            x_patch = x_patch_adapter
        
        # Step 2: Apply ViT transformer blocks with Reins feature correction
        if len(blocks) > 0:
            if vit_cls is not None:
                # Combine tokens: [cls, reg_tokens (if any), patch_tokens]
                if reg_tokens is not None:
                    x_combined = torch.cat([vit_cls, reg_tokens, x_patch], dim=1)
                else:
                    x_combined = torch.cat([vit_cls, x_patch], dim=1)
                
                # Apply ViT blocks with Reins feature correction
                for idx, blk in enumerate(blocks):
                    # Calculate the global block index
                    global_block_idx = block_start_idx + idx
                    
                    # Apply ViT block
                    x_combined = blk(x_combined)
                    
                    # Apply Reins feature correction if module is available and block index is valid
                    if (reins_module is not None and 
                        global_block_idx >= reins_module.non_adapter_layers):
                        x_combined = reins_module.forward(
                            x_combined, 
                            global_block_idx, 
                            batch_first=True, 
                            has_cls_token=True,
                            num_register_token=reg_tokens.shape[1] if reg_tokens is not None else 0
                        )
                
                # Split back to separate components
                if reg_tokens is not None:
                    vit_cls = x_combined[:, :1, :]
                    reg_tokens = x_combined[:, 1:1 + reg_tokens.shape[1], :]
                    x_patch = x_combined[:, 1 + reg_tokens.shape[1]:, :]
                else:
                    vit_cls = x_combined[:, :1, :]
                    x_patch = x_combined[:, 1:, :]
            else:
                # No cls token, just apply blocks to patch tokens with Reins correction
                for idx, blk in enumerate(blocks):
                    # Calculate the global block index
                    global_block_idx = block_start_idx + idx
                    
                    # Apply ViT block
                    x_patch = blk(x_patch)
                    
                    # Apply Reins feature correction
                    if (reins_module is not None and 
                        global_block_idx >= reins_module.non_adapter_layers):
                        x_patch = reins_module.forward(
                            x_patch, 
                            global_block_idx, 
                            batch_first=True, 
                            has_cls_token=False
                        )
            
        # Step 3: Extract enhanced features from ViT back to CNN features
        # Project to adapter dimension if needed
        if self.vit_dim != self.adapter_dim:
            x_patch_adapter = self.vit_to_adapter(x_patch)
        else:
            x_patch_adapter = x_patch
        
        # Uses single-level deformable attention: CNN features attend to ViT features
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x_patch_adapter, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        
        # Step 4: Apply additional extraction layers if enabled for better feature refinement
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x_patch_adapter, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        
        return x_patch, c, vit_cls, reg_tokens


class AdapterReinsDINOv2Adapter(nn.Module):
    """
    Combined Adapter + Reins module for DINOv2 that integrates both spatial adaptation
    and feature correction mechanisms.
    
    This module combines:
    - Spatial Prior Module (SPM) for multi-scale CNN features
    - Adapter Interaction Blocks for bidirectional feature exchange
    - Reins module for feature correction after each ViT block
    """
    def __init__(self, adapter_config, reins_config, embed_dim, norm_layer, drop_path_rate):
        super().__init__()
        self.cfg = adapter_config
        self.reins_cfg = reins_config
        self.interaction_indexes = adapter_config['interaction_indexes']
        self.add_vit_feature = adapter_config['add_vit_feature']
        
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
        
        # Enhanced interaction blocks with Reins integration
        self.interactions = nn.Sequential(*[
            AdapterReinsInteractionBlock(
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
                with_cp=with_cp
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
                nn.init.constant_(self.vit_to_adapter_proj.bias, 0)
        else:
            self.vit_to_adapter_proj = None

        # Weight initialization
        self._init_all_weights()
        normal_(self.level_embed)
        
        # Log configuration
        self.logger = logging.getLogger()
        if self.adapter_dim != embed_dim:
            self.logger.info(f'[AdapterReinsDINOv2Adapter] Using adapter bottleneck dim: {self.adapter_dim} (ViT dim: {embed_dim})')

    def _init_all_weights(self):
        """Initialize all weights in the module."""
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

    def forward(self, x, backbone: DINOv2, masks=None):
        """
        Forward pass combining Adapter and Reins mechanisms.
        
        Args:
            x (Tensor): Input image tensor, shape (B, 3, H, W)
            backbone (DINOv2): DINOv2 backbone model
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

        # Prepare ViT tokens
        x_tokens = backbone.prepare_tokens_with_masks(x, masks)
        vit_cls = x_tokens[:, :1, :]
        
        # Handle register tokens if present
        if backbone.num_register_tokens > 0:
            reg_tokens = x_tokens[:, 1:1 + backbone.num_register_tokens, :]
            x_patch_tokens = x_tokens[:, 1 + backbone.num_register_tokens:, :]
        else:
            reg_tokens = None
            x_patch_tokens = x_tokens[:, 1:, :]
        
        _, _, dim = x_patch_tokens.shape
        vit_out_scales = []

        # Process through interaction blocks with Reins feature correction
        for i, layer in enumerate(self.interactions):
            blk_range = self.interaction_indexes[i]
            current_blocks = backbone.blocks[blk_range[0]: blk_range[-1] + 1]
            
            # Apply enhanced interaction block with Reins correction
            # Pass the starting block index for correct Reins layer indexing
            x_patch_tokens, c, vit_cls, reg_tokens = layer(
                x_patch_tokens, c,
                current_blocks,
                deform_inputs1, deform_inputs2, H, W,
                vit_cls=vit_cls,
                reg_tokens=reg_tokens,
                block_start_idx=blk_range[0],  # Pass the starting block index
                reins_module=self.reins  # Pass the reins module externally
            )
            
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
        if self.add_vit_feature and len(vit_out_scales) == 4:
            x1, x2, x3, x4 = vit_out_scales
            
            # Project ViT features to adapter dimension if needed
            if self.vit_to_adapter_proj is not None:
                # x1, x2, x3, x4 are all (B, vit_dim, H, W)
                # Project each feature map
                B_vit, C_vit, H1, W1 = x1.shape
                x1 = self.vit_to_adapter_proj(x1.flatten(2).transpose(1, 2)).transpose(1, 2).view(B_vit, self.adapter_dim, H1, W1)
                
                _, _, H2, W2 = x2.shape
                x2 = self.vit_to_adapter_proj(x2.flatten(2).transpose(1, 2)).transpose(1, 2).view(B_vit, self.adapter_dim, H2, W2)
                
                _, _, H3, W3 = x3.shape
                x3 = self.vit_to_adapter_proj(x3.flatten(2).transpose(1, 2)).transpose(1, 2).view(B_vit, self.adapter_dim, H3, W3)
                
                _, _, H4, W4 = x4.shape
                x4 = self.vit_to_adapter_proj(x4.flatten(2).transpose(1, 2)).transpose(1, 2).view(B_vit, self.adapter_dim, H4, W4)
            
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


class AdapterReinsDINOv2(DINOv2):
    """
    Combined Adapter + Reins DINOv2 model that integrates both spatial adaptation
    and feature correction mechanisms for enhanced performance.
    
    This model combines:
    - DINOv2 backbone with spatial adaptation via Adapter mechanism
    - Feature correction via Reins mechanism after each ViT block
    - Modular design for easy parameter management and loading
    """
    def __init__(self, backbone_config=None, pretrained=None):
        super().__init__(**backbone_config['dinov2_config'])
        self.logger = logging.getLogger()
        
        self.enable_adapter = True
        self.save_whole_backbone = False
        
        # Extract configurations
        adapter_config = backbone_config['adapter_config']
        reins_config = backbone_config.get('reins_config', None)
        
        # Initialize combined adapter + reins module
        self.adapter = AdapterReinsDINOv2Adapter(
            adapter_config=adapter_config,
            reins_config=reins_config,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer,
            drop_path_rate=self.drop_path_rate
        )

        # Load pretrained weights
        if pretrained is not None:
            if isinstance(pretrained, dict):
                if 'dinov2' in pretrained:
                    self.load_dinov2_pretrained(pretrained['dinov2'])
                if 'adapter' in pretrained:
                    self.load_adapter_pretrained(pretrained['adapter'])
            else:
                self.load_dinov2_pretrained(pretrained)

        # Set trainable parameters
        self.train(True)

    def load_dinov2_pretrained(self, pretrained):
        """Load DINOv2 backbone pretrained parameters."""
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
            self.logger.warning(f'Missing {len(missing_keys)} keys in checkpoint:')
            for key in missing_keys:
                self.logger.warning(f'  - {key}')
        
        if unexpected_keys:
            self.logger.warning(f'Unexpected {len(unexpected_keys)} keys in checkpoint:')
            for key in unexpected_keys:
                self.logger.warning(f'  - {key}')
        
        if not missing_keys and not unexpected_keys:
            self.logger.info('Perfect match: all parameters loaded successfully!')

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
        else:
            self.logger.warning(f'No adapter parameters found in {pretrained}')

    def save_adapter(self, path):
        """Save combined adapter + reins parameters."""
        torch.save({'adapter': self.adapter.state_dict()}, path)
        self.logger.info(f'[AdapterReinsDINOv2] Adapter+Reins saved to {path}')

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


