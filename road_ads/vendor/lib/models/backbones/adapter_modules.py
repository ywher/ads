import logging
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from timm.layers import DropPath

from .ops.modules import MSDeformAttn

# _logger = logging.getLogger(__name__)


def get_reference_points(spatial_shapes, device):
    """
    Generate normalized reference points for deformable attention sampling.
    
    This function creates a grid of reference points for each feature level, which are used
    as base locations for deformable attention sampling. Each reference point represents
    the center of a feature map cell in normalized coordinates [0, 1].
    
    Args:
        spatial_shapes (list): List of tuples containing (height, width) for each feature level
                              e.g., [(H//8, W//8), (H//16, W//16), (H//32, W//32)]
        device (torch.device): Device to create tensors on
        
    Returns:
        Tensor: Reference points of shape (1, total_points, 1, 2) where:
                - total_points = sum(H_i * W_i for all levels)
                - Last dimension contains normalized (x, y) coordinates in [0, 1]
                - Third dimension is singleton for broadcasting compatibility
    """
    reference_points_list = []
    
    # Generate reference points for each spatial level
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        # Create coordinate grids: pixel centers from 0.5 to H_-0.5, W_-0.5
        # This ensures reference points are at the center of each spatial cell
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),  # Y coordinates
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device), indexing='ij')  # X coordinates
        
        # Flatten spatial dimensions and add batch dimension
        # Normalize coordinates to [0, 1] range by dividing by spatial dimensions
        ref_y = ref_y.reshape(-1)[None] / H_  # Shape: (1, H_*W_), normalized Y coords
        ref_x = ref_x.reshape(-1)[None] / W_  # Shape: (1, H_*W_), normalized X coords
        
        # Stack (x, y) coordinates: note the order is (x, y) not (y, x)
        # Shape: (1, H_*W_, 2) where last dim is [x_norm, y_norm]
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    
    # Concatenate reference points from all levels along the spatial dimension
    # Shape: (1, total_spatial_points, 2) where total_spatial_points = sum(H_i * W_i)
    reference_points = torch.cat(reference_points_list, 1)
    
    # Add singleton dimension for broadcasting compatibility with deformable attention
    # Final shape: (1, total_spatial_points, 1, 2)
    reference_points = reference_points[:, :, None]
    
    return reference_points


def deform_inputs(x):
    """
    Generate deformable attention input parameters for multi-scale feature interaction.
    
    This function creates two sets of deformable attention inputs used for bidirectional
    feature exchange between ViT and CNN features:
    - deform_inputs1: For injecting CNN features into ViT features
    - deform_inputs2: For extracting features from ViT back to CNN features
    
    Args:
        x (Tensor): Input image tensor of shape (B, 3, H, W)
        
    Returns:
        tuple: (deform_inputs1, deform_inputs2) where each contains:
            - reference_points (Tensor): Normalized reference points for attention sampling
            - spatial_shapes (Tensor): Spatial dimensions of feature levels  
            - level_start_index (Tensor): Starting indices for each feature level
    """
    bs, c, h, w = x.shape
    
    # === Prepare deform_inputs1 for CNN -> ViT injection ===
    # Multi-level spatial shapes: 3 CNN feature levels at different scales
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),      # stride 8: c2
                                      (h // 16, w // 16),    # stride 16: c3  
                                      (h // 32, w // 32)],   # stride 32: c4
                                     dtype=torch.long, device=x.device)
    
    # Level start indices: cumulative token counts for multi-level features
    # [0, N2, N2+N3] where N2=H*W/64, N3=H*W/256, N4=H*W/1024
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    
    # Reference points for ViT queries: single-level at stride 16 resolution
    # ViT patch tokens attend to multi-scale CNN features
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
    # === Prepare deform_inputs2 for ViT -> CNN extraction ===
    # Single-level spatial shape: only stride 16 for ViT features
    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    
    # Level start index for single level: just [0]
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    
    # Reference points for CNN queries: multi-level at 3 different scales
    # Multi-scale CNN features attend to single-level ViT features
    reference_points = get_reference_points([(h // 8, w // 8),      # for c2 queries
                                             (h // 16, w // 16),    # for c3 queries
                                             (h // 32, w // 32)],   # for c4 queries
                                            x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    """
    Convolutional Feed-Forward Network (ConvFFN) module that combines linear transformations
    with depthwise convolution for spatial feature processing.
    
    This module follows the typical FFN structure: Linear -> DWConv -> Activation -> Dropout -> Linear -> Dropout,
    where the depthwise convolution provides spatial inductive bias for better feature representation.
    
    Args:
        in_features (int): Number of input features
        hidden_features (int, optional): Number of hidden features. If None, defaults to in_features
        out_features (int, optional): Number of output features. If None, defaults to in_features
        act_layer (nn.Module): Activation layer. Default: GELU
        drop (float): Dropout rate. Default: 0.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        # Set default values for output and hidden features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # First linear transformation: expand to hidden dimension
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # Depthwise convolution for spatial feature processing
        self.dwconv = DWConv(hidden_features)
        
        # Activation function (typically GELU)
        self.act = act_layer()
        
        # Second linear transformation: project back to output dimension
        self.fc2 = nn.Linear(hidden_features, out_features)
        
        # Dropout layer for regularization
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Forward pass of ConvFFN.
        
        Args:
            x (Tensor): Input features of shape (B, N, in_features) where N = H*W for flattened spatial features
            H (int): Height of the feature map at stride 16
            W (int): Width of the feature map at stride 16
            
        Returns:
            Tensor: Output features of shape (B, N, out_features)
        """
        # First linear transformation: (B, N, in_features) -> (B, N, hidden_features)
        x = self.fc1(x)
        
        # Depthwise convolution with spatial reshaping: processes multi-scale features
        x = self.dwconv(x, H, W)
        
        # Apply activation function
        x = self.act(x)
        
        # Apply dropout for regularization
        x = self.drop(x)
        
        # Second linear transformation: (B, N, hidden_features) -> (B, N, out_features)
        x = self.fc2(x)
        
        # Apply final dropout
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    """
    Depthwise Convolution module for processing multi-scale concatenated features from different spatial resolutions.
    
    This module handles concatenated features from 3 different scales (stride 8, 16, 32) by:
    1. Splitting the input into 3 parts corresponding to different spatial resolutions
    2. Reshaping each part back to its original spatial dimensions
    3. Applying depthwise convolution to each part separately
    4. Flattening and concatenating the results back together
    
    The input is expected to be concatenated features where:
    - First 16/21 of tokens: stride 8 features (H*2, W*2)
    - Next 4/21 of tokens: stride 16 features (H, W)  
    - Last 1/21 of tokens: stride 32 features (H//2, W//2)
    
    Args:
        dim (int): Number of input channels/features. Default: 768
    """
    def __init__(self, dim=768):
        super().__init__()
        # Depthwise convolution: groups=dim means each channel is convolved independently
        # 3x3 kernel with padding=1 maintains spatial dimensions
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        """
        Forward pass for multi-scale depthwise convolution.
        
        Args:
            x (Tensor): Concatenated multi-scale features of shape (B, N, C) where:
                       N = 16*n + 4*n + 1*n = 21*n (n = H*W, the number of tokens at stride 16)
                       - First 16*n tokens: stride 8 features, spatial size (H*2, W*2)
                       - Next 4*n tokens: stride 16 features, spatial size (H, W)
                       - Last 1*n tokens: stride 32 features, spatial size (H//2, W//2)
            H (int): Height of feature map at stride 16
            W (int): Width of feature map at stride 16
            
        Returns:
            Tensor: Processed features of shape (B, N, C) with same concatenation order
        """
        B, N, C = x.shape
        # Calculate the base number of tokens (at stride 16 resolution)
        n = N // 21  # n = H * W
        
        # Split and reshape the concatenated features back to spatial format
        # x1: stride 8 features, 16*n tokens -> (B, C, H*2, W*2)
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        
        # x2: stride 16 features, 4*n tokens -> (B, C, H, W)  
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        
        # x3: stride 32 features, 1*n tokens -> (B, C, H//2, W//2)
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        
        # Apply depthwise convolution to each scale separately
        # Each convolution preserves spatial dimensions due to padding=1
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)  # (B, C, H*2, W*2) -> (B, 16*n, C)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, 4*n, C)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)  # (B, C, H//2, W//2) -> (B, 1*n, C)
        
        # Concatenate processed features back in the original order
        x = torch.cat([x1, x2, x3], dim=1)  # (B, 21*n, C)
        return x


class Extractor(nn.Module):
    """
    Extractor module that extracts enhanced features from Vision Transformer back to CNN features
    using multi-scale deformable attention mechanism.
    
    The Extractor takes CNN features as queries and ViT features as keys/values, allowing
    CNN features to attend to the enhanced ViT features. It optionally includes a ConvFFN
    for further feature processing and supports gradient checkpointing.
    
    Args:
        dim (int): Feature dimension/embedding size
        num_heads (int): Number of attention heads for deformable attention. Default: 6
        n_points (int): Number of sampling points per attention head. Default: 4
        n_levels (int): Number of feature levels for deformable attention. Default: 1
        deform_ratio (float): Ratio for deformable attention sampling. Default: 1.0
        with_cffn (bool): Whether to use ConvFFN for feature processing. Default: True
        cffn_ratio (float): Hidden dimension ratio for ConvFFN. Default: 0.25
        drop (float): Dropout rate. Default: 0.
        drop_path (float): Stochastic depth rate. Default: 0.
        norm_layer: Normalization layer. Default: LayerNorm with eps=1e-6
        with_cp (bool): Whether to use gradient checkpointing. Default: False
    """
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        # Normalization layers for query (CNN features) and key/value (ViT features)
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        
        # Multi-scale deformable attention for feature extraction
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        
        # Configuration flags
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        
        # Optional ConvFFN for additional feature processing
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        """
        Forward pass for extracting enhanced features from ViT back to CNN features.
        
        Args:
            query (Tensor): CNN features as queries, shape (B, N_cnn, dim)
            reference_points (Tensor): Reference points for deformable attention, shape (1, N_levels, 1, 2)
            feat (Tensor): ViT features as keys/values, shape (B, N_vit, dim)
            spatial_shapes (Tensor): Spatial shapes of feature levels, shape (n_levels, 2)
            level_start_index (Tensor): Start indices for each level, shape (n_levels,)
            H (int): Height of feature map at stride 16
            W (int): Width of feature map at stride 16
            
        Returns:
            Tensor: Enhanced CNN features, shape (B, N_cnn, dim)
        """
        
        def _inner_forward(query, feat):
            # Apply multi-scale deformable attention: CNN features attend to ViT features
            # Normalized query (CNN) attends to normalized feat (ViT) at multiple sampling points
            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            # Residual connection: add attention output to original query
            query = query + attn
    
            # Optional ConvFFN processing for additional feature transformation
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query
        
        # Use gradient checkpointing to save memory during training if enabled
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
            
        return query


class Injector(nn.Module):
    """
    Injector module that injects multi-scale CNN spatial prior features into Vision Transformer features
    using multi-level deformable attention mechanism.
    
    The Injector takes ViT features as queries and multi-scale CNN features as keys/values, allowing
    ViT features to attend to spatial priors from CNN at different scales. It includes a learnable
    scaling parameter (gamma) to control the injection strength.
    
    Args:
        dim (int): Feature dimension/embedding size
        num_heads (int): Number of attention heads for deformable attention. Default: 6
        n_points (int): Number of sampling points per attention head. Default: 4
        n_levels (int): Number of feature levels for deformable attention. Default: 1
        deform_ratio (float): Ratio for deformable attention sampling. Default: 1.0
        norm_layer: Normalization layer. Default: LayerNorm with eps=1e-6
        init_values (float): Initial value for learnable scaling parameter gamma. Default: 0.
        with_cp (bool): Whether to use gradient checkpointing. Default: False
    """
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        # Configuration flag for gradient checkpointing
        self.with_cp = with_cp
        
        # Normalization layers for query (ViT features) and key/value (CNN features)
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        
        # Multi-level deformable attention for spatial prior injection
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        
        # Learnable scaling parameter to control injection strength
        # Initialized to init_values (usually 0) for stable training
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, gamma=None):
        """
        Forward pass for injecting CNN spatial priors into ViT features.
        
        Args:
            query (Tensor): ViT features as queries, shape (B, N_vit, dim)
            reference_points (Tensor): Reference points for deformable attention, shape (1, N_vit, 1, 2)
            feat (Tensor): Concatenated multi-scale CNN features as keys/values, shape (B, N_total, dim)
                          where N_total = N2 + N3 + N4 (from stride 8, 16, 32 features)
            spatial_shapes (Tensor): Spatial shapes of CNN feature levels, shape (n_levels, 2)
                                   e.g., [(H//8, W//8), (H//16, W//16), (H//32, W//32)]
            level_start_index (Tensor): Start indices for each CNN level, shape (n_levels,)
                                      e.g., [0, N2, N2+N3] where N2=H*W/64, N3=H*W/256, N4=H*W/1024
            gamma (Tensor, optional): External gamma parameter to use instead of self.gamma.
            
        Returns:
            Tensor: Enhanced ViT features with injected spatial priors, shape (B, N_vit, dim)
        """
        
        def _inner_forward(query, feat):
            # Apply multi-level deformable attention: ViT features attend to multi-scale CNN features
            # Normalized query (ViT) attends to normalized feat (CNN) at multiple scales and sampling points
            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            # Residual connection with learnable scaling: gradually inject spatial priors
            # gamma starts from 0 and learns to control injection strength during training
            
            # Use provided gamma if available, otherwise use self.gamma
            current_gamma = gamma if gamma is not None else self.gamma
            return query + current_gamma * attn
        
        # Use gradient checkpointing to save memory during training if enabled
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
            
        return query


class InteractionBlock(nn.Module):
    """
    Interaction Block for bidirectional feature exchange between Vision Transformer (ViT) features and CNN spatial prior features using deformable attention mechanism.
    
    The block consists of:
    1. Injector: Injects multi-scale CNN features into ViT features using multi-level deformable attention
    2. Extractor: Extracts enhanced features from ViT back to CNN features using single-level deformable attention
    3. Optional extra extractors: Additional extraction layers for better feature refinement
    
    Args:
        dim (int): Feature dimension/embedding size
        num_heads (int): Number of attention heads for deformable attention. Default: 6
        n_points (int): Number of sampling points per attention head. Default: 4
        norm_layer: Normalization layer. Default: LayerNorm with eps=1e-6
        drop (float): Dropout rate. Default: 0.
        drop_path (float): Stochastic depth rate. Default: 0.
        with_cffn (bool): Whether to use ConvFFN in extractor. Default: True
        cffn_ratio (float): Hidden dimension ratio for ConvFFN. Default: 0.25
        init_values (float): Initial value for learnable scaling parameter gamma. Default: 0.
        deform_ratio (float): Ratio for deformable attention. Default: 1.0
        extra_extractor (bool): Whether to add 2 extra extractor layers. Default: False
        with_cp (bool): Whether to use gradient checkpointing. Default: False
    """
    def __init__(self, 
                 dim,  # 1024
                 num_heads=6,  # 16
                 n_points=4,  # 4
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),  # LN
                 drop=0.,  # 0
                 drop_path=0.,  # 0
                 with_cffn=True,  # True
                 cffn_ratio=0.25,  # 0.25
                 init_values=0.,  # 0
                 deform_ratio=1.0,  # 0.5
                 extra_extractor=False,  # False*3, True
                 with_cp=False):
        super().__init__()

        # Injector: Multi-level (3 levels) deformable attention to inject CNN spatial priors into ViT features
        # Takes ViT features as query and multi-scale CNN features (c2,c3,c4) as key/value
        self.injector = Injector(dim=dim,  # 1024
                                 n_levels=3,  # 3
                                 num_heads=num_heads,  # 16
                                 init_values=init_values,  # 0
                                 n_points=n_points,  # 4
                                 norm_layer=norm_layer,  # LN
                                 deform_ratio=deform_ratio,  # 0.5
                                 with_cp=with_cp)  # False
        
        # Extractor: Single-level (1 level) deformable attention to extract features from ViT back to CNN
        # Takes CNN features as query and ViT features as key/value, includes optional ConvFFN
        self.extractor = Extractor(dim=dim,  # 1024
                                   n_levels=1,  # 1
                                   num_heads=num_heads,  # 16
                                   n_points=n_points,  # 4
                                   norm_layer=norm_layer,  # LN
                                   deform_ratio=deform_ratio,  # 0.5
                                   with_cffn=with_cffn,  # True
                                   cffn_ratio=cffn_ratio,  # 0.25
                                   drop=drop,  # 0
                                   drop_path=drop_path,  # 0
                                   with_cp=with_cp)  # False
        
        # Optional extra extractors: Additional 2 extractor layers for better feature refinement
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim,  # 1024
                          num_heads=num_heads,  # 16
                          n_points=n_points,  # 4
                          norm_layer=norm_layer,  # LN
                          with_cffn=with_cffn,  # True
                          cffn_ratio=cffn_ratio,  # 0.25
                          deform_ratio=deform_ratio,  # 0.5
                          drop=drop,  # 0
                          drop_path=drop_path,  # 0
                          with_cp=with_cp)  # False
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W, vit_cls=None, reg_tokens=None):
        """
        Forward pass for bidirectional feature interaction between ViT and CNN features.
        
        Args:
            x (Tensor): ViT patch token features of shape (B, N, dim) where N = H*W
            c (Tensor): Concatenated CNN spatial prior features of shape (B, N_total, dim)
                       where N_total = N2 + N3 + N4 (from stride 8, 16, 32 features)
            blocks (nn.ModuleList): ViT transformer blocks to be applied after injection
            deform_inputs1 (list): Deformable attention inputs for injection step:
                                  [reference_points, spatial_shapes, level_start_index]
                                  - reference_points: (1, N, 1, 2) for single level
                                  - spatial_shapes: [(H//8, W//8), (H//16, W//16), (H//32, W//32)]
                                  - level_start_index: [0, N2, N2+N3]
            deform_inputs2 (list): Deformable attention inputs for extraction step:
                                  [reference_points, spatial_shapes, level_start_index]  
                                  - reference_points: (1, N_total, 1, 2) for multi-level
                                  - spatial_shapes: [(H//16, W//16)] for single level
                                  - level_start_index: [0]
            H (int): Height of feature map at stride 16 (H//16)
            W (int): Width of feature map at stride 16 (W//16)
            vit_cls (Tensor, optional): CLS token of shape (B, 1, dim). If provided, will be processed with ViT blocks
            reg_tokens (Tensor, optional): Register tokens of shape (B, num_reg, dim). If provided, will be processed with ViT blocks
            
        Returns:
            tuple: (x, c, vit_cls, reg_tokens) where:
                - x (Tensor): Enhanced ViT features after injection and transformer blocks (B, N, dim)
                - c (Tensor): Enhanced CNN features after extraction (B, N_total, dim)
                - vit_cls (Tensor): Updated CLS token (B, 1, dim) if provided, else None
                - reg_tokens (Tensor): Updated register tokens (B, num_reg, dim) if provided, else None
        """
        # Step 1: Inject multi-scale CNN spatial priors into ViT features
        # Uses multi-level deformable attention: ViT features attend to CNN features at 3 scales
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        
        # Step 2: Apply ViT transformer blocks to the injected features
        # Need to combine cls token, register tokens and patch tokens for ViT blocks
        if len(blocks) > 0:
            if vit_cls is not None:
                # Combine tokens: [cls, reg_tokens (if any), patch_tokens]
                if reg_tokens is not None:
                    x_combined = torch.cat([vit_cls, reg_tokens, x], dim=1)
                else:
                    x_combined = torch.cat([vit_cls, x], dim=1)
                
                # Apply ViT blocks
                for blk in blocks:
                    x_combined = blk(x_combined)
                
                # Split back to separate components
                if reg_tokens is not None:
                    vit_cls = x_combined[:, :1, :]
                    reg_tokens = x_combined[:, 1:1 + reg_tokens.shape[1], :]
                    x = x_combined[:, 1 + reg_tokens.shape[1]:, :]
                else:
                    vit_cls = x_combined[:, :1, :]
                    x = x_combined[:, 1:, :]
            else:
                # No cls token, just apply blocks to patch tokens
                for blk in blocks:
                    x = blk(x)
            
        # Step 3: Extract enhanced features from ViT back to CNN features  
        # Uses single-level deformable attention: CNN features attend to ViT features
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        
        # Step 4: Apply additional extraction layers if enabled for better feature refinement
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        
        return x, c, vit_cls, reg_tokens


class InteractionBlockWithCls(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, cls, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        x = torch.cat((cls, x), dim=1)
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        cls, x = x[:, :1, ], x[:, 1:, ]
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, cls
    

class SpatialPriorModule(nn.Module):
    """
    Spatial Prior Module for generating multi-scale CNN features that serve as spatial priors
    for Vision Transformer features. This module creates a feature pyramid with 4 levels:
    - c1: stride 4 (1/4 resolution)
    - c2: stride 8 (1/8 resolution) 
    - c3: stride 16 (1/16 resolution)
    - c4: stride 32 (1/32 resolution)
    
    Args:
        inplanes (int): Number of input channels for the first conv layer. Default: 64
        embed_dim (int): Embedding dimension for output features. Default: 384
        with_cp (bool): Whether to use checkpoint to save memory. Default: False
    """
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        # Stem: 3x3 conv + BN + ReLU blocks with maxpool, reduces resolution by 4x
        # Input: (B, 3, H, W) -> Output: (B, inplanes, H/4, W/4)
        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),  # /2
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # /2, total /4
        ])
        
        # Conv2: Further downsample by 2x, increase channels to 2*inplanes
        # Input: (B, inplanes, H/4, W/4) -> Output: (B, 2*inplanes, H/8, W/8)
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        
        # Conv3: Further downsample by 2x, increase channels to 4*inplanes
        # Input: (B, 2*inplanes, H/8, W/8) -> Output: (B, 4*inplanes, H/16, W/16)
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        
        # Conv4: Further downsample by 2x, keep channels at 4*inplanes
        # Input: (B, 4*inplanes, H/16, W/16) -> Output: (B, 4*inplanes, H/32, W/32)
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        
        # 1x1 convolutions to project different scale features to the same embedding dimension
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)      # c1: stride 4
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)  # c2: stride 8
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)  # c3: stride 16
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)  # c4: stride 32

    def forward(self, x):
        """
        Forward pass to generate multi-scale spatial prior features.
        
        Args:
            x (Tensor): Input image tensor of shape (B, 3, H, W)
            
        Returns:
            tuple: Four feature maps (c1, c2, c3, c4) where:
                - c1: (B, embed_dim, H/4, W/4) - stride 4, kept as spatial map
                - c2: (B, N2, embed_dim) - stride 8, flattened to sequence, N2 = H*W/64
                - c3: (B, N3, embed_dim) - stride 16, flattened to sequence, N3 = H*W/256  
                - c4: (B, N4, embed_dim) - stride 32, flattened to sequence, N4 = H*W/1024
        """
        def _inner_forward(x):
            # Generate multi-scale feature maps through progressive downsampling
            c1 = self.stem(x)      # (B, inplanes, H/4, W/4)
            c2 = self.conv2(c1)    # (B, 2*inplanes, H/8, W/8)
            c3 = self.conv3(c2)    # (B, 4*inplanes, H/16, W/16)
            c4 = self.conv4(c3)    # (B, 4*inplanes, H/32, W/32)
            
            # Project all features to the same embedding dimension
            c1 = self.fc1(c1)      # (B, embed_dim, H/4, W/4)
            c2 = self.fc2(c2)      # (B, embed_dim, H/8, W/8)
            c3 = self.fc3(c3)      # (B, embed_dim, H/16, W/16)
            c4 = self.fc4(c4)      # (B, embed_dim, H/32, W/32)
    
            bs, dim, _, _ = c1.shape
            # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # Keep c1 as spatial map for upsampling
            # Flatten spatial dimensions and transpose to sequence format for c2, c3, c4
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # (B, H*W/64, embed_dim) - stride 8
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # (B, H*W/256, embed_dim) - stride 16
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # (B, H*W/1024, embed_dim) - stride 32
    
            return c1, c2, c3, c4
        
        # Use gradient checkpointing to save memory if enabled
        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs


class InteractionBlockV3(nn.Module):
    """
    Interaction Block specifically designed for DINOv3 with rope encoding support.
    
    Key differences from InteractionBlock:
    - Supports rope position encoding for DINOv3 blocks
    - Handles storage tokens properly for DINOv3
    - Modified forward method to work with DINOv3's token structure
    - Optional injector: can skip injection to preserve original DINOv3 features
    
    The block performs bidirectional feature exchange:
    1. Injector (optional): Injects multi-scale CNN features into ViT features  
    2. ViT blocks: Processes all tokens with rope encoding support
    3. Extractor: Extracts enhanced features from ViT back to CNN features
    4. Optional extra extractors: Additional extraction layers for refinement
    
    Args:
        dim (int): Feature dimension/embedding size
        num_heads (int): Number of attention heads for deformable attention. Default: 6
        n_points (int): Number of sampling points per attention head. Default: 4
        norm_layer: Normalization layer. Default: LayerNorm with eps=1e-6
        drop (float): Dropout rate. Default: 0.
        drop_path (float): Stochastic depth rate. Default: 0.
        with_cffn (bool): Whether to use ConvFFN in extractor. Default: True
        cffn_ratio (float): Hidden dimension ratio for ConvFFN. Default: 0.25
        init_values (float): Initial value for learnable scaling parameter gamma. Default: 0.
        deform_ratio (float): Ratio for deformable attention. Default: 1.0
        extra_extractor (bool): Whether to add 2 extra extractor layers. Default: False
        with_cp (bool): Whether to use gradient checkpointing. Default: False
        use_injector (bool): Whether to use injector for CNN->ViT injection. Default: True
    """
    def __init__(self, 
                 dim,  # 1024
                 num_heads=6,  # 16
                 n_points=4,  # 4
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),  # LN
                 drop=0.,  # 0
                 drop_path=0.,  # 0
                 with_cffn=True,  # True
                 cffn_ratio=0.25,  # 0.25
                 init_values=0.,  # 0
                 deform_ratio=1.0,  # 0.5
                 extra_extractor=False,  # False*3, True
                 with_cp=False,
                 use_injector=True):  # ✅ 新增参数
        super().__init__()
        
        self.use_injector = use_injector  # ✅ 保存标志

        # Injector: Multi-level deformable attention to inject CNN spatial priors into ViT features
        # ✅ 只在需要时初始化 injector
        if use_injector:
            self.injector = Injector(dim=dim,
                                     n_levels=3,
                                     num_heads=num_heads,
                                     init_values=init_values,
                                     n_points=n_points,
                                     norm_layer=norm_layer,
                                     deform_ratio=deform_ratio,
                                     with_cp=with_cp)
        else:
            self.injector = None  # ✅ 不使用时设为 None
        
        # Extractor: Single-level deformable attention to extract features from ViT back to CNN
        self.extractor = Extractor(dim=dim,
                                   n_levels=1,
                                   num_heads=num_heads,
                                   n_points=n_points,
                                   norm_layer=norm_layer,
                                   deform_ratio=deform_ratio,
                                   with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio,
                                   drop=drop,
                                   drop_path=drop_path,
                                   with_cp=with_cp)
        
        # Optional extra extractors for better feature refinement
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim,
                          num_heads=num_heads,
                          n_points=n_points,
                          norm_layer=norm_layer,
                          with_cffn=with_cffn,
                          cffn_ratio=cffn_ratio,
                          deform_ratio=deform_ratio,
                          drop=drop,
                          drop_path=drop_path,
                          with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W, rope_embed=None, gamma=None):
        """
        Forward pass for DINOv3 bidirectional feature interaction with rope encoding support.
        
        Args:
            x (Tensor): ViT token features of shape (B, N_total, dim) including cls and optional storage tokens
            c (Tensor): Concatenated CNN spatial prior features of shape (B, N_spatial, dim)
            blocks (nn.ModuleList): DINOv3 transformer blocks to be applied after injection
            deform_inputs1 (list): Deformable attention inputs for injection step (unused if use_injector=False)
            deform_inputs2 (list): Deformable attention inputs for extraction step  
            H (int): Height of feature map at stride 16
            W (int): Width of feature map at stride 16
            rope_embed (callable, optional): Rope position embedding function for DINOv3 blocks
            gamma (Tensor, optional): External gamma parameter for injector.
            
        Returns:
            tuple: (x, c) where:
                - x (Tensor): Enhanced ViT features after optional injection and transformer blocks
                - c (Tensor): Enhanced CNN features after extraction
        """
        # Step 1: Optional Injection - inject CNN spatial priors into ViT patch features
        # ✅ 只在 use_injector=True 时执行注入
        if self.use_injector and self.injector is not None:
            # Extract patch tokens (excluding cls and storage tokens) for injection
            if x.shape[1] > H * W:  # Has cls and/or storage tokens
                # For DINOv3: x = [cls, storage_tokens, patch_tokens] or [cls, patch_tokens]
                x_patch = x[:, -H*W:, :]  # Take last H*W tokens as patch tokens
                x_non_patch = x[:, :-H*W, :]  # Take first tokens as cls/storage tokens
            else:
                x_patch = x
                x_non_patch = None
                
            # Inject CNN spatial priors into ViT patch features
            x_patch = self.injector(query=x_patch, reference_points=deform_inputs1[0],
                                   feat=c, spatial_shapes=deform_inputs1[1],
                                   level_start_index=deform_inputs1[2],
                                   gamma=gamma)
            
            # Reconstruct full token sequence after injection
            if x_non_patch is not None:
                x = torch.cat([x_non_patch, x_patch], dim=1)
            else:
                x = x_patch
        # ✅ 如果 use_injector=False, 直接跳过注入步骤，x 保持不变
        
        # Step 2: Apply DINOv3 transformer blocks to all tokens
        # Apply transformer blocks with rope encoding support for DINOv3
        for block in blocks:
            if rope_embed is not None:
                # DINOv3 blocks require rope position encoding
                rope_sincos = rope_embed(H=H*16//16, W=W*16//16)  # Adjust to patch resolution
                x = block(x, rope_sincos)
            else:
                # Fallback for blocks that don't need rope encoding
                x = block(x)
        
        # Step 3: Extraction - extract enhanced ViT features back to CNN
        # Extract patch tokens for feature extraction
        if x.shape[1] > H * W:
            x_patch = x[:, -H*W:, :]  # Take patch tokens
        else:
            x_patch = x
            
        # Extract ViT features back to CNN spatial priors
        c = self.extractor(query=c, reference_points=deform_inputs2[0], feat=x_patch, 
                          spatial_shapes=deform_inputs2[1], level_start_index=deform_inputs2[2], 
                          H=H, W=W)
        
        # Step 4: Optional additional extraction layers for feature refinement
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0], feat=x_patch, 
                             spatial_shapes=deform_inputs2[1], level_start_index=deform_inputs2[2], 
                             H=H, W=W)
        
        return x, c