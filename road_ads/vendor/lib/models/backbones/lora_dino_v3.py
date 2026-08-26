import logging
import torch
import os
import numpy as np
import math
import torch.nn as nn
from functools import partial

from .dino_v3 import DINOv3
from .peft import set_requires_grad, set_train, get_pyramid_feature

work_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


class LoRAAdapter(nn.Module):
    """LoRA adapter for DINOv3"""
    def __init__(
        self,
        embed_dim: int = 1024,
        num_layers: int = 24,
        lora_rank: int = 16,
        non_adapter_layers: int = 0,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        target_modules: list = None,  # 新增参数：指定要微调的模块
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.lora_rank = lora_rank
        self.non_adapter_layers = non_adapter_layers
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # 设置默认微调的模块为 q 和 v
        if target_modules is None:
            target_modules = ['q', 'v']
        self.target_modules = target_modules
        
        # 验证 target_modules 参数
        valid_modules = ['q', 'k', 'v']
        for module in self.target_modules:
            if module not in valid_modules:
                raise ValueError(f"Invalid target module '{module}'. Valid options: {valid_modules}")
        
        # 只对有效层创建LoRA参数
        self.valid_layers = num_layers - non_adapter_layers
        
        # 根据 target_modules 创建对应的LoRA参数
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
        
        # Dropout layer
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        
        # 缩放因子
        self.scaling = lora_alpha / lora_rank
        
        # 初始化参数
        self._reset_parameters()
    
    def _reset_parameters(self):
        """重置LoRA参数"""
        for layer_idx in range(self.valid_layers):
            # 只初始化存在的模块
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
        """
        Apply LoRA to input features for specific layer
        
        Args:
            x: input tensor [B, N, C] or [N, B, C]
            layer_idx: current layer index (0-based)
            
        Returns:
            delta_qkv: LoRA adjustments for Q, K, V (None if not in target_modules)
        """
        if layer_idx < self.non_adapter_layers:
            return None, None, None
        
        adjusted_layer = layer_idx - self.non_adapter_layers
        if adjusted_layer >= self.valid_layers:
            return None, None, None
        
        # 计算 LoRA 调整，只计算在 target_modules 中的模块
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


class LoRADINOv3(DINOv3):
    def __init__(
        self,
        backbone_config=None,
        pretrained=None,  # pretrained={'dinov3': path1, 'adapter': path2}
    ):
        # 1. 初始化 DINOv3
        super().__init__(**backbone_config['dinov3_config'])
        
        # 2. 初始化 LoRA adapter
        self.enable_adapter = False
        if backbone_config['lora_config'] is not None:
            self.enable_adapter = True
            self.save_whole_backbone = False
            self.adapter = LoRAAdapter(**backbone_config['lora_config'])
        
        self.logger = logging.getLogger()
        
        # 3. 先加载预训练参数
        if pretrained is not None:
            if isinstance(pretrained, dict):
                if 'dinov3' in pretrained and pretrained['dinov3']:
                    self.load_dinov3_pretrained(pretrained['dinov3'])
                # *** 关键：先加载adapter参数 ***
                if 'adapter' in pretrained and self.enable_adapter and pretrained['adapter']:
                    self.load_adapter_pretrained(pretrained['adapter'])
            else:
                self.load_dinov3_pretrained(pretrained)
        
        # 4. 最后替换attention层（此时adapter参数已经是pretrained的了）
        if self.enable_adapter:
            self._replace_attention_layers()
        
        # 5. 设置模型参数的requires_grad
        self.train(True)
    
    def _replace_attention_layers(self):
        """替换原始attention层为LoRA版本"""
        for layer_idx, block in enumerate(self.blocks):
            if layer_idx >= self.adapter.non_adapter_layers:
                # 保存原始的qkv层
                original_qkv = block.attn.qkv
                
                # 创建新的LoRA版本的qkv层，不保存adapter引用
                lora_qkv = LoRAQKV(
                    original_qkv=original_qkv,
                    layer_idx=layer_idx,
                    embed_dim=original_qkv.in_features,
                    non_adapter_layers=self.adapter.non_adapter_layers
                )
                
                # 使用partial预设adapter_func参数，避免在LoRAQKV中保存adapter引用
                lora_qkv.forward = partial(lora_qkv.forward, adapter_func=self.adapter.forward)
                
                # 替换attention层
                block.attn.qkv = lora_qkv
    
    def load_dinov3_pretrained(self, pretrained):
        """加载 DINOv3 backbone 的预训练参数"""
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
            self.logger.warning(f'Missing {len(missing_keys)} keys in checkpoint:')
            for key in missing_keys:
                self.logger.warning(f'  - {key}')
        if unexpected_keys:
            self.logger.warning(f'Unexpected {len(unexpected_keys)} keys in checkpoint:')
            for key in unexpected_keys:
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
        elif 'lora' in checkpoint:
            state_dict = checkpoint['lora']
        else:
            state_dict = checkpoint
            
        # 只加载adapter部分参数, 并将key中的各种前缀去掉
        adapter_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            # 处理各种可能的前缀
            prefixes_to_remove = [
                'adapter.',
                'backbone.adapter.',
                'lora.',
                'backbone.lora.',
                'model.adapter.',
                'model.lora.'
            ]
            
            for prefix in prefixes_to_remove:
                if new_key.startswith(prefix):
                    new_key = new_key.replace(prefix, '', 1)  # 只替换第一个匹配的前缀
                    break
            
            adapter_state_dict[new_key] = v
        
        if adapter_state_dict:
            missing_keys, unexpected_keys = self.adapter.load_state_dict(adapter_state_dict, strict=strict)
            self.logger.info(f'Loaded adapter checkpoint from {pretrained}')
            loaded_keys = len(state_dict) - len(unexpected_keys)
            
            self.logger.info(f'Successfully loaded {loaded_keys}/{len(state_dict)} parameters from checkpoint')
            
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
        else:
            self.logger.warning(f'No adapter parameters found in {pretrained}')
            self.logger.info(f'Available keys in checkpoint: {list(state_dict.keys())[:10]}{"..." if len(state_dict) > 10 else ""}')
    
    def forward(self, x, masks=None):
        """前向传播"""
        # masks: [B, H*W]（与 DINOv3.prepare_tokens_with_masks 对齐）
        B, _, h, w = x.shape
        x, (H, W) = self.prepare_tokens_with_masks(x, masks)

        outs = []
        for idx, blk in enumerate(self.blocks):
            rope_sincos = self.rope_embed(H=H, W=W) if hasattr(self, "rope_embed") and self.rope_embed is not None else None
            x = blk(x, rope_sincos)

            if idx in self.out_indices:
                patch = x[:, self.n_storage_tokens + 1:, :]  # [B, H*W, C]
                outs.append(patch.permute(0, 2, 1).reshape(B, -1, H, W).contiguous())

        return get_pyramid_feature(outs), outs, x[:, 0, :]
    
    def save_adapter(self, path):
        """保存adapter参数"""
        if self.enable_adapter:
            torch.save({'adapter': self.adapter.state_dict()}, path)
            self.logger.info(f'[LoRADINOv3] Adapter saved to {path}')
        else:
            self.logger.warning('[LoRADINOv3] No adapter to save')
        
    def train(self, mode: bool = True):
        """设置训练模式，只训练adapter参数"""
        super().train(mode)
        
        if not mode:
            return self
        
        # 冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 只解冻adapter参数
        if self.enable_adapter and hasattr(self, 'adapter'):
            for param in self.adapter.parameters():
                param.requires_grad = True
            self.adapter.train(True)
        
        return self


class LoRAQKV(nn.Module):
    """LoRA版本的QKV层（避免adapter参数重复）"""
    def __init__(self, original_qkv, layer_idx, embed_dim, non_adapter_layers):
        super().__init__()
        # 保持original_qkv不变
        self.original_qkv = original_qkv
        # 不保存adapter引用，避免参数重复
        self.layer_idx = layer_idx
        self.embed_dim = embed_dim
        self.non_adapter_layers = non_adapter_layers
        
        # 添加兼容属性，使其与原始Linear层兼容
        self.in_features = original_qkv.in_features
        self.out_features = original_qkv.out_features
        self.bias = original_qkv.bias
        
        # 冻结原始参数
        for param in self.original_qkv.parameters():
            param.requires_grad = False
    
    def forward(self, x, adapter_func=None):
        """
        Args:
            x: input tensor
            adapter_func: 传入的adapter函数，用于计算LoRA调整
        """
        # 计算原始QKV
        qkv = self.original_qkv(x)  # [B, N, 3*C]
        
        # 如果传入了adapter函数且当前层需要adapter
        if adapter_func is not None and self.layer_idx >= self.non_adapter_layers:
            # 通过函数调用获取LoRA调整
            delta_q, delta_k, delta_v = adapter_func(x, self.layer_idx)
            
            # 应用LoRA调整
            if delta_q is not None:
                qkv[:, :, :self.embed_dim] += delta_q  # Q部分
            if delta_k is not None:
                qkv[:, :, self.embed_dim:2*self.embed_dim] += delta_k  # K部分
            if delta_v is not None:
                qkv[:, :, 2*self.embed_dim:] += delta_v  # V部分
        
        return qkv
    
    @property
    def weight(self):
        """提供weight属性以保持兼容性"""
        return self.original_qkv.weight


def get_std_lora_dinov3_large():
    """标准的LoRADINOv3-Large配置（默认微调Q和V）"""
    lora_config = dict(
        embed_dim=1024,
        num_layers=24,
        lora_rank=16,
        non_adapter_layers=0,
        lora_alpha=16.0,
        lora_dropout=0.0,
        target_modules=['q', 'v'],  # 默认微调Q和V
    )
    
    backbone_config = {
        'lora_config': lora_config,
        'dinov3_config': {
            'patch_size': 16,
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'ffn_ratio': 4,
            'img_size': 512,
            'ffn_layer': "mlp",
            'layerscale_init': 1.0e-5,
            'qkv_bias': True,
            'proj_bias': True,
            'ffn_bias': True,
        }
    }
    return LoRADINOv3(backbone_config=backbone_config)


def get_lora_dinov3_with_custom_targets(target_modules=['q', 'v']):
    """自定义target_modules的LoRADINOv3配置"""
    lora_config = dict(
        embed_dim=1024,
        num_layers=24,
        lora_rank=16,
        non_adapter_layers=0,
        lora_alpha=16.0,
        lora_dropout=0.0,
        target_modules=target_modules,  # 自定义微调模块
    )
    
    backbone_config = {
        'lora_config': lora_config,
        'dinov3_config': {
            'patch_size': 16,
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'ffn_ratio': 4,
            'img_size': 512,
            'ffn_layer': "mlp",
            'layerscale_init': 1.0e-5,
            'qkv_bias': True,
            'proj_bias': True,
            'ffn_bias': True,
        }
    }
    return LoRADINOv3(backbone_config=backbone_config)


if __name__ == '__main__':
    backbone_cfg = {
        'lora_config': {
            'embed_dim': 1024,
            'num_layers': 24,
            'lora_rank': 8,
            'non_adapter_layers': 0,
            'lora_alpha': 16.0,
            'lora_dropout': 0.0,
            'target_modules': ['q', 'k', 'v'],  # 默认微调Q和V
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
            'layerscale_init': 1e-05,
            'ffn_layer': 'mlp',
            'ffn_bias': True,
            'proj_bias': True,
            'n_storage_tokens': 4,
            'mask_k_bias': True,
        },
    }
    
    pretrained_path = {}
    dinov3_ckpt = os.path.join(work_root, 'pretrained/dinov3/dinov3_vitl16.pth')
    # dinov3_ckpt = os.path.join(work_root, 'pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    if os.path.isfile(dinov3_ckpt):
        pretrained_path['dinov3'] = dinov3_ckpt
    else:
        print(f'[WARN] DINOv3 ckpt not found: {dinov3_ckpt}, skip loading backbone.')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LoRADINOv3(backbone_config=backbone_cfg, pretrained=pretrained_path).to(device)
    model.train(True)

    # 验证参数不重复
    param_ids = set()
    duplicate_count = 0

    for name, param in model.named_parameters():
        param_id = id(param)
        if param_id in param_ids:
            print(f"发现重复参数: {name}")
            duplicate_count += 1
        else:
            param_ids.add(param_id)

    print(f"重复参数数量: {duplicate_count}")

    # 测试代码
    import cv2
    norm = {'mean': (123.675, 116.28, 103.53), 'std': (58.395, 57.12, 57.375)}
    img_path = os.path.join(work_root, 'lib/models/backbones/images/city.png')
    
    if os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_LINEAR)
        img = img - np.array(norm['mean'], dtype=np.float32)
        img = img / np.array(norm['std'], dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        img1 = img[:, :, :, :512].to(device)
        img2 = img[:, :, :, 512:].to(device)
        x = torch.cat([img1, img2], dim=0)
        
        outs = model(x)
        
        print(f'len of outs: {len(outs)}')
        if isinstance(outs, tuple):
            for i in range(len(outs)):
                print(f'outs_{i}:')
                if isinstance(outs[i], torch.Tensor):
                    print(f'outs_{i}: {outs[i].shape}')
                elif isinstance(outs[i], list):
                    for j in range(len(outs[i])):
                        print(f'outs_{i}_{j}: {outs[i][j].shape}')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    total_params_m = total_params / 1e6
    trainable_params_m = trainable_params / 1e6
    non_trainable_params_m = non_trainable_params / 1e6

    print(f"total params: {total_params} ({total_params_m:.2f}M)")  # 305684993
    print(f"trainable params: {trainable_params} ({trainable_params_m:.2f}M)")  # 2530817
    print(f"non-trainable params: {non_trainable_params} ({non_trainable_params_m:.2f}M)")  # 303154176
    print(f"trainable params ratio: {(trainable_params / total_params)*100:.4f}%")  # 0.8279%
    '''
    len of outs: 3
    outs_0:
    outs_0_0: torch.Size([2, 1024, 128, 128])
    outs_0_1: torch.Size([2, 1024, 64, 64])
    outs_0_2: torch.Size([2, 1024, 32, 32])
    outs_0_3: torch.Size([2, 1024, 16, 16])
    outs_1:
    outs_1_0: torch.Size([2, 1024, 32, 32])
    outs_1_1: torch.Size([2, 1024, 32, 32])
    outs_1_2: torch.Size([2, 1024, 32, 32])
    outs_1_3: torch.Size([2, 1024, 32, 32])
    outs_2:
    outs_2: torch.Size([2, 1024])
    total params: 303940608
    trainable params: 786432
    trainable params: 0.7500 in MB
    non-trainable params: 303154176
    trainable params ratio: 0.2587%
    '''