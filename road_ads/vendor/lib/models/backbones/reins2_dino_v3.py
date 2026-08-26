
import logging
import torch
import os
import numpy as np

from .dino_v3 import DINOv3
from .reins2 import DualLoRAReins, DualReins
from .peft import set_requires_grad, set_train, get_pyramid_feature
work_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

class DualReinsDINOv3(DINOv3):
    def __init__(
        self,
        backbone_config=None,
        pretrained=None,  # pretrained={'dinov3': path1, 'adapter': path2, 'adapter_pretrained': path3}
        use_lora=True,  # 默认使用LoRA版本
    ):
        # init the dino v3
        super().__init__(**backbone_config['dinov3_config'])
        self.logger = logging.getLogger()
        # init the adapter
        self.enable_adapter = False
        if backbone_config['reins_config'] is not None:
            self.enable_adapter = True
            self.save_whole_backbone = False  # 不保存整个模型，只保存adapter部分
            
            # 根据use_lora选择使用DualLoRAReins或DualReins
            if use_lora:
                self.adapter = DualLoRAReins(**backbone_config['reins_config'])
            else:
                self.adapter = DualReins(**backbone_config['reins_config'])
        
        
        
        # 分别加载三部分预训练参数
        if pretrained is not None:
            if isinstance(pretrained, dict):
                # 加载DINOv3 backbone
                if 'dinov3' in pretrained and pretrained['dinov3']:
                    self.load_dinov3_pretrained(pretrained['dinov3'])
                
                # 加载adapter的默认部分（可选）
                if 'adapter' in pretrained and self.enable_adapter and pretrained['adapter']:
                    self.load_adapter_pretrained(pretrained['adapter'])
                
                # 加载adapter的预训练部分（用于reins_pretrained）
                if 'adapter_pretrained' in pretrained and self.enable_adapter and pretrained['adapter_pretrained']:
                    self.load_adapter_pretrained_part(pretrained['adapter_pretrained'])
            else:
                # 兼容原来的加载方式
                self.load_dinov3_pretrained(pretrained)
            
        # set the model params requires_grad
        self.train(True)
        
        # 重新应用freeze_pretrained设置（因为train(True)会覆盖requires_grad）
        if self.enable_adapter and hasattr(self.adapter, 'freeze_pretrained') and self.adapter.freeze_pretrained:
            for param in self.adapter.reins_pretrained.parameters():
                param.requires_grad = False
            self.logger.info('[DualReinsDINOv3] Reapplied freeze for reins_pretrained parameters after train(True)')
        # self.reins_count = 0
        
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
    
    def load_adapter_pretrained(self, pretrained, strict=False):
        """加载整个adapter的预训练参数（包括default和pretrained两部分）"""
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'adapter' in checkpoint:
            state_dict = checkpoint['adapter']
        elif 'reins' in checkpoint:
            state_dict = checkpoint['reins']
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
                'reins.',
                'backbone.reins.',
                'model.adapter.',
                'model.reins.'
            ]
            
            for prefix in prefixes_to_remove:
                if new_key.startswith(prefix):
                    new_key = new_key.replace(prefix, '', 1)  # 只替换第一个匹配的前缀
                    break
            
            adapter_state_dict[new_key] = v
        
        if adapter_state_dict:
            missing_keys, unexpected_keys = self.adapter.load_state_dict(adapter_state_dict, strict=strict)
            self.logger.info(f'Loaded adapter checkpoint from {pretrained}')
            loaded_keys = len(adapter_state_dict) - len(unexpected_keys)
            
            self.logger.info(f'Successfully loaded {loaded_keys}/{len(adapter_state_dict)} adapter parameters')
            
            if missing_keys:
                self.logger.warning(f'Missing {len(missing_keys)} keys in checkpoint:')
                for key in missing_keys[:10]:  # 只显示前10个
                    self.logger.warning(f'  - {key}')
                if len(missing_keys) > 10:
                    self.logger.warning(f'  ... and {len(missing_keys) - 10} more')
            
            if unexpected_keys:
                self.logger.warning(f'Unexpected {len(unexpected_keys)} keys in checkpoint:')
                for key in unexpected_keys[:10]:  # 只显示前10个
                    self.logger.warning(f'  - {key}')
                if len(unexpected_keys) > 10:
                    self.logger.warning(f'  ... and {len(unexpected_keys) - 10} more')
            
            if not missing_keys and not unexpected_keys:
                self.logger.info('Perfect match: all parameters loaded successfully!')
        else:
            self.logger.warning(f'No adapter parameters found in {pretrained}')
            self.logger.info(f'Available keys in checkpoint: {list(state_dict.keys())[:10]}{"..." if len(state_dict) > 10 else ""}')
    
    def load_adapter_pretrained_part(self, pretrained):
        """加载预训练的Reins参数到adapter的reins_pretrained部分"""
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'adapter' in checkpoint:
            state_dict = checkpoint['adapter']
        elif 'reins' in checkpoint:
            state_dict = checkpoint['reins']
        else:
            state_dict = checkpoint
        
        self.logger.info(f'Loading pretrained adapter part from {pretrained}')
        self.adapter.load_pretrained_reins(state_dict)
        self.logger.info('Pretrained adapter part loaded successfully!')
    
    def forward(self, x, masks=None):
        # masks: [B, H*W]（与 DINOv3.prepare_tokens_with_masks 对齐）
        B, _, h, w = x.shape
        x, (H, W) = self.prepare_tokens_with_masks(x, masks)

        outs = []
        for idx, blk in enumerate(self.blocks):
            rope_sincos = self.rope_embed(H=H, W=W) if hasattr(self, "rope_embed") and self.rope_embed is not None else None
            x = blk(x, rope_sincos)

            if self.enable_adapter and idx >= self.adapter.non_adapter_layers:
                x = self.adapter.forward(x, idx, batch_first=True, has_cls_token=True, num_register_token=self.n_storage_tokens)

            if idx in self.out_indices:
                patch = x[:, self.n_storage_tokens + 1:, :]  # [B, H*W, C]
                outs.append(patch.permute(0, 2, 1).reshape(B, -1, H, W).contiguous())

        return get_pyramid_feature(outs), outs, x[:, 0, :]
    
    def save_adapter(self, path):
        """保存adapter/reins参数"""
        if self.enable_adapter:
            torch.save({'adapter': self.adapter.state_dict()}, path)
            self.logger.info(f'[DualReinsDINOv3] Adapter saved to {path}')
        else:
            self.logger.warning('[DualReinsDINOv3] No adapter to save')
    
    def get_alpha_values(self):
        """获取当前的alpha融合系数值"""
        if self.enable_adapter and hasattr(self.adapter, 'get_alpha_values'):
            return self.adapter.get_alpha_values()
        else:
            return None
        
    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["adapter"])
        set_train(self, ["adapter"])
        
        # 重新应用freeze_pretrained设置（确保预训练参数保持冻结）
        if self.enable_adapter and hasattr(self.adapter, 'freeze_pretrained') and self.adapter.freeze_pretrained:
            for param in self.adapter.reins_pretrained.parameters():
                param.requires_grad = False


def get_std_dual_reins_dinov3_large(freeze_pretrained=False, use_lora=True, alpha_init=-10.0):
    """标准的DualReinsDINOv3-Large配置"""
    reins_config = dict(
        token_length=100,
        embed_dims=1024,
        num_layers=24,
        patch_size=16,
        lora_dim=16,
        non_adapter_layers=0,
        link_token_to_query=False,
        freeze_pretrained=freeze_pretrained,
        alpha_init=alpha_init,  # alpha融合系数的初始化值
    )
    backbone_config = {
        'reins_config': reins_config,
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
    return DualReinsDINOv3(backbone_config=backbone_config, use_lora=use_lora)


if __name__ == '__main__':
    backbone_cfg =  {
                    'reins_config': {
                        'lora_dim': 16,
                        'num_layers': 24,
                        "non_adapter_layers": 0,
                        'embed_dims': 1024,
                        'patch_size': 16,
                        'token_length': 100,
                        'link_token_to_query': False,
                        'freeze_pretrained': True,  # 是否冻结预训练参数
                        'alpha_init': -5.0,  # alpha融合系数的初始化值
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
                    },}
    pretrained_path = {}
    dinov3_ckpt = os.path.join(work_root, 'pretrained/dinov3/dinov3_vitl16.pth')
    if os.path.isfile(dinov3_ckpt):
        pretrained_path['dinov3'] = dinov3_ckpt
    else:
        print(f'[WARN] DINOv3 ckpt not found: {dinov3_ckpt}, skip loading backbone.')
    
    # 可选：添加预训练的adapter参数路径
    adapter_pretrained_ckpt = os.path.join(work_root, 'pretrained/dinov3/acdc/city4acdc_src_sup_rein.pth')
    if os.path.isfile(adapter_pretrained_ckpt):
        pretrained_path['adapter_pretrained'] = adapter_pretrained_ckpt
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DualReinsDINOv3(backbone_config=backbone_cfg, pretrained=pretrained_path, use_lora=True).to(device)
    model.train(True)

    import cv2
    norm={'mean': (123.675, 116.28, 103.53), 'std': (58.395, 57.12, 57.375)}
    img_path = os.path.join(work_root, 'lib/models/backbones/images/city.png')
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Read image in BGR format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
    # resize to 512 * 1024
    img = cv2.resize(img, (1024, 512), interpolation=cv2.INTER_LINEAR)
    # img = img[:, :512, :]  # crop to 512 * 512
    img = img - np.array(norm['mean'], dtype=np.float32)
    img = img / np.array(norm['std'], dtype=np.float32)
    img = np.transpose(img, (2, 0, 1))  # Change to (C, H, W) format
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    img1 = img[:, :, :, :512].cuda()  # [1, 3, 512, 512]
    img2 = img[:, :, :, 512:].cuda()  # [1, 3, 512, 512]
    x = torch.cat([img1, img2], dim=0)  # [2, 3, 512, 512]
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
    '''
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
    '''
    # print(outs[0][0])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"total params: {total_params}")  # 308215834
    print(f"trainable params: {trainable_params}")  # 5061658
    print(f"non-trainable params: {non_trainable_params}")  # 303154176
    print(f"trainable params ratio: {(trainable_params / total_params)*100:.4f}%")  # 1.6422%
    
    # 查看alpha融合系数
    if hasattr(model, 'get_alpha_values'):
        alpha_values = model.get_alpha_values()
        if alpha_values is not None:
            print(f"\nAlpha values (fusion coefficients):")
            print(f"Mean: {alpha_values.mean():.6f}, Std: {alpha_values.std():.6f}")
            print(f"Min: {alpha_values.min():.6f}, Max: {alpha_values.max():.6f}")
            print(f"Values: {alpha_values}")
    
    # print the trainable params name
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f'{name}: {param.shape}, {param.numel()}')
    '''
    Expected output similar to:
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
    '''