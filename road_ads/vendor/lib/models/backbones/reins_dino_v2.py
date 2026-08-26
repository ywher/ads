
import logging
import torch
import os

from .dino_v2 import DINOv2
from .reins import LoRAReins
from .peft import set_requires_grad, set_train, get_pyramid_feature
work_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

class ReinsDINOv2(DINOv2):
    def __init__(
        self,
        backbone_config=None,
        pretrained=None,  # pretrained={'dinov2': path1, 'adapter': path2}
    ):
        # init the dino v2
        super().__init__(**backbone_config['dinov2_config'])
        
        # init the adapter
        self.enable_adapter = False
        if backbone_config['reins_config'] is not None:
            self.enable_adapter = True
            self.save_whole_backbone = False  # 不保存整个模型，只保存adapter部分
            self.adapter = LoRAReins(**backbone_config['reins_config'])
        
        self.logger = logging.getLogger()
        
        # 分别加载两部分预训练参数
        if pretrained is not None:
            if isinstance(pretrained, dict):
                if 'dinov2' in pretrained:
                    self.load_dinov2_pretrained(pretrained['dinov2'])
                if 'adapter' in pretrained and self.enable_adapter:
                    self.load_adapter_pretrained(pretrained['adapter'])
            else:
                # 兼容原来的加载方式
                self.load_dinov2_pretrained(pretrained)
            
        # set the model params requires_grad
        self.train(True)
        # self.reins_count = 0
        
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
        if not self.enable_adapter:
            self.logger.warning("Adapter is not enabled, skipping adapter checkpoint loading")
            return
            
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
            self.logger.info(f'Available keys in checkpoint: {list(state_dict.keys())[:10]}{"..." if len(state_dict) > 10 else ""}')
    
    def save_adapter(self, path):
        """保存 adapter 参数"""
        if not self.enable_adapter:
            self.logger.warning("Adapter is not enabled, cannot save adapter checkpoint")
            return
        torch.save({'adapter': self.adapter.state_dict()}, path)
        self.logger.info(f'Adapter saved to {path}')
        
    def forward(self, x, masks=None):
        # masks: [B, N]
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            # 应用 adapter（如果启用且当前层在适配范围内）
            if self.enable_adapter and idx >= self.adapter.non_adapter_layers:
                x = self.adapter.forward(
                    x, 
                    idx, 
                    batch_first=True, 
                    has_cls_token=True, 
                    num_register_token=self.num_register_tokens
                )
            
            # 收集输出特征（跳过 CLS token 和 register tokens）
            if idx in self.out_indices:
                # x[:, 1+self.num_register_tokens:, :] 跳过 CLS token 和 register tokens
                feat = x[:, 1+self.num_register_tokens:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                outs.append(feat)
        
        # 返回金字塔特征、原始输出特征列表、CLS token
        pyramid_feats = get_pyramid_feature(outs)
        cls_token = x[:, 0, :]  # shape: [B, D]
        
        return pyramid_feats, outs, cls_token
        
    def train(self, mode: bool = True):
        """设置训练模式"""
        if not mode:
            # 评估模式：设置整个模型为 eval
            return super().train(mode)
        set_requires_grad(self, ["adapter"])
        set_train(self, ["adapter"])


def get_std_reins_dinov2_large():
    backbone_config = {
        'reins_config': {
            'token_length': 100,
            'embed_dims': 1024,
            'num_layers': 24,
            'non_adapter_layers': 0,
            'patch_size': 16,
            'lora_dim': 16,
            'query_dims': 256,
            'use_softmax': True,
            'link_token_to_query': False,
            'scale_init': 0.001,
            'zero_mlp_delta_f': False,
        },
        'dinov2_config': {
            'patch_size': 16,
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'mlp_ratio': 4,
            'img_size': 512,
            'ffn_layer': "mlp",
            'init_values': 1.0e-5,
            'block_chunks': 0,
            'qkv_bias': True,
            'proj_bias': True,
            'ffn_bias': True,
        }
    }
    return ReinsDINOv2(backbone_config=backbone_config)


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
                    },
                    'dinov2_config': {
                        'img_size': 512,
                        'patch_size': 16,
                        'embed_dim': 1024,
                        'depth': 24,
                        'num_heads': 16,
                        'mlp_ratio': 4,
                        'qkv_bias': True,
                        'ffn_bias': True,
                        'proj_bias': True,
                        'init_values': 1e-05,
                        'ffn_layer': 'mlp',
                        'block_chunks': 0,
                    },}
    pretrained_path = {}
    pretrained_path['dinov2'] = os.path.join(work_root, 'pretrained/dinov2_converted.pth')
    # pretrained_path['adapter'] = os.path.join(work_root, 'pretrained/rein.pth')
    model = ReinsDINOv2(backbone_config=backbone_cfg, pretrained=pretrained_path)
    model.train(True)
    model.cuda()
    x = torch.randn(2, 3, 512, 512).cuda()
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
    # for i, out in enumerate(outs):
    #     print(f'outs_{i}: {out.shape}')
        
    # from torchsummary import summary
    # summary(model, (3, 512, 512))
    
    total_params = sum(p.numel() for p in model.parameters())  # 总参数量 
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数量 
    non_trainable_params = total_params - trainable_params  # 非可训练参数量 304199680

    print(f"total params: {total_params}")  # 306730497
    print(f"trainable params: {trainable_params}")  # 2530817
    print(f"non-trainable params: {non_trainable_params}")  # 304199680
    print(f"trainable params ratio: {(trainable_params / total_params)*100:.4f}%")  #  0.8251%
    # print the trainable params name
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f'{name}: {param.shape}, {param.numel()}')
    '''
    outs_0: torch.Size([2, 1024, 128, 128])
    outs_1: torch.Size([2, 1024, 64, 64])
    outs_2: torch.Size([2, 1024, 32, 32])
    outs_3: torch.Size([2, 1024, 16, 16])
    '''