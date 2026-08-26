


import logging
import torch

from .dino_v3 import DINOv3
from .peft import get_pyramid_feature

class PureDINOv3(DINOv3):
    def __init__(
        self,
        backbone_config=None,
        pretrained=None,
    ):
        # 初始化 DINOv3 主干
        super().__init__(**backbone_config['dinov3_config'])

        self.logger = logging.getLogger()
        self.save_whole_backbone = True

        # 加载预训练
        if pretrained is not None:
            self.pretrained = pretrained
            self.load_pretrained(pretrained)

        # 可选：冻结所有参数
        if backbone_config.get('freeze_grad', False):
            self.freeze_params()
            self.save_whole_backbone = False
            return

        # 设置可训练
        self.train()
        
    def load_pretrained(self, pretrained):
        if isinstance(pretrained, dict) and pretrained.get('dinov3', None) is not None:
            pretrained = pretrained['dinov3']
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'backbone' in checkpoint:
            state_dict = checkpoint['backbone']
        else:
            state_dict = checkpoint
            
        if len(state_dict) > len(self.state_dict()):
            # extract the dino v2 related params and remove the prefix
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_k = k.replace('backbone.', '')
                    new_state_dict[new_k] = v
            state_dict = new_state_dict    
        
        self.load_state_dict(state_dict, strict=True)
        self.logger.info(f'Load dinov3 checkpoint from pretrained {pretrained}.')
        print(f'Load dinov3 checkpoint from pretrained {pretrained}.')
    
    def forward(self, x, masks=None):
        # 返回 out_indices 对应尺度的特征图列表
        outs = self.forward_features(x, masks)  # list[Tensor], 每个 [B, C, H/patch, W/patch]
        return get_pyramid_feature(outs), outs, x[:, 0, :]
    
    def freeze_params(self):
        for _, param in self.named_parameters():
            param.requires_grad = False
        self.logger.info("Freeze all params in dino v3.")


def get_std_pure_dinov3_large():
    # 仅示例，与你项目的注册工厂保持一致即可
    return PureDINOv3(
        backbone_config=dict(
            dinov3_config=dict(
                img_size=512,
                patch_size=16,
                in_chans=3,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                ffn_ratio=4.0,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path_rate=0.0,
                layerscale_init=1.0e-5,
                norm_layer="layernorm",
                ffn_layer="mlp",
                n_storage_tokens=0,
                mask_k_bias=False,
                out_indices=[7, 11, 15, 23],
            ),
            freeze_grad=False,
        ),
        pretrained=None,
    )


if __name__ == '__main__':
    backbone_cfg = {
        'dinov3_config': {
            'img_size': 512,
            'patch_size': 16,
            'in_chans': 3,
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'ffn_ratio': 4.0,
            'qkv_bias': True,
            'proj_bias': True,
            'ffn_bias': True,
            'drop_path_rate': 0.0,
            'layerscale_init': 1e-5,
            'norm_layer': 'layernorm',
            'ffn_layer': 'mlp',
            'n_storage_tokens': 0,
            'mask_k_bias': True,
            'out_indices': [7, 11, 15, 23],
        },
        'freeze_grad': False,
    }

    # 如有预训练，填写路径
    pretrained_path = None  # '/path/to/dinov3_pretrained.pth'
    model = PureDINOv3(backbone_config=backbone_cfg, pretrained=pretrained_path)
    model.train(True).cuda()

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
    outs_2: torch.Size([2, 512, 512])
    
    total params: 303150080 (303.15M)
    trainable params: 303150080 (303.15M)
    non-trainable params: 0 (0.00M)
    trainable params ratio: 100.0000%
    '''