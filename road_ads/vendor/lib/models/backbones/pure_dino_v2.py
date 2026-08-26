


import logging
import torch

from .dino_v2 import DINOv2
from .peft import get_pyramid_feature

class PureDINOv2(DINOv2):
    def __init__(
        self,
        backbone_config=None,
        pretrained=None,
    ):
        # init the dino v2
        super().__init__(**backbone_config['dinov2_config'])
        
        self.logger = logging.getLogger()
        self.save_whole_backbone = True
        
        # load the pretrained
        if pretrained is not None:
            self.pretrained = pretrained
            self.load_pretrained(pretrained)
        
        # 
        if backbone_config.get('freeze_grad', False):
            self.freeze_params()
            self.save_whole_backbone = False
            return
               
        # set the model params requires_grad
        self.train()
        
    def load_pretrained(self, pretrained):
        if isinstance(pretrained, dict) and pretrained.get('dinov2', None) is not None:
            pretrained = pretrained['dinov2']
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
            
        self.load_state_dict(state_dict, True)
        self.logger.info(f'Load dinov2 checkpoint from pretrained {pretrained}.')
        print(f'Load dinov2 checkpoint from pretrained {pretrained}.')
    
    def forward(self, x, masks=None):
        # masks: [B, N]
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in self.out_indices:
                outs.append(x[:, 1+self.num_register_tokens:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous())
        
        return get_pyramid_feature(outs), outs, x[:, 0, :]
        # return self.reins.return_auto(outs)

    # def train(self, mode: bool = True):
    #     if not mode:
    #         return super().train(mode)
    #     set_requires_grad(self, ["reins"])
    #     set_train(self, ["reins"])
    
    def freeze_params(self):
        for name, param in self.named_parameters():
            param.requires_grad = False
        self.logger.info("Freeze all params in dino v2.")


def get_std_pure_dinov2_large():
    reins_config = dict(
        token_length=100,
        embed_dims=1024,
        num_layers=24,
        patch_size=16,
        lora_dim=16,
    )
    return PureDINOv2(
        reins_config=reins_config,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        img_size=512,
        ffn_layer="mlp",
        init_values=1.0e-5,
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )


