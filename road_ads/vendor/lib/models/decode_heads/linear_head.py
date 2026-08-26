import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from lib.models.model_utils.cna import ConvModule

from lib.loss.losses import CrossEntropyLoss

class LinearHead(nn.Module):
    def __init__(self, decoder_config, pretrained=None):
        super().__init__()
        self.logger = logging.getLogger()
        self.input_transform = 'multiple_select'

        self.in_channels = decoder_config['in_channels']  # [1024, 1024, 1024, 1024]
        self.in_index = decoder_config['in_index']        # [0, 1, 2, 3]
        self.channels = decoder_config['channels']        # 256
        self.dropout_ratio = decoder_config['dropout_ratio']  # 0.1
        self.num_classes = decoder_config['num_classes']      # 19
        self.align_corners = decoder_config['align_corners']  # False
        self.interpolate = decoder_config.get('interpolate', True)
        self.norm_cfg = decoder_config['norm_cfg']

        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout2d(self.dropout_ratio)
        else:
            self.dropout = None
            
        # set the decode loss function
        self.loss_config = decoder_config['loss_decode']  # # dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        self.decode_loss = CrossEntropyLoss(ignore_index=self.loss_config.get('ignore_index', 255))
        self.loss_weight = self.loss_config.get('loss_weight', 1.0)

        # 线性融合层
        self.mid_channels = self.in_channels[-1]  # 1024
        self.fusion_conv = ConvModule(
            in_channels=sum(self.in_channels),  # 4096
            out_channels=self.mid_channels,  # 1024
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(self.mid_channels, self.mid_channels // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(self.mid_channels // 2),
            nn.GELU(),
            nn.ConvTranspose2d(self.mid_channels // 2, self.channels, kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.init_weights()
        self.init_conv_seg()
        
        self.debug = True
        self.debug_output = {}

        if pretrained is not None:
            self.pretrained = pretrained
            self.load_pretrained(pretrained)

    def load_pretrained(self, pretrained):
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, False)
        self.logger.info(f'Load decoder checkpoint from pretrained {pretrained}.')

    def init_conv_seg(self):
        if self.conv_seg is not None:
            nn.init.normal_(self.conv_seg.weight, 0, 0.01)
            if self.conv_seg.bias is not None:
                nn.init.constant_(self.conv_seg.bias, 0)
            self.logger.info('Initialize conv_seg layer.')

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for sub_m in m:
                    if isinstance(sub_m, (nn.LayerNorm, nn.BatchNorm2d)):
                        nn.init.constant_(sub_m.weight, 1)
                        nn.init.constant_(sub_m.bias, 0)
                    elif isinstance(sub_m, nn.ConvTranspose2d):
                        nn.init.kaiming_normal_(sub_m.weight, mode='fan_out', nonlinearity='relu')
                        if sub_m.bias is not None:
                            nn.init.constant_(sub_m.bias, 0)
                    
    def cls_seg(self, feat):
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs):
        # 支持 tuple 输入，选择第二个（分辨率相同的多层级特征）
        if isinstance(inputs, tuple):
            x = inputs[1]
        else:
            x = inputs

        # x: list of 4 tensors, each [B, C, H, W], H/W相同
        out = torch.cat(x, dim=1)  # [B, C*4, H, W]
        out = self.fusion_conv(out)  # [B, channels, H, W]
        out = self.output_upscaling(out)  # [B, channels//4, H*4, W*4]
        out = self.cls_seg(out)  # [B, num_classes, H*4, W*4]
        return out

    def cal_loss(self, seg_logits, seg_label, seg_weight=None, loss_key=None):
        """
        Calculate the loss of segmentation head.
        """
        loss_dict = {}
        loss = F.cross_entropy(seg_logits, seg_label, ignore_index=255)
        loss_dict['seg_loss'] = loss
        return loss_dict
    
if __name__ == '__main__':
    # in_channels= [64, 128, 320, 512]
    in_channels = [1024, 1024, 1024, 1024]
    # in_channels = [1280, 1280, 1280, 1280]
    config = {
    'in_channels': in_channels,
    'in_index': [0, 1, 2, 3],
    'channels': 256,
    'dropout_ratio': 0.1,
    'num_classes': 19,
    'norm_cfg': dict(type='GN', num_groups=32),
    'align_corners': False,
    'interpolate': True,
    'loss_decode': {
        'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0
    }
    }
    
    model = LinearHead(config)
    total_params = sum(p.numel() for p in model.parameters())  # 总参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数量
    non_trainable_params = total_params - trainable_params  # 非可训练参数量

    print(f"total params: {total_params}")  # 6824467
    print(f"trainable params: {trainable_params}")  # 6824467
    print(f"non-trainable params: {non_trainable_params}")  # 0
    print(f"trainable params ratio: {(trainable_params / total_params)*100:.4f}%")  # 100%
    # for m in model.modules():
    #     print(m)
    
    x = []
    x.append(torch.randn(2, in_channels[0], 128, 128))
    x.append(torch.randn(2, in_channels[1], 64, 64))
    x.append(torch.randn(2, in_channels[2], 32, 32))
    x.append(torch.randn(2, in_channels[3], 16, 16))
    
    x2 = []
    orin_shape = (32, 32)
    for i in range(len(in_channels)):
        x2.append(torch.randn(2, in_channels[i], orin_shape[0], orin_shape[1]))
    cls_token = torch.randn(2, in_channels[-1])  # cls token
    inputs = (x, x2, cls_token)  # multi-scale features, multi-level but same scale features
    
    # save_model_params_summary(model, filename="param_snapshot.txt", show_values=20)
    
    seg_logits = model(x2)  # 【2， 19， 512， 512】
    print(f'seg_logits.shape: {seg_logits.shape}')
    seg_logits = model(inputs)
    print(f'seg_logits.shape: {seg_logits.shape}')
    
