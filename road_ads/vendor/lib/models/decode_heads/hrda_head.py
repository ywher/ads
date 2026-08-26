"""HRDA decode head.

HRDA 解码头。

This head decodes low-resolution context features and high-resolution detail
features, predicts an attention map, and fuses both branches into final logits.

该解码头分别解码低分辨率上下文特征和高分辨率细节特征，预测尺度注意力图，
并融合两条分支得到最终分割 logits。
"""

from copy import deepcopy
import logging
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from lib.models.model_utils.funcs import add_prefix, match_shape, crop, resize as _resize
from lib.models.decode_heads.daformer_head import DAFormerHead
from lib.models.decode_heads.segformer_head import SegFormerHead


def scale_box(box, scale):
    """Scale a crop box from image coordinates to feature coordinates.

    将图像坐标系中的 crop box 缩放到特征坐标系。
    """
    y1, y2, x1, x2 = box
    # assert y1 % scale == 0
    # assert y2 % scale == 0
    # assert x1 % scale == 0
    # assert x2 % scale == 0
    y1 = int(y1 / scale)
    y2 = int(y2 / scale)
    x1 = int(x1 / scale)
    x2 = int(x2 / scale)
    return y1, y2, x1, x2


class HRDAHead(nn.Module):
    """Fuse low-resolution and high-resolution segmentation logits.

    融合低分辨率与高分辨率分割 logits。

    The forward path returns `(fused_seg, lr_seg, hr_seg)` during training so
    auxiliary low/high-resolution losses can be added. Test-time `forward_test`
    returns only `fused_seg`.

    训练阶段前向返回 `(fused_seg, lr_seg, hr_seg)`，便于附加低/高分辨率
    辅助损失；测试阶段 `forward_test` 只返回 `fused_seg`。
    """

    def __init__(self, decoder_config, pretrained=None):
        super(HRDAHead, self).__init__()
        self.logger = logging.getLogger()
        '''
        single_scale_head,  # single_scale_head
        lr_loss_weight=0,  # 0
        hr_loss_weight=0,  # 0.1
        scales=[1],  # [0.5, 1]
        attention_embed_dim=256,  # 256
        attention_classwise=True,  # True
        enable_hr_crop=False,  # True
        hr_slide_inference=True,  # True
        fixed_attention=None,
        debug_output_attention=False,
        **kwargs
        '''
        self.single_scale_head = decoder_config['single_scale_head']  # DAFormerHead
        self.lr_loss_weight = decoder_config['lr_loss_weight']  # 0
        self.hr_loss_weight = decoder_config['hr_loss_weight']  # 0.1
        self.scales = decoder_config['scales']  # [0.5, 1]
        self.attention_embed_dim = decoder_config.get('attention_embed_dim', 256)  # 256
        self.attention_classwise = decoder_config.get('attention_classwise', True)  # True
        self.enable_hr_crop = decoder_config['enable_hr_crop']  # True
        self.hr_slide_inference = decoder_config['hr_slide_inference']  # True
        self.hr_slide_overlapping = decoder_config.get('hr_slide_overlapping', True)  # True
        self.hr_crop_size = decoder_config.get('hr_crop_size', [512, 512])  # [512, 512]
        self.hr_crop_box = None  # None
        self.fixed_attention = decoder_config.get('fixed_attention', None)  # None
        self.debug_output_attention = decoder_config.get('debug_output_attention', False)  # False
        self.crop_coord_divisible = decoder_config.get('crop_coord_divisible', 8)  # 8
        self.blur_hr_crop = decoder_config.get('blur_hr_crop', False)  # False
        self.feature_scale = decoder_config.get('feature_scale', 0.5)  # 0.5
        # self.align_corners = decoder_config.get('align_corners', False)  # False
        
        head_cfg = {
            'in_channels': decoder_config['in_channels'],  # [64, 128, 320, 512]
            'in_index': decoder_config['in_index'],  # [0, 1, 2, 3]
            'channels': decoder_config['channels'],  # 256
            'dropout_ratio': decoder_config['dropout_ratio'],  # 
            'num_classes': decoder_config['num_classes'],  # 19
            'norm_cfg': decoder_config['norm_cfg'],  # dict(type='BN', requires_grad=True)
            'align_corners': decoder_config['align_corners'],  # False
            'interpolate': decoder_config.get('interpolate', True),  # False
            'loss_decode': decoder_config['loss_decode'],  # dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
            'decoder_params': decoder_config['decoder_params'],
        }
        if 'unsup_loss_decode' in decoder_config:
            head_cfg['unsup_loss_decode'] = decoder_config['unsup_loss_decode']
        if 'target_loss_decode' in decoder_config:
            head_cfg['target_loss_decode'] = decoder_config['target_loss_decode']
        
        attn_cfg = deepcopy(head_cfg)
        if self.single_scale_head == 'DAFormerHead':
            attn_cfg['channels'] = self.attention_embed_dim  # 256
            attn_cfg['decoder_params']['embed_dims'] = self.attention_embed_dim  #256
            if attn_cfg['decoder_params']['fusion_cfg']['type'] == 'aspp':
                attn_cfg['decoder_params']['fusion_cfg'] = dict(
                    type='conv',
                    kernel_size=1,
                    act_cfg=dict(type='ReLU'),
                    norm_cfg=attn_cfg['decoder_params']['fusion_cfg']
                    ['norm_cfg'])
            self.os = 4  # 512 / 128 = 4
        elif self.single_scale_head == 'SegFormerHead':
            attn_cfg['channels'] = self.attention_embed_dim  # 256
            attn_cfg['decoder_params']['embed_dim'] = self.attention_embed_dim  #256
            self.os = 4  # 512 / 128 = 4
        elif self.single_scale_head == 'DLV2Head':
            # kwargs['init_cfg'] = None
            # kwargs.pop('dilations')
            # kwargs['channels'] = 1
            self.os = 8
        else:
            raise NotImplementedError(self.single_scale_head)
        
        head_cfg['type'] = self.single_scale_head
        self.head = eval(head_cfg['type'])(head_cfg)  # DAFormerHead aspp

        attn_cfg['type'] = self.single_scale_head
        if not self.attention_classwise:  # False
            attn_cfg['num_classes'] = 1
        if self.fixed_attention is None:
            self.scale_attention = eval(attn_cfg['type'])(attn_cfg)  # DAFormerHead conv
        else:
            self.scale_attention = None
            # self.fixed_attention = fixed_attention
        self.debug = False
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
        elif 'decoder' in checkpoint:
            state_dict = checkpoint['decoder']
        else:
            state_dict = checkpoint
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, False)
        self.logger.info(f'Load decoder checkpoint from pretrained {pretrained}.')
        
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

    def set_hr_crop_box(self, boxes):
        """Store the high-resolution crop box selected by the segmentor.

        保存 segmentor 选择的高分辨率 crop box。
        """
        self.hr_crop_box = boxes

    def hr_crop_slice(self, scale):
        """Return feature-map slices for the stored crop box.

        根据已保存 crop box 返回特征图上的切片。
        """
        crop_y1, crop_y2, crop_x1, crop_x2 = scale_box(self.hr_crop_box, scale)
        return slice(crop_y1, crop_y2), slice(crop_x1, crop_x2)

    def resize(self, input, scale_factor):
        """Resize logits or attention maps with the head interpolation policy.

        使用当前 head 的插值策略 resize logits 或注意力图。
        """
        return _resize(
            input=input,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=self.head.align_corners)

    def decode_hr(self, inp, bs):
        """Decode high-resolution features and merge crop logits if needed.

        解码高分辨率特征；如果输入来自滑窗 crop，则将 crop logits 聚合回整图。
        """
        if isinstance(inp, dict) and 'boxes' in inp.keys():
            # multi-level, crop * bs, c, h, w
            features = inp['features']  # [18,1024,128,128], [18,1024,64,64], [18,1024,32,32], [18,1024,16,16]
            # inp['boxes'] is a list of boxes, each box is a list of [y1, y2, x1, x2], [[y1, y2, x1, x2], ...]
            boxes = inp['boxes']
            dev = features[0][0].device
            h_img, w_img = 0, 0  # 256, 256
            for i in range(len(boxes)):
                boxes[i] = scale_box(boxes[i], self.os)  # scale boxes from original image to feature map size, self.os = 4
                y1, y2, x1, x2 = boxes[i]
                if h_img < y2:
                    h_img = y2
                if w_img < x2:
                    w_img = x2
            preds = torch.zeros((bs, self.head.num_classes, h_img, w_img), device=dev)  # [2, 19, 256, 256]
            count_mat = torch.zeros((bs, 1, h_img, w_img), device=dev)  # [2, 1, 256, 256]

            crop_seg_logits = self.head(features)  # [18, 19, 128, 128]
            for i in range(len(boxes)):
                y1, y2, x1, x2 = boxes[i]
                crop_seg_logit = crop_seg_logits[i * bs:(i + 1) * bs]  # [2, 19, 128, 128]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))  # pad 0 to the left, right, top, bottom

                count_mat[:, :, y1:y2, x1:x2] += 1

            assert (count_mat == 0).sum() == 0
            preds = preds / count_mat
            return preds  # [2, 19, 256, 256]
        else:
            return self.head(inp)

    def get_scale_attention(self, inp):
        """Predict or reuse the low/high-resolution fusion attention.

        预测或复用低/高分辨率融合注意力。
        """
        if self.scale_attention is not None:  # default this
            att = torch.sigmoid(self.scale_attention(inp))
        else:
            att = self.fixed_attention
        return att

    def forward(self, inputs):
        """Fuse low-resolution and high-resolution branch outputs.

        融合低分辨率与高分辨率分支输出。
        """
        assert len(inputs) == 2
        hr_inp = inputs[1]  # hight resolution, list of four scale features for detail image [512, 512] (crop from 1024 1024)
        hr_scale = self.scales[1]  # 1
        lr_inp = inputs[0]  # low resolution, list of four scale features for context image [512, 512] (resize from 1024 1024)
        lr_sc_att_inp = inputs[0]  # low resolution four scale features, used for scale attention
        lr_scale = self.scales[0]  # 0.5
        batch_size = lr_inp[0].shape[0]  # 2
        assert lr_scale <= hr_scale

        has_crop = self.hr_crop_box is not None  # True
        if has_crop:
            crop_y1, crop_y2, crop_x1, crop_x2 = self.hr_crop_box
            # 将crop_box除以2的示例
            # half_crop_box = tuple(coord // 2 for coord in self.hr_crop_box)

        # print_log(f'lr_inp {[f.shape for f in lr_inp]}', 'mmseg')
        lr_seg = self.head(lr_inp)  # [2, 19, 128, 128] [3,19,128,128] for acdc eval
        # print_log(f'lr_seg {lr_seg.shape}', 'mmseg')

        hr_seg = self.decode_hr(hr_inp, batch_size)  # [2, 19, 128, 128] for hr training logits, [2, 19, 256, 256] for ema logits [3, 19, 256, 256] for acdc test

        att = self.get_scale_attention(lr_sc_att_inp)  # [2, 19, 128, 128] for gta eval, [3, 19, 128, 128] for acdc eval
        if has_crop:  # True for model train, False for EMA model infer
            mask = lr_seg.new_zeros([lr_seg.shape[0], 1, *lr_seg.shape[2:]])  # [2, 1, 128, 128]
            sc_os = self.os / (lr_scale / hr_scale)  # 8, from lr image to original image size
            slc = self.hr_crop_slice(sc_os)  # scale hr crop box to low resolution scale
            mask[:, :, slc[0], slc[1]] = 1
            att = att * mask  # [2, 19, 128, 128], select the attention region for the crop box
        # print_log(f'att {att.shape}', 'mmseg')
        lr_seg = (1 - att) * lr_seg  # [2, 19, 128, 128]
        # print_log(f'scaled lr_seg {lr_seg.shape}', 'mmseg')
        up_lr_seg = self.resize(lr_seg, hr_scale / lr_scale)  # [2, 19, 256, 256]
        if torch.is_tensor(att):
            att = self.resize(att, hr_scale / lr_scale)  # [2, 19, 256, 256]

        if has_crop:  # True for model train
            hr_seg_inserted = torch.zeros_like(up_lr_seg)  # [2, 19, 256, 256]
            slc = self.hr_crop_slice(self.os)  # scale hr crop box to output stride scale
            hr_seg_inserted[:, :, slc[0], slc[1]] = hr_seg  # [2, 19, 256, 256]
        else:  # True for model infer
            hr_seg_inserted = hr_seg

        if hr_seg_inserted.shape != up_lr_seg.shape: ###new added
            hr_seg_inserted = F.interpolate(hr_seg_inserted, up_lr_seg.shape[2:])
        fused_seg = att * hr_seg_inserted + up_lr_seg  # [2, 19, 256, 256]

        if self.debug_output_attention:  # False 
            att = torch.sum(
                att * torch.softmax(fused_seg, dim=1), dim=1, keepdim=True)
            return att, None, None

        if self.debug:
            if 'High Res' in self.debug_output:
                self.debug_output['High Res'] = torch.cat(
                    [self.debug_output['High Res'], hr_seg.detach()], dim=0)
                self.debug_output['High Res Inserted'] = torch.cat(
                    [self.debug_output['High Res Inserted'], hr_seg_inserted.detach()], dim=0)
                self.debug_output['Low Res'] = torch.cat(
                    [self.debug_output['Low Res'], lr_seg.detach()], dim=0)
                self.debug_output['Fused'] = torch.cat(
                    [self.debug_output['Fused'], fused_seg.detach()], dim=0)
            else:
                self.debug_output.update({
                    'High Res':
                    torch.max(hr_seg, dim=1)[1].detach().cpu().numpy(),  # (19, 128, 128)
                    'High Res Inserted':
                    torch.max(hr_seg_inserted, dim=1)[1].detach().cpu().numpy(),  # (19, 256, 256)
                    'Low Res': 
                    torch.max(lr_seg, dim=1)[1].detach().cpu().numpy(),  # (19, 128, 128)
                    'Fused':
                    torch.max(fused_seg, dim=1)[1].detach().cpu().numpy(),  # (19, 256, 256)
                })
            if torch.is_tensor(att):
                if 'Attention' in self.debug_output:
                    self.debug_output['Attention'] = np.concatenate(
                        [self.debug_output['Attention'],
                         torch.sum(att * torch.softmax(fused_seg, dim=1), dim=1).detach().cpu().numpy()], axis=0)
                else:
                    self.debug_output['Attention'] = torch.sum(
                        att * torch.softmax(fused_seg, dim=1), dim=1,
                        keepdim=True).detach().cpu().numpy()

        return fused_seg, lr_seg, hr_seg

    def reset_crop(self):
        """Clear the stored HR crop after loss calculation.

        在损失计算后清理已保存的高分辨率 crop。
        """
        del self.hr_crop_box
        self.hr_crop_box = None

    def forward_train(self,
                      inputs,):
        """Forward function for training.

        训练阶段前向函数。
        """
        if self.enable_hr_crop:
            assert self.hr_crop_box is not None
        seg_logits = self.forward(inputs)
        # self.reset_crop()
        return seg_logits  # three seg logits: fused_seg, lr_seg, hr_seg

    def forward_test(self, inputs):
        """Forward function for testing, only `fused_seg` is used.

        测试阶段前向函数，只返回 `fused_seg`。
        """
        return self.forward(inputs)[0]

    def cal_loss(self, seg_logit, seg_label, seg_weight=None, loss_key=None):
        """Compute fused, low-resolution, and high-resolution losses.

        计算融合输出、低分辨率分支和高分辨率分支的损失。
        """
        fused_seg, lr_seg, hr_seg = seg_logit  # [2, 19, 256, 256], [2, 19, 128, 128], [2, 19, 128, 128]
        
        # check if the shapes match
        seg_weight = match_shape(seg_weight, seg_label.shape[-2:], mode='bilinear', align_corners=self.head.align_corners) if seg_weight is not None else seg_weight
            
        fused_seg = match_shape(fused_seg, seg_label.shape[-2:], mode='bilinear', align_corners=self.head.align_corners)
        
        loss_dict = self.head.cal_loss(
            fused_seg, seg_label, seg_weight, loss_key=loss_key)
        if self.hr_loss_weight == 0 and self.lr_loss_weight == 0:
            return loss_dict

        if self.lr_loss_weight > 0:  # 0
            lr_seg = match_shape(lr_seg, seg_label.shape[-2:], mode='bilinear', align_corners=self.head.align_corners)
            loss_dict.update(add_prefix(
                self.head.cal_loss(lr_seg, seg_label, seg_weight, loss_key=loss_key),
                'lr'))
            
        
        if self.hr_loss_weight > 0 and self.enable_hr_crop:  # 0.1 True
            scale_hr_crop_box = scale_box(self.hr_crop_box, self.scales[1])  # scale crop box to input size
            cropped_seg_label = crop(seg_label, scale_hr_crop_box)  # [2, 1, 512, 512]
            if seg_weight is not None:
                cropped_seg_weight = crop(seg_weight, scale_hr_crop_box)
                self.debug_output['Cropped GT Weight'] = cropped_seg_weight.squeeze(1).detach().cpu().numpy()
            else:
                cropped_seg_weight = seg_weight
            self.debug_output['Cropped GT'] = cropped_seg_label.squeeze(1).detach().cpu().numpy()
            
            # check if the shapes match
            hr_seg = match_shape(hr_seg, cropped_seg_label.shape[-2:], mode='bilinear', align_corners=self.head.align_corners)
            loss_dict.update(add_prefix(
                self.head.cal_loss(
                    hr_seg,
                    cropped_seg_label,
                    cropped_seg_weight,
                    loss_key=loss_key),
                'hr'))
        elif self.hr_loss_weight > 0:
            # check if the shapes match
            hr_seg = match_shape(hr_seg, seg_label.shape[-2:], mode='bilinear', align_corners=self.head.align_corners)
            loss_dict.update(add_prefix(
                self.head.cal_loss(hr_seg, seg_label, seg_weight, loss_key=loss_key),
                'hr'))
            
        loss_dict['seg_loss'] *= (1 - self.lr_loss_weight - self.hr_loss_weight)  # 0.9
        if self.lr_loss_weight > 0:
            loss_dict['lr_seg_loss'] *= self.lr_loss_weight
        if self.hr_loss_weight > 0:
            loss_dict['hr_seg_loss'] *= self.hr_loss_weight  # 0.1
            
        if self.debug:
            self.debug_output['GT'] = seg_label.squeeze(1).detach().cpu().numpy()
            # Remove debug output from cross entropy loss, maybe no use
            self.debug_output.pop('Seg. Pred.', None)
            self.debug_output.pop('Seg. GT', None)
            
        self.reset_crop()
        
        return loss_dict


def save_model_params_summary(model, filename="param_snapshot.txt", show_values=20):
    with open(filename, "w") as f:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            data = param.data.view(-1).cpu().numpy()
            f.write(f"Name: {name}\n")
            f.write(f"  Shape: {list(param.shape)}\n")
            f.write(f"  Mean: {data.mean():.6f}, Std: {data.std():.6f}, Min: {data.min():.6f}, Max: {data.max():.6f}\n")
            shown_values = data[:min(len(data), show_values)]
            f.write(f"  Values (first {len(shown_values)}): {np.array2string(shown_values, precision=4, separator=', ')}\n")
            f.write("-" * 80 + "\n")

if __name__ == '__main__':
    # in_channels= [64, 128, 320, 512]
    in_channels = [1024, 1024, 1024, 1024]
    config = {
    'single_scale_head': 'DAFormerHead',
    'lr_loss_weight': 0,
    'hr_loss_weight': 0.1,
    'scales': [0.5, 1],
    'attention_embed_dim': 256,
    'attention_classwise': True,
    'enable_hr_crop': True,
    'hr_crop_size': [512, 512],
    'hr_slide_inference': True,
    'hr_slide_overlapping': True,
    'crop_coord_divisible': 8,
    'blur_hr_crop': False,
    'feature_scale': 0.5,
    'fixed_attention': None,
    'debug_output_attention': False,
    'in_channels': in_channels,
    'in_index': [0, 1, 2, 3],
    'channels': 256,
    'dropout_ratio': 0.1,
    'num_classes': 19,
    'norm_cfg': dict(type='BN', requires_grad=True),
    'align_corners': False,
    'interpolate': False,
    'decoder_params': {
        'embed_dims': 256,
        'embed_cfg': {'type': 'mlp', 'act_cfg': None, 'norm_cfg': None},
        'embed_neck_cfg': {'type': 'mlp', 'act_cfg': None, 'norm_cfg': None},
        'fusion_cfg': {
            'type': 'aspp',
            'sep': True,
            'dilations': (1, 6, 12, 18),
            'pool': False,
            'act_cfg': dict(type='ReLU'),
            'norm_cfg': dict(type='BN', requires_grad=True)
        }
    },
    'loss_decode': {
        'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0
    }
    }
    
    model = HRDAHead(config)
    total_params = sum(p.numel() for p in model.parameters())  # 总参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数量
    non_trainable_params = total_params - trainable_params  # 非可训练参数量

    print(f"total params: {total_params}")  # 4498707 for 1024 * 4 channels, 4242982
    print(f"trainable params: {trainable_params}")  # 4498707, 4242982
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
    
    seg_logits = model([x, x])
    if isinstance(seg_logits, tuple):
        for i, seg_logit in enumerate(seg_logits):
            print(f'seg_logits[{i}].shape: {seg_logit.shape}')
        # fused_seg, lr_seg, hr_seg
        # [2, 19, 256, 256], # [2, 19, 128, 128], [2, 19, 128, 128]
    else:
        print(f'seg_logits.shape: {seg_logits.shape}')  # [2, 19, 128, 128]
    # seg_logits = model(inputs)
    # print(f'seg_logits.shape: {seg_logits.shape}')  # [2, 19, 128, 128]
