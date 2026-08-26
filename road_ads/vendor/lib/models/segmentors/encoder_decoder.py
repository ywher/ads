# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0
# Modifications: Support for seg_weight
"""Base encoder-decoder segmentor.

基础 encoder-decoder 分割模型。

This module defines the common segmentation path used by supervised training,
semi training, pseudo-label generation, and evaluation. Inputs are normalized
image tensors in `NCHW` format; logits use `N x num_classes x H x W`.

本模块定义监督训练、semi 训练、伪标签生成和评估共用的分割主流程。输入为
`NCHW` 格式的归一化图像张量；分割 logits 的形状为
`N x num_classes x H x W`。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import copy
import numpy as np
from matplotlib import pyplot as plt

from lib.models.model_utils.funcs import resize, add_prefix, crop
from lib.models.model_utils.token_masking import TokenMasking
from lib.models.model_utils.dacs_transforms import  get_mean_std_self
from lib.models.model_utils.visualization import subplotimg, prepare_debug_out

img_mean = (123.675, 116.28, 103.53)
img_std = (58.395, 57.12, 57.375)

class EncoderDecoder(nn.Module):
    """Standard backbone + decode-head segmentor.

    标准 backbone + decode head 分割模型。

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    `backbone` extracts one tensor or a list of multi-level feature tensors,
    `decode_head` maps those features to semantic logits, and the optional
    auxiliary head only contributes extra training supervision.

    `backbone` 提取单层或多层特征，`decode_head` 将特征映射为语义分割
    logits，可选 auxiliary head 只在训练阶段提供额外监督。
    """
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 token_mask_ratio=None,
                 train_cfg=None,
                 test_cfg=None,
                 ):
        super(EncoderDecoder, self).__init__()
        self.logger = logging.getLogger()
        
        # backbone
        self.backbone = backbone
        self.token_mask_ratio = token_mask_ratio
        if self.token_mask_ratio is not None:
            self.token_masking = TokenMasking(self.token_mask_ratio)
        
        # neck, normally no neck for encoder-decoder
        self.with_neck = False
        if neck is not None:
            self.neck = neck
            self.with_neck = True
            
        # init decoder head
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.automatic_debug = True
        # self.debug = True
        # self.debug_output = {}
        if train_cfg is not None and 'log_config' in train_cfg:
            self.debug_img_interval = train_cfg['log_config'].get('img_interval', 1000)
        self.local_iter = 0
        
        self.logger.info(f'EncoderDecoder model has been initialized')
        # print(f'EncoderDecoder model has been initialized')

    # init decode head
    def _init_decode_head(self, decode_head):
        """Initialize decode head metadata.

        初始化 decode head 及其元信息。
        """
        self.decode_head = decode_head
        # Some wrapper heads expose the real segmentation head as `.head`.
        # 部分包装型 head 会把真实分割头放在 `.head` 中。
        if hasattr(decode_head, 'head'):
            self.align_corners = decode_head.head.align_corners
            self.num_classes = decode_head.head.num_classes
        else:
            # Plain heads store metadata directly on themselves.
            # 普通 head 直接在自身保存元信息。
            self.align_corners = decode_head.align_corners
            self.num_classes = decode_head.num_classes
    
    # init auxiliary head for deep supervision
    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize optional auxiliary head.

        初始化可选辅助分割头。
        """
        self.with_auxiliary_head = auxiliary_head is not None
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(head_cfg)
            else:
                self.auxiliary_head = auxiliary_head
        else:
            self.auxiliary_head = None

    def extract_feat(self, img, enable_token_masking=False):
        """Extract backbone features from an image batch.

        从一个图像 batch 中提取 backbone 特征。

        Args:
            img (Tensor): Normalized image tensor with shape `N x C x H x W`.
                形状为 `N x C x H x W` 的归一化图像张量。
            enable_token_masking (bool): Whether to apply token masking during
                training when `token_mask_ratio` is configured.
                当配置了 `token_mask_ratio` 时，是否在训练阶段启用 token masking。

        Returns:
            Tensor | list[Tensor] | tuple: Raw backbone features after optional
            neck processing. 返回经过可选 neck 处理后的 backbone 特征。
        """
        if self.token_mask_ratio is not None and enable_token_masking and self.training:
            # print(img.shape)
            B, _, _, _ = img.shape
            token_length = self.backbone.patch_embed.num_patches
            masks = self.token_masking(shape=(B, token_length)).to(img.device)
            # record the ratio of masks  # (B, L)
            # self.logger.info(f'Token masking ratio: {masks.sum().item() / (B * token_length)}')
            x = self.backbone(img, masks=masks)
        else:
            x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x
    
    def generate_pseudo_label(self, img):
        """Generate raw segmentation logits for pseudo-labeling.

        生成用于伪标签的原始分割 logits。
        """
        # self.update_debug_state()
        # if self.debug:
        #     self.debug_output = {
        #         'Image': img,
        #     }
        out = self.encode_decode(img)
        if isinstance(out, dict):
            out = out['seg_logits']
        elif isinstance(out, (tuple, list)):
            out = out[0]
        # if self.debug:
        #     self.debug_output.update(self.decode_head.debug_output)
        #     self.debug_output['Pred'] = out.cpu().numpy()

        return out
    
    def generate_cls_logits(self, img, upscale_pred=False):
        """Generate class probabilities from segmentation logits.

        根据分割 logits 生成类别概率。
        """
        out = self.encode_decode(img, upscale_pred=upscale_pred)
        out = F.softmax(out, dim=1)
        return out

    def _build_complementary_dropout_mask(self, feat, kept_ratio=0.5):
        """Build paired channel masks for UniMatch-style complementary dropout."""
        batch_size, channels = feat.shape[:2]
        if batch_size < 2 or batch_size % 2 != 0:
            return None

        half = batch_size // 2
        mask1 = torch.bernoulli(
            feat.new_full((half, channels), 0.5)) * 2.0
        mask2 = 2.0 - mask1

        kept = int(round(half * float(kept_ratio)))
        if kept > 0:
            kept = min(kept, half)
            keep_idx = torch.randperm(half, device=feat.device)[:kept]
            mask1[keep_idx] = 1.0
            mask2[keep_idx] = 1.0

        return torch.cat([mask1, mask2], dim=0).view(
            batch_size, channels, *([1] * (feat.dim() - 2)))

    def _apply_complementary_dropout(self, feats, kept_ratio=0.5):
        """Apply complementary channel dropout to tensor/list feature trees."""
        if torch.is_tensor(feats):
            mask = self._build_complementary_dropout_mask(
                feats, kept_ratio=kept_ratio)
            return feats if mask is None else feats * mask

        if isinstance(feats, list):
            if feats and all(torch.is_tensor(feat) for feat in feats):
                mask = self._build_complementary_dropout_mask(
                    feats[0], kept_ratio=kept_ratio)
                if mask is not None:
                    return [
                        feat * mask if feat.shape[:2] == feats[0].shape[:2]
                        else self._apply_complementary_dropout(
                            feat, kept_ratio=kept_ratio)
                        for feat in feats
                    ]
            return [
                self._apply_complementary_dropout(
                    feat, kept_ratio=kept_ratio)
                for feat in feats
            ]

        if isinstance(feats, tuple):
            return tuple(
                self._apply_complementary_dropout(
                    feat, kept_ratio=kept_ratio)
                for feat in feats
            )

        if isinstance(feats, dict):
            return {
                key: (
                    self._apply_complementary_dropout(
                        value, kept_ratio=kept_ratio)
                    if key == 'features' else value
                )
                for key, value in feats.items()
            }

        return feats

    def encode_decode(self, img, return_feat=False, enable_token_masking=False, upscale_pred=True, comp_drop=False):
        """Encode images with backbone and decode into a semantic segmentation map.

        使用 backbone 编码图像，并通过 decode head 解码为语义分割图。
        
        Args:
            img (Tensor): Input tensor with shape `N x C x H x W`.
                输入图像张量，形状为 `N x C x H x W`。
            return_feat (bool): Whether to return backbone features.
                是否返回 backbone 特征。
            enable_token_masking (bool): Whether to enable token masking.
                是否启用 token masking。
            upscale_pred (bool): Whether to resize logits to input image size.
                是否将 logits 上采样到输入图像尺寸。
            
        Returns:
            dict: Dictionary containing output tensors with keys:
                - seg_logits: Segmentation logits
                - aux_logits: Auxiliary head outputs (optional) 
                - features: Backbone features (optional)

            返回字典，包含分割 logits、可选辅助头输出和可选 backbone 特征。
        """
        output_dict = {}
        
        # extract features from backbone
        x = self.extract_feat(img, enable_token_masking=enable_token_masking)
        if comp_drop:
            kept_ratio = 0.5 if isinstance(comp_drop, bool) else \
                float(comp_drop.get('kept_ratio', 0.5))
            x = self._apply_complementary_dropout(
                x, kept_ratio=kept_ratio)
        
        # get segmentation logits from decode head
        out = self.decode_head(x)
        
        if upscale_pred:
            if isinstance(out, dict):
                out = {k: self._resize_if_needed(v, img.shape[2:]) for k, v in out.items()}
            elif isinstance(out, list):
                out = [self._resize_if_needed(v, img.shape[2:]) for v in out]
            else:
                out = self._resize_if_needed(out, img.shape[2:])  # from (2, 19, 128, 128) to (2, 19, 512, 512)
                
        output_dict['seg_logits'] = out
        
        # auxiliary head for mlcls task
        if self.auxiliary_head is not None:
            aux_out = self.auxiliary_head(x)
            output_dict['aux_logits'] = aux_out
        
        if return_feat:
            output_dict['features'] = x
            
        return output_dict

    def forward_train(self, 
                      data_batch, 
                      seg_weight=None, 
                      return_feat=False,
                      enable_token_masking=False, 
                      upscale_pred=True,
                      loss_key=None,
                      comp_drop=False):
        """Forward function for training.

        训练阶段前向函数。

        Args:
            data_batch (tuple): (img, gt_semantic_seg) tuple containing:
                - img (Tensor): Input images of shape (N, C, H, W)
                - gt_semantic_seg (Tensor): GT semantic segmentation of shape (N, H, W)
            seg_weight (Tensor): Pixel-wise segmentation weight, used for loss calculation
            return_feat (bool): Whether to return backbone features
            enable_token_masking (bool): Whether to enable token masking
            upscale_pred (bool): Whether to upscale predictions to input size

        Returns:
            dict: Dictionary containing losses and intermediate results with keys:
                - seg_loss: Main segmentation loss
                - aux_loss: Auxiliary segmentation loss (if auxiliary head exists)
                - features: Backbone features (if return_feat=True)
                
                remove the following keys
                - seg_logits: Segmentation logits
                - aux_logits: Auxiliary head logits (if auxiliary head exists)
                
        """
        # Extract image and ground truth from batch
        img, gt_semantic_seg = data_batch
        
        results = {}
        
        # Get model predictions and features
        output_dict = self.encode_decode(img, 
                                       return_feat=return_feat, 
                                       enable_token_masking=enable_token_masking, 
                                       upscale_pred=upscale_pred,
                                       comp_drop=comp_drop)
        
        # Calculate main segmentation loss
        if gt_semantic_seg is not None:
            seg_loss_dict = self.decode_head.cal_loss(
                output_dict['seg_logits'],
                gt_semantic_seg,
                seg_weight,
                loss_key=loss_key,
            )
        
            results.update(seg_loss_dict)
        results['seg_logits'] = output_dict['seg_logits']
        
        # Calculate auxiliary loss if auxiliary head exists
        if self.auxiliary_head is not None and 'aux_logits' in output_dict:
            aux_loss_dict = self.auxiliary_head.cal_loss(
                output_dict['aux_logits'],
                gt_semantic_seg,
                seg_weight
            )
            results.update(add_prefix(aux_loss_dict, 'aux'))
        
        if return_feat:
            results['features'] = output_dict['features']
        
        self.local_iter += 1
        
        return results
    
    def process_debug(self, img):
        self.debug_output = {
            'Image': img,
            **self.decode_head.debug_output,
        }
        if self.auxiliary_head is not None:
            self.debug_output.update(
                add_prefix(self.auxiliary_head.debug_output, 'Aux'))
        if self.automatic_debug:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'encdec_debug')
            os.makedirs(out_dir, exist_ok=True)
            means, stds = get_mean_std_self(img_mean, img_std, img.shape[0], img.device)
            for j in range(img.shape[0]):
                rows, cols = 1, len(self.debug_output)
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.92,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                for k, (n, v) in enumerate(self.debug_output.items()):
                    subplotimg(axs[k], **prepare_debug_out(n, v[j], means, stds))
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
            del self.debug_output

    def _get_crop_coords(self, idx, stride, crop_size, img_size):
        """Calculate crop coordinates for sliding-window inference.

        计算滑窗推理中单个窗口的裁剪坐标。
        """
        start = idx * stride
        end = min(start + crop_size, img_size)
        start = max(end - crop_size, 0)
        return start, end
    
    def slide_inference(self, img, rescale=None):
        """Inference by sliding-window with overlap.

        使用带重叠区域的滑窗方式进行推理。

        Args:
            img (dict): A dictionary containing:
                - 'img': The input image tensor of shape (N, C, H, W).
                - 'lb_shape': The original label shape (H, W).
            crop_size (tuple): The size of the sliding window (height, width).
            rescale (bool, optional): Whether to rescale the output to the original shape.

        Returns:
            Tensor: The output segmentation map after sliding-window inference.

            返回经过重叠区域平均后的分割 logits。
        """
        # Extract the original label shape and image tensor
        lb_shape = img['lb_shape']  # (1024, 2048)
        lb = img.get('lb', None)  # (1, 1024, 2048), optional for inference
        img = img['img']  # (1, 3, 1024, 2048)
        # print(f"lb shape: {lb_shape}, img shape: {img.shape}")
        

        # Define the sliding window stride and crop size
        h_stride, w_stride = self.test_cfg['stride']  # hrda infer: 512, 512; 256, 256
        h_crop, w_crop = self.test_cfg['crop_size']  # hrda infer: 1024, 1024; 512, 512
        batched_slide = self.test_cfg.get('batched_slide', False)  # True for hrda
        half_batched_slide = self.test_cfg.get('half_batched_slide', False)  # True for hrda

        # Get the dimensions of the input image
        batch_size, _, h_img, w_img = img.size()  # e.g., [1, 3, 1080, 1920] for hrda; [1, 3, 512, 1024]
        num_classes = self.num_classes  # 19

        # Calculate the number of sliding windows needed in height and width
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1  # eval: 2 for acdc hrda
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1  # eval: 3 for acdc hrda

        # Initialize tensors to store predictions and count overlaps
        device = img.device
        dtype = img.dtype
        preds = torch.zeros((batch_size, num_classes, h_img, w_img), 
                        device=device, dtype=dtype)  # (1, 19, 1080, 1920) for acdc hrda
        count_mat = torch.zeros((batch_size, 1, h_img, w_img), 
                            device=device, dtype=torch.int16)  # (1, 1, 1080, 1920) for acdc hrda

        # Perform sliding-window inference
        if batched_slide:
            crop_imgs = []
            crops = []
            
            # 预先计算所有裁剪坐标
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1, y2 = self._get_crop_coords(h_idx, h_stride, h_crop, h_img)
                    x1, x2 = self._get_crop_coords(w_idx, w_stride, w_crop, w_img)
                    
                    crop_img = img[:, :, y1:y2, x1:x2]  # [1, 3, 1024, 1024]
                    crop_imgs.append(crop_img)
                    crops.append((y1, y2, x1, x2))
            
            # 使用torch.cat一次性拼接，减少内存操作
            # [6, 3, 1024, 1024] for acdc hrda;
            # Example batched crop tensor for semi training
            crop_imgs = torch.cat(crop_imgs, dim=0)  
            
            with torch.no_grad():
                if half_batched_slide:
                    # For half-batched sliding window, we split the crops into two halves and infer them separately
                    # This is useful for large crops to avoid OOM
                    half_size = len(crops) // 2
                    first_half = self._encode_decode_with_upscale(crop_imgs[:half_size])
                    decoder_output_debug_first_half = {k: copy.deepcopy(v) for k, v in self.decode_head.debug_output.items()}
                    second_half = self._encode_decode_with_upscale(crop_imgs[half_size:])
                    for k, v in self.decode_head.debug_output.items():
                        if isinstance(v, torch.Tensor):
                            self.decode_head.debug_output[k] = torch.cat(
                                [decoder_output_debug_first_half[k], self.decode_head.debug_output[k]], dim=0)
                        elif isinstance(v, np.ndarray):
                            self.decode_head.debug_output[k] = np.concatenate(
                                [decoder_output_debug_first_half[k], self.decode_head.debug_output[k]], axis=0)
                    # 合并两个半批次的结果
                    # [6, 19, 1024, 1024] for acdc hrda;
                    crop_seg_logits = torch.cat([first_half, second_half], dim=0)
                else:
                    crop_seg_logits = self._encode_decode_with_upscale(crop_imgs)
            
            if lb is not None:
                crop_seg_lb = []
            for i, (y1, y2, x1, x2) in enumerate(crops):
                crop_seg_logit = crop_seg_logits[i * batch_size:(i + 1) * batch_size]
                preds[:, :, y1:y2, x1:x2].add_(crop_seg_logit)
                count_mat[:, :, y1:y2, x1:x2].add_(1)
                if lb is not None:
                    crop_seg_lb.append(crop(lb, (y1, y2, x1, x2)))
                    
            if lb is not None:
                # print(f"the shape of crop seg lb list: {[c.shape for c in crop_seg_lb]}")
                crop_seg_lb = torch.cat(crop_seg_lb, dim=0)  # [6, 1024, 1024] for acdc hrda
                self.decode_head.debug_output['Cropped GT'] = crop_seg_lb.squeeze(1).detach().cpu().numpy()

        else:
            for h_idx in range(h_grids):  # Iterate over height grids
                for w_idx in range(w_grids):  # Iterate over width grids
                    # Calculate the crop coordinates for the current window
                    y1, y2 = self._get_crop_coords(h_idx, h_stride, h_crop, h_img)
                    x1, x2 = self._get_crop_coords(w_idx, w_stride, w_crop, w_img)

                    # Extract the cropped image
                    crop_img = img[:, :, y1:y2, x1:x2]

                    # Perform inference on the cropped image
                    crop_seg_logit = self._encode_decode_with_upscale(crop_img)

                    preds[:, :, y1:y2, x1:x2].add_(crop_seg_logit)
                    count_mat[:, :, y1:y2, x1:x2].add_(1)
                    
                    # Add the predictions to the corresponding region in the output tensor
                    # preds += F.pad(
                    #     crop_seg_logit,
                    #     (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2))
                    # )

                    # # Update the count matrix to track overlaps
                    # count_mat[:, :, y1:y2, x1:x2] += 1

        # Ensure there are no zero values in the count matrix (all regions are covered)
        assert (count_mat == 0).sum() == 0

        # Normalize the predictions by the count matrix to handle overlaps
        # preds = preds / count_mat
        # 原地除法，避免创建新张量
        preds.div_(count_mat.clamp_(min=1).to(preds.dtype))

        # Resize the predictions to the original label shape if needed
        return self._resize_if_needed(preds, lb_shape if rescale else None)
    
    def slide_inference_with_feats(self, img, rescale=None):
        """Run sliding-window inference and aggregate backbone features.

        滑窗推理，并返回分割 logits 与聚合后的 backbone 特征。

        Overlapped regions are averaged in both segmentation logits and feature
        maps. This path is used by feature export/evaluation utilities.

        对分割 logits 和特征图的重叠区域都做平均；该路径主要服务于特征导出
        和评估分析工具。
        """
        lb_shape = img['lb_shape']
        lb = img.get('lb', None)  # 仅用于debug可视化，逻辑不依赖
        img = img['img']

        h_stride, w_stride = self.test_cfg['stride']
        h_crop, w_crop = self.test_cfg['crop_size']

        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes

        device = img.device
        dtype = img.dtype

        # 主输出累计
        preds = torch.zeros((batch_size, num_classes, h_img, w_img), device=device, dtype=dtype)
        count_mat = torch.zeros((batch_size, 1, h_img, w_img), device=device, dtype=torch.int16)

        # 特征累计容器（按层）
        feats_accum = None  # List[Tensor], 每层形状 [B, C_i, H_full_i, W_full_i]
        feats_count = None  # List[Tensor], 每层形状 [B, 1, H_full_i, W_full_i]

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1, y2 = self._get_crop_coords(h_idx, h_stride, h_crop, h_img)
                x1, x2 = self._get_crop_coords(w_idx, w_stride, w_crop, w_img)

                crop_img = img[:, :, y1:y2, x1:x2]  # [B, C, h_crop, w_crop]
                crop_seg_logit, crop_feats = self._encode_decode_with_feats(crop_img, upscale_pred=True)
                if isinstance(crop_feats, tuple) and len(crop_feats) == 3:
                    crop_feats = crop_feats[1]  # four multi-layer same-scale features

                # 聚合分割输出
                preds[:, :, y1:y2, x1:x2].add_(crop_seg_logit)
                count_mat[:, :, y1:y2, x1:x2].add_(1)

                # 初始化特征累计容器
                if feats_accum is None:
                    # 统一成 list
                    if isinstance(crop_feats, (tuple, list)):
                        feats_list = list(crop_feats)
                    else:
                        feats_list = [crop_feats]

                    feats_accum, feats_count = [], []
                    # 为每层建立与整图对齐的累计张量（按比例映射裁剪坐标）
                    for f in feats_list:
                        _, C_i, h_i, w_i = f.shape
                        # 层尺度相对于crop的比例
                        sy = h_i / float(h_crop)
                        sx = w_i / float(w_crop)
                        # 对应整图的特征尺寸
                        H_full_i = int(np.ceil(h_img * sy))
                        W_full_i = int(np.ceil(w_img * sx))
                        feats_accum.append(torch.zeros((batch_size, C_i, H_full_i, W_full_i), device=device, dtype=f.dtype))
                        feats_count.append(torch.zeros((batch_size, 1, H_full_i, W_full_i), device=device, dtype=torch.int16))

                # 累计每层特征（滑动平均）
                if not isinstance(crop_feats, (tuple, list)):
                    feats_list = [crop_feats]
                else:
                    feats_list = crop_feats

                for i, f in enumerate(feats_list):
                    _, _, h_i, w_i = f.shape
                    sy = h_i / float(h_crop)
                    sx = w_i / float(w_crop)

                    y1_i = int(round(y1 * sy))
                    x1_i = int(round(x1 * sx))
                    y2_i = y1_i + h_i
                    x2_i = x1_i + w_i

                    H_full_i = feats_accum[i].shape[-2]
                    W_full_i = feats_accum[i].shape[-1]
                    # 边界保护
                    y2_i = min(y2_i, H_full_i)
                    x2_i = min(x2_i, W_full_i)

                    feats_accum[i][:, :, y1_i:y2_i, x1_i:x2_i].add_(f[:, :, :y2_i - y1_i, :x2_i - x1_i])
                    feats_count[i][:, :, y1_i:y2_i, x1_i:x2_i].add_(1)

                # 可选：debug裁剪GT（保持与原实现一致）
                if lb is not None:
                    pass  # 不更改原有debug逻辑

        # 归一化分割输出
        preds.div_(count_mat.clamp_(min=1).to(preds.dtype))

        # 归一化特征输出
        agg_feats = []
        for acc, cnt in zip(feats_accum, feats_count):
            agg_feats.append(acc / cnt.clamp_(min=1).to(acc.dtype))

        # resize seg 到原label形状
        preds = self._resize_if_needed(preds, lb_shape if rescale else None)
        return preds, agg_feats
    
    def _extract_seg_logits(self, output):
        """Extract segmentation logits from common output containers.

        提取分割 logits，统一处理不同类型的输出。
        
        Args:
            output: encode_decode的输出，可能是tensor、tuple或dict
            
        Returns:
            torch.Tensor: 分割logits张量
        """
        if isinstance(output, (tuple, list)):
            return output[0]
        elif isinstance(output, dict):
            if isinstance(output['seg_logits'], (tuple, list)):
                return output['seg_logits'][0]
            else:
                return output['seg_logits']
        return output

    def _encode_decode_with_upscale(self, imgs):
        """Encode, decode, and resize logits to input size.

        编码解码并上采样，支持单张或批量图像处理。
        
        Args:
            imgs (torch.Tensor): 输入图像张量，形状为 [N, C, H, W]
            
        Returns:
            torch.Tensor: 上采样后的分割logits
        """
        output = self.encode_decode(imgs, upscale_pred=True)  # [3, 3, 1024, 1024] for acdc hrda infer,
        return self._extract_seg_logits(output)
    
    def _encode_decode_with_feats(self, imgs, upscale_pred=True):
        """Encode/decode and return both logits and backbone features.

        编码解码并同时返回分割 logits 与 backbone 特征。
        """
        output = self.encode_decode(imgs, return_feat=True, upscale_pred=upscale_pred)
        seg_logit = self._extract_seg_logits(output)
        feats = output.get('features', None)
        return seg_logit, feats

    def whole_inference(self, img, rescale=None, upscale_pred=True):
        """Run full-image inference without sliding windows.

        使用整图推理，不进行滑窗裁剪。
        """
        lb_shape = img['lb_shape']
        img = img['img']
        seg_logit = self.encode_decode(img, upscale_pred=upscale_pred)
        seg_logit = self._extract_seg_logits(seg_logit)
        return self._resize_if_needed(seg_logit, lb_shape if rescale else None)
    
    def whole_inference_with_feats(self, img, rescale=None, upscale_pred=True):
        """Run full-image inference and return raw backbone features.

        整图推理并返回原始 backbone 特征，不做滑窗聚合。
        """
        lb_shape = img['lb_shape']
        img = img['img']
        seg_logit, feats = self._encode_decode_with_feats(img, upscale_pred=upscale_pred)
        if isinstance(feats, tuple) and len(feats) == 3:
            feats = feats[1]  # four layer same resolution feats
        seg_logit = self._resize_if_needed(seg_logit, lb_shape if rescale else None)
        # 特征直接返回（保持原生分辨率/多层列表）
        if not isinstance(feats, (tuple, list)):
            feats = [feats]
        return seg_logit, list(feats)
    
    def _resize_if_needed(self, seg_logit, target_size=None):
        """Resize segmentation logits if `target_size` is provided.

        如果提供了 `target_size`，则将分割 logits resize 到目标尺寸。
        """
        if target_size is not None and seg_logit.shape[-2:] != target_size:
            return resize(
                seg_logit,
                size=target_size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False
            )
        return seg_logit

    def inference(self, img, rescale=None, return_backbone_feat=False):
        """Run inference with the configured `whole` or `slide` mode.

        按配置的 `whole` 或 `slide` 模式执行推理。

        When `return_backbone_feat=True`, this returns both class probabilities
        and backbone features. Otherwise it returns class probabilities only.

        当 `return_backbone_feat=True` 时，同时返回类别概率和 backbone 特征；
        否则只返回类别概率。
        """
        assert self.test_cfg['mode'] in ['slide', 'whole']
        
        was_training = self.training
        if was_training:
            self.eval()
        
        try:
            with torch.no_grad():
                if return_backbone_feat:
                    if self.test_cfg['mode'] == 'slide':
                        seg_logit, agg_feats = self.slide_inference_with_feats(img, rescale=rescale)
                    else:
                        seg_logit, agg_feats = self.whole_inference_with_feats(img, rescale=rescale, upscale_pred=True)
                else:
                    if self.test_cfg['mode'] == 'slide':
                        seg_logit = self.slide_inference(img, rescale=rescale)
                    else:
                        seg_logit = self.whole_inference(img, rescale=rescale)

                if hasattr(self.decode_head, 'debug_output_attention') and \
                        self.decode_head.debug_output_attention:
                    output = seg_logit
                else:
                    output = F.softmax(seg_logit, dim=1)

                if self.test_cfg.get('scale', 1.0) != 1.0 and rescale:
                    output = resize(
                        output,
                        scale_factor=self.test_cfg['scale'],
                        mode='bilinear',
                        align_corners=self.align_corners,
                        warning=False
                    )
        finally:
            if was_training:
                self.train()

        if return_backbone_feat:
            return output, agg_feats
        return output
    
    def inference_cls(self, img, rescale=None):
        """Inference with slide/whole style.

        使用 whole/slide 风格进行类别 logits 推理。

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.

        """
        assert self.test_cfg['mode'] in ['slide', 'whole']
        cls_logit = self.whole_inference(img, rescale=None, upscale_pred=False)

        return cls_logit

    def simple_test(self, img, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
