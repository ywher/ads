"""HRDA encoder-decoder segmentor.

HRDA 多尺度 encoder-decoder 分割模型。

HRDA combines a low-resolution context branch and a high-resolution detail
branch. During training the detail branch usually uses one aligned crop; during
inference it can use sliding-window crops and fuse them in the decode head.

HRDA 会组合低分辨率上下文分支和高分辨率细节分支。训练时细节分支通常
使用一个对齐 crop；推理时可以使用滑窗 crop，并在 decode head 中融合。
"""

import numpy as np
import torch

from .encoder_decoder import EncoderDecoder
from lib.models.model_utils.funcs import resize, match_shape, crop, get_crop_bbox

class HRDAEncoderDecoder(EncoderDecoder):
    """Multi-scale HRDA segmentor built on top of `EncoderDecoder`.

    基于 `EncoderDecoder` 的多尺度 HRDA 分割模型。

    `self.scales` describes the image scales consumed by the decode head. A
    common setting is `[0.5, 1.0]`, where the first branch provides global
    context and the second branch provides high-resolution local detail.

    `self.scales` 描述 decode head 消费的图像尺度。常见设置是
    `[0.5, 1.0]`：第一个分支提供全局上下文，第二个分支提供高分辨率
    局部细节。
    """

    last_train_crop_box = {}

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 token_mask_ratio=None,
                 train_cfg=None,
                 test_cfg=None,
                 ):
        """Initialize HRDA-specific scale and crop settings.

        初始化 HRDA 特有的尺度与 crop 设置。

        scales=[1],
        hr_crop_size=None,
        hr_slide_inference=True,
        hr_slide_overlapping=True,
        crop_coord_divisible=1,
        blur_hr_crop=False,
        feature_scale=1
        """
        super(HRDAEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            token_mask_ratio=token_mask_ratio,
            train_cfg=train_cfg,
            test_cfg=test_cfg)
        
        self.feature_scale_all_strs = ['all']  # [‘all’]
        self.feature_scale = decode_head.feature_scale  # 0.5
        if isinstance(self.feature_scale, str):
            assert self.feature_scale in self.feature_scale_all_strs
        self.scales = sorted(decode_head.scales)  # [0.5, 1]
        self.crop_size = decode_head.hr_crop_size  #[512, 512]
        self.hr_slide_inference = decode_head.hr_slide_inference  # True
        self.hr_slide_overlapping = decode_head.hr_slide_overlapping  # True
        self.crop_coord_divisible = decode_head.crop_coord_divisible  # 8
        self.blur_hr_crop = decode_head.blur_hr_crop  # False

    def extract_unscaled_feat(self, img, enable_token_masking=False):
        """Extract features without changing image scale.

        不改变图像尺度，直接提取 backbone 特征。

        This is the HRDA equivalent of `EncoderDecoder.extract_feat`, but it
        also unwraps backbone outputs that package features together with
        auxiliary metadata.

        这是 HRDA 版本的 `EncoderDecoder.extract_feat`，同时会展开部分
        backbone 返回的“特征 + 辅助信息”组合输出。
        """
        # x = self.backbone(img)  # list of four multi-scale feature maps eval: [N,64,128,128], [N,128,64,64], [N,256,32,32], [N,512,16,16]
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
        if isinstance(x, tuple) and len(x) == 3:  # rein dinov2
            x = x[0]  # Extract features from the backbone
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_slide_feat(self, img, enable_token_masking=False):
        """Extract high-resolution features with sliding-window crops.

        使用滑窗 crop 提取高分辨率分支特征。

        The image is divided into overlapping or non-overlapping crops, and features
        are extracted from each crop independently.

        图像会被切分为重叠或不重叠的 crop，并分别提取每个 crop 的特征。

        Args:
            img (Tensor): Input image tensor of shape (B, C, H, W)

        Returns:
            dict: Dictionary containing:
                - features: List of feature tensors extracted from each crop
                - boxes: List of crop coordinates [y1, y2, x1, x2] for each crop
        """
        # Calculate stride size based on overlapping setting
        if self.hr_slide_overlapping:  # True
            # Half overlap 
            # [256, 256] for semi, hrda
            h_stride, w_stride = [e // 2 for e in self.crop_size]  # 256, 256
        else:
            h_stride, w_stride = self.crop_size  # No overlap

        # Get crop dimensions and image size
        h_crop, w_crop = self.crop_size  # Crop size [512, 512]
        bs, _, h_img, w_img = img.size()  # 512, 512; eval: 1024, 1024

        # Calculate number of grid cells in height and width directions
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1  # hrda train: 3, eval: 3
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1  # hrda train: 3, eval: 3

        # Initialize lists to store crops and their information
        crop_imgs, crop_feats, crop_boxes = [], [], []

        # Iterate through grid cells to extract crops, 9 crops in total for each image
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                # Calculate initial crop coordinates
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)

                # Adjust coordinates to ensure fixed crop size
                y1 = max(y2 - h_crop, 0)  # Adjust top coordinate if needed
                x1 = max(x2 - w_crop, 0)  # Adjust left coordinate if needed

                # Extract and store crop
                crop_imgs.append(img[:, :, y1:y2, x1:x2])
                crop_boxes.append([y1, y2, x1, x2])

        # Concatenate all crops and extract features
        # Shape: (N*B, C, H_crop, W_crop) 
        # ema: [18, 3, 512, 512]
        # eval: [27, 3, 512, 512]
        crop_imgs = torch.cat(crop_imgs, dim=0)
        
        # list of four multi-scale feature maps 
        # Shape of crop_feats: feature levels, (N*B) x C x H x W
        # where N is number of crops, B is batch size
        # ema: [18, C, 128, 128], [18, C, 64, 64], [18, C, 32, 32], [18, C, 16, 16]
        # eval: [27, C, 128, 128], [27, C, 64, 64], [27, C, 32, 32], [27, C, 16, 16]
        crop_feats = self.extract_unscaled_feat(crop_imgs, enable_token_masking=enable_token_masking)  

        return {'features': crop_feats, 'boxes': crop_boxes}
    
    def blur_downup(self, img, s=0.5):
        """Apply blur by downsampling and then upsampling the image.

        通过先下采样再上采样实现平滑模糊。

        This creates a smoothing effect that can help reduce high-frequency noise.

        Args:
            img (Tensor): Input image tensor of shape (B, C, H, W)
            s (float): Scale factor for downsampling. Default: 0.5
                    The image is first downscaled by this factor and then 
                    upscaled back to the original size.

        Returns:
            Tensor: Blurred image tensor of the same shape as input
        """
        # First downsample the image by factor s
        img = resize(
            input=img,
            scale_factor=s,
            mode='bilinear',  # Use bilinear interpolation for smooth scaling
            align_corners=self.align_corners)
        
        # Then upsample back to original size (scale by 1/s)
        img = resize(
            input=img,
            scale_factor=1 / s,
            mode='bilinear',
            align_corners=self.align_corners)
        
        return img

    def resize(self, img, s):
        """Resize an image tensor by a scale factor.

        按尺度因子 resize 图像张量。
        
        Args:
            img (Tensor): Input image tensor of shape (B, C, H, W)
            s (float): Scale factor for resizing
                    - s > 1: upsampling
                    - s < 1: downsampling
                    - s = 1: no change
        
        Returns:
            Tensor: Resized image tensor
                    - If s = 1: Returns original image
                    - Otherwise: Returns resized image using bilinear interpolation
        """
        # Return original image if scale factor is 1
        if s == 1:
            return img
        else:
            # Disable gradient computation for resize operation
            with torch.no_grad():
                return resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',  # Use bilinear interpolation for smooth scaling
                    align_corners=self.align_corners  # Maintain consistency in corner pixel handling
                )

    def extract_feat(self, img, enable_token_masking=False):
        """Extract HRDA features from one or more image scales.

        从一个或多个图像尺度提取 HRDA 特征。
        
        Args:
            img (Tensor): Input image tensor of shape (B, C, H, W)
            
        Returns:
            list or Tensor: 
                - If feature_scale is 'all': Returns list of features at different scales
                - Otherwise: Returns features at specified scale
        """
        # Handle multi-scale feature extraction
        if self.feature_scale in self.feature_scale_all_strs:
            mres_feats = []  # List to store multi-resolution features
            
            # Iterate through different scales
            for i, s in enumerate(self.scales):
                # Apply blur effect for high-resolution crops if enabled
                if s == 1 and self.blur_hr_crop:
                    scaled_img = self.blur_downup(img)
                else:
                    scaled_img = self.resize(img, s)
                    
                # Apply cropping for scales > 1 if crop_size is specified
                if self.crop_size is not None and i >= 1:
                    scaled_img = crop(
                        scaled_img, 
                        HRDAEncoderDecoder.last_train_crop_box[i]  # Use stored crop box
                    )
                    
                # Extract features and store in list
                mres_feats.append(self.extract_unscaled_feat(scaled_img, enable_token_masking=enable_token_masking))
                
            return mres_feats
        
        # Handle single-scale feature extraction
        else:
            scaled_img = self.resize(img, self.feature_scale)
            return self.extract_unscaled_feat(scaled_img, enable_token_masking=enable_token_masking)

    def encode_decode(self, img, return_feat=False, enable_token_masking=False, upscale_pred=True, comp_drop=False):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input.

        使用多尺度 backbone 特征解码语义分割图。

        Returns:
            dict: `seg_logits` plus optional `features`. When
            `hr_slide_inference=True`, high-resolution features are represented
            as a dict with crop features and crop boxes.

            返回包含 `seg_logits` 和可选 `features` 的字典。当
            `hr_slide_inference=True` 时，高分辨率特征会以包含 crop 特征和
            crop 坐标的字典表示。
        """
        
        # 初始化返回字典
        output_dict = {}
        
        mres_feats = []
        self.decode_head.debug_output = {}
        for i, s in enumerate(self.scales):  # 0.5, 1.0; 1.0, 2.0
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(img)
            else:
                # train: [2, 3, 512, 512], 
                # eval: lr-[N, 3, 512, 512] hr-[N, 3, 1024, 1024]
                scaled_img = self.resize(img, s)  
            if i >= 1 and self.hr_slide_inference:  
                mres_feats.append(self.extract_slide_feat(scaled_img, enable_token_masking=enable_token_masking))  # high resolution features
            else:
                # eval: lr: [[N, C, 128, 128], [N, C, 64, 64], [N, C, 32, 32], [N, C, 16, 16]]
                mres_feats.append(self.extract_unscaled_feat(scaled_img, enable_token_masking=enable_token_masking))  # low resolution features
            if self.decode_head.debug:
                if f'Img {i} Scale {s}' in self.decode_head.debug_output:
                    self.decode_head.debug_output[f'Img {i} Scale {s}'] = torch.cat(
                        [self.decode_head.debug_output[f'Img {i} Scale {s}'],
                         scaled_img.detach()], dim=0)  # from [3, 3, 1024, 1024] to [6, 3, 1024, 1024], half batched
                else:
                    self.decode_head.debug_output[f'Img {i} Scale {s}'] = scaled_img.detach()
        if comp_drop:
            kept_ratio = 0.5 if isinstance(comp_drop, bool) else \
                float(comp_drop.get('kept_ratio', 0.5))
            mres_feats = self._apply_complementary_dropout(
                mres_feats, kept_ratio=kept_ratio)

        out = self._decode_head_forward_test(mres_feats)  # [3, 19, 256, 256]; 
        # input: mres_feats[0]: list of four multi-scale features; 
        # mres_feats[1]: dict, features and boxes; 
        #   features: list of four multi-scale features, [N, 64, 128, 128], [N, 128, 64, 64], [N, 256, 32, 32], [N, 512, 16, 16], 
        #   boxes: list of crop coordinates [y1, y2, x1, x2] for each crop

        if upscale_pred:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        output_dict['seg_logits'] = out
        if return_feat and self.feature_scale in self.feature_scale_all_strs:
            output_dict['features'] = mres_feats
        elif return_feat:
            output_dict['features'] = mres_feats[0]  # low resolution four multi-scale features
            
        return output_dict

    def _forward_train_features(self, img, enable_token_masking=False):
        """Extract aligned multi-resolution features for training.

        提取训练阶段对齐后的多尺度特征。

        The high-resolution branch records one crop box and shares it with the
        decode head so logits, labels, and optional visualization stay aligned.

        高分辨率分支会记录一个 crop box，并传给 decode head，保证 logits、
        标签和可视化内容对齐。
        """
        mres_feats = []
        self.decode_head.debug_output = {}
        assert len(self.scales) <= 2, 'Only up to 2 scales are supported.'
        prob_vis = None
        for i, s in enumerate(self.scales):  # 0.5, 1.0; 1.0, 2.0
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(img)
            else:
                scaled_img = resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)
                # 0.5 context scaled image: from [2, 3, 1024, 1024] resize to [2, 3, 512, 512]; 1.0: from [2, 3, 512, 512] remain as [2, 3, 512, 512]
                # 1.0 detail scaled image: remain as [2, 3, 1024, 1024]; 2.0: from [2, 3, 512, 512] resize to [2, 3, 1024, 1024]
                
            # detail crop: crop [512, 512] from [1024, 1024]
            if self.crop_size is not None and i >= 1:  
                crop_box = get_crop_bbox(*scaled_img.shape[-2:],
                                         self.crop_size,
                                         self.crop_coord_divisible)
                # 0.5 all, False
                if self.feature_scale in self.feature_scale_all_strs:
                    HRDAEncoderDecoder.last_train_crop_box[i] = crop_box
                self.decode_head.set_hr_crop_box(crop_box)
                scaled_img = crop(scaled_img, crop_box)  # detail crop image, [2, 3, 512, 512]
            if self.decode_head.debug:
                self.decode_head.debug_output[f'Img {i} Scale {s}'] = \
                    scaled_img.detach()
            mres_feats.append(self.extract_unscaled_feat(scaled_img, enable_token_masking=enable_token_masking))
        return mres_feats, prob_vis

    def forward_style(self, img):
        scaled_img = resize(
            input=img,
            scale_factor=self.feature_scale,
            mode='bilinear',
            align_corners=self.align_corners)
        return self.backbone.forward_features(scaled_img, return_style=True)
    
    def forward_train(self,
                      data_batch,
                      seg_weight=None,
                      return_feat=False,
                      enable_token_masking=False,
                      loss_key=None,
                      comp_drop=False,
                      ):
        """Forward function for training.

        HRDA 训练阶段前向函数。

        Args:
            data_batch (tuple): (img, gt_semantic_seg) tuple containing:
                - img (Tensor): Input images of shape (N, C, H, W)
                - gt_semantic_seg (Tensor): GT semantic segmentation of shape (N, H, W)
            seg_weight (Tensor): Pixel-wise segmentation weight, used for loss calculation
            return_feat (bool): Whether to return backbone features
            enable_token_masking (bool): Whether to enable token masking

        Returns:
            dict[str, Tensor]: a dictionary of loss components

            返回包含 loss、seg_logits 和可选 features 的字典。
        """
        results = dict()
        
        img, gt_semantic_seg = data_batch

        # mres_feats: list of two multi-scale features. 
        #   mres_feats[0] features for low resolution image [2,768,128,128] [2,768,64,64] [2,768,64,64] [2,768,32,32] [2,768,16,16]
        #   mres_feats[1] features for high resolution image [2,768,128,128] [2,768,64,64] [2,768,64,64] [2,768,32,32] [2,768,16,16]
        # prob_vis: None
        mres_feats, prob_vis = self._forward_train_features(img, enable_token_masking=enable_token_masking)
        if comp_drop:
            kept_ratio = 0.5 if isinstance(comp_drop, bool) else \
                float(comp_drop.get('kept_ratio', 0.5))
            mres_feats = self._apply_complementary_dropout(
                mres_feats, kept_ratio=kept_ratio)
        # 0.5 1.0
        for i, s in enumerate(self.scales):
            # False, 0.5 ['all']
            if return_feat and self.feature_scale in self.feature_scale_all_strs:
                if 'features' not in results:
                    results['features'] = []
                results['features'].append(mres_feats[i])
            # 0.5, return context crop image features
            if return_feat and s == self.feature_scale:  
                results['features'] = mres_feats[i]
                break

        decode_loss_dict = self._decode_head_forward_train(mres_feats,
                                                      gt_semantic_seg,
                                                      seg_weight,
                                                      loss_key=loss_key)
        results.update(decode_loss_dict)

        if self.decode_head.debug and prob_vis is not None:
            self.decode_head.debug_output['Crop Prob.'] = prob_vis

        # if self.with_auxiliary_head:
        #     raise NotImplementedError

        self.local_iter += 1
        
        return results

    def forward_with_aux(self, img):
        """Forward helper used by auxiliary/debug paths.

        辅助或调试路径使用的前向 helper。
        """
        assert not self.with_auxiliary_head
        mres_feats, _ = self._forward_train_features(img)
        out = self.decode_head.forward(mres_feats)
        # out = resize(
        #     input=out,
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        return {'main': out}

    def _decode_head_forward_train(self,
                                   x,
                                   gt_semantic_seg,
                                   seg_weight=None,
                                   loss_key=None):
        """Run forward function and calculate loss for decode head in
        training.

        运行 decode head 训练前向并计算损失。
        """
        loss_dict = dict()
        seg_logits = self.decode_head.forward_train(x)
        loss_dict.update({'seg_logits': seg_logits})
        
        # seg_logits = match_shape(seg_logits, gt_semantic_seg.shape[-2:], mode='bilinear', align_corners=self.align_corners)
        # seg_weight = match_shape(seg_weight, gt_semantic_seg.shape[-2:], mode='bilinear', align_corners=self.align_corners) if seg_weight is not None else None
        decode_loss = self.decode_head.cal_loss(
            seg_logits, gt_semantic_seg, seg_weight, loss_key=loss_key)
        # loss_decode = self.decode_head.forward_train(x,
        #                                              gt_semantic_seg,
        #                                              self.train_cfg,
        #                                              seg_weight)

        loss_dict.update(decode_loss)
        return loss_dict
    
    def _decode_head_forward_test(self, x):
        """Run forward function and calculate loss for decode head in
        inference.

        运行 decode head 推理前向，返回分割 logits。
        """
        seg_logits = self.decode_head.forward_test(x)
        return seg_logits
