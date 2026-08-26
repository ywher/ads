# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from copy import deepcopy
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn as nn
from torch.nn.parallel import DistributedDataParallel

import numpy as np
import logging
from tqdm import tqdm
import scipy.ndimage as ndimage
from scipy.spatial.distance import directed_hausdorff

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

from lib.models.segmentors import *
from lib.models.backbones import *
from lib.models.decode_heads import *
from lib.models.model_utils.dacs_transforms import get_mean_std_self, denorm
from lib.models.model_utils.visualization import (
    get_debug_palette,
    save_debug_sup_dep_images,
    save_debug_sup_predictions,
    save_debug_cls_predictions,
    save_debug_hrda_images,
)
from lib.loss.losses import parse_losses
from utils.classes import CLASSES
from utils.util import compute_ious_from_hist, get_dataset_eval_splits
dataset_class = CLASSES['cityscapes']

def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, DistributedDataParallel):
        return module.module
    
    return module

class ProjHead(nn.Module):
    """
    1x1 -> Norm -> Act -> 1x1 的瓶颈残差投影头
    将 C -> C//2 -> C，并可选残差。最后一层权重/归一化零初始化=起步近似恒等。
    """
    def __init__(
        self,
        channels: int,
        *,
        bottleneck_ratio: float = 1.0,
        act: str = 'relu',         # 'relu' | 'gelu' | 'silu'
        residual: bool = True,
        zero_init_last: bool = True,
        normalize_feat: bool = True,
    ):
        super().__init__()
        hidden = max(1, int(channels * bottleneck_ratio))
        self.residual = residual
        self.norm_feat = normalize_feat
        
        if act == 'relu':
            act_func = nn.ReLU()
        elif act == 'gelu':
            act_func = nn.GELU()
        elif act == 'silu':
            act_func = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act}")

        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            act_func,
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        
        # 初始化（nonlinearity 与激活匹配；GELU/SiLU 用 'linear' 更稳妥）
        nonlin = 'relu' if isinstance(act_func, nn.ReLU) else 'linear'
        for m in self.layers:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nonlin)

        if zero_init_last:
            # 最后一层 conv 权重置零（bias=False 无需处理 bias）
            last_conv = self.layers[-1]
            nn.init.zeros_(last_conv.weight)

    def forward(self, x):
        identity = x
        y = self.layers(x)
        if self.residual:
            y = y + identity
        if self.norm_feat:
            # 对输出进行归一化
            assert y.dim() == 4, f"Expected 4D tensor, got {y.dim()}D tensor"
            y = F.normalize(y, p=2, dim=1, eps=1e-12)
        return y

class MultiScaleProjHead(nn.Module):
    """
    针对多尺度特征的投影头包装：对每个尺度使用一个独立的 ProjHead1x1。
    """
    def __init__(self, channels_per_level, **kwargs):
        super().__init__()
        self.heads = nn.ModuleList([ProjHead(c, **kwargs) for c in channels_per_level])

    def forward(self, feats):
        assert len(feats) == len(self.heads)
        return [h(f) for h, f in zip(self.heads, feats)]
    
class SharedProjHead(nn.Module):
    """
    单个共享 Project Head：所有尺度复用同一套参数。
    适用于各尺度通道数完全一致的场景（例如 [1024, 1024, 1024, 1024]）。
    """
    def __init__(self, channels: int, **kwargs):
        super().__init__()
        self.head = ProjHead(channels, **kwargs)

    def forward(self, feats):
        # 支持 Tensor 或 list/tuple of Tensor
        if isinstance(feats, (list, tuple)):
            return [self.head(f) for f in feats]
        return self.head(feats)
    
    def align_loss(self, feat_student, feat_teacher, return_sim=False):
        if self.head.norm_feat:
            feat_student = feat_student.permute((0,2,3,1))  # [b,c,h,w] -> [b,h,w,c]
            feat_teacher = feat_teacher.permute((0,2,3,1))  # [b,c,h,w] -> [b,h,w,c]
            sim = (feat_student * feat_teacher).sum(dim=-1)
            loss = (1-sim).mean()

        else:
            sim = torch.linalg.norm(feat_student-feat_teacher, dim=1, ord=2)
            loss = sim.mean() / 100

        if return_sim:
            return loss, sim
        else:
            return loss

class SemiSegmentor(nn.Module):
    @staticmethod
    def _is_master_process():
        return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

    def __init__(self, model_cfg):
        super(SemiSegmentor, self).__init__()
        self.logger = logging.getLogger()
        self.model_cfg = model_cfg
        self.model_type = self.model_cfg['type']  # EncoderDecoder
       
        self.train_cfg = self.model_cfg['train_cfg']
        self.test_cfg = self.model_cfg['test_cfg']
        self.num_classes = self.model_cfg['decode_head']['decoder_config']['num_classes']
        self.class_set = self.model_cfg.get('class_set', None)
        self.debug_palette = get_debug_palette(self.num_classes, self.class_set)
        
        self.logger.info(f'Building model: {self.model_type}')
        self.model = self.build_model()
        self.logger.info(f'Model built successfully.\n')
        # aux head
        if self.get_model().auxiliary_head is not None:
            self.with_aux_head = True
            self.aux_loss_weight = self.model_cfg['aux_head']['loss_decode']['loss_weight']
            self.debug_imgs = self.model_cfg['aux_head'].get('debug_imgs', False)
        else:
            self.with_aux_head = False
            self.aux_loss_weight = 0
            self.debug_imgs = False
        
        self.local_iter = 0
        self.respth = model_cfg.get('respth', './debug')
        self.model_id = model_cfg.get('model_id', '')  # For decoupled finetuning: 'm1' or 'm2'
        self.img_mean = model_cfg.get('rgb_mean', (123.675, 116.28, 103.53))
        self.img_std = model_cfg.get('rgb_std', (58.395, 57.12, 57.375))
        self.debug_img_interval = int(model_cfg.get('debug_img_interval', 500) or 0)
        
        
    def build_project_head(self):
        # embed_dims = self.model_cfg['decode_head']['decoder_config']['in_channels']  # [1024, 1024, 1024, 1024]
        # # 进行特征维度的变换，两层，从embed到一半再恢复
        # self.proj_head = MultiScaleProjHead(
        #                     channels_per_level=embed_dims,
        #                     bottleneck_ratio=0.5,     # 1024 -> 512 -> 1024
        #                     norm='gn',                # 小 batch 建议 GN
        #                     gn_groups=32,
        #                     act='relu',
        #                     dropout=0.0,
        #                     residual=True,
        #                     zero_init_last=True
        #                 )
        
        embed_dims = self.model_cfg['decode_head']['decoder_config']['in_channels']  # 例: [1024, 1024, 1024, 1024]
        # 由于每层的特征维度相同，使用共享的投影头即可
        ch = embed_dims[0] if isinstance(embed_dims, (list, tuple)) else embed_dims
        self.proj_head = SharedProjHead(
            channels=ch,
            bottleneck_ratio=1.0,     # 1024 -> 512 -> 1024
            act='relu',
            residual=True,
            zero_init_last=True,
            normalize_feat=True,
        )

    def build_model(self, cfg=None):
        """构建模型
        
        Args:
            cfg (dict, optional): 模型配置。如果为None则使用self.model_cfg。
            
        Returns:
            nn.Module: 构建的模型
        """
        # 使用传入的配置或默认配置
        cfg = cfg if cfg is not None else self.model_cfg
        
        # 获取预训练路径
        backbone_pretrained = cfg.get('backbone_pretrained', None)
        decoder_pretrained = cfg.get('decoder_pretrained', None)
        token_mask_ratio = cfg.get('token_mask_ratio', None)
        
        # 构建backbone
        backbone_type = cfg['backbone']['type']
        backbone = eval(backbone_type)(cfg['backbone']['backbone_config'], backbone_pretrained)
        
        # 构建decoder
        decoder_type = cfg['decode_head']['type']
        if 'loss_decode' in cfg['decode_head']:
            cfg['decode_head']['decoder_config']['loss_decode'] = \
                cfg['decode_head']['loss_decode']
        decoder = eval(decoder_type)(cfg['decode_head']['decoder_config'], decoder_pretrained)
        
        # 构建auxiliary decoder(如果存在)
        aux_decoder_cfg = cfg.get('aux_head', None)
        aux_decoder_pretrained = cfg.get('aux_decoder_pretrained', None)
        if aux_decoder_cfg is not None:
            aux_decoder_type = aux_decoder_cfg['type']
            aux_decoder = eval(aux_decoder_type)(cfg['aux_head']['decoder_config'], aux_decoder_pretrained)
        else:
            aux_decoder = None

        # 构建完整模型
        model_type = cfg['type']
        model = eval(model_type)(
            backbone=backbone,
            decode_head=decoder,
            auxiliary_head=aux_decoder,
            token_mask_ratio=token_mask_ratio,
            test_cfg=self.test_cfg
        )
        
        return model
    
    def get_model(self):
        return get_module(self.model)

    def extract_feat(self, img):
        """Extract features from images."""
        return self.get_model().extract_feat(img)

    def encode_decode(self, img):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_model().encode_decode(img)

    def forward_train_dep_step(self, data_batch, critetia):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # seg_logit = self.get_model().forward_train(img, return_feat=return_feat)
        
        # if return_feat:
        #     return seg_logit[0], seg_logit[1]
        # else:
        #     return seg_logit
        
        img, gt_seg, gt_depth = data_batch
        log_vars = {}
        
        if self.with_aux_head:
            criteria_seg, critetia_dep, critetia_mlcls = critetia
            pred, aux_pred = self.get_model().forward_train(img, return_feat=False)
            multi_cls_lb = self.get_model().auxiliary_head.convert_seg_to_multilabel(gt_seg)
            mlcls_loss = critetia_mlcls(aux_pred, multi_cls_lb)
            if torch.isnan(mlcls_loss).any():
                self.logger.info(f"LOSS NAN in loss_mlcls! aux_pred: {aux_pred}, multi_cls_lb: {multi_cls_lb}")
                mlcls_loss= torch.nan_to_num(mlcls_loss)
            # apply the loss weight
            mlcls_loss = mlcls_loss * self.aux_loss_weight
            log_vars['mlcls_loss'] = mlcls_loss
            mlcls_loss.backward(retain_graph=True)
        else:
            criteria_seg, critetia_dep = critetia
            pred = self.get_model().forward_train(img, return_feat=False)
        # if pred.shape != gt_seg.shape:
        #     pred = torch.nn.functional.interpolate(pred, size=gt_
        
        if isinstance(pred, dict):
            seg_pred = pred['S']
            dep_pred = pred['D']
            seg_init_pred = pred['initial_S']
            dep_init_pred = pred['initial_D']
        
        seg_loss = criteria_seg(seg_pred, gt_seg)
        seg_init_loss = criteria_seg(seg_init_pred, gt_seg)
        dep_loss = critetia_dep(dep_pred, gt_depth)
        dep_init_loss = critetia_dep(dep_init_pred, gt_depth)
        
        log_vars['seg_loss'] = seg_loss
        log_vars['seg_init_loss'] = seg_init_loss
        log_vars['dep_loss'] = dep_loss
        log_vars['dep_init_loss'] = dep_init_loss
        total_seg_loss = seg_loss + seg_init_loss
        total_dep_loss = (dep_loss + dep_init_loss) * 0.001
        total_loss = total_seg_loss + total_dep_loss
        log_vars['loss'] = total_loss
        total_loss.backward()
        
        # to visualize the mlcls in debug mode
        if self._is_master_process() and self.debug_img_interval > 0 and \
                (self.local_iter + 1) % self.debug_img_interval == 0:  #  and self.with_aux_head
            batch_size = img.size(0)
            means, stds = get_mean_std_self(self.img_mean, self.img_std, batch_size, img.device)
            save_debug_sup_dep_images(self, batch_size, means, stds, img, gt_seg, gt_depth.squeeze(1).detach().cpu().numpy(), torch.argmax(seg_pred, dim=1), dep_pred.squeeze(1).detach().cpu().numpy(), torch.argmax(seg_init_pred, dim=1), dep_init_pred.squeeze(1).detach().cpu().numpy())
        
        self.local_iter += 1
        
        # torch.cuda.empty_cache()
        return log_vars
        
    def train_dep_step(self, data_batch, critetia):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
    
        log_vars = self.forward_train_dep_step(data_batch, critetia)
        # torch.cuda.empty_cache()
        return log_vars
    
    def forward_train_cls_step(self, data_batch, critetia):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # seg_logit = self.get_model().forward_train(img, return_feat=return_feat)
        
        # if return_feat:
        #     return seg_logit[0], seg_logit[1]
        # else:
        #     return seg_logit
        
        img, gt_seg = data_batch
        log_vars = {}
        
        criteria_cls_sup = critetia
        cls_pred = self.get_model().forward_train(img, return_feat=False, upscale_pred=False)
        
        cls_loss, gt_cls = criteria_cls_sup(cls_pred, gt_seg, return_seg_lb=True)
        log_vars['cls_loss'] = cls_loss
        cls_loss.backward()
        
        if self._is_master_process() and self.debug_img_interval > 0 and \
                (self.local_iter + 1) % self.debug_img_interval == 0:
            batch_size = img.size(0)
            debug_folder = f'debug_{self.model_id}' if self.model_id else 'debug'
            out_dir = os.path.join(self.respth, debug_folder)
            os.makedirs(out_dir, exist_ok=True)
            means, stds = get_mean_std_self(self.img_mean, self.img_std, batch_size, img.device)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)

            save_debug_cls_predictions(
                out_dir=out_dir,
                local_iter=self.local_iter,
                batch_size=batch_size,
                vis_img=vis_img,
                gt_seg=gt_seg,
                cls_pred=cls_pred,
                cls_lb=gt_cls,
                critetia_cls=criteria_cls_sup,
                dataset_class=dataset_class,
                debug_imgs=self.debug_imgs,
                adjust_y_range=True)
                
        self.local_iter += 1
        
        # torch.cuda.empty_cache()
        return log_vars
    
    def train_cls_step(self, data_batch, critetia):
        log_vars = self.forward_train_cls_step(data_batch, critetia)
        # torch.cuda.empty_cache()
        return log_vars
    
    def forward_train_step(
        self,
        data_batch,
        seg_weight=None,
        backward_scale=1.0,
        update_iter=True,
        enable_debug=True,
    ):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # seg_logit = self.get_model().forward_train(img, return_feat=return_feat)
        
        # if return_feat:
        #     return seg_logit[0], seg_logit[1]
        # else:
        #     return seg_logit
        
        img, gt_seg = data_batch
        log_vars = {}
        seg_debug = {}
        save_debug = (
            enable_debug
            and
            self._is_master_process()
            and self.debug_img_interval > 0
            and (self.local_iter + 1) % self.debug_img_interval == 0
        )
        decode_head = getattr(self.get_model(), 'decode_head', None)
        if hasattr(decode_head, 'debug'):
            decode_head.debug = save_debug
        
        pred_results = self.get_model().forward_train(
            data_batch,
            seg_weight,
            return_feat=False,
        )
        # if pred.shape != gt_seg.shape:
        #     pred = torch.nn.functional.interpolate(pred, size=gt_
        
        seg_debug['Sup'] = self.get_model().decode_head.debug_output
        pred = pred_results.pop('seg_logits')
        seg_loss, seg_log_vars = parse_losses(pred_results)
        
        log_vars.update(seg_log_vars)
        (seg_loss * float(backward_scale)).backward()
        
        if save_debug:
            batch_size = img.size(0)
            debug_folder = f'debug_{self.model_id}' if self.model_id else 'debug'
            out_dir = os.path.join(self.respth, debug_folder)
            os.makedirs(out_dir, exist_ok=True)
            means, stds = get_mean_std_self(self.img_mean, self.img_std, batch_size, img.device)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)

            if self.with_aux_head:
                pass
                """save_debug_sup_mlcls_predictions(
                    out_dir=out_dir,
                    local_iter=self.local_iter,
                    batch_size=batch_size,
                    vis_img=vis_img,
                    gt_seg=gt_seg,
                    pred=pred,
                    aux_pred=aux_pred,
                    multi_cls_lb=multi_cls_lb,
                    criteria_sup=criteria_sup,
                    critetia_mlcls=critetia_mlcls,
                    dataset_class=dataset_class,
                    debug_imgs=self.debug_imgs
                )"""
            else:
                if isinstance(self.get_model(), HRDAEncoderDecoder):
                    # print('key of seg_debug:', seg_debug['Sup'].keys())
                    save_debug_hrda_images(
                        out_dir=out_dir,
                        local_iter=self.local_iter,
                        seg_debug=seg_debug,
                        batch_size=batch_size,
                        means=means,
                        stds=stds,
                        palette=self.debug_palette,
                    )
                else:
                    save_debug_sup_predictions(
                        out_dir=out_dir,
                        local_iter=self.local_iter,
                        batch_size=batch_size,
                        vis_img=vis_img,
                        gt_seg=gt_seg,
                        pred=pred,
                        palette=self.debug_palette,
                    )
                
            
        if update_iter:
            self.local_iter += 1
        
        # torch.cuda.empty_cache()
        return log_vars
    
    def train_step(self, data_batch):
        log_vars = self.forward_train_step(data_batch)
        # torch.cuda.empty_cache()
        return log_vars
    
    def inference(self, img, rescale=None):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        return self.get_model().inference(img, rescale)

    def simple_test(self, img, rescale=True):
        """Simple test with single image."""
        return self.get_model().simple_test(img, rescale)

    def aug_test(self, imgs, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        return self.get_model().aug_test(imgs, rescale)

    def compute_hist(self, pred, lb):
        assert pred.shape == lb.shape
        pred = pred.flatten()
        lb = lb.flatten()
        keep = lb < self.num_classes
        pred = pred[keep]
        lb = lb[keep]
        hist = np.zeros((self.num_classes, self.num_classes))
        hist_tmp = np.bincount(
            self.num_classes * lb.reshape(-1) + pred.reshape(-1),
            minlength=self.num_classes**2)
        hist = hist + hist_tmp.reshape(self.num_classes, self.num_classes)
        return hist
    
    def eval(self, val_loader, rescale=True):
        self.logger.info(f'Start evaluating model, mode: {self.test_cfg["mode"]}')
        
        # 设置模型为评估模式
        was_training = self.training
        self.get_model().eval()
        
        hist = np.zeros((self.num_classes, self.num_classes))
        split_order, eval_splits, split_name = get_dataset_eval_splits(val_loader.dataset)
        split_hists = None
        split_counts = None
        if eval_splits is not None:
            split_hists = {
                split: np.zeros((self.num_classes, self.num_classes))
                for split in split_order
            }
            split_counts = {split: 0 for split in split_order}
            self.logger.info(f'{split_name} split-wise evaluation enabled: all/' + '/'.join(split_order))
        self.last_eval_ious_by_scene = None
        self.last_eval_split_order = split_order
        self.last_eval_scene_counts = None
        sample_offset = 0
        
        # 预分配内存，避免频繁分配
        device_predictions = []
        device_labels = []
        
        try:
            with torch.inference_mode():  # 改为 inference_mode
                for val_data in tqdm(val_loader, total=len(val_loader)):
                    im = val_data['im'].cuda(non_blocking=True)
                    lb = val_data['lb']  # 保持在CPU
                    
                    im_input = {'img': im, 'lb_shape': lb.size()[1:3]}
                    pred_logit = self.get_model().inference(im_input, rescale=rescale)
                    
                    if isinstance(pred_logit, dict):
                        pred_logit = pred_logit.get('seg_logits', pred_logit.get('S'))
                    
                    # 立即转CPU并处理
                    pred = torch.argmax(pred_logit, dim=1).cpu().numpy()
                    lb_np = lb.numpy()
                    
                    # 立即计算直方图，不累积
                    hist += self.compute_hist(pred, lb_np)
                    if split_hists is not None:
                        batch_size = pred.shape[0]
                        for batch_idx in range(batch_size):
                            split_idx = sample_offset + batch_idx
                            if split_idx >= len(eval_splits):
                                continue
                            split = eval_splits[split_idx]
                            if split not in split_hists:
                                continue
                            split_hists[split] += self.compute_hist(
                                pred[batch_idx:batch_idx + 1],
                                lb_np[batch_idx:batch_idx + 1])
                            split_counts[split] += 1
                        sample_offset += batch_size
                    
                    # 显式清理
                    del pred_logit, im
                    
        finally:
            if was_training:
                self.get_model().train()

        # 计算IoU
        denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
        iu = np.diag(hist) / np.maximum(denominator, 1) * 100
        mean_iu = np.nanmean(iu)
        if split_hists is not None:
            self.last_eval_ious_by_scene = {
                split: compute_ious_from_hist(split_hists[split])
                for split in split_order
            }
            self.last_eval_scene_counts = split_counts

        return mean_iu, iu
    
    def _process_batch_hist(self, predictions, labels, hist):
        """Accumulate histograms without assuming a fixed image resolution."""
        # Mapillary keeps each validation image at its native aspect ratio.
        # Consecutive predictions can therefore have different H/W and cannot
        # be concatenated into one tensor.
        for prediction, label in zip(predictions, labels):
            pred_np = prediction.cpu().numpy()
            label_np = label.cpu().numpy()
            hist += self.compute_hist(pred_np, label_np)

    def eval_cls(self, val_loader, criteria):
        self.logger.info('Start evaluating validation loss...')
        self.get_model().eval()

        all_losses = []
        # 计算需要评估的样本数量
        total_samples = len(val_loader.dataset)
        eval_samples = total_samples // 10
        current_samples = 0

        for im, lb in tqdm(val_loader, total=eval_samples, leave=False):
            # 检查是否已经处理了足够的样本
            if current_samples >= eval_samples:
                break

            im = im.cuda(non_blocking=True)
            lb = lb.cuda(non_blocking=True)

            with torch.no_grad():
                cls_score = self.get_model().inference_cls(im)  # [B, C]
                # prob = F.softmax(cls_score, dim=1)
                val_loss = criteria(cls_score, lb)
                all_losses.append(val_loss.item())

            B, H, W = lb.shape
            # C = cls_score.shape[1]

            # lb_onehot = torch.zeros(B, H, W, self.num_classes, device=lb.device)
            # for c in range(self.num_classes):
            #     lb_onehot[:, :, :, c] = (lb == c)

            # lb_ratio = lb_onehot.view(B, -1, self.num_classes).mean(dim=1)

            # all_preds.append(prob)
            # all_labels.append(lb_ratio)
            
            current_samples += B

        self.get_model().train()
        
        val_loss = np.mean(all_losses)

        # all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
        # all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

        # try:
        #     auc = roc_auc_score(all_labels, all_preds, average='macro')
        # except ValueError as e:
        #     self.logger.warning(f"AUC computation failed: {e}")
        #     auc = float('nan')

        return val_loss

    def eval_med(self, val_loader, rescale=True):
        """
        Medical image evaluation with Dice Score, IoU, and HD95 computation.
        
        Args:
            val_loader: Validation data loader
            rescale (bool): Whether to rescale predictions to original size
            
        Returns:
            tuple: (mean_dice, dice_per_class, mean_iou, iou_per_class, mean_hd95, hd95_per_class)
        """
        self.logger.info(f'Start medical evaluation, mode: {self.test_cfg["mode"]}')
        
        # 设置模型为评估模式
        was_training = self.training
        self.get_model().eval()
        
        # 初始化累计指标 [背景, 前景1, 前景2, ...]
        total_intersection = np.zeros(self.num_classes)
        total_union = np.zeros(self.num_classes)
        total_pred_sum = np.zeros(self.num_classes)
        total_gt_sum = np.zeros(self.num_classes)
        
        # HD95 指标累计
        hd95_accumulator = [[] for _ in range(self.num_classes)]
        
        try:
            with torch.inference_mode():  # 使用 inference_mode 而非 no_grad
                for batch_idx, val_data in enumerate(tqdm(val_loader, total=len(val_loader), desc="Medical Evaluation")):
                    im = val_data['im'].cuda(non_blocking=True)
                    lb = val_data['lb']  # 保持在CPU上
                    
                    # 构建输入字典
                    im_input = {'img': im, 'lb_shape': lb.size()[1:3]}
                    
                    # 推理
                    pred_logit = self.get_model().inference(im_input, rescale=rescale)
                    
                    if isinstance(pred_logit, dict):
                        pred_logit = pred_logit.get('seg_logits', pred_logit.get('S'))
                    
                    # 立即转CPU，不在GPU上累积
                    pred = torch.argmax(pred_logit, dim=1).cpu()  # 立即转CPU
                    
                    # 立即处理，不批量累积
                    self._process_med_metrics_immediate(pred, lb, 
                                                    total_intersection, total_union,
                                                    total_pred_sum, total_gt_sum, 
                                                    hd95_accumulator)
                    
                    # 显式删除GPU张量
                    del pred_logit, im
                    
                    # 定期清理GPU缓存（可选）
                    if len(val_loader.dataset) > 100 and batch_idx % 20 == 0 and batch_idx > 0:
                        torch.cuda.empty_cache()
                        
        finally:
            if was_training:
                self.get_model().train()

        # 计算 Dice Score
        dice_per_class = 2 * total_intersection / np.maximum(
            total_pred_sum + total_gt_sum, 1e-8
        ) * 100
        
        # 计算 IoU
        iou_per_class = total_intersection / np.maximum(total_union, 1e-8) * 100
        
        # 计算 HD95
        hd95_per_class = np.zeros(self.num_classes)
        for class_id in range(self.num_classes):
            if len(hd95_accumulator[class_id]) > 0:
                # 计算95百分位数
                hd95_per_class[class_id] = np.percentile(hd95_accumulator[class_id], 95)
            else:
                # 没有该类别的样本时设为NaN
                hd95_per_class[class_id] = np.nan
        
        # 计算均值（通常医学分割关注前景类，可选择跳过背景）
        if self.num_classes == 2:
            # 二分类：只关注前景类
            mean_dice = dice_per_class[1]
            mean_iou = iou_per_class[1]
            mean_hd95 = hd95_per_class[1] if not np.isnan(hd95_per_class[1]) else 0.0
        else:
            # 多分类：可选择包含或排除背景类
            # 排除背景类 (索引0)
            mean_dice = np.nanmean(dice_per_class[1:]) 
            mean_iou = np.nanmean(iou_per_class[1:])
            mean_hd95 = np.nanmean(hd95_per_class[1:])

        self.logger.info(f'Medical Evaluation Results:')
        self.logger.info(f'Mean Dice Score: {mean_dice:.2f}%')
        self.logger.info(f'Mean IoU: {mean_iou:.2f}%')
        self.logger.info(f'Mean HD95: {mean_hd95:.2f}mm')
        
        for i, (dice, iou, hd95) in enumerate(zip(dice_per_class, iou_per_class, hd95_per_class)):
            class_name = f'Background' if i == 0 else f'Foreground_{i}' if self.num_classes == 2 else f'Class_{i}'
            hd95_str = f'{hd95:.2f}mm' if not np.isnan(hd95) else 'N/A'
            self.logger.info(f'{class_name}: Dice={dice:.2f}%, IoU={iou:.2f}%, HD95={hd95_str}')

        return mean_dice, dice_per_class, mean_iou, iou_per_class, mean_hd95, hd95_per_class

    def _process_med_metrics_immediate(self, pred_cpu, label_cpu, total_intersection, 
                                    total_union, total_pred_sum, total_gt_sum, hd95_accumulator):
        """立即处理单个批次的指标，避免GPU内存累积"""
        pred_np = pred_cpu.numpy()
        label_np = label_cpu.numpy()
        
        batch_size = pred_np.shape[0]
        
        # 逐样本处理HD95
        for b in range(batch_size):
            pred_sample = pred_np[b]
            gt_sample = label_np[b]
            
            for class_id in range(self.num_classes):
                if class_id == 0 and self.num_classes > 2:
                    continue
                    
                pred_binary = (pred_sample == class_id).astype(np.uint8)
                gt_binary = (gt_sample == class_id).astype(np.uint8)
                
                valid_mask = (gt_sample != 255)
                pred_binary = pred_binary & valid_mask
                gt_binary = gt_binary & valid_mask
                
                hd95_value = self._compute_hd95_single(pred_binary, gt_binary)
                if hd95_value is not None:
                    hd95_accumulator[class_id].append(hd95_value)
        
        # 计算Dice和IoU指标
        for class_id in range(self.num_classes):
            pred_binary = (pred_np == class_id).astype(np.uint8)
            gt_binary = (label_np == class_id).astype(np.uint8)
            
            valid_mask = (label_np != 255)
            pred_binary = pred_binary[valid_mask]
            gt_binary = gt_binary[valid_mask]
            
            intersection = np.sum(pred_binary & gt_binary)
            union = np.sum(pred_binary | gt_binary)
            pred_sum = np.sum(pred_binary)
            gt_sum = np.sum(gt_binary)
            
            total_intersection[class_id] += intersection
            total_union[class_id] += union
            total_pred_sum[class_id] += pred_sum
            total_gt_sum[class_id] += gt_sum

    def _compute_hd95_single(self, pred_binary, gt_binary, spacing=(1.0, 1.0)):
        """
        计算单个样本单个类别的HD95
        
        Args:
            pred_binary (np.ndarray): 二值预测 [H, W]
            gt_binary (np.ndarray): 二值真实标签 [H, W]
            spacing (tuple): 像素间距，默认(1.0, 1.0)表示1mm/pixel
            
        Returns:
            float or None: HD95距离（mm），如果无法计算则返回None
        """
        # 检查是否有前景像素
        if np.sum(pred_binary) == 0 and np.sum(gt_binary) == 0:
            return 0.0  # 都为空，距离为0
        elif np.sum(pred_binary) == 0 or np.sum(gt_binary) == 0:
            return np.inf  # 一个为空，距离为无穷大
        
        try:
            # 提取边界点
            pred_edges = self._get_boundary_points(pred_binary)
            gt_edges = self._get_boundary_points(gt_binary)
            
            if len(pred_edges) == 0 or len(gt_edges) == 0:
                return np.inf
            
            # 应用像素间距
            pred_edges = pred_edges * np.array(spacing)
            gt_edges = gt_edges * np.array(spacing)
            
            # 计算双向Hausdorff距离
            hd1 = directed_hausdorff(pred_edges, gt_edges)[0]
            hd2 = directed_hausdorff(gt_edges, pred_edges)[0]
            
            # 取最大值作为Hausdorff距离
            hd_distance = max(hd1, hd2)
            
            return hd_distance
            
        except Exception as e:
            # 计算失败时返回None
            self.logger.warning(f"HD95 computation failed: {e}")
            return None

    def _get_boundary_points(self, binary_mask):
        """
        提取二值mask的边界点坐标
        
        Args:
            binary_mask (np.ndarray): 二值mask [H, W]
            
        Returns:
            np.ndarray: 边界点坐标 [N, 2]，格式为(y, x)
        """
        # 使用形态学操作提取边界
        eroded = ndimage.binary_erosion(binary_mask)
        boundary = binary_mask ^ eroded
        
        # 获取边界点坐标
        boundary_coords = np.column_stack(np.where(boundary))  # [N, 2], (y, x)
        
        return boundary_coords.astype(np.float32)
