#!/usr/bin/python
# -*- encoding: utf-8 -*-

from torch.utils.data import DataLoader

from .base_dataset import BaseDataset
from .base_depth_dataset import BaseDepthDataset

NUM_CLASSES = 19
IGNORE_INDEX = 255
NORM_CFG = {'mean': (123.675, 116.28, 103.53), 'std': (58.395, 57.12, 57.375)}
CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
)
PALETTE = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    [0, 0, 230], [119, 11, 32]
]


class MapillaryDataset(BaseDataset):
    """
    MapillaryDataset 语义分割数据集
    """
    def __init__(self, dataroot, annpath=None, trans_func=None, trans_func_strong=None, mode='train',
                 norm=NORM_CFG, return_img_name=False, rcs_cfg=None, crop_pseudo_margins=None, aug_mode=False, weather_aug_cfg=None):
        super().__init__(dataroot, annpath, trans_func, trans_func_strong, mode, norm, return_img_name, rcs_cfg, aug_mode, weather_aug_cfg)
        self.num_classes = NUM_CLASSES
        self.ignore_index = IGNORE_INDEX
        self.norm_cfg = norm
        self.CLASSES = CLASSES
        self.PALETTE = PALETTE


class MapillaryDepDataset(BaseDepthDataset):
    """
    MapillaryDataset 语义分割数据集（RGB + 深度）
    """
    def __init__(self, dataroot, annpath, trans_func=None, mode='train',
                 norm=NORM_CFG, return_img_name=False, rcs_cfg=None):
        super().__init__(dataroot, annpath, trans_func, mode, norm, return_img_name, rcs_cfg)
        self.num_classes = NUM_CLASSES
        self.ignore_index = IGNORE_INDEX
        self.norm_cfg = norm
        self.CLASSES = CLASSES
        self.PALETTE = PALETTE
        
# ---------- 示例测试 ----------
