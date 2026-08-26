#!/usr/bin/python
# -*- encoding: utf-8 -*-

from .base_dataset import BaseDataset
from .cityscapes_cv2 import (
    CLASSES,
    IGNORE_INDEX,
    NORM_CFG,
    NUM_CLASSES,
    PALETTE,
)


class GTADataset(BaseDataset):
    """GTA5/GTAV semantic segmentation dataset with Cityscapes trainIds.

    The local `data/gta/train.txt` is expected to contain image/label pairs
    such as `images/00001.png,labels/00001_labelTrainIds.png`.
    """

    def __init__(self, dataroot, annpath=None, trans_func=None,
                 trans_func_strong=None, mode='train', norm=NORM_CFG,
                 return_img_name=False, rcs_cfg=None, crop_pseudo_margins=None,
                 aug_mode=False, weather_aug_cfg=None):
        super().__init__(
            dataroot,
            annpath,
            trans_func,
            trans_func_strong,
            mode,
            norm,
            return_img_name,
            rcs_cfg,
            aug_mode,
            weather_aug_cfg,
        )
        self.num_classes = NUM_CLASSES
        self.ignore_index = IGNORE_INDEX
        self.norm_cfg = norm
        self.CLASSES = CLASSES
        self.PALETTE = PALETTE
        self.pseudo_margins = crop_pseudo_margins
