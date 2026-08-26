#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import numpy as np

from utils.classes import CLASSES as CLASS_REGISTRY
from utils.color_map import color_map

from .base_dataset import BaseDataset
from .cityscapes_cv2 import CityscapesDataset, IGNORE_INDEX, NORM_CFG


SYN_CITY_CLASSES = tuple(CLASS_REGISTRY['syn_city'])
SYN_CITY_PALETTE = [
    list(color_map['syn_city'][idx])
    for idx in range(len(SYN_CITY_CLASSES))
]
SYN_CITY_NUM_CLASSES = len(SYN_CITY_CLASSES)

CITYSCAPES_TO_SYN_CITY = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for city_id, syn_id in {
    0: 0,    # road
    1: 1,    # sidewalk
    2: 2,    # building
    3: 3,    # wall
    4: 4,    # fence
    5: 5,    # pole
    6: 6,    # traffic light
    7: 7,    # traffic sign
    8: 8,    # vegetation
    # Cityscapes terrain/truck/train are ignored for SYNTHIA->Cityscapes.
    10: 9,   # sky
    11: 10,  # person
    12: 11,  # rider
    13: 12,  # car
    15: 13,  # bus
    17: 14,  # motorcycle
    18: 15,  # bicycle
}.items():
    CITYSCAPES_TO_SYN_CITY[city_id] = syn_id
CITYSCAPES_TO_SYN_CITY[IGNORE_INDEX] = IGNORE_INDEX


class SYNTHIADataset(BaseDataset):
    """SYNTHIA source dataset using pre-converted 16-class trainIds.

    `data/synthia/train.txt` is expected to contain image/label pairs pointing
    to `GT/LABELS_trainid_16cls`, whose label ids follow the `syn_city` order.
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
        self.num_classes = SYN_CITY_NUM_CLASSES
        self.ignore_index = IGNORE_INDEX
        self.norm_cfg = norm
        self.CLASSES = SYN_CITY_CLASSES
        self.PALETTE = SYN_CITY_PALETTE
        self.pseudo_margins = crop_pseudo_margins


class CityscapesSynCityDataset(CityscapesDataset):
    """Cityscapes target/val dataset remapped to SYNTHIA's 16 shared classes."""

    def __init__(self, dataroot, annpath=None, trans_func=None,
                 trans_func_strong=None, mode='train', norm=NORM_CFG,
                 return_img_name=False, rcs_cfg=None, crop_pseudo_margins=None,
                 aug_mode=False, weather_aug_cfg=None):
        if rcs_cfg is not None:
            rcs_cfg = rcs_cfg.copy()
            rcs_cfg['label_map'] = CITYSCAPES_TO_SYN_CITY
            if annpath is not None:
                # Keep 16-class RCS stats separate from existing Cityscapes-19
                # stats in the same split folder.
                # 将 16 类 RCS 统计与同一 split 下已有的 19 类统计分开保存。
                rcs_cfg.setdefault(
                    'stats_dir',
                    os.path.join(os.path.dirname(os.path.abspath(annpath)),
                                 'rcs_syn_city'),
                )
        super().__init__(
            dataroot,
            annpath,
            trans_func,
            trans_func_strong,
            mode,
            norm,
            return_img_name,
            rcs_cfg,
            crop_pseudo_margins,
            aug_mode,
            weather_aug_cfg,
        )
        self.num_classes = SYN_CITY_NUM_CLASSES
        self.CLASSES = SYN_CITY_CLASSES
        self.PALETTE = SYN_CITY_PALETTE
        self.lb_map = CITYSCAPES_TO_SYN_CITY
