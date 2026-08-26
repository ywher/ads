# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import os.path as osp
import logging

import numpy as np
import torch

from lib.data import *
from .get_dataloader import get_dataset, get_data_loader
from .split_utils import ensure_rcs_class_stats, get_path_match_keys


def get_rcs_class_probs(stats_dir, temperature):
    with open(osp.join(stats_dir, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s = s.copy()
        s.pop('file', None)
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()

def get_crop_bbox(img_size, crop_size):
    """Randomly get a crop bounding box."""
    assert len(img_size) == len(crop_size)
    assert len(img_size) == 2
    margin_h = max(img_size[0] - crop_size[0], 0)
    margin_w = max(img_size[1] - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2


class SemiDataset(object):
    """Pair source and target datasets for semi training.

    为 semi 训练配对 source 与 target 数据集。

    Returned samples keep each inner dataset field and add a domain prefix:
    `src_im`, `src_lb`, `tar_im`, `tar_lb`, plus optional prefixed fields such
    as `tar_mask`, `src_dep`, `src_specified_scale`, or strong/weak keys.

    返回的样本会保留内部数据集字段，并添加域前缀：`src_im`、`src_lb`、
    `tar_im`、`tar_lb`，以及可能存在的 `tar_mask`、`src_dep`、
    `src_specified_scale` 或强弱增强字段。
    """

    def __init__(self, cfg):
        self.logger = logging.getLogger()
        self.cfg = cfg
        self.source = self.get_src_dataset()  # gta
        self.target = self.get_tar_dataset()  # city
        self.val_ld = self.get_val_loader()   # city val
        self.logger.info(f'len(self.source): {len(self.source)}')
        self.logger.info(f'len(self.target): {len(self.target)}')
        self.logger.info(f'len(self.val_ld): {len(self.val_ld)}')
        
        self.scales = self.cfg['source']['scale'] if 'scale' in self.cfg['source'] else None
        
        # check the source and target dataset
        self.ignore_index = self.target.ignore_index  # 255
        self.CLASSES = self.target.CLASSES  # 19 class names
        self.PALETTE = self.target.PALETTE  # 19 class colors
        
        assert self.target.ignore_index == self.source.ignore_index
        assert self.target.CLASSES == self.source.CLASSES
        assert self.target.PALETTE == self.source.PALETTE
                
        rcs_cfg = self.cfg.get('rare_class_sampling')
        self.rcs_enabled = rcs_cfg is not None
        self.logger.info(f'Rare class sampling: {self.rcs_enabled}')
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']  # 0.01
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']  # 0.5
            self.rcs_min_pixels = rcs_cfg['min_pixels']  # 3000
            self.rcs_stats_dir = ensure_rcs_class_stats(
                self.cfg['source']['data_root'],
                self.cfg['source']['im_anns'],
                rcs_cfg=rcs_cfg,
                num_classes=rcs_cfg.get('num_classes', 19),
                ignore_index=self.ignore_index,
            )

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                self.rcs_stats_dir, self.rcs_class_temp)
            self.logger.info(f'RCS Classes: {self.rcs_classes}')
            self.logger.info(f'RCS ClassProb: {self.rcs_classprob}')

            with open(osp.join(self.rcs_stats_dir, 'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }  # {id: [[lb_path, id_num], [],...], ...}
            
            self.samples_with_class = {}  # {id: [filename, ...], ...}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:  # 3000
                        self.samples_with_class[c].append(file)
                assert len(self.samples_with_class[c]) > 0

            self.file_to_idx = {}  # {filename: idx, ...}
            for i, lb_path in enumerate(self.source.lb_paths):
                for key in get_path_match_keys(lb_path, self.cfg['source']['data_root']):
                    self.file_to_idx[key] = i

    def _build_split_cfg(self, split):
        """Build a `get_dataset` config for one data split.

        为 source/target/val 中的一个 split 构造 `get_dataset` 所需配置。
        """
        split_cfg = self.cfg[split]
        dataset_cfg = {
            'rgb_mean': self.cfg['rgb_mean'],
            'rgb_std': self.cfg['rgb_std'],

            'dataset': split_cfg['type'],
            'data_root': split_cfg['data_root'],
            'im_anns': split_cfg['im_anns'],
            'resize': split_cfg['resize'],
            'keep_ratio': split_cfg.get('keep_ratio', False),
            'scale': split_cfg.get('scale', None),
            'cropsize': split_cfg['cropsize'],
            'cat_max_ratio': split_cfg['cat_max_ratio'],
            'flip': split_cfg['flip'],
            'photo_metric': split_cfg['photo_metric'],
            'rotate': split_cfg['rotate'],
        }
        if 'crop_pseudo_margins' in split_cfg:
            dataset_cfg['crop_pseudo_margins'] = split_cfg['crop_pseudo_margins']
        if 'offline_teacher_pseudo' in split_cfg:
            dataset_cfg['offline_teacher_pseudo'] = dict(
                split_cfg['offline_teacher_pseudo'])
        return dataset_cfg

    @staticmethod
    def _merge_domain_samples(src_data, tar_data):
        """Prefix and merge source/target sample dictionaries.

        给 source/target 样本字段添加前缀并合并为一个 semi batch 字典。
        """
        semi_data = {}
        semi_data.update({f'src_{k}': v for k, v in src_data.items()})
        semi_data.update({f'tar_{k}': v for k, v in tar_data.items()})
        return semi_data

    # get the source dataset
    def get_src_dataset(self):
        return get_dataset(self._build_split_cfg('source'), mode='train')
    
    # get the target dataset
    def get_tar_dataset(self):
        return get_dataset(self._build_split_cfg('target'), mode='train')

    # get the target dataloader
    def get_val_loader(self):
        val_dict = self._build_split_cfg('val')
        val_dict.update({
            'ims_per_gpu': 1,
            'num_works': self.cfg['workers_per_gpu'],
        })
        return get_data_loader(val_dict, mode='val', distributed=False)

    # get the rare class sample
    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = None
        for key in get_path_match_keys(f1, self.cfg['source']['data_root']):
            if key in self.file_to_idx:
                i1 = self.file_to_idx[key]
                break
        if i1 is None:
            self.logger.warning('RCS sample %s was not found, falling back to random source sample', f1)
            i1 = np.random.choice(range(len(self.source)))
        # print(f'c: {c}, f1: {f1}, i1: {i1}')
        
        use_decode_once_rcs = (
            hasattr(self.source, 'get_item_with_rcs_crop')
            and not (
                getattr(self.source, 'aug_mode', False)
                and getattr(self.source, 'trans_func_strong', None) is not None
            )
        )
        if use_decode_once_rcs:
            src_data = self.source.get_item_with_rcs_crop(
                i1,
                c,
                self.rcs_min_pixels,
                self.rcs_min_crop_ratio,
                max_retries=10,
            )
        else:
            src_data = self.source[i1]
            if self.rcs_min_crop_ratio > 0:  # 0.5
                for _ in range(10):
                    n_class = torch.sum(src_data['lb'].data == c)
                    if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                        break
                    src_data = self.source[i1]
        i2 = np.random.choice(range(len(self.target)))
        # print(f'i2: {i2}')
        tar_data = self.target[i2]
        
        
        return self._merge_domain_samples(src_data, tar_data)

    def __getitem__(self, idx):
        if self.scales is not None:
            # Randomly select a scale from the predefined scales
            scale = np.random.choice(self.scales)
            self.source.set_specified_scale(scale)
            self.target.set_specified_scale(scale)
            # print(f'{idx} Selected scale: {scale}')
        
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            src_data = self.source[idx % len(self.source)]
            tar_data = self.target[idx % len(self.target)]
            
            return self._merge_domain_samples(src_data, tar_data)

    def __len__(self):
        return len(self.source) * len(self.target)


class SSDADataset(object):
    """Triplet dataset for source-supervised domain adaptation.

    SSDA 训练同时使用三类数据：
    - `source`: 有标注源域数据；
    - `target_labeled`: 极少量有标注目标域数据；
    - `target_unlabeled`: 目标域无标注数据。

    返回字段使用 `src_`、`tgt_l_`、`tgt_u_` 前缀，训练脚本可以清晰区分
    三个分支，日志中也能分别记录三份 split 的来源和长度。
    """

    def __init__(self, cfg):
        self.logger = logging.getLogger()
        self.cfg = cfg
        self.source = self.get_src_dataset()
        self.target_labeled = self.get_target_labeled_dataset()
        self.target_unlabeled = self.get_target_unlabeled_dataset()
        self.val_ld = self.get_val_loader()
        self.logger.info(f'len(self.source): {len(self.source)}')
        self.logger.info(f'len(self.target_labeled): {len(self.target_labeled)}')
        self.logger.info(f'len(self.target_unlabeled): {len(self.target_unlabeled)}')
        self.logger.info(f'len(self.val_ld): {len(self.val_ld)}')

        self.scales = self.cfg['source']['scale'] if 'scale' in self.cfg['source'] else None

        self.ignore_index = self.target_unlabeled.ignore_index
        self.CLASSES = self.target_unlabeled.CLASSES
        self.PALETTE = self.target_unlabeled.PALETTE

        for name, dataset in (
            ('source', self.source),
            ('target_labeled', self.target_labeled),
        ):
            assert self.target_unlabeled.ignore_index == dataset.ignore_index, name
            assert self.target_unlabeled.CLASSES == dataset.CLASSES, name
            assert self.target_unlabeled.PALETTE == dataset.PALETTE, name

        source_rcs_cfg = self.cfg.get(
            'source_rare_class_sampling',
            self.cfg.get('rare_class_sampling'),
        )
        target_labeled_rcs_cfg = self.cfg.get(
            'target_labeled_rare_class_sampling')
        self.source_rcs_state = self._init_rcs_state(
            'source',
            self.source,
            self.cfg['source'],
            source_rcs_cfg,
        )
        self.target_labeled_rcs_state = self._init_rcs_state(
            'target_labeled',
            self.target_labeled,
            self.cfg['target_labeled'],
            target_labeled_rcs_cfg,
        )
        self.source_rcs_enabled = self.source_rcs_state is not None
        self.target_labeled_rcs_enabled = self.target_labeled_rcs_state is not None
        self.rcs_enabled = self.source_rcs_enabled or self.target_labeled_rcs_enabled
        self.logger.info(
            'SSDA rare class sampling: source=%s, target_labeled=%s',
            self.source_rcs_enabled,
            self.target_labeled_rcs_enabled,
        )
        self.rcs_stats_dir = (
            self.source_rcs_state['stats_dir']
            if self.source_rcs_state is not None else None)

    def _init_rcs_state(self, branch_name, dataset, split_cfg, rcs_cfg):
        enabled = rcs_cfg is not None
        self.logger.info('%s rare class sampling: %s', branch_name, enabled)
        if not enabled:
            return None

        rcs_cfg = dict(rcs_cfg)
        rcs_class_temp = rcs_cfg['class_temp']
        rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
        rcs_min_pixels = rcs_cfg['min_pixels']
        stats_dir = ensure_rcs_class_stats(
            split_cfg['data_root'],
            split_cfg['im_anns'],
            rcs_cfg=rcs_cfg,
            num_classes=rcs_cfg.get('num_classes', 19),
            ignore_index=self.ignore_index,
        )

        rcs_classes, rcs_classprob = get_rcs_class_probs(
            stats_dir, rcs_class_temp)
        self.logger.info('%s RCS Classes: %s', branch_name, rcs_classes)
        self.logger.info('%s RCS ClassProb: %s', branch_name, rcs_classprob)

        with open(osp.join(stats_dir, 'samples_with_class.json'), 'r') as of:
            samples_with_class_and_n = json.load(of)
        samples_with_class_and_n = {
            int(k): v
            for k, v in samples_with_class_and_n.items()
            if int(k) in rcs_classes
        }

        samples_with_class = {}
        for c in rcs_classes:
            samples_with_class[c] = []
            for file, pixels in samples_with_class_and_n[c]:
                if pixels > rcs_min_pixels:
                    samples_with_class[c].append(file)
            assert len(samples_with_class[c]) > 0

        file_to_idx = {}
        for i, lb_path in enumerate(dataset.lb_paths):
            for key in get_path_match_keys(lb_path, split_cfg['data_root']):
                file_to_idx[key] = i

        return {
            'branch_name': branch_name,
            'stats_dir': stats_dir,
            'classes': rcs_classes,
            'classprob': rcs_classprob,
            'min_crop_ratio': rcs_min_crop_ratio,
            'min_pixels': rcs_min_pixels,
            'samples_with_class': samples_with_class,
            'file_to_idx': file_to_idx,
        }

    def _build_split_cfg(self, split):
        split_cfg = self.cfg[split]
        dataset_cfg = {
            'rgb_mean': self.cfg['rgb_mean'],
            'rgb_std': self.cfg['rgb_std'],
            'dataset': split_cfg['type'],
            'data_root': split_cfg['data_root'],
            'im_anns': split_cfg['im_anns'],
            'resize': split_cfg['resize'],
            'keep_ratio': split_cfg.get('keep_ratio', False),
            'scale': split_cfg.get('scale', None),
            'cropsize': split_cfg['cropsize'],
            'cat_max_ratio': split_cfg['cat_max_ratio'],
            'flip': split_cfg['flip'],
            'photo_metric': split_cfg['photo_metric'],
            'rotate': split_cfg['rotate'],
        }
        if 'crop_pseudo_margins' in split_cfg:
            dataset_cfg['crop_pseudo_margins'] = split_cfg['crop_pseudo_margins']
        if 'offline_teacher_pseudo' in split_cfg:
            dataset_cfg['offline_teacher_pseudo'] = dict(
                split_cfg['offline_teacher_pseudo'])
        return dataset_cfg

    @staticmethod
    def _prefix_sample(prefix, sample):
        return {f'{prefix}_{k}': v for k, v in sample.items()}

    def get_src_dataset(self):
        return get_dataset(self._build_split_cfg('source'), mode='train')

    def get_target_labeled_dataset(self):
        return get_dataset(self._build_split_cfg('target_labeled'), mode='train')

    def get_target_unlabeled_dataset(self):
        return get_dataset(self._build_split_cfg('target_unlabeled'), mode='train')

    def get_val_loader(self):
        val_dict = self._build_split_cfg('val')
        val_dict.update({
            'ims_per_gpu': 1,
            'num_works': self.cfg['workers_per_gpu'],
        })
        return get_data_loader(val_dict, mode='val', distributed=False)

    def get_rare_sample(self, dataset, split_cfg, rcs_state):
        c = np.random.choice(rcs_state['classes'], p=rcs_state['classprob'])
        f1 = np.random.choice(rcs_state['samples_with_class'][c])
        i1 = None
        for key in get_path_match_keys(f1, split_cfg['data_root']):
            if key in rcs_state['file_to_idx']:
                i1 = rcs_state['file_to_idx'][key]
                break
        if i1 is None:
            self.logger.warning(
                '%s RCS sample %s was not found, falling back to random sample',
                rcs_state['branch_name'],
                f1,
            )
            i1 = np.random.choice(range(len(dataset)))

        use_decode_once_rcs = (
            hasattr(dataset, 'get_item_with_rcs_crop')
            and not (
                getattr(dataset, 'aug_mode', False)
                and getattr(dataset, 'trans_func_strong', None) is not None
            )
        )
        if use_decode_once_rcs:
            return dataset.get_item_with_rcs_crop(
                i1,
                c,
                rcs_state['min_pixels'],
                rcs_state['min_crop_ratio'],
                max_retries=10,
            )

        data = dataset[i1]
        if rcs_state['min_crop_ratio'] > 0:
            for _ in range(10):
                n_class = torch.sum(data['lb'].data == c)
                if n_class > rcs_state['min_pixels'] * rcs_state['min_crop_ratio']:
                    break
                data = dataset[i1]
        return data

    def get_rare_source_sample(self):
        return self.get_rare_sample(
            self.source,
            self.cfg['source'],
            self.source_rcs_state,
        )

    def get_rare_target_labeled_sample(self):
        return self.get_rare_sample(
            self.target_labeled,
            self.cfg['target_labeled'],
            self.target_labeled_rcs_state,
        )

    def _merge_triplet_samples(self, src_data, tgt_l_data, tgt_u_data):
        ssda_data = {}
        ssda_data.update(self._prefix_sample('src', src_data))
        ssda_data.update(self._prefix_sample('tgt_l', tgt_l_data))
        ssda_data.update(self._prefix_sample('tgt_u', tgt_u_data))
        return ssda_data

    def __getitem__(self, idx):
        if self.scales is not None:
            scale = np.random.choice(self.scales)
            self.source.set_specified_scale(scale)
            self.target_labeled.set_specified_scale(scale)
            self.target_unlabeled.set_specified_scale(scale)

        if self.source_rcs_enabled:
            src_data = self.get_rare_source_sample()
        else:
            src_data = self.source[idx % len(self.source)]
        if self.target_labeled_rcs_enabled:
            tgt_l_data = self.get_rare_target_labeled_sample()
        else:
            tgt_l_data = self.target_labeled[idx % len(self.target_labeled)]
        tgt_u_data = self.target_unlabeled[idx % len(self.target_unlabeled)]
        return self._merge_triplet_samples(src_data, tgt_l_data, tgt_u_data)

    def __len__(self):
        return max(
            len(self.source),
            len(self.target_labeled),
            len(self.target_unlabeled),
        )
