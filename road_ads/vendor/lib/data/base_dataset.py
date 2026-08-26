#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import logging
import json
import cv2
import numpy as np

import torch
from torch.utils.data import Dataset

from .split_utils import (
    ensure_rcs_class_stats,
    get_path_match_keys,
    load_ann_pairs,
    resolve_data_path,
)


def get_rcs_class_probs(stats_dir, temperature):
    with open(os.path.join(stats_dir, 'sample_class_stats.json'), 'r') as of:
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


class BaseDataset(Dataset):
    """
    Base class for datasets, providing common functionality for loading and processing images and labels.
    """
    def __init__(self, dataroot, annpath, trans_func=None, trans_func_strong=None, mode='train',
                 norm={'mean': (123.675, 116.28, 103.53), 'std': (58.395, 57.12, 57.375)},
                 return_img_name=False, rcs_cfg=None, aug_mode=False, weather_aug_cfg=None):
        super(BaseDataset, self).__init__()
        assert mode in ('train', 'val', 'test'), f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'."
        self.logger = logging.getLogger()
        self.mode = mode
        self.trans_func = trans_func
        self.trans_func_strong = trans_func_strong  # 强增强变换
        self.ignore_index = 255
        self.lb_map = None
        self.norm_cfg = norm
        self.return_img_name = return_img_name
        self.rcs_enabled = rcs_cfg is not None
        self.aug_mode = aug_mode  # 是否返回双增强图像
        self.dataroot = dataroot
        self.annpath = annpath
        
        # Weather augmentation configuration
        self.weather_aug_cfg = weather_aug_cfg
        self.weather_aug_enabled = weather_aug_cfg is not None and mode == 'train' and aug_mode
        if self.weather_aug_enabled:
            self.normal_folder = weather_aug_cfg.get('normal_folder', 'train')
            self.weather_folder = weather_aug_cfg.get('weather_folder', 'train_weather_all')
            self.weather_conditions = weather_aug_cfg.get('conditions', ['dawn', 'dusk', 'foggy', 'glare', 'night', 'rainy', 'snowy', 'sunny'])
            self.weather_aug_prob = weather_aug_cfg.get('prob', 0.5)
            self.weather_root = weather_aug_cfg.get('weather_root', None)
            self.weather_mix_ratio = weather_aug_cfg.get('mix_ratio', 0.0)  # 原图融合系数，0表示纯天气图，1表示纯原图
            self.weather_class_mix = weather_aug_cfg.get('class_mix', 0.0)  # 类别混合比例，0表示不使用类别混合
            self.logger.info(f'Weather augmentation enabled: prob={self.weather_aug_prob}, mix_ratio={self.weather_mix_ratio}, class_mix={self.weather_class_mix}, conditions={self.weather_conditions}')
        
        # Cache mean and std as numpy arrays for fast broadcasting
        self._mean = np.array(self.norm_cfg['mean'], dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(self.norm_cfg['std'], dtype=np.float32).reshape(1, 1, 3)

        # Load image and label paths
        self.img_paths, self.lb_paths = self._load_paths(dataroot, annpath)
        self._validate_image_paths(dataroot, annpath)
        self.len = len(self.img_paths)
        
        # for resize src and tar image
        self.specified_scale = None  # 添加这个属性
        
        # 初始化 rare class 相关信息
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg.get('class_temp', 0.01)
            self.rcs_min_crop_ratio = rcs_cfg.get('min_crop_ratio', 0.5)
            self.rcs_min_pixels = rcs_cfg.get('min_pixels', 3000)
            self.rcs_stats_dir = ensure_rcs_class_stats(
                dataroot,
                annpath,
                rcs_cfg=rcs_cfg,
                num_classes=rcs_cfg.get('num_classes', 19),
                ignore_index=self.ignore_index,
            )
            
            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                self.rcs_stats_dir, self.rcs_class_temp)
            self.logger.info(f'RCS Classes: {self.rcs_classes}')
            self.logger.info(f'RCS ClassProb: {self.rcs_classprob}')
            # print(f'RCS Classes: {self.rcs_classes}')
            # print(f'RCS ClassProb: {self.rcs_classprob}')
                        
            with open(os.path.join(self.rcs_stats_dir, 'samples_with_class.json'), 'r') as of:
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
            for i, lb_path in enumerate(self.lb_paths):
                for key in get_path_match_keys(lb_path, dataroot):
                    self.file_to_idx[key] = i
        
        
    def set_specified_scale(self, scale):
        """设置指定的缩放因子"""
        self.specified_scale = scale
    
    def _load_paths(self, dataroot, annpath):
        """
        Load image and label paths from the annotation file.
        """
        img_paths, lb_paths = [], []
        for imgpth, lbpth in load_ann_pairs(annpath):
            img_paths.append(resolve_data_path(dataroot, imgpth))
            lb_paths.append(resolve_data_path(dataroot, lbpth))
        assert len(img_paths) == len(lb_paths), "Mismatch between image and label paths."
        return img_paths, lb_paths

    def _validate_image_paths(self, dataroot, annpath):
        """Fail early when the split points to missing images.

        Labels may be absent for true unlabeled data, but images must always be
        readable. Catching missing images during dataset construction gives a
        useful error on the main process instead of a later DataLoader worker
        `NoneType.shape` failure.
        """
        missing_count = 0
        missing_examples = []
        for img_path in self.img_paths:
            if os.path.isfile(img_path):
                continue
            missing_count += 1
            if len(missing_examples) < 8:
                missing_examples.append(img_path)

        if missing_count == 0:
            return

        examples = '\n'.join(f'  - {path}' for path in missing_examples)
        raise FileNotFoundError(
            'Dataset image path check failed: '
            f'{missing_count}/{len(self.img_paths)} images are missing.\n'
            f'data_root: {dataroot}\n'
            f'annpath: {annpath}\n'
            f'first missing images:\n{examples}\n'
            'Please fix the data_root/data symlink or set the dataset root '
            'environment variable, e.g. SSDA_CITYSCAPES_ROOT=/path/to/cityscapes.'
        )
    
    def _get_index_by_file(self, file_name):
        # 支持多种分隔符，确保文件名一致
        for key in get_path_match_keys(file_name, self.dataroot):
            if key in self.file_to_idx:
                return self.file_to_idx[key]
        self.logger.warning(f"File {file_name} not found in file_to_idx mapping!")
        return None

    def get_item_with_rcs_crop(self, idx, class_id, min_pixels,
                               min_crop_ratio, max_retries=10):
        """Return an RCS sample without reopening it for every crop retry.

        The selected image, class, acceptance threshold, retry count and
        stochastic augmentation suffix are identical to the outer RCS loop.
        Only deterministic work (decode and cacheable resize operations) is
        reused.  Weak/strong augmentation mode keeps the historical path
        because it may intentionally load a second weather image.
        """
        if self.aug_mode and self.trans_func_strong is not None:
            raise NotImplementedError(
                'decode-once RCS is not enabled for weak/strong augmentation')

        impth, lbpth = self.img_paths[idx], self.lb_paths[idx]
        img, label = self.get_image(
            impth, lbpth, use_weather_for_strong=False)
        if self.lb_map is not None:
            label = self.lb_map[label]

        im_lb = dict(im=img, lb=label)
        if self.specified_scale is not None:
            im_lb['specified_scale'] = self.specified_scale

        prepared = im_lb
        split_index = None
        if self.trans_func is not None and hasattr(
                self.trans_func, 'prepare_rcs_retry'):
            prepared, split_index = self.trans_func.prepare_rcs_retry(im_lb)

        attempts = max_retries + 1 if min_crop_ratio > 0 else 1
        transformed = prepared
        for _ in range(attempts):
            if self.trans_func is None:
                transformed = prepared
            elif split_index is not None:
                transformed = self.trans_func.apply_rcs_retry(
                    prepared, split_index)
            else:
                transformed = self.trans_func(im_lb.copy())

            if min_crop_ratio <= 0:
                break
            n_class = np.count_nonzero(transformed['lb'] == class_id)
            if n_class > min_pixels * min_crop_ratio:
                break

        im = self._process_image(transformed['im'])
        lb = torch.from_numpy(transformed['lb'].astype(np.int64))
        data = {'im': im, 'lb': lb, 'im_path': impth}
        if 'specified_scale' in transformed:
            data['specified_scale'] = transformed['specified_scale']
        if self.return_img_name:
            data['im_name'] = os.path.basename(impth)
        return data
    
    def get_rcs_sample(self):
        """采样一个稀有类别样本，并保证 crop 后该类别像素足够"""
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        file_name = np.random.choice(self.samples_with_class[c])
        idx = self._get_index_by_file(file_name)
        if idx is None:
            # fallback: 随机采样一个样本
            idx = np.random.randint(0, self.len)
        impth, lbpth = self.img_paths[idx], self.lb_paths[idx]
        
        # 如果是 aug_mode，返回弱增强和强增强两个版本
        if self.aug_mode and self.trans_func_strong is not None:
            # 弱增强 - 使用原始图像
            img_weak, label_weak = self.get_image(impth, lbpth, use_weather_for_strong=False)
            if self.lb_map is not None:
                label_weak = self.lb_map[label_weak]
            
            im_lb_weak = dict(im=img_weak.copy(), lb=label_weak.copy())
            if self.specified_scale is not None:
                im_lb_weak['specified_scale'] = self.specified_scale
            # crop保证
            if self.trans_func is not None and self.rcs_min_crop_ratio > 0:
                for j in range(10):
                    tmp_im_lb = self.trans_func(im_lb_weak.copy())
                    n_class = np.sum(tmp_im_lb['lb'] == c)
                    if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                        im_lb_weak = tmp_im_lb
                        break
                else:
                    im_lb_weak = tmp_im_lb
            elif self.trans_func is not None:
                im_lb_weak = self.trans_func(im_lb_weak)
            
            # 提取crop_bbox用于强增强的空间对齐（flip可以不一致）
            crop_bbox = im_lb_weak.get('crop_bbox', None)
            
            im_weak = self._process_image(im_lb_weak['im'])
            lb_weak = torch.from_numpy(im_lb_weak['lb'].astype(np.int64))
            
            # 强增强 - 可能使用天气增强图像
            img_strong, label_strong = self.get_image(impth, lbpth, use_weather_for_strong=True)
            if self.lb_map is not None:
                label_strong = self.lb_map[label_strong]
            
            im_lb_strong = dict(im=img_strong.copy(), lb=label_strong.copy())
            if self.specified_scale is not None:
                im_lb_strong['specified_scale'] = self.specified_scale
            if crop_bbox is not None:
                im_lb_strong['crop_bbox'] = crop_bbox  # 使用相同的crop位置（flip可以不同）
                
            if self.trans_func_strong is not None:
                im_lb_strong = self.trans_func_strong(im_lb_strong)
            im_strong = self._process_image(im_lb_strong['im'])
            lb_strong = torch.from_numpy(im_lb_strong['lb'].astype(np.int64))
            
            data = {
                'im_weak': im_weak,
                'lb_weak': lb_weak,
                'im_weak_did_flip': im_lb_weak['did_flip'],
                'im_strong': im_strong,
                'lb_strong': lb_strong,
                'im_strong_did_flip': im_lb_strong['did_flip'],
            }
            if 'specified_scale' in im_lb_weak:
                data['specified_scale'] = im_lb_weak['specified_scale']
            if self.return_img_name:
                data['im_name'] = os.path.basename(impth)
            return data
        else:
            # 标准模式 - 不使用天气增强
            img, label = self.get_image(impth, lbpth, use_weather_for_strong=False)
            if self.lb_map is not None:
                label = self.lb_map[label]
            
            im_lb = dict(im=img, lb=label)
            if self.specified_scale is not None:
                im_lb['specified_scale'] = self.specified_scale
            # crop保证
            if self.trans_func is not None and self.rcs_min_crop_ratio > 0:
                for j in range(10):
                    tmp_im_lb = self.trans_func(im_lb)
                    n_class = np.sum(tmp_im_lb['lb'] == c)
                    if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                        im_lb = tmp_im_lb
                        break
                else:
                    # self.logger.warning(f"RCS crop failed for class {c}, file {file_name}, using last crop.")
                    im_lb = tmp_im_lb
            elif self.trans_func is not None:
                im_lb = self.trans_func(im_lb)
            im = self._process_image(im_lb['im'])
            lb = torch.from_numpy(im_lb['lb'].astype(np.int64))
            data = {'im': im, 'lb': lb, 'im_path': impth}
            if 'specified_scale' in im_lb:
                data['specified_scale'] = im_lb['specified_scale']
            if self.return_img_name:
                data['im_name'] = os.path.basename(impth)
            return data
    
    
    def __getitem__(self, idx):
        """
        Get an item by index, including image, label, and optionally the image name.
        If aug_mode is True, returns both weak and strong augmented images.
        """
        if self.rcs_enabled:
            return self.get_rcs_sample()
        else:
            impth, lbpth = self.img_paths[idx], self.lb_paths[idx]
            img, label = self.get_image(impth, lbpth)
            # print(f"Loading image: {impth}, label: {lbpth}")
            # print(f"Image shape: {img.shape}, Label shape: {label.shape}")

            # Apply label mapping if specified
            if self.lb_map is not None:
                label = self.lb_map[label]

            # 如果是 aug_mode，返回弱增强和强增强两个版本
            if self.aug_mode and self.trans_func_strong is not None:
                # 弱增强 - 使用原始图片
                im_lb_weak = dict(im=img.copy(), lb=label.copy())
                if self.specified_scale is not None:
                    im_lb_weak['specified_scale'] = self.specified_scale
                if self.trans_func is not None:
                    im_lb_weak = self.trans_func(im_lb_weak)
                
                # 提取crop_bbox用于强增强的空间对齐（flip可以不一致）
                crop_bbox = im_lb_weak.get('crop_bbox', None)
                
                im_weak = self._process_image(im_lb_weak['im'])
                lb_weak = torch.from_numpy(im_lb_weak['lb'].astype(np.int64))
                
                # 强增强 - 可能使用天气图片作为初始图
                # 重新加载图片，可能使用天气变化版本
                img_strong, label_strong = self.get_image(impth, lbpth, use_weather_for_strong=True)
                
                im_lb_strong = dict(im=img_strong.copy(), lb=label_strong.copy())
                if self.specified_scale is not None:
                    im_lb_strong['specified_scale'] = self.specified_scale
                if crop_bbox is not None:
                    im_lb_strong['crop_bbox'] = crop_bbox  # 使用相同的crop位置（flip可以不同）
                    
                if self.trans_func_strong is not None:
                    im_lb_strong = self.trans_func_strong(im_lb_strong)
                im_strong = self._process_image(im_lb_strong['im'])
                lb_strong = torch.from_numpy(im_lb_strong['lb'].astype(np.int64))

                data = {
                    'im_weak': im_weak,
                    'lb_weak': lb_weak,
                    'im_weak_did_flip': im_lb_weak['did_flip'],
                    'im_strong': im_strong,
                    'lb_strong': lb_strong,
                    'im_strong_did_flip': im_lb_strong['did_flip'],
                }
                if 'specified_scale' in im_lb_weak:
                    data['specified_scale'] = im_lb_weak['specified_scale']
                if self.return_img_name:
                    data['im_name'] = os.path.basename(impth)
                return data
            else:
                # 标准模式，只返回单个增强图像
                # Apply transformations
                im_lb = dict(im=img, lb=label)
                offline_mask = self._offline_teacher_transform_mask(idx)
                if offline_mask is not None:
                    im_lb['mask'] = offline_mask
                if self.specified_scale is not None:
                    im_lb['specified_scale'] = self.specified_scale
                    
                if self.trans_func is not None:
                    im_lb = self.trans_func(im_lb)

                # Process image and label
                im = self._process_image(im_lb['im'])
                lb = torch.from_numpy(im_lb['lb'].astype(np.int64))  # 只需astype，无需copy/clone

                data = {'im': im, 'lb': lb, 'im_path': impth}
                self._decode_offline_teacher_transform_mask(
                    data, im_lb.get('mask'))
                
                if 'specified_scale' in im_lb:
                    data['specified_scale'] = im_lb['specified_scale']
                
                if self.return_img_name:
                    data['im_name'] = os.path.basename(impth)
                return data

    def _apply_class_mix(self, original_img, weather_img, label, class_mix_ratio):
        """
        从原图中随机选择一定比例的类别区域，粘贴到天气变化图上
        Args:
            original_img: 原始图片 (H, W, 3)
            weather_img: 天气变化后的图片 (H, W, 3)
            label: 语义标签 (H, W)
            class_mix_ratio: 从原图中选择类别的比例 (0-1)
        Returns:
            mixed_img: 混合后的图片 (H, W, 3)
        """
        # 获取所有有效的类别 (忽略255)
        valid_classes = np.unique(label)
        valid_classes = valid_classes[valid_classes != 255]  # 排除ignore类别
        
        if len(valid_classes) == 0:
            return weather_img  # 如果没有有效类别，直接返回天气图
        
        # 计算需要从原图中选择多少个类别
        num_classes_to_select = max(1, int(len(valid_classes) * class_mix_ratio))
        
        # 随机选择类别
        selected_classes = np.random.choice(valid_classes, size=num_classes_to_select, replace=False)
        
        # 创建混合掩码
        class_mask = np.zeros(label.shape, dtype=bool)
        for cls in selected_classes:
            class_mask |= (label == cls)
        
        # 将选中类别的区域从原图复制到天气图
        mixed_img = weather_img.copy()
        mixed_img[class_mask] = original_img[class_mask]
        
        return mixed_img
    
    def _process_image(self, img):
        # --- 优化点2：BGR2RGB和归一化 ---
        img = img[:, :, ::-1]  # BGR to RGB
        img = img.astype(np.float32)
        img = (img - self._mean) / self._std  # normalization
        img = np.ascontiguousarray(img.transpose(2, 0, 1))  # HWC->CHW，make sure C-contiguous
        return torch.from_numpy(img)

    def _offline_teacher_transform_mask(self, idx, valid_mask=None):
        """Pack confidence and validity for identical spatial transforms.

        The low byte stores uint8 teacher confidence and bit 8 stores the
        geometric valid region. Existing mask transforms use nearest-neighbor
        interpolation, so both maps remain aligned with the pseudo label.
        """
        paths = getattr(self, 'offline_teacher_confidence_paths', None)
        if paths is None:
            return valid_mask
        confidence = cv2.imread(paths[idx], cv2.IMREAD_GRAYSCALE)
        if confidence is None:
            raise FileNotFoundError(
                f'Failed to read offline teacher confidence: {paths[idx]}')
        if valid_mask is None:
            valid_mask = np.ones(confidence.shape, dtype=np.uint8)
        elif valid_mask.shape != confidence.shape:
            valid_mask = cv2.resize(
                valid_mask,
                (confidence.shape[1], confidence.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return (
            confidence.astype(np.uint16)
            | ((valid_mask > 0).astype(np.uint16) << 8)
        ).astype(np.uint16, copy=False)

    @staticmethod
    def _decode_offline_teacher_transform_mask(data, transformed_mask):
        """Expose geometric validity and aligned uint8 confidence tensors."""
        if transformed_mask is None:
            return
        if transformed_mask.dtype == np.uint16:
            data['mask'] = torch.from_numpy(
                ((transformed_mask >> 8) & 1).astype(np.uint8))
            data['offline_teacher_confidence'] = torch.from_numpy(
                (transformed_mask & 255).astype(np.uint8))
        else:
            data['mask'] = torch.from_numpy(
                transformed_mask.astype(np.uint8))

    def get_weather_image_path(self, original_img_path):
        """
        根据原始图片路径，随机选择一个天气条件下的对应图片路径
        Args:
            original_img_path: 原始图片路径，例如: 'datasets/cityscapes/leftImg8bit/train/aachen/xxx.png'
        Returns:
            weather_img_path: 天气变化后的图片路径，例如: 'datasets/cityscapes/leftImg8bit/train_weather/rainy/aachen/xxx.png'
        """
        if not self.weather_aug_enabled or np.random.rand() > self.weather_aug_prob:
            return None  # 不使用天气增强
        
        # 随机选择一个天气条件
        weather_condition = np.random.choice(self.weather_conditions)
        
        # 构建天气图片路径
        # 原始路径: .../leftImg8bit/train/aachen/xxx.png
        # 目标路径: .../leftImg8bit/train_weather/rainy/aachen/xxx.png
        
        # 分解路径
        path_parts = original_img_path.split(os.sep)
        
        # 找到 'train' 的位置并替换为 'train_weather/condition'
        for i, part in enumerate(path_parts):
            if part == self.normal_folder:
                # 替换 'train' 为 'train_weather/condition'
                weather_path_parts = path_parts[:i] + [self.weather_folder, weather_condition] + path_parts[i+1:]
                weather_img_path = os.sep.join(weather_path_parts)
                
                # 检查文件是否存在
                if os.path.exists(weather_img_path):
                    return weather_img_path
                else:
                    self.logger.warning(f'Weather image not found: {weather_img_path}, using original image')
                    return None
        
        return None

    def get_image(self, impth, lbpth, use_weather_for_strong=False):
        """
        Load an image and its corresponding label.
        Args:
            impth: 原始图片路径
            lbpth: 标签路径
            use_weather_for_strong: 是否为强增强使用天气图片
        """
        # 先读取原始图片
        # print(f"Loading image: {impth}")
        original_img = cv2.imread(impth, cv2.IMREAD_COLOR)
        if original_img is None:
            raise FileNotFoundError(
                f'Failed to read image with cv2.imread: {impth}\n'
                f'data_root: {self.dataroot}\n'
                f'annpath: {self.annpath}\n'
                'Check whether the file exists, the symlink is valid, and the '
                'image is not corrupted.'
            )
        img = original_img
        
        # 如果需要使用天气增强，尝试加载天气图片并融合
        if use_weather_for_strong and self.weather_aug_enabled:
            weather_img_path = self.get_weather_image_path(impth)
            if weather_img_path is not None:
                weather_img = cv2.imread(weather_img_path, cv2.IMREAD_COLOR)
                if weather_img is not None:
                    # 提取天气条件用于日志
                    path_parts = weather_img_path.split(os.sep)
                    weather_condition = None
                    for i, part in enumerate(path_parts):
                        if part == self.weather_folder and i + 1 < len(path_parts):
                            weather_condition = path_parts[i + 1]
                            break
                    
                    # Step 1: 像素级混合 (mix_ratio)
                    if self.weather_mix_ratio > 0:
                        img = cv2.addWeighted(
                            original_img, self.weather_mix_ratio,
                            weather_img, 1.0 - self.weather_mix_ratio,
                            0
                        )
                        # print(f'🌦️✨ [Weather Aug] Blending {weather_condition} image (ratio={self.weather_mix_ratio:.2f}): {os.path.basename(weather_img_path)}')
                    else:
                        img = weather_img
                        # print(f'🌦️  [Weather Aug] Using pure {weather_condition} image: {os.path.basename(weather_img_path)}')
                    
                    # Step 2: 类别级混合 (class_mix) - 从原图中随机选择类别区域粘贴到天气图上
                    if self.weather_class_mix > 0:
                        label = cv2.imread(lbpth, cv2.IMREAD_GRAYSCALE)
                        if label is not None:
                            img = self._apply_class_mix(original_img, img, label, self.weather_class_mix)
                            # print(f'🎨 [Class Mix] Applied class mix (ratio={self.weather_class_mix:.2f})')
                else:
                    # 如果读取失败，使用原始图片
                    pass
                    # print(f'⚠️  [Weather Aug] Failed to load weather image, using original: {os.path.basename(impth)}')
        
        label = cv2.imread(lbpth, cv2.IMREAD_GRAYSCALE)  # Read label in grayscale
        if label is None:
            label = np.ones(img.shape[:2], dtype=np.uint8)
            label *= self.ignore_index
        
        return img, label

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return self.len


if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    # Example usage
    class CityScapes(BaseDataset):
        pass

    ds = CityScapes('./data/', './data/annotations.txt', mode='val')
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4, drop_last=True)

    for imgs, labels in tqdm(dl, desc="Testing DataLoader"):
        print(f"Batch size: {len(imgs)}")
        for img in imgs:
            print(f"Image size: {img.size()}")
        break
