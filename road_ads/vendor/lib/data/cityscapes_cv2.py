#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import numpy as np

import torch
from torch.utils.data import DataLoader

from .base_dataset import BaseDataset
from .base_depth_dataset import BaseDepthDataset


labels_info = [
    {"hasInstances": False, "category": "void", "catid": 0, "name": "unlabeled", "ignoreInEval": True, "id": 0, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "ego vehicle", "ignoreInEval": True, "id": 1, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "rectification border", "ignoreInEval": True, "id": 2, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "out of roi", "ignoreInEval": True, "id": 3, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "static", "ignoreInEval": True, "id": 4, "color": [0, 0, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "dynamic", "ignoreInEval": True, "id": 5, "color": [111, 74, 0], "trainId": 255},
    {"hasInstances": False, "category": "void", "catid": 0, "name": "ground", "ignoreInEval": True, "id": 6, "color": [81, 0, 81], "trainId": 255},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "road", "ignoreInEval": False, "id": 7, "color": [128, 64, 128], "trainId": 0},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "sidewalk", "ignoreInEval": False, "id": 8, "color": [244, 35, 232], "trainId": 1},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "parking", "ignoreInEval": True, "id": 9, "color": [250, 170, 160], "trainId": 255},
    {"hasInstances": False, "category": "flat", "catid": 1, "name": "rail track", "ignoreInEval": True, "id": 10, "color": [230, 150, 140], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "building", "ignoreInEval": False, "id": 11, "color": [70, 70, 70], "trainId": 2},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "wall", "ignoreInEval": False, "id": 12, "color": [102, 102, 156], "trainId": 3},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "fence", "ignoreInEval": False, "id": 13, "color": [190, 153, 153], "trainId": 4},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "guard rail", "ignoreInEval": True, "id": 14, "color": [180, 165, 180], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "bridge", "ignoreInEval": True, "id": 15, "color": [150, 100, 100], "trainId": 255},
    {"hasInstances": False, "category": "construction", "catid": 2, "name": "tunnel", "ignoreInEval": True, "id": 16, "color": [150, 120, 90], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "pole", "ignoreInEval": False, "id": 17, "color": [153, 153, 153], "trainId": 5},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "polegroup", "ignoreInEval": True, "id": 18, "color": [153, 153, 153], "trainId": 255},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic light", "ignoreInEval": False, "id": 19, "color": [250, 170, 30], "trainId": 6},
    {"hasInstances": False, "category": "object", "catid": 3, "name": "traffic sign", "ignoreInEval": False, "id": 20, "color": [220, 220, 0], "trainId": 7},
    {"hasInstances": False, "category": "nature", "catid": 4, "name": "vegetation", "ignoreInEval": False, "id": 21, "color": [107, 142, 35], "trainId": 8},
    {"hasInstances": False, "category": "nature", "catid": 4, "name": "terrain", "ignoreInEval": False, "id": 22, "color": [152, 251, 152], "trainId": 9},
    {"hasInstances": False, "category": "sky", "catid": 5, "name": "sky", "ignoreInEval": False, "id": 23, "color": [70, 130, 180], "trainId": 10},
    {"hasInstances": True, "category": "human", "catid": 6, "name": "person", "ignoreInEval": False, "id": 24, "color": [220, 20, 60], "trainId": 11},
    {"hasInstances": True, "category": "human", "catid": 6, "name": "rider", "ignoreInEval": False, "id": 25, "color": [255, 0, 0], "trainId": 12},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "car", "ignoreInEval": False, "id": 26, "color": [0, 0, 142], "trainId": 13},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "truck", "ignoreInEval": False, "id": 27, "color": [0, 0, 70], "trainId": 14},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bus", "ignoreInEval": False, "id": 28, "color": [0, 60, 100], "trainId": 15},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "caravan", "ignoreInEval": True, "id": 29, "color": [0, 0, 90], "trainId": 255},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "trailer", "ignoreInEval": True, "id": 30, "color": [0, 0, 110], "trainId": 255},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "train", "ignoreInEval": False, "id": 31, "color": [0, 80, 100], "trainId": 16},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "motorcycle", "ignoreInEval": False, "id": 32, "color": [0, 0, 230], "trainId": 17},
    {"hasInstances": True, "category": "vehicle", "catid": 7, "name": "bicycle", "ignoreInEval": False, "id": 33, "color": [119, 11, 32], "trainId": 18},
    {"hasInstances": False, "category": "vehicle", "catid": 7, "name": "license plate", "ignoreInEval": True, "id": -1, "color": [0, 0, 142], "trainId": -1}
]

# Pre-computed global constants
NUM_CLASSES = 19
IGNORE_INDEX = 255
NORM_CFG = {'mean': (123.675, 116.28, 103.53), 'std': (58.395, 57.12, 57.375)}
CLASSES = (
    'road', 'sidewalk', 'building', 'wall', 'fence', 
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 
    'sky', 'person', 'rider', 'car', 'truck', 
    'bus', 'train', 'motorcycle', 'bicycle'
)
PALETTE = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
]


class CityscapesDataset(BaseDataset):
    """Cityscapes 语义分割数据集"""
    def __init__(self, dataroot, annpath=None, trans_func=None, trans_func_strong=None, mode='train',
                 norm=NORM_CFG, return_img_name=False, rcs_cfg=None, crop_pseudo_margins=None, aug_mode=False, weather_aug_cfg=None):
        super().__init__(dataroot, annpath, trans_func, trans_func_strong, mode, norm, return_img_name, rcs_cfg, aug_mode, weather_aug_cfg)
        self.num_classes = NUM_CLASSES
        self.ignore_index = IGNORE_INDEX
        self.norm_cfg = norm
        self.CLASSES = CLASSES
        self.PALETTE = PALETTE
        self.pseudo_margins = crop_pseudo_margins
        self.valid_mask_size = [1024, 2048]
        self.get_valid_pseudo_mask()

    def get_valid_pseudo_mask(self):
        if self.pseudo_margins:
            assert len(self.pseudo_margins) == 4, "pseudo_margins 应为 [top, bottom, left, right]"
            top, bottom, left, right = self.pseudo_margins
            self.valid_pseudo_mask = np.zeros(self.valid_mask_size, dtype=np.uint8)
            self.valid_pseudo_mask[top:-bottom, left:-right] = 1
        else:
            self.valid_pseudo_mask = None
            
    def get_rcs_sample(self):
        """采样一个稀有类别样本，并保证 crop 后该类别像素足够，同时处理 pseudo mask"""
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        file_name = np.random.choice(self.samples_with_class[c])
        idx = self._get_index_by_file(file_name)
        # print(f"RCS Sample: Class {c}, File {file_name}, Index {idx}")
        if idx is None:
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
            if self.valid_pseudo_mask is not None:
                im_lb_weak['mask'] = self.valid_pseudo_mask.copy()
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
            if self.valid_pseudo_mask is not None:
                im_lb_strong['mask'] = self.valid_pseudo_mask.copy()
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
            if self.valid_pseudo_mask is not None:
                data['mask_weak'] = torch.from_numpy(im_lb_weak['mask'].astype(np.uint8))
                data['mask_strong'] = torch.from_numpy(im_lb_strong['mask'].astype(np.uint8))
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
            if self.valid_pseudo_mask is not None:
                im_lb['mask'] = self.valid_pseudo_mask.copy()
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
            data = {'im': im, 'lb': lb}
            if self.valid_pseudo_mask is not None:
                data['mask'] = torch.from_numpy(im_lb['mask'].astype(np.uint8))
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
        
        if getattr(self, "rcs_enabled", False):
            data = self.get_rcs_sample()
        else:
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
                if self.valid_pseudo_mask is not None:
                    im_lb_weak['mask'] = self.valid_pseudo_mask.copy()
                if self.trans_func is not None:
                    im_lb_weak = self.trans_func(im_lb_weak)
                
                # 提取crop_bbox和did_flip用于强增强的空间对齐
                crop_bbox = im_lb_weak.get('crop_bbox', None)
                did_flip = im_lb_weak.get('did_flip', False)
                
                im_weak = self._process_image(im_lb_weak['im'])
                lb_weak = torch.from_numpy(im_lb_weak['lb'].astype(np.int64))
                
                # 强增强 - 可能使用天气增强图像
                img_strong, label_strong = self.get_image(impth, lbpth, use_weather_for_strong=True)
                if self.lb_map is not None:
                    label_strong = self.lb_map[label_strong]
                
                im_lb_strong = dict(im=img_strong.copy(), lb=label_strong.copy())
                if self.specified_scale is not None:
                    im_lb_strong['specified_scale'] = self.specified_scale
                if self.valid_pseudo_mask is not None:
                    im_lb_strong['mask'] = self.valid_pseudo_mask.copy()
                if crop_bbox is not None:
                    im_lb_strong['crop_bbox'] = crop_bbox  # 使用相同的crop位置
                if did_flip is not None:
                    im_lb_strong['do_flip'] = did_flip  # 使用相同的flip决策
                    
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
                    'im_strong_did_flip': im_lb_strong['did_flip'],  # 记录是否进行了flip，用于蒸馏损失对齐
                }
                if self.valid_pseudo_mask is not None:
                    data['mask_weak'] = torch.from_numpy(im_lb_weak['mask'].astype(np.uint8))
                    data['mask_strong'] = torch.from_numpy(im_lb_strong['mask'].astype(np.uint8))
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
                
                # Apply transformations
                im_lb = dict(im=img, lb=label)
                if self.specified_scale is not None:
                    im_lb['specified_scale'] = self.specified_scale
                offline_mask = self._offline_teacher_transform_mask(
                    idx, self.valid_pseudo_mask)
                if offline_mask is not None:
                    im_lb['mask'] = offline_mask
                    
                if self.trans_func is not None:
                    im_lb = self.trans_func(im_lb)

                # Process image and label
                im = self._process_image(im_lb['im'])
                lb = torch.from_numpy(im_lb['lb'].astype(np.int64))  # 只需astype，无需copy/clone

                data = {'im': im, 'lb': lb}
                
                # Add pseudo mask if available
                if 'mask' in im_lb:
                    self._decode_offline_teacher_transform_mask(
                        data, im_lb['mask'])
                    
                if 'specified_scale' in im_lb:
                    data['specified_scale'] = im_lb['specified_scale']
            
            if self.return_img_name:
                data['im_name'] = os.path.basename(impth)
        return data
        
            # # Return image name if required
            # if self.return_img_name:
            #     im_name = os.path.basename(impth)
            #     return im, lb, im_name
            # return im, lb
        
class CityscapesDepDataset(BaseDepthDataset):
    """Cityscapes 带深度的语义分割数据集"""
    def __init__(self, dataroot, annpath=None, trans_func=None, mode='train',
                 norm=NORM_CFG, return_img_name=False):
        super().__init__(dataroot, annpath, trans_func, mode, norm, return_img_name)
        self.num_classes = NUM_CLASSES
        self.ignore_index = IGNORE_INDEX
        self.norm_cfg = norm
        self.CLASSES = CLASSES
        self.PALETTE = PALETTE



if __name__ == "__main__":
    # --- 优化4: 性能测试和基准代码 ---
    import time
    from tqdm import tqdm
    
    def benchmark_dataset(dataset_class, *args, **kwargs):
        """Benchmark dataset loading performance"""
        print(f"Benchmarking {dataset_class.__name__}...")
        
        # Create dummy data for testing if needed
        try:
            ds = dataset_class(*args, **kwargs)
        except FileNotFoundError:
            print("Dataset files not found, creating dummy dataset for testing...")
            # Create a minimal test dataset
            ds = dataset_class(
                dataroot="./dummy_data",
                annpath="./dummy_data/test.txt",
                mode='train'
            )
        
        # Test different DataLoader configurations
        configs = [
            # {'batch_size': 1, 'num_workers': 0, 'pin_memory': False},
            # {'batch_size': 4, 'num_workers': 0, 'pin_memory': False},
            {'batch_size': 2, 'num_workers': 2, 'pin_memory': True},
            # {'batch_size': 4, 'num_workers': 4, 'pin_memory': True},
        ]
        
        for config in configs:
            print(f"\nTesting config: {config}")
            
            # --- 优化5: DataLoader优化配置 ---
            dataloader = DataLoader(
                ds,
                shuffle=False,
                drop_last=True,
                persistent_workers=config['num_workers'] > 0,  # Keep workers alive
                prefetch_factor=2 if config['num_workers'] > 0 else 2,  # Prefetch batches
                **config
            )
            
            # Measure loading time
            start_time = time.time()
            n_batches = min(10, len(dataloader))  # Limit to 10 batches for quick test
            
            try:
                for i, batch in enumerate(tqdm(dataloader, desc="Loading", total=n_batches)):
                    if i >= n_batches:
                        break
                    # Simulate some processing
                    if isinstance(batch, (list, tuple)):
                        for item in batch:
                            if torch.is_tensor(item):
                                print(f"Batch item shape: {item.shape}")
                        
            except Exception as e:
                print(f"Error during loading: {e}")
                continue
                
            end_time = time.time()
            total_time = end_time - start_time
            time_per_batch = total_time / n_batches if n_batches > 0 else 0
            
            print(f"  Total time: {total_time:.4f}s")
            print(f"  Time per batch: {time_per_batch:.4f}s")
            print(f"  Throughput: {config['batch_size'] / time_per_batch:.2f} samples/s" if time_per_batch > 0 else "N/A")
    
    # Run benchmarks
    print("=== Cityscapes Dataset Optimization Benchmark ===")
    
    # Example usage - replace with actual data paths
    data_root = "datasets/cityscapes"
    ann_file = "datasets/cityscapes/train.txt"
    dep_ann_file = "datasets/cityscapes/train_depth.txt"
    rare_class_sampling = {
        'min_pixels': 3000, 'class_temp': 0.01, 'min_crop_ratio': 0.5
    }
    
    if os.path.exists(data_root) and os.path.exists(ann_file):
        benchmark_dataset(CityscapesDataset, data_root, ann_file, mode='train', rcs_cfg=rare_class_sampling)
        # print("\n" + "="*50)
        # benchmark_dataset(CityscapesDepDataset, data_root, dep_ann_file, mode='train')
    else:
        print(f"Dataset not found at {data_root}")
        print("Please update the paths or create dummy data for testing")
