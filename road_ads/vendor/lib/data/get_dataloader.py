import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .transform_cv2 import (
    TransformationTrain,
    TransformationVal,
    TransformationTest,
)
from .sampler import RepeatedDistSampler, Sampler
from .offline_teacher_pseudo import attach_offline_teacher_pseudo_labels

from lib.data import *


def _require_keys(cfg, keys, context):
    """Validate that a config dictionary contains required keys.

    校验配置字典是否包含必需字段，缺失时给出更清晰的报错信息。
    """
    missing = [key for key in keys if key not in cfg]
    if missing:
        raise KeyError(f"Missing required key(s) in {context}: {missing}")


class RepeatedSampler(Sampler):
    """
    A sampler that repeats the dataset for a specified number of iterations.
    """
    def __init__(self, data_source, max_iter):
        self.data_source = data_source
        self.max_iter = max_iter
        self.num_samples = len(data_source)
        self.current_iter = 0

    def __iter__(self):
        while self.current_iter < self.max_iter:
            for idx in range(self.num_samples):
                yield idx
            self.current_iter += 1

    def __len__(self):
        return self.max_iter * self.num_samples


def get_transformation(cfg, mode):
    """
    Get the transformation function based on the mode (train/val/test).

    根据模式（train/val/test）构建对应的数据增强流水线。
    """
    _require_keys(
        cfg,
        ['resize', 'cropsize', 'cat_max_ratio', 'flip', 'photo_metric', 'rotate'],
        f'{mode} transformation config',
    )
    if mode == 'train':
        return TransformationTrain(
            resize_shape=cfg['resize'],
            keep_ratio=cfg.get('keep_ratio', False),
            scale=cfg.get('scale', None),  # Optional scale for training
            cropsize=cfg['cropsize'],
            cat_max_ratio=cfg['cat_max_ratio'],
            flip=cfg['flip'],
            photo_metric=cfg['photo_metric'],
            rotate=cfg['rotate']
        )
    elif mode == 'val':
        return TransformationVal(
            resize_shape=cfg['resize'],
            keep_ratio=cfg.get('keep_ratio', False),
            cropsize=cfg['cropsize'],
            cat_max_ratio=cfg['cat_max_ratio'],
            flip=cfg['flip'],
            photo_metric=cfg['photo_metric'],
            rotate=cfg['rotate']
        )
    elif mode == 'test':
        return TransformationTest(
            resize_shape=cfg['resize'],
            keep_ratio=cfg.get('keep_ratio', False),
            cropsize=cfg['cropsize'],
            cat_max_ratio=cfg['cat_max_ratio'],
            flip=cfg['flip'],
            photo_metric=cfg['photo_metric'],
            rotate=cfg['rotate']
        )
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'train', 'val', or 'test'.")


def get_dataset(cfg, mode='train', save_pred=False):
    """
    Create and return the dataset based on the configuration and mode.

    根据配置和模式创建数据集对象。

    The returned dataset uses dict samples in the base segmentation path:
    `im` for image tensor, `lb` for semantic label, plus optional fields such
    as `im_name`, `dep`, `mask`, or weak/strong augmentation outputs depending
    on the concrete dataset and config.

    基础语义分割路径返回字典样本：`im` 表示图像张量，`lb` 表示语义标签；
    根据具体数据集和配置，还可能包含 `im_name`、`dep`、`mask` 或强弱增强输出。
    """
    _require_keys(
        cfg,
        ['dataset', 'data_root', 'im_anns', 'rgb_mean', 'rgb_std'],
        f'{mode} dataset config',
    )
    dataset_name = cfg['dataset']
    data_root = cfg['data_root']
    annpath = cfg['im_anns']
    norm = {'mean': cfg['rgb_mean'], 'std': cfg['rgb_std']}
    rare_class_sampling = cfg.get('rare_class_sampling', None)
    weather_aug_cfg = cfg.get('weather_aug', None)  # 天气增强配置
    
    # 检查是否需要强增强（aug_mode）
    aug_mode = cfg.get('aug_mode', False)
    trans_func_strong = None
    
    if aug_mode and mode == 'train':
        # aug_mode: 弱增强只包含空间变换，不包含PhotoMetricDistortion
        # 创建修改后的cfg，关闭photo_metric
        cfg_weak = cfg.copy()
        cfg_weak['photo_metric'] = False  # 弱增强不使用PhotoMetricDistortion
        trans_func = get_transformation(cfg_weak, mode)
    else:
        # 标准模式：使用原始cfg
        trans_func = get_transformation(cfg, mode)
    
    if aug_mode and mode == 'train':
        # 为强增强创建单独的transformation，包含更多增强操作
        from .transform_cv2 import (
            Compose, RandomHorizontalFlip, Resize, RandomResize, RandomCrop, 
            PhotoMetricDistortion, ColorJitter, RandomGrayscale, GaussianBlur
        )
        
        # 构建强增强pipeline
        transforms_strong = []
        
        # 1. Resize (与弱增强相同)
        if cfg.get('keep_ratio', False):
            transforms_strong.append(Resize(resize_shape=cfg['resize'], keep_ratio=True))
        else:
            if cfg.get('scale', None):
                transforms_strong.append(RandomResize(scales=cfg['scale'], return_scale=True))
            else:
                transforms_strong.append(Resize(resize_shape=cfg['resize'], keep_ratio=False))
        
        # 2. 强颜色增强 - ColorJitter (相比弱增强更强)
        transforms_strong.append(ColorJitter(
            brightness=0.4,  # 增强亮度变化
            contrast=0.4,    # 增强对比度变化
            saturation=0.4,  # 增强饱和度变化
            hue=0.1,         # 色调抖动，改变颜色种类
            p=0.5            # 50%概率应用ColorJitter
        ))
        
        # 3. RandomGrayscale - 随机转灰度
        transforms_strong.append(RandomGrayscale(p=0.3))
        
        # 4. GaussianBlur - 高斯模糊
        transforms_strong.append(GaussianBlur(
            kernel_size=(3, 3),
            sigma=(1.0, 2.0),
            p=0.8
        ))
        
        # 5. PhotoMetricDistortion - 光度变换
        if cfg.get('photo_metric', False):
            transforms_strong.append(PhotoMetricDistortion())
        
        # 6. RandomCrop (与弱增强相同)
        if cfg.get('cropsize', None):
            transforms_strong.append(RandomCrop(
                crop_size=cfg['cropsize'],
                cat_max_ratio=cfg.get('cat_max_ratio', 0.75)
            ))
        
        # 7. RandomHorizontalFlip (与弱增强相同)
        if cfg.get('flip', 0) > 0:
            transforms_strong.append(RandomHorizontalFlip(p=cfg['flip']))
        
        trans_func_strong = Compose(transforms_strong)

    # Dynamically evaluate the dataset class
    dataset_class = eval(dataset_name)
    if 'crop_pseudo_margins' in cfg:
        # If crop_pseudo_margins is specified, pass it to the dataset class
        dataset = dataset_class(data_root, annpath, trans_func=trans_func, trans_func_strong=trans_func_strong,
                                mode=mode, norm=norm, return_img_name=save_pred, rcs_cfg=rare_class_sampling,
                                crop_pseudo_margins=cfg['crop_pseudo_margins'], aug_mode=aug_mode, weather_aug_cfg=weather_aug_cfg)
    else:
        # Otherwise, return the dataset without crop_pseudo_margins
        dataset = dataset_class(data_root, annpath, trans_func=trans_func, trans_func_strong=trans_func_strong,
                                mode=mode, norm=norm, return_img_name=save_pred, rcs_cfg=rare_class_sampling,
                                aug_mode=aug_mode, weather_aug_cfg=weather_aug_cfg)

    return attach_offline_teacher_pseudo_labels(
        dataset,
        cfg.get('offline_teacher_pseudo'),
    )


def get_data_loader(cfg, mode='train', save_pred=False, distributed=None):
    """
    Create and return the DataLoader based on the configuration and mode.

    根据配置和模式创建 DataLoader。
    """
    required_loader_keys = ['num_works']
    if mode == 'train':
        required_loader_keys.append('ims_per_gpu')
    _require_keys(cfg, required_loader_keys, f'{mode} dataloader config')

    dataset = get_dataset(cfg, mode, save_pred)
    batch_size = cfg['ims_per_gpu'] if mode == 'train' else 1
    num_workers = cfg['num_works']
    shuffle = mode == 'train'
    drop_last = mode == 'train'

    if distributed is None:
        distributed = dist.is_available() and dist.is_initialized()

    # Distributed training support
    if distributed:
        assert dist.is_available(), "Distributed training is not available."
        if mode == 'train':
            assert cfg.get('max_iters') is not None, "max_iters must be specified for training."
            n_train_imgs = cfg['ims_per_gpu'] * dist.get_world_size() * cfg['max_iters']
            sampler = RepeatedDistSampler(dataset, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=drop_last)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    # Non-distributed DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
