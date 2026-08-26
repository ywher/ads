"""Utilities for attaching precomputed teacher pseudo labels to a dataset."""

import logging
import os
from pathlib import Path


def _relative_image_path(image_path, data_root):
    image_path = Path(image_path)
    data_root = Path(data_root)
    try:
        return image_path.resolve().relative_to(data_root.resolve())
    except ValueError:
        rel_path = os.path.relpath(str(image_path), str(data_root))
        if rel_path.startswith('..'):
            raise ValueError(
                f'Image path is outside data_root: image={image_path}, '
                f'data_root={data_root}')
        return Path(rel_path)


def attach_offline_teacher_pseudo_labels(dataset, cache_cfg):
    """Replace an unlabeled dataset's placeholder labels with teacher labels.

    The teacher labels are confidence-filtered before training and stored with
    the same relative paths as the target images. Reusing the regular label
    pipeline guarantees that resize, crop, and flip stay aligned with images.
    """
    if not cache_cfg or not cache_cfg.get('enabled', True):
        return dataset

    cache_root = Path(cache_cfg['root']).expanduser()
    if not cache_root.is_absolute():
        cache_root = (Path.cwd() / cache_root).resolve()
    label_subdir = cache_cfg.get('label_subdir', 'labelTrainIds')
    label_root = cache_root / label_subdir
    strict = bool(cache_cfg.get('strict', True))

    if not label_root.is_dir():
        raise FileNotFoundError(
            f'Offline teacher label directory not found: {label_root}. '
            'Run tools/acquisition/pseudo_labels/build_offline_teacher_pseudo.py first.')

    teacher_paths = []
    confidence_paths = []
    confidence_subdir = cache_cfg.get('confidence_subdir')
    confidence_root = (
        cache_root / confidence_subdir if confidence_subdir else None)
    missing = []
    for image_path in dataset.img_paths:
        rel_path = _relative_image_path(image_path, dataset.dataroot)
        teacher_path = label_root / rel_path.with_suffix('.png')
        teacher_paths.append(str(teacher_path))
        if strict and not teacher_path.is_file() and len(missing) < 8:
            missing.append(str(teacher_path))
        if confidence_root is not None:
            confidence_path = confidence_root / rel_path.with_suffix('.png')
            confidence_paths.append(str(confidence_path))
            if (strict and not confidence_path.is_file()
                    and len(missing) < 8):
                missing.append(str(confidence_path))

    if missing:
        examples = '\n'.join(f'  - {path}' for path in missing)
        raise FileNotFoundError(
            'Offline teacher cache does not cover the target split. '
            f'First missing labels:\n{examples}')

    dataset.lb_paths = teacher_paths
    if confidence_root is not None:
        dataset.offline_teacher_confidence_paths = confidence_paths
    dataset.offline_teacher_pseudo_cfg = dict(cache_cfg)
    logging.getLogger().info(
        '[OfflineTeacher] Attached %d pseudo labels from %s%s',
        len(teacher_paths),
        label_root,
        (f' with uint8 confidence from {confidence_root}'
         if confidence_root is not None else ''),
    )
    return dataset
