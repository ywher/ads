#!/usr/bin/python
# -*- encoding: utf-8 -*-

import json
import logging
import os
import os.path as osp

import cv2
import numpy as np


RCS_STATS_FILES = (
    'sample_class_stats.json',
    'sample_class_stats_dict.json',
    'samples_with_class.json',
)


def parse_ann_line(line):
    """Parse one annotation line as either `image,label` or `image label`."""
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    if ',' in line:
        parts = [item.strip() for item in line.split(',', 1)]
    else:
        parts = line.split()

    if len(parts) < 2:
        raise ValueError(f'Invalid annotation line: {line!r}')
    return parts[0], parts[1]


def load_ann_pairs(annpath):
    pairs = []
    with open(annpath, 'r') as fr:
        for line_no, line in enumerate(fr, start=1):
            parsed = parse_ann_line(line)
            if parsed is None:
                continue
            pairs.append(parsed)
    if not pairs:
        raise ValueError(f'No image/label pairs found in annotation file: {annpath}')
    return pairs


def resolve_data_path(data_root, path):
    if osp.isabs(path):
        return path
    return osp.join(data_root, path)


def resolve_rcs_stats_dir(rcs_cfg, annpath):
    stats_dir = None
    if rcs_cfg:
        stats_dir = rcs_cfg.get('stats_dir', rcs_cfg.get('stats_root', None))
    if stats_dir is None:
        stats_dir = osp.dirname(osp.abspath(annpath))
    return stats_dir


def rcs_stats_exist(stats_dir):
    return all(osp.exists(osp.join(stats_dir, name)) for name in RCS_STATS_FILES)


def get_path_match_keys(path, data_root=None):
    """Return stable keys used to map RCS stats entries back to dataset indices."""
    keys = set()
    if not path:
        return keys

    norm_path = osp.normpath(path)
    keys.add(norm_path)
    keys.add(osp.basename(norm_path))
    keys.add(norm_path.replace(os.sep, '/'))

    abs_path = norm_path if osp.isabs(norm_path) else (
        osp.abspath(osp.join(data_root, norm_path)) if data_root else osp.abspath(norm_path)
    )
    keys.add(abs_path)
    keys.add(abs_path.replace(os.sep, '/'))

    if data_root:
        try:
            rel_path = osp.relpath(abs_path, osp.abspath(data_root))
            keys.add(rel_path)
            keys.add(rel_path.replace(os.sep, '/'))
        except ValueError:
            pass

    return keys


def compute_label_stats(label_path, file_key, num_classes, ignore_index=255,
                        label_map=None):
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(f'Could not read label image: {label_path}')
    if label_map is not None:
        label = np.asarray(label_map, dtype=np.uint8)[label]

    stats = {'file': file_key}
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue
        pixels = int(np.sum(label == class_id))
        if pixels > 0:
            stats[int(class_id)] = pixels
    return stats


def save_rcs_class_stats(out_dir, sample_class_stats):
    os.makedirs(out_dir, exist_ok=True)
    valid_stats = [stats for stats in sample_class_stats if len(stats) > 1]
    valid_stats = sorted(valid_stats, key=lambda item: item['file'])

    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as f:
        json.dump(valid_stats, f, indent=2)

    sample_class_stats_dict = {}
    for stats in valid_stats:
        stats_copy = stats.copy()
        filename = stats_copy.pop('file')
        sample_class_stats_dict[filename] = {
            str(k): int(v)
            for k, v in sorted(stats_copy.items(), key=lambda item: int(item[0]))
        }

    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as f:
        json.dump(sample_class_stats_dict, f, indent=2)

    samples_with_class = {}
    for filename, class_stats in sample_class_stats_dict.items():
        for class_id, pixel_count in class_stats.items():
            samples_with_class.setdefault(str(class_id), []).append(
                [filename, int(pixel_count)])

    samples_with_class = {
        str(class_id): sorted(samples, key=lambda item: item[0])
        for class_id, samples in sorted(
            samples_with_class.items(), key=lambda item: int(item[0]))
    }
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as f:
        json.dump(samples_with_class, f, indent=2)


def ensure_rcs_class_stats(data_root, annpath, rcs_cfg=None, num_classes=19,
                           ignore_index=255):
    """Ensure split-local RCS JSON files exist and return their directory."""
    stats_dir = resolve_rcs_stats_dir(rcs_cfg, annpath)
    if rcs_stats_exist(stats_dir):
        return stats_dir

    online_generate = True if rcs_cfg is None else rcs_cfg.get('online_generate', True)
    if not online_generate:
        missing = [
            name for name in RCS_STATS_FILES
            if not osp.exists(osp.join(stats_dir, name))
        ]
        raise FileNotFoundError(
            f'RCS stats missing in {stats_dir}: {missing}. '
            'Set rare_class_sampling.online_generate=True to generate them.')

    logger = logging.getLogger()
    logger.info('Generating split-local RCS stats: ann=%s, out=%s',
                annpath, stats_dir)

    label_map = None if rcs_cfg is None else rcs_cfg.get('label_map', None)
    sample_class_stats = []
    for _, label_rel in load_ann_pairs(annpath):
        label_path = resolve_data_path(data_root, label_rel)
        try:
            sample_class_stats.append(
                compute_label_stats(
                    label_path,
                    label_rel,
                    num_classes,
                    ignore_index,
                    label_map=label_map,
                ))
        except FileNotFoundError as exc:
            logger.warning(str(exc))

    if not sample_class_stats:
        raise RuntimeError(
            f'Failed to generate RCS stats for {annpath}; no readable labels found.')

    save_rcs_class_stats(stats_dir, sample_class_stats)
    return stats_dir
