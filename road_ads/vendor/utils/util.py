import os
import json
import re
import numpy as np
from collections import abc

ACDC_EVAL_SCENES = ('fog', 'rain', 'snow', 'night')
OUR_EVAL_SCENES = ('fog', 'rain', 'snow', 'dawn-dusk', 'glare', 'night')
OUR_SCENE_ALIASES = {
    'dawn': 'dawn-dusk',
    'dusk': 'dawn-dusk',
    'dawn_dusk': 'dawn-dusk',
    'dawn-dusk': 'dawn-dusk',
    # Common typo kept as an alias so old manually written mapping files still work.
    'duan-dusk': 'dawn-dusk',
}


def csv_ious(ious, class_names, csv_file):
    # first row is header, miou, class names
    # second row is iou values, miou, iou of each class
    parent = os.path.dirname(csv_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(csv_file, 'w') as f:
        f.write('mIoU,')
        f.write(','.join(class_names))
        f.write('\n')
        # keep .4f
        mIoU = f'{ious.mean():.4f}'
        f.write(f'{mIoU},')
        ious = [f'{iou:.4f}' for iou in ious]
        f.write(','.join(ious))
        # f.write('\n')


def _format_float(value):
    if np.isnan(value):
        return 'nan'
    return f'{value:.4f}'


def csv_ious_rows(rows, class_names, csv_file, split_header='split'):
    """Save multiple IoU rows into one CSV.

    `rows` is an iterable of `(split_name, ious)` pairs. The first column is
    usually `all/fog/rain/snow/night` for ACDC validation.
    """
    parent = os.path.dirname(csv_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(csv_file, 'w') as f:
        f.write(f'{split_header},mIoU,')
        f.write(','.join(class_names))
        f.write('\n')
        for split_name, ious in rows:
            ious = np.asarray(ious, dtype=np.float64)
            mIoU = np.nanmean(ious)
            iou_values = [_format_float(iou) for iou in ious]
            f.write(f'{split_name},{_format_float(mIoU)},')
            f.write(','.join(iou_values))
            f.write('\n')


def csv_ious_with_splits(ious, class_names, csv_file, split_ious=None,
                         split_order=None):
    """Save overall IoU, optionally with dataset split-wise rows.

    Evaluations without `split_ious` keep the historical two-line CSV format.
    Split-aware evaluations get a left `split` column with an `all` row first,
    followed by the requested scene/weather/light rows.
    """
    if not split_ious:
        csv_ious(ious, class_names, csv_file)
        return

    rows = [('all', ious)]
    if split_order is None:
        split_order = ACDC_EVAL_SCENES
    for scene in split_order:
        if scene in split_ious:
            rows.append((scene, split_ious[scene]))
    csv_ious_rows(rows, class_names, csv_file)


def get_scene_from_path(path, scenes, aliases=None):
    """Parse a scene/weather/light split name from an image or label path."""
    norm_path = str(path).replace('\\', '/')
    parts = norm_path.split('/')
    for part in parts:
        normalized = aliases.get(part, part) if aliases else part
        if normalized in scenes:
            return normalized
    basename = os.path.basename(norm_path).lower()
    candidate_names = set(scenes)
    if aliases:
        candidate_names.update(aliases.keys())
    for scene in candidate_names:
        if re.search(rf'(^|[_\-.]){re.escape(scene)}($|[_\-.])', basename):
            return aliases.get(scene, scene) if aliases else scene
    return None


def get_acdc_scene_from_path(path):
    """Parse ACDC condition name from an image or label path."""
    return get_scene_from_path(path, ACDC_EVAL_SCENES)


def get_our_scene_from_path(path):
    """Parse our adverse weather/light split from an image or label path."""
    return get_scene_from_path(path, OUR_EVAL_SCENES, aliases=OUR_SCENE_ALIASES)


def _infer_acdc_root(path):
    norm_path = os.path.abspath(str(path)).replace('\\', '/')
    parts = norm_path.split('/')
    for marker in ('rgb_anon', 'gt'):
        if marker in parts:
            return '/'.join(parts[:parts.index(marker)])
    if 'acdc' in parts:
        return '/'.join(parts[:parts.index('acdc') + 1])
    return None


def _load_acdc_scene_lookup_file(data_root, split='val'):
    """Load optional filename -> scene mapping from val_type json/txt files."""
    candidate_files = (
        os.path.join(data_root, f'{split}_type.json'),
        os.path.join(data_root, f'{split}_type.txt'),
        os.path.join(data_root, 'val_type.json'),
        os.path.join(data_root, 'val_type.txt'),
    )
    for candidate_file in candidate_files:
        if not os.path.isfile(candidate_file):
            continue

        scene_by_filename = {}
        if candidate_file.endswith('.json'):
            with open(candidate_file, 'r') as f:
                mapping = json.load(f)
            if isinstance(mapping, dict):
                iterable = mapping.items()
            elif isinstance(mapping, list):
                iterable = []
                for item in mapping:
                    if not isinstance(item, dict):
                        continue
                    filename = item.get('filename') or item.get('file') or item.get('image')
                    scene = item.get('scene') or item.get('type') or item.get('condition')
                    iterable.append((filename, scene))
            else:
                iterable = []
            for filename, scene in iterable:
                if scene in ACDC_EVAL_SCENES:
                    scene_by_filename[os.path.basename(str(filename))] = scene
        else:
            with open(candidate_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if ',' in line:
                        filename, scene = [part.strip() for part in line.split(',', 1)]
                    else:
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        filename, scene = parts[0], parts[1]
                    if scene in ACDC_EVAL_SCENES:
                        scene_by_filename[os.path.basename(filename)] = scene

        if scene_by_filename:
            return scene_by_filename
    return None


def _build_acdc_scene_lookup_from_rgb_anon(data_root, split='val'):
    scene_by_filename = {}
    for scene in ACDC_EVAL_SCENES:
        candidate_root = os.path.join(data_root, 'rgb_anon', scene, split)
        if not os.path.isdir(candidate_root):
            continue
        for root, _, files in os.walk(candidate_root):
            for filename in files:
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    scene_by_filename.setdefault(filename, scene)
    return scene_by_filename


def _build_acdc_scene_lookup(data_root, split='val'):
    scene_by_filename = _load_acdc_scene_lookup_file(data_root, split=split)
    if scene_by_filename is not None:
        return scene_by_filename
    return _build_acdc_scene_lookup_from_rgb_anon(data_root, split=split)


def get_acdc_eval_scenes(dataset):
    """Return per-sample ACDC scenes for validation datasets, or None."""
    dataset_name = dataset.__class__.__name__
    if dataset_name not in ('ACDCDataset', 'ACDCDepDataset'):
        return None

    img_paths = getattr(dataset, 'img_paths', None)
    if img_paths is None:
        return None

    scenes = [get_acdc_scene_from_path(path) for path in img_paths]
    if not any(scene in ACDC_EVAL_SCENES for scene in scenes):
        data_root = _infer_acdc_root(img_paths[0]) if img_paths else None
        if data_root is None:
            return None
        split = getattr(dataset, 'mode', 'val')
        scene_by_filename = _build_acdc_scene_lookup(data_root, split=split)
        scenes = [
            scene_by_filename.get(os.path.basename(str(path)))
            for path in img_paths
        ]
        if not any(scene in ACDC_EVAL_SCENES for scene in scenes):
            return None
    return scenes


def get_our_eval_scenes(dataset):
    """Return per-sample our dataset weather/light splits, or None."""
    dataset_name = dataset.__class__.__name__
    if dataset_name not in ('OurDataset', 'OurDepDataset'):
        return None

    img_paths = getattr(dataset, 'img_paths', None)
    if img_paths is None:
        return None

    scenes = [get_our_scene_from_path(path) for path in img_paths]
    if not any(scene in OUR_EVAL_SCENES for scene in scenes):
        return None
    return scenes


def get_dataset_eval_splits(dataset):
    """Return `(split_order, per_sample_splits, display_name)` for known datasets."""
    acdc_scenes = get_acdc_eval_scenes(dataset)
    if acdc_scenes is not None:
        return ACDC_EVAL_SCENES, acdc_scenes, 'ACDC'

    our_scenes = get_our_eval_scenes(dataset)
    if our_scenes is not None:
        return OUR_EVAL_SCENES, our_scenes, 'Our'

    return None, None, None


def compute_ious_from_hist(hist):
    denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
    return np.diag(hist) / np.maximum(denominator, 1) * 100


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)

def log_model_params(model, logger=None):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # 转换为M单位
    total_params_m = total_params / 1e6
    trainable_params_m = trainable_params / 1e6
    non_trainable_params_m = non_trainable_params / 1e6
    
    logger.info(f'total params: {total_params} ({total_params_m:.2f}M), trainable params: {trainable_params} ({trainable_params_m:.2f}M)')
    logger.info(f'non-trainable params: {non_trainable_params} ({non_trainable_params_m:.2f}M)')
    logger.info(f'trainable params ratio: {(trainable_params / total_params) * 100:.4f}%')
    
def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    # 转换为M单位
    total_params_m = total_params / 1e6
    trainable_params_m = trainable_params / 1e6
    non_trainable_params_m = non_trainable_params / 1e6
    
    print(f'total params: {total_params} ({total_params_m:.2f}M), trainable params: {trainable_params} ({trainable_params_m:.2f}M)')
    print(f'non-trainable params: {non_trainable_params} ({non_trainable_params_m:.2f}M)')
    print(f'trainable params ratio: {(trainable_params / total_params) * 100:.4f}%')

# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self, length=0):
#         self.length = length
#         self.reset()

#     def reset(self):
#         if self.length > 0:
#             self.history = []
#         else:
#             self.count = 0
#             self.sum = 0.0
#         self.val = 0.0
#         self.avg = 0.0

#     def update(self, val, num=1):
#         if self.length > 0:
#             # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
#             assert num == 1
#             self.history.append(val)
#             if len(self.history) > self.length:
#                 del self.history[0]

#             self.val = self.history[-1]
#             self.avg = np.mean(self.history)
#         else:
#             self.val = val
#             self.sum += val * num
#             self.count += num
#             self.avg = self.sum / self.count
