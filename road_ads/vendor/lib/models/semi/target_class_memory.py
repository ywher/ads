"""Target-labeled class memory for target-target mixing.

This helper stores compact class masks from target-labeled crops and pastes
high-need target classes into target-unlabeled images. It is deliberately
target-only, so it can be used in SSDA and also degenerates naturally to a
semi-supervised setting without source data.
"""
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F


SKY_CLASS_ID = 10
ROAD_SUPPORT_CLASSES = {0, 1, 9}
LOWER_OBJECT_CLASSES = {11, 12, 13, 14, 15, 16, 17, 18}
SIGN_LIGHT_CLASSES = {6, 7}


@dataclass
class TargetClassMemoryMixResult:
    images: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor
    mix_masks: list
    num_class_choice: list
    mix_ratio: float
    num_replaced: int
    memory_hits: int
    memory_attempts: int
    selected_classes: list
    selected_class_counts: dict
    selected_class_pixels: dict
    gate_candidate_count: int = 0
    gate_skipped_class: int = 0
    gate_skipped_conf: int = 0

    @property
    def log_vars(self):
        return {
            'target_class_memory_replaced': float(self.num_replaced),
            'target_class_memory_hit_ratio': (
                float(self.memory_hits) / max(1, self.memory_attempts)),
            'target_class_memory_mix_ratio': float(self.mix_ratio),
            'target_class_memory_selected_count': float(
                len(self.selected_classes)),
            'target_class_memory_hit_count': float(self.memory_hits),
            'target_class_memory_attempt_count': float(self.memory_attempts),
            'target_class_memory_gate_candidate_count': float(
                self.gate_candidate_count),
            'target_class_memory_gate_skipped_class': float(
                self.gate_skipped_class),
            'target_class_memory_gate_skipped_conf': float(
                self.gate_skipped_conf),
        }


def _squeeze_map(value):
    if value.dim() == 4 and value.shape[1] == 1:
        return value.squeeze(1)
    return value


def _valid_classes(label, num_classes, ignore_index):
    values = torch.unique(label.detach().long())
    values = values[values != int(ignore_index)]
    return values[(values >= 0) & (values < int(num_classes))]


def _bbox_from_mask(mask):
    ys, xs = torch.where(mask)
    if ys.numel() == 0:
        return None
    return (
        int(ys.min().item()),
        int(ys.max().item()) + 1,
        int(xs.min().item()),
        int(xs.max().item()) + 1,
    )


class TargetClassMemoryBank:
    """FIFO class-wise target-labeled mask memory."""

    def __init__(self, capacity_per_class=4, sample_strategy='latest'):
        self.capacity_per_class = int(capacity_per_class)
        self.sample_strategy = str(sample_strategy).lower()
        self.entries = defaultdict(list)
        self._insert_order = 0
        self.offline_loaded_entries = 0

    def __len__(self):
        return sum(len(items) for items in self.entries.values())

    def available_classes(self):
        return [
            class_id
            for class_id, items in self.entries.items()
            if items
        ]

    def update_from_labeled_batch(
        self,
        images,
        labels,
        num_classes,
        min_pixels=32,
        max_area_ratio=1.0,
        ignore_index=255,
    ):
        labels = _squeeze_map(labels).long()
        _, _, height, width = images.shape
        image_area = float(height * width)
        max_area = image_area * float(max_area_ratio)

        for batch_idx in range(labels.shape[0]):
            label = labels[batch_idx]
            for class_id_tensor in _valid_classes(label, num_classes, ignore_index):
                class_id = int(class_id_tensor.item())
                mask = label.eq(class_id)
                num_pixels = int(mask.sum().item())
                if num_pixels < int(min_pixels):
                    continue
                bbox = _bbox_from_mask(mask)
                if bbox is None:
                    continue
                y0, y1, x0, x1 = bbox
                if float((y1 - y0) * (x1 - x0)) > max_area:
                    continue
                mask_crop = mask[y0:y1, x0:x1].detach().cpu().clone()
                entry = {
                    'class_id': class_id,
                    'img': images[batch_idx, :, y0:y1, x0:x1].detach().cpu().clone(),
                    'mask': mask_crop,
                    'label': torch.full_like(mask_crop.long(), class_id),
                    'bbox': (y0, y1, x0, x1),
                    'pixels': num_pixels,
                    'insert_order': self._insert_order,
                    'offline': False,
                    'quality': float(num_pixels),
                }
                self._insert_order += 1
                class_entries = self.entries[class_id]
                class_entries.append(entry)
                if len(class_entries) > self.capacity_per_class:
                    class_entries.pop(0)

    def push_offline_entry(self, entry):
        class_id = int(entry['class_id'])
        img = entry['img'].detach().cpu().clone().float()
        mask = entry['mask'].detach().cpu().clone().bool()
        label = entry.get('label', None)
        if label is None:
            label = torch.full(mask.shape, class_id, dtype=torch.long)
        else:
            label = label.detach().cpu().clone().long()
        stored = {
            'class_id': class_id,
            'img': img,
            'mask': mask,
            'label': label,
            'bbox': entry.get('bbox', (0, mask.shape[0], 0, mask.shape[1])),
            'pixels': int(entry.get('pixels', int(mask.sum().item()))),
            'insert_order': self._insert_order,
            'offline': True,
            'quality': float(entry.get('quality', 0.0)),
            'y_center': float(entry.get('y_center', 0.5)),
            'x_center': float(entry.get('x_center', 0.5)),
            'image_height': int(entry.get('image_height', mask.shape[0])),
            'image_width': int(entry.get('image_width', mask.shape[1])),
            'entry_id': entry.get('entry_id', ''),
            'object_complete': bool(entry.get('object_complete', False)),
        }
        self._insert_order += 1
        class_entries = self.entries[class_id]
        class_entries.append(stored)
        if len(class_entries) > self.capacity_per_class:
            class_entries.pop(0)
        self.offline_loaded_entries += 1

    @staticmethod
    def _read_offline_crop(bank_root, rel_path, mode='image'):
        path = bank_root / str(rel_path)
        if mode == 'image':
            value = cv2.imread(str(path), cv2.IMREAD_COLOR)
        else:
            value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if value is None:
            raise FileNotFoundError(f'Could not read offline target memory crop: {path}')
        if mode != 'image' and value.ndim == 3:
            value = value[..., 0]
        return value

    @staticmethod
    def _normalize_offline_image(image_bgr, rgb_mean, rgb_std):
        image_rgb = image_bgr[:, :, ::-1].astype(np.float32)
        mean = np.asarray(rgb_mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(rgb_std, dtype=np.float32).reshape(1, 1, 3)
        image_rgb = (image_rgb - mean) / std
        image_rgb = np.ascontiguousarray(image_rgb.transpose(2, 0, 1))
        return torch.from_numpy(image_rgb).float()

    def load_offline_jsonl(
        self,
        metadata_path,
        rgb_mean,
        rgb_std,
        allowed_classes=None,
        min_pixels=1,
        max_area_ratio=1.0,
        require_object_complete=False,
        strict=True,
    ):
        metadata_path = Path(metadata_path)
        bank_root = metadata_path.parent
        if not metadata_path.exists():
            if strict:
                raise FileNotFoundError(f'Target class offline bank not found: {metadata_path}')
            return 0
        allowed = (
            {int(class_id) for class_id in allowed_classes}
            if allowed_classes else None)
        grouped = defaultdict(list)
        for line in metadata_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                bool(require_object_complete)
                and not bool(row.get('object_complete', False))
            ):
                continue
            class_id = int(row['class_id'])
            if allowed is not None and class_id not in allowed:
                continue
            pixels = int(row.get('pixels', 0))
            if pixels < int(min_pixels):
                continue
            area_ratio = float(row.get('area_ratio', 0.0))
            if area_ratio > float(max_area_ratio):
                continue
            image = self._read_offline_crop(bank_root, row['image_crop'], 'image')
            mask = self._read_offline_crop(bank_root, row['mask_crop'], 'mask') > 0
            label = self._read_offline_crop(bank_root, row['label_crop'], 'label').astype(np.int64)
            if image.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if label.shape[:2] != image.shape[:2]:
                label = cv2.resize(
                    label,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            if int(mask.sum()) < int(min_pixels):
                continue
            grouped[class_id].append({
                'class_id': class_id,
                'img': self._normalize_offline_image(image, rgb_mean, rgb_std),
                'mask': torch.from_numpy(mask.copy()).bool(),
                'label': torch.from_numpy(label.copy()).long(),
                'bbox': (
                    int(row.get('bbox_y0', 0)),
                    int(row.get('bbox_y1', image.shape[0])),
                    int(row.get('bbox_x0', 0)),
                    int(row.get('bbox_x1', image.shape[1])),
                ),
                'pixels': int(mask.sum()),
                'quality': float(row.get('quality', pixels)),
                'y_center': float(row.get('y_center', 0.5)),
                'x_center': float(row.get('x_center', 0.5)),
                'image_height': int(row.get('image_height', image.shape[0])),
                'image_width': int(row.get('image_width', image.shape[1])),
                'entry_id': row.get('entry_id', ''),
                'object_complete': bool(row.get('object_complete', False)),
            })

        loaded = 0
        for class_id, entries in grouped.items():
            entries.sort(key=lambda item: float(item.get('quality', 0.0)), reverse=True)
            for entry in reversed(entries[:self.capacity_per_class]):
                self.push_offline_entry(entry)
                loaded += 1
        return loaded

    def sample(self, class_id, strategy=None):
        class_entries = self.entries.get(int(class_id), [])
        if not class_entries:
            return None
        strategy = str(strategy or self.sample_strategy).lower()
        if strategy == 'random':
            return random.choice(class_entries)
        if strategy in ('quality', 'best'):
            return max(class_entries, key=lambda item: float(item.get('quality', 0.0)))
        return class_entries[-1]


def _resize_donor_if_needed(donor, target_shape, max_area_ratio):
    target_h, target_w = target_shape
    img = donor['img']
    mask = donor['mask']
    label = donor.get(
        'label',
        torch.full(mask.shape, int(donor['class_id']), dtype=torch.long),
    )
    crop_h, crop_w = int(mask.shape[0]), int(mask.shape[1])
    max_area = float(target_h * target_w) * float(max_area_ratio)
    area = float(crop_h * crop_w)
    if area <= max_area and crop_h <= target_h and crop_w <= target_w:
        return img, mask, label
    scale = (max_area / max(1.0, area)) ** 0.5
    scale = min(scale, target_h / max(1, crop_h), target_w / max(1, crop_w))
    new_h = max(1, int(round(crop_h * scale)))
    new_w = max(1, int(round(crop_w * scale)))
    img = F.interpolate(
        img.unsqueeze(0),
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False,
    ).squeeze(0)
    mask = F.interpolate(
        mask.float().unsqueeze(0).unsqueeze(0),
        size=(new_h, new_w),
        mode='nearest',
    ).squeeze(0).squeeze(0).bool()
    label = F.interpolate(
        label.float().unsqueeze(0).unsqueeze(0),
        size=(new_h, new_w),
        mode='nearest',
    ).squeeze(0).squeeze(0).long()
    return img, mask, label


def _label_hist(label, bbox, num_classes, ignore_index):
    y0, y1, x0, x1 = bbox
    patch = label[y0:y1, x0:x1]
    valid = (
        (patch >= 0)
        & (patch < int(num_classes))
        & (patch != int(ignore_index))
    )
    hist = torch.zeros(int(num_classes), device=label.device, dtype=torch.float32)
    if not bool(valid.any().item()):
        return hist
    values, counts = torch.unique(patch[valid].long(), return_counts=True)
    hist[values] = counts.float()
    return hist / hist.sum().clamp_min(1.0)


def _context_score(label, class_id, bbox, donor_y_center, num_classes, ignore_index):
    y0, y1, x0, x1 = bbox
    height = max(1, int(label.shape[0]))
    y_center = (float(y0 + y1) * 0.5) / float(height)
    hist = _label_hist(label, bbox, num_classes, ignore_index)
    sky = float(hist[SKY_CLASS_ID].detach().item()) if SKY_CLASS_ID < hist.numel() else 0.0
    road_support = sum(
        float(hist[c].detach().item())
        for c in ROAD_SUPPORT_CLASSES
        if c < hist.numel()
    )
    y_score = max(0.0, 1.0 - abs(y_center - float(donor_y_center)) / 0.35)
    score = 0.35 + 0.65 * y_score
    class_id = int(class_id)
    if class_id in SIGN_LIGHT_CLASSES:
        upper_prior = max(0.0, 1.0 - abs(y_center - 0.38) / 0.34)
        score += 0.7 * upper_prior - 0.5 * max(0.0, y_center - 0.62)
    elif class_id in LOWER_OBJECT_CLASSES:
        score += 0.9 * road_support - 1.2 * sky
    else:
        score -= 0.5 * sky
    return float(score)


def _candidate_bboxes(donor, crop_shape, target_shape, context_candidates, context_y_jitter):
    crop_h, crop_w = crop_shape
    target_h, target_w = target_shape
    if crop_h > target_h or crop_w > target_w:
        return []
    y_center = float(donor.get('y_center', 0.5))
    x_center = float(donor.get('x_center', 0.5))
    y_offsets = [0.0, -float(context_y_jitter), float(context_y_jitter)]
    x_centers = [x_center, 0.25, 0.5, 0.75]
    bboxes = []
    for dy in y_offsets:
        cy = min(1.0, max(0.0, y_center + dy))
        y0 = int(round(cy * target_h - crop_h * 0.5))
        y0 = max(0, min(target_h - crop_h, y0))
        for cx in x_centers:
            x0 = int(round(float(cx) * target_w - crop_w * 0.5))
            x0 = max(0, min(target_w - crop_w, x0))
            bbox = (y0, y0 + crop_h, x0, x0 + crop_w)
            if bbox not in bboxes:
                bboxes.append(bbox)
            if len(bboxes) >= int(context_candidates):
                return bboxes
    return bboxes


def _choose_context_paste(donor, target_label, num_classes, ignore_index,
                          max_area_ratio, context_candidates,
                          context_y_jitter):
    img, mask, label = _resize_donor_if_needed(
        donor,
        target_shape=tuple(target_label.shape[-2:]),
        max_area_ratio=max_area_ratio,
    )
    crop_h, crop_w = int(mask.shape[0]), int(mask.shape[1])
    candidates = _candidate_bboxes(
        donor,
        crop_shape=(crop_h, crop_w),
        target_shape=tuple(target_label.shape[-2:]),
        context_candidates=context_candidates,
        context_y_jitter=context_y_jitter,
    )
    if not candidates:
        return None
    scored = [
        (
            _context_score(
                target_label,
                int(donor['class_id']),
                bbox,
                float(donor.get('y_center', 0.5)),
                num_classes,
                ignore_index,
            ),
            bbox,
        )
        for bbox in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return img, mask, label, scored[0][1]


def _choose_memory_classes(
    memory_bank,
    class_scores,
    num_classes,
    max_classes,
    random_prob=0.0,
    min_score=0.0,
    pseudo_label=None,
    pseudo_conf=None,
    allowed_classes=None,
    blocked_classes=None,
    min_pseudo_conf=0.0,
):
    available = [
        class_id
        for class_id in memory_bank.available_classes()
        if 0 <= int(class_id) < int(num_classes)
    ]
    if not available:
        return [], {
            'candidate_count': 0,
            'skipped_class': 0,
            'skipped_conf': 0,
        }
    allowed = (
        {int(class_id) for class_id in allowed_classes}
        if allowed_classes else None)
    blocked = (
        {int(class_id) for class_id in blocked_classes}
        if blocked_classes else set())
    pseudo_label = (
        _squeeze_map(pseudo_label).long()
        if pseudo_label is not None else None)
    pseudo_conf = (
        _squeeze_map(pseudo_conf).float()
        if pseudo_conf is not None else None)
    scores = torch.as_tensor(class_scores, dtype=torch.float32).detach().cpu()
    candidates = []
    skipped_class = 0
    skipped_conf = 0
    for class_id in available:
        class_id = int(class_id)
        if allowed is not None and class_id not in allowed:
            skipped_class += 1
            continue
        if class_id in blocked:
            skipped_class += 1
            continue
        score = float(scores[int(class_id)].item())
        if score >= float(min_score):
            if (
                pseudo_label is not None
                and pseudo_conf is not None
                and float(min_pseudo_conf) > 0
            ):
                class_mask = pseudo_label.eq(class_id)
                if not bool(class_mask.any().item()):
                    skipped_conf += 1
                    continue
                class_conf = float(
                    pseudo_conf[class_mask].mean().detach().item())
                if class_conf < float(min_pseudo_conf):
                    skipped_conf += 1
                    continue
            candidates.append((score, class_id))
    if not candidates:
        return [], {
            'candidate_count': 0,
            'skipped_class': skipped_class,
            'skipped_conf': skipped_conf,
        }
    if random.random() < float(random_prob):
        random.shuffle(candidates)
    else:
        candidates.sort(key=lambda item: item[0], reverse=True)
    return [class_id for _, class_id in candidates[:int(max_classes)]], {
        'candidate_count': len(candidates),
        'skipped_class': skipped_class,
        'skipped_conf': skipped_conf,
    }


def apply_target_class_memory_mix(
    unlabeled_img,
    pseudo_label,
    pseudo_weight,
    memory_bank,
    class_scores,
    num_classes,
    max_classes=2,
    random_prob=0.0,
    min_score=0.0,
    pseudo_conf=None,
    allowed_classes=None,
    blocked_classes=None,
    min_pseudo_conf=0.0,
    context_paste=False,
    context_candidates=9,
    context_y_jitter=0.08,
    max_paste_area_ratio=0.08,
    sample_strategy=None,
    mask_only_weight=False,
    ignore_index=255,
):
    """Paste high-need target-labeled class masks into target-unlabeled images."""
    pseudo_label = _squeeze_map(pseudo_label).long()
    pseudo_weight = _squeeze_map(pseudo_weight).float()
    mixed_img = unlabeled_img.clone()
    mixed_label = pseudo_label.clone()
    if bool(mask_only_weight):
        mixed_weight = torch.zeros_like(pseudo_weight)
    else:
        mixed_weight = pseudo_weight.clone()
    batch_size, _, height, width = mixed_img.shape

    selected_classes, gate_stats = _choose_memory_classes(
        memory_bank,
        class_scores,
        num_classes=num_classes,
        max_classes=max_classes,
        random_prob=random_prob,
        min_score=min_score,
        pseudo_label=pseudo_label,
        pseudo_conf=pseudo_conf,
        allowed_classes=allowed_classes,
        blocked_classes=blocked_classes,
        min_pseudo_conf=min_pseudo_conf,
    )

    mix_masks = []
    num_class_choice = []
    num_replaced = 0
    memory_hits = 0
    memory_attempts = int(batch_size * len(selected_classes))
    selected_class_counts = defaultdict(int)
    selected_class_pixels = defaultdict(int)

    for batch_idx in range(batch_size):
        sample_mask = torch.zeros(
            1, 1, height, width,
            device=mixed_img.device,
            dtype=torch.long,
        )
        chosen_this_sample = 0
        for class_id in selected_classes:
            donor = memory_bank.sample(class_id, strategy=sample_strategy)
            if donor is None:
                continue
            if bool(donor.get('offline', False)) and context_paste:
                paste = _choose_context_paste(
                    donor,
                    pseudo_label[batch_idx],
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    max_area_ratio=max_paste_area_ratio,
                    context_candidates=context_candidates,
                    context_y_jitter=context_y_jitter,
                )
                if paste is None:
                    continue
                img_crop, mask, label_crop, bbox = paste
                y0, y1, x0, x1 = bbox
            else:
                y0, y1, x0, x1 = donor['bbox']
                if y1 > height or x1 > width:
                    continue
                img_crop = donor['img']
                mask = donor['mask']
                label_crop = donor.get(
                    'label',
                    torch.full(mask.shape, int(class_id), dtype=torch.long),
                )
            if y1 > height or x1 > width:
                continue
            mask = mask.to(device=mixed_img.device).bool()
            if not bool(mask.any().item()):
                continue
            img_crop = img_crop.to(
                device=mixed_img.device,
                dtype=mixed_img.dtype,
            )
            label_crop = label_crop.to(
                device=mixed_label.device,
                dtype=mixed_label.dtype,
            )
            crop_view = mixed_img[batch_idx, :, y0:y1, x0:x1]
            crop_mask = mask.unsqueeze(0).expand_as(crop_view)
            crop_view[crop_mask] = img_crop[crop_mask]
            mixed_label[batch_idx, y0:y1, x0:x1][mask] = label_crop[mask]
            mixed_weight[batch_idx, y0:y1, x0:x1][mask] = 1.0
            sample_mask[0, 0, y0:y1, x0:x1][mask] = 1
            class_pixels = int(mask.sum().item())
            num_replaced += class_pixels
            memory_hits += 1
            selected_class_counts[int(class_id)] += 1
            selected_class_pixels[int(class_id)] += class_pixels
            chosen_this_sample += 1
        mix_masks.append(sample_mask)
        num_class_choice.append(chosen_this_sample)

    total_pixels = batch_size * height * width
    mix_ratio = (
        float(sum(mask.float().sum().item() for mask in mix_masks))
        / max(1, total_pixels)
    )
    return TargetClassMemoryMixResult(
        images=mixed_img,
        labels=mixed_label,
        weights=mixed_weight,
        mix_masks=mix_masks,
        num_class_choice=num_class_choice,
        mix_ratio=mix_ratio,
        num_replaced=num_replaced,
        memory_hits=memory_hits,
        memory_attempts=memory_attempts,
        selected_classes=selected_classes,
        selected_class_counts=dict(selected_class_counts),
        selected_class_pixels=dict(selected_class_pixels),
        gate_candidate_count=int(gate_stats['candidate_count']),
        gate_skipped_class=int(gate_stats['skipped_class']),
        gate_skipped_conf=int(gate_stats['skipped_conf']),
    )
