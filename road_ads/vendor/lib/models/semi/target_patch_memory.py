"""Target-domain patch memory mixing helpers.

The helper is intentionally model-agnostic: it receives target labeled images
and target unlabeled pseudo labels, then returns a patch-replaced mixed batch
that can reuse the existing SSDA mix loss.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TargetPatchMemoryMixResult:
    images: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor
    mix_masks: list
    num_class_choice: list
    mix_ratio: float
    num_replaced: int
    memory_hits: int
    kl_mean: float

    @property
    def log_vars(self):
        return {
            'target_patch_memory_replaced': float(self.num_replaced),
            'target_patch_memory_hit_ratio': (
                float(self.memory_hits) / max(1, self.num_replaced)),
            'target_patch_memory_mix_ratio': float(self.mix_ratio),
            'target_patch_memory_kl_mean': float(self.kl_mean),
        }


def _patch_slices(height, width, grid_size):
    grid_size = int(grid_size)
    if grid_size <= 0:
        raise ValueError('grid_size must be positive.')
    y_edges = torch.linspace(0, height, grid_size + 1).round().long().tolist()
    x_edges = torch.linspace(0, width, grid_size + 1).round().long().tolist()
    slices = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            y0, y1 = y_edges[gy], y_edges[gy + 1]
            x0, x1 = x_edges[gx], x_edges[gx + 1]
            if y1 > y0 and x1 > x0:
                slices.append((y0, y1, x0, x1))
    return slices


def _label_distribution(label_patch, num_classes, ignore_index=255):
    valid = (label_patch >= 0) & (label_patch < int(num_classes))
    valid = valid & (label_patch != int(ignore_index))
    if not bool(valid.any().item()):
        return None
    hist = torch.bincount(
        label_patch[valid].reshape(-1).long(),
        minlength=int(num_classes),
    ).float()
    return hist / hist.sum().clamp_min(1.0)


def _mean_patch_value(value_map, patch_slice):
    y0, y1, x0, x1 = patch_slice
    patch = value_map[y0:y1, x0:x1]
    if patch.numel() == 0:
        return 0.0
    return float(patch.float().mean().detach().item())


def _kl_divergence(candidate_dist, query_dist, eps=1e-8):
    candidate = candidate_dist.float().clamp_min(eps)
    query = query_dist.to(device=candidate.device, dtype=candidate.dtype).clamp_min(eps)
    return float(F.kl_div(candidate.log(), query, reduction='sum').item())


class TargetPatchMemoryBank:
    """A small FIFO patch memory for target-domain patches."""

    def __init__(self, capacity=256):
        self.capacity = int(capacity)
        self.entries = []
        self._insert_order = 0

    def __len__(self):
        return len(self.entries)

    def push_entries(self, entries):
        for entry in entries:
            stored = {
                'img': entry['img'].detach().cpu().clone(),
                'label': entry['label'].detach().cpu().clone().long(),
                'weight': entry.get(
                    'weight',
                    torch.ones_like(entry['label'], dtype=torch.float32),
                ).detach().cpu().clone().float(),
                'class_dist': entry['class_dist'].detach().cpu().clone().float(),
                'conf': float(entry.get('conf', 1.0)),
                'source': entry.get('source', 'target_labeled'),
                'class_id': int(entry.get('class_id', -1)),
                'insert_order': self._insert_order,
            }
            self._insert_order += 1
            self.entries.append(stored)
            if len(self.entries) > self.capacity:
                self.entries.pop(0)

    def sample_by_kl(self, query_class_dist, source=None, patch_shape=None):
        if not self.entries:
            return None

        query = query_class_dist.detach().cpu().float()
        candidates = []
        for entry in self.entries:
            if source is not None and entry.get('source') != source:
                continue
            if patch_shape is not None and tuple(entry['label'].shape) != tuple(patch_shape):
                continue
            kl_value = _kl_divergence(entry['class_dist'], query)
            candidates.append((kl_value, entry))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], -float(item[1].get('conf', 0.0))))
        return candidates[0][1]


def _build_labeled_entries(
    images,
    labels,
    num_classes,
    grid_size,
    min_class_ratio=0.05,
    ignore_index=255,
):
    _, _, height, width = images.shape
    entries = []
    for b in range(images.shape[0]):
        best_by_class = {}
        for patch_slice in _patch_slices(height, width, grid_size):
            y0, y1, x0, x1 = patch_slice
            label_patch = labels[b, y0:y1, x0:x1]
            class_dist = _label_distribution(label_patch, num_classes, ignore_index)
            if class_dist is None:
                continue
            class_ratio, class_id = torch.max(class_dist, dim=0)
            if float(class_ratio.item()) < float(min_class_ratio):
                continue
            class_id_int = int(class_id.item())
            quality = float(class_ratio.item())
            current = best_by_class.get(class_id_int)
            if current is not None and quality <= current[0]:
                continue
            best_by_class[class_id_int] = (quality, {
                'img': images[b, :, y0:y1, x0:x1],
                'label': label_patch,
                'weight': torch.ones_like(label_patch, dtype=torch.float32),
                'class_dist': class_dist,
                'conf': 1.0,
                'source': 'target_labeled',
                'class_id': class_id_int,
            })
        entries.extend(entry for _, entry in best_by_class.values())
    return entries


def apply_target_patch_memory_mix(
    labeled_img,
    labeled_label,
    unlabeled_img,
    pseudo_label,
    pseudo_weight,
    pseudo_conf,
    memory_bank,
    num_classes,
    grid_size=8,
    replace_ratio=0.125,
    min_labeled_class_ratio=0.05,
    ignore_index=255,
):
    """Replace low-confidence target-unlabeled patches with target memory.

    The current version is deliberately target-only: memory is filled from
    target-labeled GT patches, and replacement is performed on `T_u` only.
    """
    if labeled_label.dim() == 4 and labeled_label.shape[1] == 1:
        labeled_label = labeled_label.squeeze(1)
    if pseudo_label.dim() == 4 and pseudo_label.shape[1] == 1:
        pseudo_label = pseudo_label.squeeze(1)
    if pseudo_weight.dim() == 4 and pseudo_weight.shape[1] == 1:
        pseudo_weight = pseudo_weight.squeeze(1)
    if pseudo_conf is None:
        pseudo_conf = pseudo_weight
    if pseudo_conf.dim() == 4 and pseudo_conf.shape[1] == 1:
        pseudo_conf = pseudo_conf.squeeze(1)

    if memory_bank is None:
        memory_bank = TargetPatchMemoryBank()

    entries = _build_labeled_entries(
        labeled_img,
        labeled_label,
        num_classes=num_classes,
        grid_size=grid_size,
        min_class_ratio=min_labeled_class_ratio,
        ignore_index=ignore_index,
    )
    memory_bank.push_entries(entries)

    mixed_img = unlabeled_img.clone()
    mixed_label = pseudo_label.clone()
    mixed_weight = pseudo_weight.clone().float()
    batch_size, _, height, width = unlabeled_img.shape
    patch_slices = _patch_slices(height, width, grid_size)
    num_patches = len(patch_slices)
    replace_count = int(round(num_patches * float(replace_ratio)))
    replace_count = min(max(1, replace_count), num_patches) if replace_ratio > 0 else 0

    mix_masks = []
    num_class_choice = []
    kl_values = []
    num_replaced = 0
    memory_hits = 0

    for b in range(batch_size):
        mask = torch.zeros(1, 1, height, width, device=unlabeled_img.device, dtype=torch.long)
        patch_scores = []
        for patch_index, patch_slice in enumerate(patch_slices):
            conf = _mean_patch_value(pseudo_conf[b], patch_slice)
            patch_scores.append((conf, patch_index))
        patch_scores.sort(key=lambda item: item[0])
        selected_indices = [idx for _, idx in patch_scores[:replace_count]]

        replaced_this_sample = 0
        for patch_index in selected_indices:
            y0, y1, x0, x1 = patch_slices[patch_index]
            query_label_patch = pseudo_label[b, y0:y1, x0:x1]
            query_dist = _label_distribution(query_label_patch, num_classes, ignore_index)
            if query_dist is None:
                continue
            donor = memory_bank.sample_by_kl(
                query_dist,
                source='target_labeled',
                patch_shape=query_label_patch.shape,
            )
            if donor is None:
                continue
            mixed_img[b, :, y0:y1, x0:x1] = donor['img'].to(
                device=mixed_img.device,
                dtype=mixed_img.dtype,
            )
            mixed_label[b, y0:y1, x0:x1] = donor['label'].to(
                device=mixed_label.device,
                dtype=mixed_label.dtype,
            )
            mixed_weight[b, y0:y1, x0:x1] = donor['weight'].to(
                device=mixed_weight.device,
                dtype=mixed_weight.dtype,
            )
            mask[:, :, y0:y1, x0:x1] = 1
            kl_values.append(_kl_divergence(donor['class_dist'], query_dist))
            num_replaced += 1
            memory_hits += 1
            replaced_this_sample += 1

        mix_masks.append(mask)
        num_class_choice.append(replaced_this_sample)

    total_pixels = batch_size * height * width
    mix_ratio = float(sum(mask.float().sum().item() for mask in mix_masks)) / max(1, total_pixels)
    kl_mean = float(sum(kl_values) / len(kl_values)) if kl_values else 0.0
    return TargetPatchMemoryMixResult(
        images=mixed_img,
        labels=mixed_label,
        weights=mixed_weight,
        mix_masks=mix_masks,
        num_class_choice=num_class_choice,
        mix_ratio=mix_ratio,
        num_replaced=num_replaced,
        memory_hits=memory_hits,
        kl_mean=kl_mean,
    )
