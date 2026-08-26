from typing import Iterable, Optional, Tuple

import torch


def _to_class_id_set(class_ids: Optional[Iterable[int]]):
    if class_ids is None:
        return set()
    return {int(class_id) for class_id in class_ids}


def build_class_conditional_anchor_weight_map(
    labels: torch.Tensor,
    enhance_class_ids: Optional[Iterable[int]] = None,
    enhance_weight: float = 1.5,
    protect_class_ids: Optional[Iterable[int]] = None,
    protect_weight: float = 0.5,
    default_weight: float = 1.0,
    ignore_index: int = 255,
    normalize_mean: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """Build a pixel-wise target-anchor replay loss weight map.

    Enhanced classes receive larger weights, protected classes receive smaller
    weights, and ignored pixels are set to zero. Optional mean normalization
    keeps the valid-pixel loss scale close to the original labeled branch.
    """
    if labels.dim() == 4 and labels.shape[1] == 1:
        label_map = labels[:, 0]
    else:
        label_map = labels
    label_map = label_map.detach().long()

    enhance_ids = _to_class_id_set(enhance_class_ids)
    protect_ids = _to_class_id_set(protect_class_ids)
    valid = label_map.ne(int(ignore_index))
    weights = torch.full(label_map.shape, float(default_weight), device=label_map.device)
    weights = weights.to(dtype=torch.float32)
    weights[~valid] = 0.0

    enhance_mask = torch.zeros_like(valid, dtype=torch.bool)
    protect_mask = torch.zeros_like(valid, dtype=torch.bool)
    for class_id in enhance_ids:
        enhance_mask |= label_map.eq(class_id)
    for class_id in protect_ids:
        protect_mask |= label_map.eq(class_id)
    enhance_mask &= valid
    protect_mask &= valid

    weights[enhance_mask] = float(enhance_weight)
    weights[protect_mask] = float(protect_weight)

    valid_count = valid.float().sum().clamp_min(1.0)
    valid_mean_before = weights[valid].mean() if valid.any() else weights.new_tensor(0.0)
    if normalize_mean and valid.any() and float(valid_mean_before.item()) > 0:
        weights[valid] = weights[valid] / valid_mean_before.clamp_min(1e-6)
    valid_mean_after = weights[valid].mean() if valid.any() else weights.new_tensor(0.0)

    stats = {
        'valid_pixel_ratio': float(valid.float().mean().item()),
        'enhance_pixel_ratio': float(enhance_mask.float().sum().item() / valid_count.item()),
        'protect_pixel_ratio': float(protect_mask.float().sum().item() / valid_count.item()),
        'weight_mean_before': float(valid_mean_before.item()),
        'weight_mean_after': float(valid_mean_after.item()),
        'weight_min': float(weights[valid].min().item()) if valid.any() else 0.0,
        'weight_max': float(weights[valid].max().item()) if valid.any() else 0.0,
    }
    return weights, stats
