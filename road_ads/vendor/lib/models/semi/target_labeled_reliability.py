"""Target-labeled reliability calibration helpers for SSDA.

These utilities use the model's current correctness on labeled target pixels to
derive class-wise reliability factors, then apply those factors to target
unlabeled pseudo-label weights.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _squeeze_label(labels: torch.Tensor) -> torch.Tensor:
    if labels.dim() == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.dim() != 3:
        raise ValueError(f'Expected labels with shape [B,H,W], got {labels.shape}.')
    return labels.long()


def _select_logits(logits) -> torch.Tensor:
    """Pick the segmentation logits tensor from tensor/list/tuple/dict outputs."""
    candidates = []

    def collect(item):
        if item is None:
            return
        if torch.is_tensor(item):
            if item.dim() == 4:
                candidates.append(item)
            return
        if isinstance(item, dict):
            for key in ('seg_logits', 'logits', 'S'):
                if key in item:
                    collect(item[key])
            for value in item.values():
                collect(value)
            return
        if isinstance(item, (list, tuple)):
            for value in item:
                collect(value)

    collect(logits)
    if not candidates:
        raise ValueError('Could not find a [B,C,H,W] logits tensor.')
    return candidates[0]


def _resize_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = _select_logits(logits)
    if logits.dim() != 4:
        raise ValueError(f'Expected logits with shape [B,C,H,W], got {logits.shape}.')
    if logits.shape[-2:] == labels.shape[-2:]:
        return logits
    return F.interpolate(
        logits.float(),
        size=labels.shape[-2:],
        mode='bilinear',
        align_corners=False,
    )


def target_labeled_class_reliability(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    min_reliability: float = 0.2,
    max_reliability: float = 1.0,
    default_reliability: float = 1.0,
    affine_floor: float = None,
    ignore_index: int = 255,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Compute class-wise target reliability from labeled target predictions."""
    labels = _squeeze_label(labels)
    logits = _resize_logits(logits, labels).detach()
    pred = logits.argmax(dim=1)
    device = labels.device
    reliability = torch.full(
        (int(num_classes),),
        float(default_reliability),
        device=device,
        dtype=torch.float32,
    )
    valid_class = torch.zeros(int(num_classes), device=device, dtype=torch.bool)

    valid = labels.ne(ignore_index) & labels.ge(0) & labels.lt(int(num_classes))
    if not valid.any():
        return reliability, valid_class, {
            'valid_class_count': 0.0,
            'reliability_mean': float(reliability.mean().detach().item()),
            'reliability_min': float(reliability.min().detach().item()),
            'reliability_max': float(reliability.max().detach().item()),
        }

    for class_id in range(int(num_classes)):
        class_mask = valid & labels.eq(class_id)
        if not class_mask.any():
            continue
        acc = pred[class_mask].eq(labels[class_mask]).float().mean()
        if affine_floor is not None:
            floor = min(1.0, max(0.0, float(affine_floor)))
            acc = floor + (1.0 - floor) * acc
        reliability[class_id] = acc.clamp(
            float(min_reliability),
            float(max_reliability),
        )
        valid_class[class_id] = True

    valid_values = reliability[valid_class] if valid_class.any() else reliability
    return reliability, valid_class, {
        'valid_class_count': float(valid_class.float().sum().detach().item()),
        'reliability_mean': float(valid_values.mean().detach().item()),
        'reliability_min': float(valid_values.min().detach().item()),
        'reliability_max': float(valid_values.max().detach().item()),
    }


def apply_target_labeled_reliability(
    pseudo_label: torch.Tensor,
    pseudo_weight: torch.Tensor,
    class_reliability: torch.Tensor,
    blend: float = 1.0,
    min_weight: float = 0.2,
    max_weight: float = 1.0,
    ignore_index: int = 255,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Apply class reliability factors to pseudo-label weights."""
    labels = _squeeze_label(pseudo_label)
    weights = pseudo_weight
    if weights.dim() == 4 and weights.shape[1] == 1:
        weights = weights[:, 0]
    if weights.dim() != 3:
        raise ValueError(f'Expected weights with shape [B,H,W], got {weights.shape}.')

    reliability = class_reliability.to(
        device=weights.device,
        dtype=torch.float32,
    ).flatten().clamp(float(min_weight), float(max_weight))
    factor = torch.ones_like(weights, dtype=torch.float32)
    valid = labels.ne(ignore_index) & labels.ge(0) & labels.lt(reliability.numel())
    if valid.any():
        factor[valid] = reliability[labels[valid].long()]

    blend = min(1.0, max(0.0, float(blend)))
    final_factor = (1.0 - blend) + blend * factor
    weighted = weights * final_factor.to(device=weights.device, dtype=weights.dtype)
    valid_factor = final_factor[valid] if valid.any() else final_factor.flatten()
    return weighted, {
        'tlrc_blend': float(blend),
        'tlrc_factor_mean': float(valid_factor.detach().float().mean().item()),
        'tlrc_factor_min': float(valid_factor.detach().float().min().item()),
        'tlrc_factor_max': float(valid_factor.detach().float().max().item()),
        'tlrc_weight_mean_before': float(weights.detach().float().mean().item()),
        'tlrc_weight_mean_after': float(weighted.detach().float().mean().item()),
    }
