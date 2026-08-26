"""Feature-prototype helpers for target-calibrated source assistance."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class ClassFeaturePrototypes:
    class_ids: torch.Tensor
    prototypes: torch.Tensor
    counts: torch.Tensor


def _collect_4d_features(features) -> Sequence[torch.Tensor]:
    if torch.is_tensor(features):
        return [features] if features.dim() == 4 else []
    if isinstance(features, dict):
        collected = []
        for value in features.values():
            collected.extend(_collect_4d_features(value))
        return collected
    if isinstance(features, (list, tuple)):
        collected = []
        for value in features:
            collected.extend(_collect_4d_features(value))
        return collected
    return []


def select_feature_tensor(features, feature_level=0) -> torch.Tensor:
    """Select one 4D tensor from nested backbone/HRDA feature structures."""
    tensors = list(_collect_4d_features(features))
    if not tensors:
        raise ValueError('No 4D feature tensor found for prototype extraction.')
    if feature_level in ('last', -1):
        return tensors[-1]
    if feature_level in ('first', None):
        return tensors[0]
    level = int(feature_level)
    level = max(0, min(level, len(tensors) - 1))
    return tensors[level]


def _resize_labels(labels: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if labels.dim() == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.dim() != 3:
        raise ValueError(f'Expected labels with shape [B,H,W], got {labels.shape}.')
    return F.interpolate(
        labels.unsqueeze(1).float(),
        size=size,
        mode='nearest',
    ).squeeze(1).long()


def _resize_weights(
    weights: Optional[torch.Tensor],
    size: Tuple[int, int],
    device,
    dtype,
) -> Optional[torch.Tensor]:
    if weights is None:
        return None
    if weights.dim() == 4 and weights.shape[1] == 1:
        weights = weights[:, 0]
    if weights.dim() != 3:
        raise ValueError(f'Expected weights with shape [B,H,W], got {weights.shape}.')
    return F.interpolate(
        weights.unsqueeze(1).to(device=device, dtype=dtype),
        size=size,
        mode='nearest',
    ).squeeze(1)


def compute_class_feature_prototypes(
    features,
    labels: torch.Tensor,
    num_classes: int,
    feature_level=0,
    weights: Optional[torch.Tensor] = None,
    ignore_index: int = 255,
    min_pixels: int = 8,
    detach: bool = True,
) -> ClassFeaturePrototypes:
    """Average normalized features per semantic class.

    The returned prototypes are L2-normalized. Pixel weights are useful for
    confident pseudo labels: low-confidence pixels contribute less or zero.
    """
    feat = select_feature_tensor(features, feature_level)
    if detach:
        feat = feat.detach()
    feat = F.normalize(feat.float(), dim=1)
    resized_label = _resize_labels(labels.to(device=feat.device), feat.shape[-2:])
    resized_weight = _resize_weights(
        weights,
        feat.shape[-2:],
        feat.device,
        feat.dtype,
    )

    class_ids = []
    prototypes = []
    counts = []
    flat_feat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
    flat_label = resized_label.reshape(-1)
    flat_weight = None if resized_weight is None else resized_weight.reshape(-1)

    for class_id in range(num_classes):
        mask = flat_label.eq(class_id)
        if flat_weight is not None:
            mask = mask & flat_weight.gt(0)
        pixel_count = int(mask.sum().detach().item())
        if pixel_count < int(min_pixels):
            continue
        selected = flat_feat[mask]
        if flat_weight is not None:
            weight = flat_weight[mask].unsqueeze(1).clamp_min(0)
            denom = weight.sum().clamp_min(1e-6)
            proto = (selected * weight).sum(dim=0) / denom
            count_value = float(weight.sum().detach().item())
        else:
            proto = selected.mean(dim=0)
            count_value = float(pixel_count)
        class_ids.append(class_id)
        prototypes.append(F.normalize(proto, dim=0))
        counts.append(count_value)

    if not class_ids:
        return ClassFeaturePrototypes(
            class_ids=torch.empty(0, dtype=torch.long, device=feat.device),
            prototypes=torch.empty(0, feat.shape[1], dtype=feat.dtype, device=feat.device),
            counts=torch.empty(0, dtype=feat.dtype, device=feat.device),
        )

    return ClassFeaturePrototypes(
        class_ids=torch.tensor(class_ids, dtype=torch.long, device=feat.device),
        prototypes=torch.stack(prototypes, dim=0),
        counts=torch.tensor(counts, dtype=feat.dtype, device=feat.device),
    )


def scatter_class_prototypes(
    class_ids: torch.Tensor,
    prototypes: torch.Tensor,
    num_classes: int,
    feature_dim: Optional[int] = None,
):
    if feature_dim is None:
        feature_dim = int(prototypes.shape[1]) if prototypes.numel() > 0 else 0
    dense = prototypes.new_zeros((num_classes, feature_dim))
    valid = torch.zeros(num_classes, dtype=torch.bool, device=prototypes.device)
    if class_ids.numel() > 0:
        dense[class_ids.long()] = prototypes
        valid[class_ids.long()] = True
    return dense, valid


def update_prototype_bank(
    bank: Optional[torch.Tensor],
    valid_bank: Optional[torch.Tensor],
    batch_proto: torch.Tensor,
    batch_valid: torch.Tensor,
    momentum: float = 0.9,
):
    """EMA-update a dense class prototype bank."""
    batch_proto = F.normalize(batch_proto.detach().float(), dim=1)
    batch_valid = batch_valid.detach().bool()
    if bank is None or valid_bank is None:
        bank = batch_proto.clone()
        valid_bank = batch_valid.clone()
        return bank, valid_bank

    bank = bank.to(device=batch_proto.device, dtype=batch_proto.dtype).clone()
    valid_bank = valid_bank.to(device=batch_proto.device).bool().clone()
    update_mask = batch_valid
    if update_mask.any():
        keep_mask = valid_bank & update_mask
        replace_mask = update_mask & ~valid_bank
        if keep_mask.any():
            bank[keep_mask] = F.normalize(
                momentum * bank[keep_mask]
                + (1.0 - momentum) * batch_proto[keep_mask],
                dim=1,
            )
        if replace_mask.any():
            bank[replace_mask] = batch_proto[replace_mask]
        valid_bank = valid_bank | update_mask
    return bank, valid_bank


def combine_target_prototypes(
    labeled_proto: Optional[torch.Tensor],
    labeled_valid: Optional[torch.Tensor],
    unlabeled_proto: Optional[torch.Tensor] = None,
    unlabeled_valid: Optional[torch.Tensor] = None,
    labeled_weight: float = 1.0,
    unlabeled_weight: float = 0.5,
):
    """Combine target labeled and confident-unlabeled feature prototypes."""
    if labeled_proto is None and unlabeled_proto is None:
        return None, None

    ref = labeled_proto if labeled_proto is not None else unlabeled_proto
    target = torch.zeros_like(ref)
    valid = torch.zeros(ref.shape[0], dtype=torch.bool, device=ref.device)

    if labeled_proto is not None and labeled_valid is not None:
        labeled_proto = labeled_proto.to(device=ref.device, dtype=ref.dtype)
        labeled_valid = labeled_valid.to(device=ref.device).bool()
        target[labeled_valid] += float(labeled_weight) * labeled_proto[labeled_valid]
        valid |= labeled_valid

    if unlabeled_proto is not None and unlabeled_valid is not None:
        unlabeled_proto = unlabeled_proto.to(device=ref.device, dtype=ref.dtype)
        unlabeled_valid = unlabeled_valid.to(device=ref.device).bool()
        target[unlabeled_valid] += float(unlabeled_weight) * unlabeled_proto[unlabeled_valid]
        valid |= unlabeled_valid

    target = F.normalize(target, dim=1)
    target[~valid] = 0
    return target, valid


def prototype_similarity_logits(
    features,
    prototypes: torch.Tensor,
    valid_mask: torch.Tensor,
    feature_level=0,
    temperature: float = 0.1,
    invalid_logit: float = -1e4,
) -> torch.Tensor:
    """Classify pixels by cosine similarity to class feature prototypes."""
    feat = select_feature_tensor(features, feature_level)
    feat = F.normalize(feat.float(), dim=1)
    proto = F.normalize(
        prototypes.to(device=feat.device, dtype=feat.dtype).float(),
        dim=1,
    )
    logits = torch.einsum('bdhw,cd->bchw', feat, proto)
    logits = logits / max(float(temperature), 1e-6)

    valid = valid_mask.to(device=feat.device).bool()
    if valid.numel() != logits.shape[1]:
        raise ValueError(
            f'valid_mask length={valid.numel()} does not match '
            f'prototype classes={logits.shape[1]}.')
    logits[:, ~valid, :, :] = float(invalid_logit)
    return logits


def feature_prototype_contrastive_loss(
    features,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    valid_mask: torch.Tensor,
    feature_level=0,
    temperature: float = 0.1,
    weights: Optional[torch.Tensor] = None,
    ignore_index: int = 255,
    min_valid_pixels: int = 1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pull target pixels toward their class prototype with CE over cosine logits."""
    logits = prototype_similarity_logits(
        features,
        prototypes,
        valid_mask,
        feature_level=feature_level,
        temperature=temperature,
    )
    labels = _resize_labels(labels.to(device=logits.device), logits.shape[-2:])
    valid_proto = valid_mask.to(device=logits.device).bool()
    label_valid = (
        labels.ne(ignore_index)
        & labels.ge(0)
        & labels.lt(logits.shape[1])
    )
    label_safe = labels.clamp(0, max(0, logits.shape[1] - 1))
    label_valid = label_valid & valid_proto[label_safe]

    resized_weight = _resize_weights(
        weights,
        logits.shape[-2:],
        logits.device,
        logits.dtype,
    )
    if resized_weight is not None:
        label_valid = label_valid & resized_weight.gt(0)

    valid_pixel_count = int(label_valid.sum().detach().item())
    if valid_pixel_count < int(min_valid_pixels):
        zero = logits.sum() * 0.0
        return zero, {
            'valid_pixel_count': 0,
            'valid_class_count': 0,
            'mean_weight': 0.0,
        }

    ce_labels = labels.clone()
    ce_labels[~label_valid] = int(ignore_index)
    loss_map = F.cross_entropy(
        logits,
        ce_labels.long(),
        ignore_index=ignore_index,
        reduction='none',
    )
    if resized_weight is None:
        loss = loss_map[label_valid].mean()
        mean_weight = 1.0
    else:
        pixel_weight = resized_weight.to(device=loss_map.device, dtype=loss_map.dtype)
        denom = pixel_weight[label_valid].sum().clamp_min(1e-6)
        loss = (loss_map[label_valid] * pixel_weight[label_valid]).sum() / denom
        mean_weight = float(pixel_weight[label_valid].mean().detach().item())

    valid_classes = torch.unique(labels[label_valid])
    return loss, {
        'valid_pixel_count': valid_pixel_count,
        'valid_class_count': int(valid_classes.numel()),
        'mean_weight': mean_weight,
    }


def source_target_class_scores(
    source_proto: Optional[torch.Tensor],
    source_valid: Optional[torch.Tensor],
    target_proto: Optional[torch.Tensor],
    target_valid: Optional[torch.Tensor],
    min_score: float = 0.1,
    default_score: float = 1.0,
    score_norm: str = 'none',
    score_temperature: float = 1.0,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
):
    """Return per-class source usefulness scores in [min_score, 1]."""
    if (
        source_proto is None or source_valid is None
        or target_proto is None or target_valid is None
    ):
        num_classes = 0
        ref = None
        if source_proto is not None:
            num_classes = source_proto.shape[0]
            ref = source_proto
        elif target_proto is not None:
            num_classes = target_proto.shape[0]
            ref = target_proto
        return None if num_classes == 0 else ref.new_full((num_classes,), default_score)

    source_proto = source_proto.float()
    target_proto = target_proto.to(device=source_proto.device, dtype=source_proto.dtype)
    source_valid = source_valid.to(device=source_proto.device).bool()
    target_valid = target_valid.to(device=source_proto.device).bool()
    valid = source_valid & target_valid
    scores = source_proto.new_full((source_proto.shape[0],), float(default_score))
    if valid.any():
        cosine = F.cosine_similarity(source_proto[valid], target_proto[valid], dim=1)
        cosine = cosine.clamp(float(min_score), 1.0)
        scores[valid] = cosine
    return normalize_class_scores(
        scores,
        valid_mask=valid,
        mode=score_norm,
        min_score=min_score,
        default_score=default_score,
        temperature=score_temperature,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
    )


def normalize_class_scores(
    scores: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    mode: str = 'none',
    min_score: float = 0.1,
    default_score: float = 1.0,
    temperature: float = 1.0,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Normalize narrow cosine scores into a more discriminative range.

    ``none`` preserves the v1 behavior. ``quantile`` and ``minmax`` stretch
    valid scores into ``[min_score, 1]`` so high-dimensional VFM prototype
    cosine values such as 0.92-0.98 do not collapse to near-constant weights.
    """
    mode = str(mode or 'none').lower()
    out = scores.detach().float().clone()
    if valid_mask is None:
        valid = torch.isfinite(out)
    else:
        valid = valid_mask.to(device=out.device).bool() & torch.isfinite(out)

    out[~valid] = float(default_score)
    if mode in ('none', 'raw', 'cosine', 'identity'):
        out[valid] = out[valid].clamp(float(min_score), 1.0)
        return out

    valid_scores = out[valid]
    if valid_scores.numel() < 2:
        out[valid] = out[valid].clamp(float(min_score), 1.0)
        return out

    if mode in ('minmax', 'min_max'):
        low = valid_scores.min()
        high = valid_scores.max()
        normalized = (out[valid] - low) / (high - low).clamp_min(eps)
    elif mode in ('quantile', 'robust_minmax', 'robust'):
        q_low = max(0.0, min(1.0, float(quantile_low)))
        q_high = max(q_low, min(1.0, float(quantile_high)))
        if q_high <= q_low:
            q_low, q_high = 0.0, 1.0
        low = torch.quantile(valid_scores, q_low)
        high = torch.quantile(valid_scores, q_high)
        normalized = (out[valid] - low) / (high - low).clamp_min(eps)
    elif mode in ('zscore', 'zscore_sigmoid', 'sigmoid'):
        mean = valid_scores.mean()
        std = valid_scores.std(unbiased=False).clamp_min(eps)
        tau = max(float(temperature), eps)
        normalized = torch.sigmoid((out[valid] - mean) / (std * tau))
    else:
        raise ValueError(
            f'Invalid feature prototype score_norm={mode!r}. Choose '
            'none, minmax, quantile, or zscore_sigmoid.')

    normalized = normalized.clamp(0.0, 1.0)
    out[valid] = float(min_score) + (1.0 - float(min_score)) * normalized
    return out


def target_rare_class_scores(
    target_counts: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    gamma: float = 0.5,
    min_score: float = 0.25,
    max_score: float = 4.0,
    default_score: float = 1.0,
    normalize_mean: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute class-level rarity multipliers from target-domain counts."""
    counts = target_counts.detach().float().clamp_min(0.0)
    if valid_mask is None:
        valid = torch.isfinite(counts)
    else:
        valid = valid_mask.to(device=counts.device).bool() & torch.isfinite(counts)
    rarity = counts.new_full(counts.shape, float(default_score))
    if not valid.any():
        return rarity

    valid_counts = counts[valid]
    positive = valid_counts.gt(0)
    if positive.any():
        ref = valid_counts[positive].mean().clamp_min(eps)
    else:
        ref = counts.new_tensor(1.0)
    raw = (ref / valid_counts.clamp_min(eps)).pow(max(0.0, float(gamma)))
    raw = raw.clamp(float(min_score), float(max_score))
    if normalize_mean and raw.numel() > 0:
        raw = raw / raw.mean().clamp_min(eps)
        raw = raw.clamp(float(min_score), float(max_score))
    rarity[valid] = raw
    return rarity


def protect_structure_class_scores(
    scores: torch.Tensor,
    structure_classes: Optional[Sequence[int]] = None,
    mode: str = 'floor',
    min_score: float = 1.0,
    default_score: float = 1.0,
) -> torch.Tensor:
    """Keep scene-structure classes from being suppressed in source ClassMix.

    Prototype scores are useful for selecting transferable source classes, but
    strong suppression of road/layout classes can remove the source-domain
    regularization that stabilizes driving-scene structure. This helper only
    edits the class scores used for source-target mix selection.
    """
    out = scores.detach().float().clone()
    if not structure_classes:
        return out

    class_ids = []
    for class_id in structure_classes:
        class_id = int(class_id)
        if 0 <= class_id < out.numel():
            class_ids.append(class_id)
    if not class_ids:
        return out

    idx = torch.as_tensor(class_ids, device=out.device, dtype=torch.long)
    mode = str(mode or 'floor').lower()
    if mode in ('floor', 'min', 'lower_bound'):
        out[idx] = torch.maximum(
            out[idx],
            out.new_full((idx.numel(),), float(min_score)),
        )
    elif mode in ('constant', 'default', 'restore', 'uniform'):
        out[idx] = float(default_score)
    elif mode in ('none', 'off', 'disable', 'disabled'):
        return out
    else:
        raise ValueError(
            f'Invalid structure score protection mode={mode!r}. Choose '
            'floor, constant, or none.')
    return out


def class_score_weight_map(
    labels: torch.Tensor,
    class_scores: Optional[torch.Tensor],
    min_weight: float = 0.5,
    max_weight: float = 1.0,
    gamma: float = 1.0,
    default_weight: float = 1.0,
    ignore_index: int = 255,
):
    """Map class scores to a pixel-wise loss weight map."""
    if class_scores is None:
        return torch.full(labels.shape, float(default_weight), device=labels.device)
    scores = class_scores.to(device=labels.device, dtype=torch.float32)
    labels_long = labels.long()
    weights = torch.full(labels_long.shape, float(default_weight), device=labels.device)
    valid = labels_long.ne(ignore_index) & labels_long.ge(0) & labels_long.lt(scores.numel())
    if valid.any():
        scaled = scores[labels_long[valid]].clamp(0.0, 1.0)
        scaled = scaled.pow(max(1e-6, float(gamma)))
        mapped = float(min_weight) + (float(max_weight) - float(min_weight)) * scaled
        weights[valid] = mapped
    return weights
