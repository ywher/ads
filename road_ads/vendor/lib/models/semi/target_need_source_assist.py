"""Target-Need Source Assistance helpers.

TNSA computes per-class source-mix scores from target-domain supervision gaps:
classes with low target coverage and high pseudo-label uncertainty receive
larger scores, so source-target ClassMix preferentially pastes source classes
that the target domain currently needs.
"""

from typing import Dict, Optional, Tuple

import torch

from .feature_prototype import target_rare_class_scores


def _squeeze_label(labels: torch.Tensor) -> torch.Tensor:
    if labels.dim() == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.dim() != 3:
        raise ValueError(f'Expected labels with shape [B,H,W], got {labels.shape}.')
    return labels.long()


def _squeeze_weight(weights: Optional[torch.Tensor], labels: torch.Tensor) -> Optional[torch.Tensor]:
    if weights is None:
        return None
    if weights.dim() == 4 and weights.shape[1] == 1:
        weights = weights[:, 0]
    if weights.dim() != 3:
        raise ValueError(f'Expected weights with shape [B,H,W], got {weights.shape}.')
    return weights.to(device=labels.device, dtype=torch.float32)


def _class_weighted_counts(
    labels: torch.Tensor,
    num_classes: int,
    weights: Optional[torch.Tensor] = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    labels = _squeeze_label(labels)
    weights = _squeeze_weight(weights, labels)
    counts = torch.zeros(num_classes, device=labels.device, dtype=torch.float32)
    valid = labels.ne(ignore_index) & labels.ge(0) & labels.lt(num_classes)
    if not valid.any():
        return counts

    flat_labels = labels[valid].long()
    if weights is None:
        flat_weights = torch.ones_like(flat_labels, dtype=torch.float32)
    else:
        flat_weights = weights[valid].float().clamp_min(0.0)
    counts.scatter_add_(0, flat_labels, flat_weights)
    return counts


def _class_uncertainty(
    pseudo_label: Optional[torch.Tensor],
    pseudo_weight: Optional[torch.Tensor],
    pseudo_conf: Optional[torch.Tensor],
    num_classes: int,
    ignore_index: int = 255,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = (
        pseudo_label.device
        if pseudo_label is not None
        else torch.device('cpu')
    )
    uncertainty = torch.zeros(num_classes, device=device, dtype=torch.float32)
    valid_classes = torch.zeros(num_classes, device=device, dtype=torch.bool)
    if pseudo_label is None or pseudo_conf is None:
        return uncertainty, valid_classes

    labels = _squeeze_label(pseudo_label)
    conf = _squeeze_weight(pseudo_conf, labels)
    weight = _squeeze_weight(pseudo_weight, labels)
    if weight is None:
        weight = torch.ones_like(labels, dtype=torch.float32)
    valid = (
        labels.ne(ignore_index)
        & labels.ge(0)
        & labels.lt(num_classes)
        & weight.gt(0)
    )
    if not valid.any():
        return uncertainty, valid_classes

    for class_id in range(num_classes):
        mask = valid & labels.eq(class_id)
        if not mask.any():
            continue
        class_weight = weight[mask].float().clamp_min(0.0)
        denom = class_weight.sum().clamp_min(1e-6)
        mean_conf = (conf[mask].float().clamp(0.0, 1.0) * class_weight).sum() / denom
        uncertainty[class_id] = 1.0 - mean_conf.clamp(0.0, 1.0)
        valid_classes[class_id] = True
    return uncertainty, valid_classes


def _normalize_mean(scores: torch.Tensor, min_score: float, max_score: float) -> torch.Tensor:
    scores = scores.float().clamp(float(min_score), float(max_score))
    return (scores / scores.mean().clamp_min(1e-6)).clamp(
        float(min_score),
        float(max_score),
    )


def _as_class_scores(scores: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
    if scores is None:
        return None
    scores = scores.to(device=ref.device, dtype=torch.float32).flatten()
    if scores.numel() != ref.numel():
        raise ValueError(
            f'Expected {ref.numel()} class scores, got {scores.numel()}.')
    return scores


def class_score_pixel_weights(
    labels: torch.Tensor,
    class_scores: torch.Tensor,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
    gamma: float = 1.0,
    ignore_index: int = 255,
    default_weight: float = 1.0,
) -> torch.Tensor:
    """Map per-class scores to a dense pixel weight map.

    The helper is used by target-deficit source pixel reweighting. Invalid /
    ignored labels keep ``default_weight`` so the segmentation loss ignore mask
    remains responsible for ignoring them.
    """
    labels = _squeeze_label(labels)
    scores = torch.as_tensor(
        class_scores,
        device=labels.device,
        dtype=torch.float32,
    ).flatten()
    weights = torch.full(
        labels.shape,
        float(default_weight),
        device=labels.device,
        dtype=torch.float32,
    )
    if scores.numel() == 0:
        return weights
    scores = scores.clamp(float(min_weight), float(max_weight))
    if float(gamma) != 1.0:
        scores = scores.pow(max(0.0, float(gamma)))
        scores = scores.clamp(float(min_weight), float(max_weight))
    valid = (
        labels.ne(ignore_index)
        & labels.ge(0)
        & labels.lt(int(scores.numel()))
    )
    if valid.any():
        weights[valid] = scores[labels[valid].long()]
    return weights


def _valid_class_ids(class_ids, num_classes: int, device) -> torch.Tensor:
    if class_ids is None:
        return torch.empty(0, device=device, dtype=torch.long)
    if torch.is_tensor(class_ids):
        ids = class_ids.to(device=device, dtype=torch.long).flatten()
    else:
        ids = torch.as_tensor(list(class_ids), device=device, dtype=torch.long)
    if ids.numel() == 0:
        return ids
    return ids[(ids >= 0) & (ids < int(num_classes))]


def class_conditional_source_route_scores(
    num_classes: int,
    enhance_classes=None,
    suppress_classes=None,
    enhance_score: float = 1.5,
    suppress_score: float = 0.5,
    min_score: float = 0.25,
    max_score: float = 4.0,
    device=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return explicit class-routing scores for source assistance.

    The scores are normalized to mean one, so they can multiply other
    class-level signals without changing the global source-mix loss scale.
    """
    if device is None:
        device = torch.device('cpu')
    scores = torch.ones(int(num_classes), device=device, dtype=torch.float32)
    enhance_ids = _valid_class_ids(enhance_classes, num_classes, device)
    suppress_ids = _valid_class_ids(suppress_classes, num_classes, device)
    if enhance_ids.numel() > 0:
        scores[enhance_ids] = float(enhance_score)
    if suppress_ids.numel() > 0:
        scores[suppress_ids] = float(suppress_score)
    scores = _normalize_mean(scores, min_score=min_score, max_score=max_score)
    return scores, {
        'enhance_count': float(enhance_ids.numel()),
        'suppress_count': float(suppress_ids.numel()),
        'score_mean': float(scores.detach().mean().item()),
        'score_min': float(scores.detach().min().item()),
        'score_max': float(scores.detach().max().item()),
    }


def per_class_gradient_cosines(
    source_losses: Dict[int, torch.Tensor],
    target_losses: Dict[int, torch.Tensor],
    params,
    num_classes: int,
    class_ids=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Measure source/target gradient agreement for shared semantic classes.

    Loss dictionaries contain differentiable scalar losses indexed by class.
    Only classes available in both domains are evaluated. Missing/zero-gradient
    classes remain invalid so callers can fall back to neutral routing.
    """
    params = tuple(params)
    device = params[0].device if params else torch.device('cpu')
    cosines = torch.zeros(int(num_classes), device=device, dtype=torch.float32)
    valid = torch.zeros(int(num_classes), device=device, dtype=torch.bool)
    if not params:
        return cosines, valid

    shared = set(source_losses).intersection(target_losses)
    if class_ids is not None:
        shared.intersection_update(int(class_id) for class_id in class_ids)
    shared = sorted(
        class_id for class_id in shared
        if 0 <= int(class_id) < int(num_classes)
    )

    def flatten(grads):
        parts = []
        for grad, param in zip(grads, params):
            if grad is None:
                grad = torch.zeros_like(param)
            parts.append(grad.detach().float().reshape(-1))
        return torch.cat(parts) if parts else None

    for class_id in shared:
        source_loss = source_losses[class_id]
        target_loss = target_losses[class_id]
        if not source_loss.requires_grad or not target_loss.requires_grad:
            continue
        source_grads = torch.autograd.grad(
            source_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        target_grads = torch.autograd.grad(
            target_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        source_vec = flatten(source_grads)
        target_vec = flatten(target_grads)
        if source_vec is None or target_vec is None:
            continue
        if source_vec.norm().item() <= 0 or target_vec.norm().item() <= 0:
            continue
        cosine = torch.nn.functional.cosine_similarity(
            source_vec,
            target_vec,
            dim=0,
        ).clamp(-1.0, 1.0)
        cosines[class_id] = cosine
        valid[class_id] = True
    return cosines, valid


def class_gradient_conflict_route_scores(
    target_deficit_scores: torch.Tensor,
    gradient_cosines: torch.Tensor,
    valid_classes: torch.Tensor,
    assist_strength: float = 0.5,
    reject_strength: float = 0.5,
    min_abs_cosine: float = 0.05,
    min_score: float = 0.25,
    max_score: float = 1.75,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Route source classes using target need and gradient compatibility."""
    deficit = torch.as_tensor(target_deficit_scores).float().flatten()
    cosine = torch.as_tensor(
        gradient_cosines,
        device=deficit.device,
        dtype=torch.float32,
    ).flatten()
    valid = torch.as_tensor(
        valid_classes,
        device=deficit.device,
        dtype=torch.bool,
    ).flatten()
    if cosine.numel() != deficit.numel() or valid.numel() != deficit.numel():
        raise ValueError('deficit, cosine, and valid vectors must have equal size.')

    route = torch.ones_like(deficit)
    finite = torch.isfinite(deficit) & torch.isfinite(cosine)
    valid = valid & finite
    if valid.any():
        need = deficit.clamp_min(0.0)
        need = need / need[valid].max().clamp_min(1e-6)
        active = valid & cosine.abs().ge(float(min_abs_cosine))
        assist = active & cosine.gt(0)
        reject = active & cosine.lt(0)
        route[assist] += (
            float(assist_strength) * need[assist] * cosine[assist]
        )
        route[reject] -= (
            float(reject_strength) * need[reject] * cosine[reject].abs()
        )
    else:
        assist = reject = torch.zeros_like(valid)
    route = route.clamp(float(min_score), float(max_score))
    active = assist | reject
    return route, {
        'valid_class_count': float(valid.sum().item()),
        'assist_count': float(assist.sum().item()),
        'reject_count': float(reject.sum().item()),
        'neutral_count': float((~active).sum().item()),
        'cosine_mean': float(
            cosine[valid].mean().item() if valid.any() else 0.0),
        'cosine_min': float(
            cosine[valid].min().item() if valid.any() else 0.0),
        'cosine_max': float(
            cosine[valid].max().item() if valid.any() else 0.0),
        'score_mean': float(route.mean().item()),
        'score_min': float(route.min().item()),
        'score_max': float(route.max().item()),
    }


def target_class_reliability_scores(
    target_labeled: torch.Tensor,
    pseudo_label: Optional[torch.Tensor],
    pseudo_weight: Optional[torch.Tensor],
    pseudo_conf: Optional[torch.Tensor],
    num_classes: int,
    labeled_weight: float = 1.0,
    unlabeled_weight: float = 0.5,
    ignore_index: int = 255,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Estimate per-class reliability from available target supervision.

    Ground-truth pixels provide full support, while pseudo-label pixels are
    weighted by both their validity mask and confidence. Log normalization
    prevents large stuff classes from completely suppressing small objects.
    """
    labels = _squeeze_label(target_labeled)
    device = labels.device
    labeled_counts = _class_weighted_counts(
        labels,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    pseudo_counts = torch.zeros(
        num_classes, device=device, dtype=torch.float32)
    if pseudo_label is not None:
        pseudo_labels = _squeeze_label(pseudo_label).to(device=device)
        pseudo_valid = _squeeze_weight(pseudo_weight, pseudo_labels)
        if pseudo_valid is None:
            pseudo_valid = torch.ones_like(
                pseudo_labels, dtype=torch.float32)
        confidence = _squeeze_weight(pseudo_conf, pseudo_labels)
        if confidence is None:
            confidence = torch.ones_like(
                pseudo_labels, dtype=torch.float32)
        pseudo_counts = _class_weighted_counts(
            pseudo_labels,
            num_classes=num_classes,
            weights=(pseudo_valid * confidence.clamp(0.0, 1.0)),
            ignore_index=ignore_index,
        )

    support = (
        float(labeled_weight) * labeled_counts
        + float(unlabeled_weight) * pseudo_counts
    ).clamp_min(0.0)
    valid = support.gt(0)
    reliability = torch.zeros_like(support)
    if valid.any():
        log_support = torch.log1p(support[valid])
        reliability[valid] = (
            log_support / log_support.max().clamp_min(1e-6)
        ).clamp(0.0, 1.0)
    return reliability, valid, {
        'reliable_class_count': float(valid.sum().item()),
        'reliability_mean': float(
            reliability[valid].mean().item() if valid.any() else 0.0),
        'reliability_min': float(
            reliability[valid].min().item() if valid.any() else 0.0),
        'reliability_max': float(
            reliability[valid].max().item() if valid.any() else 0.0),
    }


def _average_tie_percentile(values: torch.Tensor) -> torch.Tensor:
    """Map a 1-D tensor to [0, 1] ranks while preserving ties."""
    values = values.float().flatten()
    if values.numel() <= 1:
        return torch.full_like(values, 0.5)
    lower = (values[:, None] > values[None, :]).float().sum(dim=1)
    equal = (values[:, None] == values[None, :]).float().sum(dim=1) - 1.0
    return (lower + 0.5 * equal) / float(values.numel() - 1)


def self_calibrated_class_route_scores(
    target_deficit_scores: torch.Tensor,
    source_transfer_scores: torch.Tensor,
    reliability_scores: torch.Tensor,
    reliability_min: float = 0.25,
    intervention_quantile: float = 0.25,
    max_delta: float = 0.5,
    min_need_percentile: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Build adaptive assist/neutral/reject multipliers for source ClassMix.

    A class is assisted only when it is needed by the target and transferable
    from the source. A needed but poorly transferable class is rejected. Low
    reliability or insufficient variation falls back to a neutral multiplier.
    """
    deficit = torch.as_tensor(target_deficit_scores).float().flatten()
    transfer = _as_class_scores(source_transfer_scores, deficit)
    reliability = _as_class_scores(reliability_scores, deficit)
    if transfer is None or reliability is None:
        raise ValueError('source transfer and reliability scores are required.')
    if not 0.0 < float(intervention_quantile) <= 0.5:
        raise ValueError('intervention_quantile must be in (0, 0.5].')
    if not 0.0 <= float(max_delta) < 1.0:
        raise ValueError('max_delta must be in [0, 1).')

    route = torch.ones_like(deficit)
    finite = (
        torch.isfinite(deficit)
        & torch.isfinite(transfer)
        & torch.isfinite(reliability)
    )
    valid = finite & reliability.ge(float(reliability_min))
    base_stats = {
        'assist_count': 0.0,
        'reject_count': 0.0,
        'neutral_count': float(route.numel()),
        'reliable_class_count': float(valid.sum().item()),
        'intervention_ratio': 0.0,
        'fallback': 1.0,
        'score_mean': 1.0,
        'score_min': 1.0,
        'score_max': 1.0,
    }
    if valid.sum().item() < 2:
        return route, base_stats

    valid_deficit = deficit[valid]
    valid_transfer = transfer[valid]
    if (
        torch.allclose(valid_deficit, valid_deficit[:1])
        or torch.allclose(valid_transfer, valid_transfer[:1])
    ):
        return route, base_stats

    need_rank = torch.zeros_like(deficit)
    affinity_rank = torch.full_like(deficit, 0.5)
    need_rank[valid] = _average_tie_percentile(valid_deficit)
    affinity_rank[valid] = _average_tie_percentile(valid_transfer)
    reliability = reliability.clamp(0.0, 1.0)
    utility = reliability * need_rank * (2.0 * affinity_rank - 1.0)
    needed = valid & need_rank.ge(float(min_need_percentile))
    positive = needed & utility.gt(0)
    negative = needed & utility.lt(0)

    assist = torch.zeros_like(valid)
    reject = torch.zeros_like(valid)
    if positive.any():
        threshold = torch.quantile(
            utility[positive], 1.0 - float(intervention_quantile))
        assist = positive & utility.ge(threshold)
    if negative.any():
        threshold = torch.quantile(
            utility[negative], float(intervention_quantile))
        reject = negative & utility.le(threshold)

    delta = (
        float(max_delta)
        * reliability
        * need_rank
        * (2.0 * affinity_rank - 1.0).abs()
    ).clamp(0.0, float(max_delta))
    route[assist] = 1.0 + delta[assist]
    route[reject] = 1.0 - delta[reject]
    intervened = assist | reject
    stats = {
        'assist_count': float(assist.sum().item()),
        'reject_count': float(reject.sum().item()),
        'neutral_count': float((~intervened).sum().item()),
        'reliable_class_count': float(valid.sum().item()),
        'intervention_ratio': float(
            intervened.float().mean().item()),
        'fallback': 0.0,
        'utility_min': float(utility[valid].min().item()),
        'utility_max': float(utility[valid].max().item()),
        'score_mean': float(route.mean().item()),
        'score_min': float(route.min().item()),
        'score_max': float(route.max().item()),
    }
    return route, stats


def combine_target_need_source_scores(
    target_need_scores: torch.Tensor,
    source_transfer_scores: Optional[torch.Tensor] = None,
    target_loss_scores: Optional[torch.Tensor] = None,
    source_transfer_weight: float = 1.0,
    target_loss_weight: float = 0.0,
    min_score: float = 0.25,
    max_score: float = 4.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combine target demand with source transferability.

    Target-need alone answers which classes the target domain lacks. Source
    transferability answers whether source features for those classes are close
    enough to the current target prototypes. The product keeps source assistance
    focused on classes that are both needed and transferable.
    """
    scores = target_need_scores.float().flatten()
    scores = _normalize_mean(scores, min_score=min_score, max_score=max_score)
    stats = {
        'target_need_mean': float(scores.detach().mean().item()),
        'target_need_min': float(scores.detach().min().item()),
        'target_need_max': float(scores.detach().max().item()),
        'source_transfer_used': 0.0,
        'target_loss_used': 0.0,
    }

    transfer = _as_class_scores(source_transfer_scores, scores)
    if transfer is not None and float(source_transfer_weight) > 0:
        transfer = _normalize_mean(
            transfer,
            min_score=min_score,
            max_score=max_score,
        )
        scores = scores * transfer.pow(float(source_transfer_weight))
        stats.update({
            'source_transfer_used': 1.0,
            'source_transfer_mean': float(transfer.detach().mean().item()),
            'source_transfer_min': float(transfer.detach().min().item()),
            'source_transfer_max': float(transfer.detach().max().item()),
        })

    target_loss = _as_class_scores(target_loss_scores, scores)
    if target_loss is not None and float(target_loss_weight) > 0:
        target_loss = _normalize_mean(
            target_loss.clamp_min(0.0) + 1e-6,
            min_score=min_score,
            max_score=max_score,
        )
        scores = scores * target_loss.pow(float(target_loss_weight))
        stats.update({
            'target_loss_used': 1.0,
            'target_loss_score_mean': float(target_loss.detach().mean().item()),
            'target_loss_score_min': float(target_loss.detach().min().item()),
            'target_loss_score_max': float(target_loss.detach().max().item()),
        })

    scores = _normalize_mean(scores, min_score=min_score, max_score=max_score)
    stats.update({
        'score_mean': float(scores.detach().mean().item()),
        'score_min': float(scores.detach().min().item()),
        'score_max': float(scores.detach().max().item()),
    })
    return scores, stats


def target_need_class_scores(
    target_labeled: torch.Tensor,
    pseudo_label: Optional[torch.Tensor],
    pseudo_weight: Optional[torch.Tensor],
    pseudo_conf: Optional[torch.Tensor],
    num_classes: int,
    labeled_weight: float = 1.0,
    unlabeled_weight: float = 0.5,
    coverage_gamma: float = 0.5,
    uncertainty_weight: float = 0.5,
    uncertainty_gamma: float = 1.0,
    min_score: float = 0.25,
    max_score: float = 4.0,
    ignore_index: int = 255,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return class-level source-mix scores from target-domain need.

    Coverage need is high for classes that are absent or rare in target labeled
    plus confident target-unlabeled pseudo labels. Uncertainty need increases
    scores for classes whose target-unlabeled pseudo labels have low confidence.
    """
    target_labeled = _squeeze_label(target_labeled)
    num_classes = int(num_classes)
    labeled_counts = _class_weighted_counts(
        target_labeled,
        num_classes,
        weights=None,
        ignore_index=ignore_index,
    )
    if pseudo_label is not None:
        unlabeled_counts = _class_weighted_counts(
            pseudo_label,
            num_classes,
            weights=pseudo_weight,
            ignore_index=ignore_index,
        ).to(device=target_labeled.device)
    else:
        unlabeled_counts = torch.zeros_like(labeled_counts)

    coverage_counts = (
        float(labeled_weight) * labeled_counts
        + float(unlabeled_weight) * unlabeled_counts
    )
    valid_signal = coverage_counts.gt(0)
    if not valid_signal.any():
        scores = torch.ones(num_classes, device=target_labeled.device)
        return scores, {
            'valid_class_count': 0.0,
            'coverage_score_mean': 1.0,
            'uncertainty_mean': 0.0,
            'score_mean': 1.0,
            'score_min': 1.0,
            'score_max': 1.0,
        }

    coverage_need = target_rare_class_scores(
        coverage_counts,
        valid_mask=torch.ones_like(coverage_counts, dtype=torch.bool),
        gamma=coverage_gamma,
        min_score=min_score,
        max_score=max_score,
        default_score=1.0,
        normalize_mean=True,
    )
    uncertainty, uncertainty_valid = _class_uncertainty(
        pseudo_label,
        pseudo_weight,
        pseudo_conf,
        num_classes,
        ignore_index=ignore_index,
    )
    uncertainty = uncertainty.to(device=target_labeled.device)
    uncertainty_valid = uncertainty_valid.to(device=target_labeled.device)
    uncertainty_factor = torch.ones_like(coverage_need)
    if uncertainty_valid.any() and float(uncertainty_weight) > 0:
        uncertainty_factor[uncertainty_valid] = (
            1.0
            + float(uncertainty_weight)
            * uncertainty[uncertainty_valid].clamp(0.0, 1.0).pow(
                max(0.0, float(uncertainty_gamma)))
        )

    scores = _normalize_mean(
        coverage_need * uncertainty_factor,
        min_score=min_score,
        max_score=max_score,
    )
    valid_unc = uncertainty[uncertainty_valid]
    return scores, {
        'valid_class_count': float(valid_signal.sum().detach().item()),
        'coverage_score_mean': float(coverage_need.detach().mean().item()),
        'coverage_score_min': float(coverage_need.detach().min().item()),
        'coverage_score_max': float(coverage_need.detach().max().item()),
        'uncertainty_mean': float(valid_unc.mean().detach().item())
        if valid_unc.numel() > 0 else 0.0,
        'uncertainty_valid_count': float(
            uncertainty_valid.sum().detach().item()),
        'score_mean': float(scores.detach().mean().item()),
        'score_min': float(scores.detach().min().item()),
        'score_max': float(scores.detach().max().item()),
    }
