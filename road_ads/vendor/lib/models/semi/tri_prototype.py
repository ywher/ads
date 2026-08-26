"""Tri-prototype helpers for SSDA structured supervision.

The first implementation uses class-distribution prototypes. It is deliberately
lightweight: source and target-labeled prototypes come from ground-truth labels,
while target-unlabeled prototypes can come from high-confidence pseudo labels.
Student logits are converted to differentiable class histograms and aligned to
the detached prototypes with Jensen-Shannon divergence.
"""

import torch
import torch.nn.functional as F


def normalize_histogram(hist, eps=1e-6):
    """L1-normalize a histogram tensor along the class dimension."""
    return hist / hist.sum(dim=-1, keepdim=True).clamp_min(eps)


def label_histograms(labels, num_classes, pixel_weights=None, eps=1e-6):
    """Build per-image class histograms from labels.

    Args:
        labels: Tensor shaped ``B x H x W`` or ``B x 1 x H x W``.
        num_classes: Number of semantic classes.
        pixel_weights: Optional tensor shaped like labels. Pixels with weight
            ``<= 0`` are ignored; positive weights contribute proportionally.
    """
    if labels.dim() == 4 and labels.shape[1] == 1:
        labels = labels.squeeze(1)
    labels = labels.detach().long()

    if pixel_weights is not None:
        if pixel_weights.dim() == 4 and pixel_weights.shape[1] == 1:
            pixel_weights = pixel_weights.squeeze(1)
        pixel_weights = pixel_weights.detach().float()

    histograms = []
    for idx in range(labels.shape[0]):
        label = labels[idx]
        valid = (label >= 0) & (label < num_classes)
        weights = None
        if pixel_weights is not None:
            weights = pixel_weights[idx].to(device=label.device)
            valid = valid & (weights > 0)

        if not torch.any(valid):
            hist = torch.zeros(num_classes, device=label.device, dtype=torch.float32)
        elif weights is None:
            hist = torch.bincount(
                label[valid],
                minlength=num_classes,
            ).float()
        else:
            hist = torch.bincount(
                label[valid],
                weights=weights[valid],
                minlength=num_classes,
            ).float()
        histograms.append(hist / hist.sum().clamp_min(eps))

    return torch.stack(histograms, dim=0)


def prediction_histograms(seg_logits, num_classes, temperature=1.0,
                          pixel_weights=None, eps=1e-6):
    """Convert segmentation logits to differentiable per-image class histograms."""
    if isinstance(seg_logits, (tuple, list)):
        seg_logits = seg_logits[0]
    if seg_logits.dim() != 4:
        raise ValueError(
            f'seg_logits must be shaped BxCxHxW, got {tuple(seg_logits.shape)}')

    probs = F.softmax(seg_logits / max(float(temperature), eps), dim=1)
    if pixel_weights is None:
        hist = probs.flatten(2).mean(dim=2)
    else:
        if pixel_weights.dim() == 4 and pixel_weights.shape[1] == 1:
            pixel_weights = pixel_weights.squeeze(1)
        weight = pixel_weights.to(device=probs.device, dtype=probs.dtype)
        if weight.shape[-2:] != probs.shape[-2:]:
            weight = F.interpolate(
                weight.unsqueeze(1),
                size=probs.shape[-2:],
                mode='nearest',
            ).squeeze(1)
        weighted = probs * weight.unsqueeze(1)
        hist = weighted.flatten(2).sum(dim=2)
        hist = hist / weight.flatten(1).sum(dim=1, keepdim=True).clamp_min(eps)

    if hist.shape[1] != num_classes:
        raise ValueError(
            f'num_classes={num_classes} does not match logits channels={hist.shape[1]}')
    return normalize_histogram(hist, eps=eps)


def histogram_js_divergence(pred_hist, target_hist, eps=1e-6):
    """Return mean Jensen-Shannon divergence between class histograms."""
    pred_hist = normalize_histogram(pred_hist, eps=eps).clamp_min(eps)
    if target_hist.dim() == 1:
        target_hist = target_hist.unsqueeze(0).expand_as(pred_hist)
    else:
        target_hist = target_hist.expand_as(pred_hist)
    target_hist = normalize_histogram(target_hist.detach(), eps=eps).clamp_min(eps)
    midpoint = 0.5 * (pred_hist + target_hist)

    kl_pred = (pred_hist * (pred_hist.log() - midpoint.log())).sum(dim=1)
    kl_tgt = (target_hist * (target_hist.log() - midpoint.log())).sum(dim=1)
    return 0.5 * (kl_pred + kl_tgt).mean()
