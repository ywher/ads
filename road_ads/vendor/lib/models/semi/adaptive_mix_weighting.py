"""Adaptive mix-branch weighting helpers for SSDA training."""


def _clamp(value, lower=0.0, upper=1.0):
    return min(upper, max(lower, float(value)))


def compute_adaptive_target_mix_share(
    lower_share,
    upper_share,
    target_reliability,
    source_affinity,
    reliability_weight=1.0,
    affinity_weight=1.0,
    scheduled_share=None,
    schedule_floor=False,
):
    """Compute target-mix share from target reliability and source affinity.

    A reliable target pseudo state should make the target semi branch more
    dominant, while a source batch that remains target-affine should still keep
    useful source-transfer mix supervision.
    """
    lower_share = _clamp(lower_share)
    upper_share = _clamp(upper_share)
    if lower_share > upper_share:
        lower_share, upper_share = upper_share, lower_share

    target_reliability = _clamp(target_reliability)
    source_affinity = _clamp(source_affinity)
    reliability_weight = max(0.0, float(reliability_weight))
    affinity_weight = max(0.0, float(affinity_weight))
    denom = reliability_weight + affinity_weight
    if denom <= 0:
        signal = 0.5
    else:
        signal = (
            reliability_weight * target_reliability
            + affinity_weight * (1.0 - source_affinity)
        ) / denom

    share = lower_share + (upper_share - lower_share) * _clamp(signal)
    if schedule_floor and scheduled_share is not None:
        share = max(share, _clamp(scheduled_share))
    return share


def compute_reliable_ratio_target_mix_share(
    base_share,
    reliable_pixel_ratio,
    progress=1.0,
):
    """Interpolate from balanced mixing using gated target reliability.

    ``reliable_pixel_ratio`` is the fraction of valid target pixels whose
    teacher confidence passes the training pseudo-label threshold. ``progress``
    optionally delays its influence over training; its default preserves the
    original adaptive-TDM behavior.
    """
    base_share = _clamp(base_share)
    reliable_pixel_ratio = _clamp(reliable_pixel_ratio)
    progress = _clamp(progress)
    return (
        base_share
        + (1.0 - base_share) * progress * reliable_pixel_ratio
    )


def compute_bounded_residual_target_mix_share(
    scheduled_share,
    reliability,
    reference_reliability,
    max_residual=0.05,
    lower_share=0.0,
    upper_share=1.0,
):
    """Apply a bounded reliability residual to a stable base schedule."""
    lower_share = _clamp(lower_share)
    upper_share = _clamp(upper_share)
    if lower_share > upper_share:
        lower_share, upper_share = upper_share, lower_share
    scheduled_share = min(
        upper_share, max(lower_share, float(scheduled_share)))
    max_residual = max(0.0, float(max_residual))
    residual = float(reliability) - float(reference_reliability)
    residual = min(max_residual, max(-max_residual, residual))
    return min(upper_share, max(lower_share, scheduled_share + residual))


def rebalance_mix_weights(source_mix_weight, target_mix_weight, target_mix_share,
                          total_weight=None):
    """Re-balance source/target mix branches while preserving total scale."""
    if total_weight is None:
        total_weight = float(source_mix_weight) + float(target_mix_weight)
    else:
        total_weight = float(total_weight)
    target_mix_share = _clamp(target_mix_share)
    return (
        total_weight * (1.0 - target_mix_share),
        total_weight * target_mix_share,
    )
