"""Budget-aware scheduling utilities for direct target supervision.

The target-labeled loader is sampled with replacement during a fixed-length
SSDA run.  A constant direct-supervision weight therefore reuses every labeled
image much more aggressively at small budgets.  This module turns the number
of *distinct* labeled target images into a saturating trust gate and uses the
gate to control both the final T-Sup weight and its activation time.
"""

import math
from pathlib import Path


def count_annotation_samples(annotation_path):
    """Count non-empty, non-comment records in an annotation list."""
    annotation_path = Path(annotation_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f'Target-labeled annotation file does not exist: {annotation_path}')

    with annotation_path.open('r', encoding='utf-8') as handle:
        records = {
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith('#')
        }
    return len(records)


def effective_number(sample_count, beta=0.99):
    """Return the saturating effective number ``(1-beta**n)/(1-beta)``."""
    sample_count = int(sample_count)
    beta = float(beta)
    if sample_count < 0:
        raise ValueError('sample_count must be nonnegative')
    if not 0.0 <= beta < 1.0:
        raise ValueError('beta must be in [0, 1)')
    if sample_count == 0:
        return 0.0
    if beta == 0.0:
        return 1.0
    return -math.expm1(sample_count * math.log(beta)) / (1.0 - beta)


def effective_budget_gate(sample_count, reference_samples=256, beta=0.99):
    """Map the labeled-target count to a monotonic gate in ``[0, 1]``."""
    reference_samples = int(reference_samples)
    if reference_samples <= 0:
        raise ValueError('reference_samples must be positive')
    reference_effective = effective_number(reference_samples, beta)
    if reference_effective <= 0:
        raise ValueError('reference effective sample count must be positive')
    return min(
        1.0,
        effective_number(sample_count, beta) / reference_effective,
    )


def resolve_effective_budget_target_sup(
    sample_count,
    max_iters,
    reference_samples=256,
    beta=0.99,
    max_weight=1.0,
    max_begin_fraction=0.5,
    ramp_fraction=0.5,
):
    """Resolve a unified count-aware T-Sup schedule.

    Let ``g`` be the effective-budget gate.  The final loss weight is
    ``max_weight * g``.  Its linear ramp starts at
    ``max_begin_fraction * (1-g)`` of training and lasts ``ramp_fraction`` of
    training.  Consequently, large labeled sets receive earlier/stronger
    clean supervision, while very small sets receive later/weaker supervision.
    """
    sample_count = int(sample_count)
    max_iters = int(max_iters)
    max_weight = float(max_weight)
    max_begin_fraction = float(max_begin_fraction)
    ramp_fraction = float(ramp_fraction)

    if sample_count <= 0:
        raise ValueError('sample_count must be positive for budget-aware T-Sup')
    if max_iters <= 0:
        raise ValueError('max_iters must be positive')
    if not 0.0 <= max_weight <= 1.0:
        raise ValueError('max_weight must be in [0, 1]')
    if not 0.0 <= max_begin_fraction < 1.0:
        raise ValueError('max_begin_fraction must be in [0, 1)')
    if not 0.0 < ramp_fraction <= 1.0:
        raise ValueError('ramp_fraction must be in (0, 1]')

    gate = effective_budget_gate(
        sample_count,
        reference_samples=reference_samples,
        beta=beta,
    )
    begin_iter = int(round(
        max_iters * max_begin_fraction * (1.0 - gate)
    ))
    ramp_iters = max(1, int(round(max_iters * ramp_fraction)))
    end_iter = min(max_iters, begin_iter + ramp_iters)
    if end_iter <= begin_iter:
        raise ValueError(
            f'Invalid T-Sup ramp interval: {begin_iter} -> {end_iter}')

    return {
        'sample_count': sample_count,
        'effective_count': effective_number(sample_count, beta),
        'gate': gate,
        'initial_weight': 0.0,
        'final_weight': max_weight * gate,
        'begin_iter': begin_iter,
        'end_iter': end_iter,
    }


def resolve_effective_budget_target_sup_config(cfg):
    """Resolve and persist a budget-aware T-Sup schedule in a full config.

    This must run after CLI split and max-iteration overrides, but before
    ``config.json`` is dumped, so the saved experiment config contains the
    actual labeled count and resolved schedule rather than placeholders.
    """
    data_cfg = cfg.data if hasattr(cfg, 'data') else cfg['data']
    semi_cfg = cfg.semi if hasattr(cfg, 'semi') else cfg['semi']
    ssda_cfg = semi_cfg.get('ssda', {})
    if not ssda_cfg.get('effective_budget_target_sup', False):
        return None
    max_iters = cfg.max_iters if hasattr(cfg, 'max_iters') else cfg['max_iters']
    count_override = ssda_cfg.get(
        'target_sup_budget_labeled_count_override', None)
    labeled_ann = data_cfg.get('target_labeled', {}).get('im_anns', None)
    if count_override is None:
        if not labeled_ann:
            raise ValueError(
                'effective-budget T-Sup requires '
                'data[target_labeled][im_anns]')
        sample_count = count_annotation_samples(labeled_ann)
    else:
        sample_count = int(count_override)

    unlabeled_count = None
    unlabeled_ann = data_cfg.get('target_unlabeled', {}).get('im_anns', None)
    if unlabeled_ann:
        try:
            unlabeled_count = count_annotation_samples(unlabeled_ann)
        except FileNotFoundError:
            # The labeled list is required to resolve the method.  The
            # unlabeled list is used only to record the nominal ratio.
            unlabeled_count = None

    resolved = resolve_effective_budget_target_sup(
        sample_count=sample_count,
        max_iters=max_iters,
        reference_samples=ssda_cfg.get(
            'target_sup_budget_reference_samples', 256),
        beta=ssda_cfg.get('target_sup_budget_beta', 0.99),
        max_weight=ssda_cfg.get('target_sup_budget_max_weight', 1.0),
        max_begin_fraction=ssda_cfg.get(
            'target_sup_budget_max_begin_fraction', 0.5),
        ramp_fraction=ssda_cfg.get(
            'target_sup_budget_ramp_fraction', 0.5),
    )
    total_count = (
        sample_count + unlabeled_count
        if unlabeled_count is not None
        else None
    )
    labeled_ratio = (
        sample_count / total_count
        if total_count is not None and total_count > 0
        else None
    )
    ssda_cfg.update({
        'target_sup_weight': resolved['initial_weight'],
        'target_sup_weight_final': resolved['final_weight'],
        'target_sup_weight_schedule': 'linear_between',
        'target_sup_weight_begin_iter': resolved['begin_iter'],
        'target_sup_weight_end_iter': resolved['end_iter'],
        'target_sup_budget_sample_count': sample_count,
        'target_sup_budget_effective_count': resolved['effective_count'],
        'target_sup_budget_gate': resolved['gate'],
        'target_sup_budget_labeled_ratio': labeled_ratio,
        'target_sup_budget_runtime_resolved': True,
    })
    return resolved
