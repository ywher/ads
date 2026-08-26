"""Relative source-target calibration for direct target supervision."""

from .effective_budget_target_sup import (
    count_annotation_samples,
    effective_number,
)


def resolve_relative_target_supervision(
    target_count,
    source_count,
    beta=0.99,
    neutral_ratio=0.5,
    min_weight=0.0,
    max_weight=2.0,
):
    """Compute the constant T-Sup weight from relative effective coverage."""
    target_count = int(target_count)
    source_count = int(source_count)
    beta = float(beta)
    neutral_ratio = float(neutral_ratio)
    min_weight = float(min_weight)
    max_weight = float(max_weight)

    if target_count <= 0:
        raise ValueError('target_count must be positive')
    if source_count <= 0:
        raise ValueError('source_count must be positive')
    if not 0.0 <= beta < 1.0:
        raise ValueError('beta must be in [0, 1)')
    if neutral_ratio <= 0.0:
        raise ValueError('neutral_ratio must be positive')
    if min_weight < 0.0 or max_weight < min_weight:
        raise ValueError('weights must satisfy 0 <= min_weight <= max_weight')

    target_effective = effective_number(target_count, beta)
    source_effective = effective_number(source_count, beta)
    if source_effective <= 0.0:
        raise ValueError('source effective sample count must be positive')

    relative_coverage = target_effective / source_effective
    raw_weight = relative_coverage / neutral_ratio
    weight = min(max(raw_weight, min_weight), max_weight)
    return {
        'target_count': target_count,
        'source_count': source_count,
        'target_effective_count': target_effective,
        'source_effective_count': source_effective,
        'relative_coverage': relative_coverage,
        'raw_weight': raw_weight,
        'weight': weight,
    }


def resolve_relative_target_supervision_config(cfg):
    """Resolve the rule after CLI split overrides select the final lists."""
    data_cfg = cfg.data if hasattr(cfg, 'data') else cfg['data']
    semi_cfg = cfg.semi if hasattr(cfg, 'semi') else cfg['semi']
    ssda_cfg = semi_cfg.get('ssda', {})
    if not ssda_cfg.get('relative_target_supervision_calibration', False):
        return None

    target_override = ssda_cfg.get(
        'relative_target_sup_target_count_override', None)
    source_override = ssda_cfg.get(
        'relative_target_sup_source_count_override', None)
    target_ann = data_cfg.get('target_labeled', {}).get('im_anns', None)
    source_ann = data_cfg.get('source', {}).get('im_anns', None)

    if target_override is None:
        if not target_ann:
            raise ValueError(
                'relative T-Sup requires data[target_labeled][im_anns]')
        target_count = count_annotation_samples(target_ann)
    else:
        target_count = int(target_override)

    if source_override is None:
        if not source_ann:
            raise ValueError('relative T-Sup requires data[source][im_anns]')
        source_count = count_annotation_samples(source_ann)
    else:
        source_count = int(source_override)

    resolved = resolve_relative_target_supervision(
        target_count=target_count,
        source_count=source_count,
        beta=ssda_cfg.get('relative_target_sup_beta', 0.99),
        neutral_ratio=ssda_cfg.get(
            'relative_target_sup_neutral_ratio', 0.5),
        min_weight=ssda_cfg.get('relative_target_sup_min_weight', 0.0),
        max_weight=ssda_cfg.get('relative_target_sup_max_weight', 2.0),
    )
    ssda_cfg.update({
        'target_sup_weight': resolved['weight'],
        'target_sup_weight_final': resolved['weight'],
        'target_sup_weight_schedule': 'constant',
        'redistribute_target_sup_from_target_mix': False,
        'relative_target_sup_target_count': resolved['target_count'],
        'relative_target_sup_source_count': resolved['source_count'],
        'relative_target_sup_target_effective_count': resolved[
            'target_effective_count'],
        'relative_target_sup_source_effective_count': resolved[
            'source_effective_count'],
        'relative_target_sup_coverage': resolved['relative_coverage'],
        'relative_target_sup_raw_weight': resolved['raw_weight'],
        'relative_target_sup_resolved_weight': resolved['weight'],
        'relative_target_sup_runtime_resolved': True,
    })
    return resolved
