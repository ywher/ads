"""Target-label-count-aware source calibration for SSDA."""

from .effective_budget_target_sup import (
    count_annotation_samples,
    effective_budget_gate,
    effective_number,
)


def resolve_effective_budget_source_calibration(
        sample_count, reference_samples=256, beta=0.99,
        min_source_weight=0.2, max_target_mix_share=0.8):
    """Resolve source decay and TDM endpoints from distinct target labels."""
    min_source_weight = float(min_source_weight)
    max_target_mix_share = float(max_target_mix_share)
    if not 0.0 <= min_source_weight <= 1.0:
        raise ValueError('min_source_weight must be in [0, 1]')
    if not 0.5 <= max_target_mix_share < 1.0:
        raise ValueError('max_target_mix_share must be in [0.5, 1)')

    gate = effective_budget_gate(
        sample_count, reference_samples=reference_samples, beta=beta)
    return {
        'sample_count': int(sample_count),
        'effective_count': effective_number(sample_count, beta),
        'gate': gate,
        'source_weight_final': 1.0 - gate * (1.0 - min_source_weight),
        'target_mix_share_final': (
            0.5 + gate * (max_target_mix_share - 0.5)),
    }


def resolve_effective_budget_source_calibration_config(cfg):
    """Resolve EBSC after CLI split overrides select the labeled list."""
    data_cfg = cfg.data if hasattr(cfg, 'data') else cfg['data']
    semi_cfg = cfg.semi if hasattr(cfg, 'semi') else cfg['semi']
    ssda_cfg = semi_cfg.get('ssda', {})
    if not ssda_cfg.get('effective_budget_source_calibration', False):
        return None

    count_override = ssda_cfg.get(
        'source_calibration_labeled_count_override', None)
    labeled_ann = data_cfg.get('target_labeled', {}).get('im_anns', None)
    if count_override is None:
        if not labeled_ann:
            raise ValueError(
                'effective-budget source calibration requires '
                'data[target_labeled][im_anns]')
        sample_count = count_annotation_samples(labeled_ann)
    else:
        sample_count = int(count_override)
    if sample_count <= 0:
        raise ValueError('source calibration requires target labels')

    resolved = resolve_effective_budget_source_calibration(
        sample_count=sample_count,
        reference_samples=ssda_cfg.get(
            'source_calibration_reference_samples', 256),
        beta=ssda_cfg.get('source_calibration_budget_beta', 0.99),
        min_source_weight=ssda_cfg.get(
            'source_calibration_min_source_weight', 0.2),
        max_target_mix_share=ssda_cfg.get(
            'source_calibration_max_target_mix_share', 0.8),
    )
    ssda_cfg.update({
        'source_sup_weight': 1.0,
        'source_sup_weight_final': resolved['source_weight_final'],
        'source_sup_weight_schedule': 'linear_decay',
        'target_dominant_mix': True,
        'target_mix_share': 0.5,
        'target_mix_share_final': resolved['target_mix_share_final'],
        'target_mix_share_schedule': 'linear_decay',
        'target_dominant_mix_total_weight': 2.0,
        'source_calibration_sample_count': resolved['sample_count'],
        'source_calibration_effective_count': resolved['effective_count'],
        'source_calibration_budget_gate': resolved['gate'],
        'source_calibration_runtime_resolved': True,
    })
    return resolved
