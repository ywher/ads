"""Diagnostics for feature-prototype source calibration.

The training loop owns the prototype banks. This module only turns a snapshot
of those banks into stable CSV/JSON records that can be inspected offline.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


CLASS_TABLE_FIELDS = [
    "iter",
    "class_id",
    "class_name",
    "src_valid",
    "tl_valid",
    "tu_valid",
    "target_valid",
    "src_count",
    "tl_count",
    "tu_count",
    "tu_conf_mean",
    "score_src_t",
    "score_src_tl",
    "score_src_tu",
    "score_tl_tu",
    "source_weight",
    "source_mix_select_count",
    "source_mix_select_freq",
    "target_mix_select_count",
    "target_mix_select_freq",
    "src_loss_sum",
    "src_loss_count",
    "src_loss_mean",
    "tgt_loss_sum",
    "tgt_loss_count",
    "tgt_loss_mean",
    "src_mix_loss_sum",
    "src_mix_loss_count",
    "src_mix_loss_mean",
    "tgt_mix_loss_sum",
    "tgt_mix_loss_count",
    "tgt_mix_loss_mean",
    "is_default_score",
]


def _as_cpu_float(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.detach().float().cpu()


def _as_cpu_bool(value: Optional[torch.Tensor], length: int) -> torch.Tensor:
    if value is None:
        return torch.zeros(length, dtype=torch.bool)
    return value.detach().bool().cpu()


def _as_cpu_count(value: Optional[torch.Tensor], length: int) -> torch.Tensor:
    if value is None:
        return torch.zeros(length, dtype=torch.float32)
    return value.detach().float().cpu()


def _safe_float(value: float | int | torch.Tensor | None) -> float:
    if value is None:
        return math.nan
    if torch.is_tensor(value):
        if value.numel() == 0:
            return math.nan
        value = float(value.detach().cpu().item())
    return float(value)


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.8f}"


def _pairwise_cosine(
    left: Optional[torch.Tensor],
    left_valid: torch.Tensor,
    right: Optional[torch.Tensor],
    right_valid: torch.Tensor,
    index: int,
) -> float:
    if left is None or right is None:
        return math.nan
    if not bool(left_valid[index].item()) or not bool(right_valid[index].item()):
        return math.nan
    lhs = F.normalize(left[index].float(), dim=0, eps=1e-6)
    rhs = F.normalize(right[index].float(), dim=0, eps=1e-6)
    return float(torch.dot(lhs, rhs).item())


def _coverage(valid: torch.Tensor) -> float:
    if valid.numel() == 0:
        return 0.0
    return float(valid.float().mean().item())


class FeaturePrototypeDiagnosticExporter:
    """Write feature-prototype per-class diagnostics.

    Output files:
    - ``class_table.csv`` appends one row per class per exported iteration.
    - ``iter_XXXXXX_summary.json`` stores a compact snapshot for quick reading.
    """

    def __init__(
        self,
        out_dir: str | Path,
        class_names: Optional[Iterable[str]] = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = list(class_names) if class_names is not None else None
        self.class_table_path = self.out_dir / "class_table.csv"

    def _class_name(self, class_id: int) -> str:
        if self.class_names is None or class_id >= len(self.class_names):
            return str(class_id)
        return str(self.class_names[class_id])

    def _write_rows(self, rows: list[dict[str, str]]) -> None:
        write_header = not self.class_table_path.exists()
        with self.class_table_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CLASS_TABLE_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def export(
        self,
        iteration: int,
        source_proto: Optional[torch.Tensor],
        source_valid: Optional[torch.Tensor],
        target_labeled_proto: Optional[torch.Tensor],
        target_labeled_valid: Optional[torch.Tensor],
        target_unlabeled_proto: Optional[torch.Tensor],
        target_unlabeled_valid: Optional[torch.Tensor],
        target_proto: Optional[torch.Tensor],
        target_valid: Optional[torch.Tensor],
        class_scores: Optional[torch.Tensor],
        class_weights: Optional[torch.Tensor],
        source_counts: Optional[torch.Tensor] = None,
        target_labeled_counts: Optional[torch.Tensor] = None,
        target_unlabeled_counts: Optional[torch.Tensor] = None,
        target_unlabeled_confidence: Optional[torch.Tensor] = None,
        source_mix_selected_counts: Optional[torch.Tensor] = None,
        source_mix_total_count: float = 0.0,
        target_mix_selected_counts: Optional[torch.Tensor] = None,
        target_mix_total_count: float = 0.0,
        loss_contributions: Optional[dict[str, dict[str, torch.Tensor]]] = None,
    ) -> dict[str, float | int]:
        tensors = [
            source_proto,
            target_labeled_proto,
            target_unlabeled_proto,
            target_proto,
            class_scores,
            class_weights,
            source_valid,
            target_labeled_valid,
            target_unlabeled_valid,
            target_valid,
        ]
        lengths = []
        for tensor in tensors:
            if tensor is not None and tensor.numel() > 0:
                lengths.append(int(tensor.shape[0]))
        if self.class_names is not None:
            lengths.append(len(self.class_names))
        num_classes = max(lengths) if lengths else 0

        src_proto = _as_cpu_float(source_proto)
        tl_proto = _as_cpu_float(target_labeled_proto)
        tu_proto = _as_cpu_float(target_unlabeled_proto)
        tgt_proto = _as_cpu_float(target_proto)
        scores = _as_cpu_float(class_scores)
        weights = _as_cpu_float(class_weights)

        src_valid = _as_cpu_bool(source_valid, num_classes)
        tl_valid = _as_cpu_bool(target_labeled_valid, num_classes)
        tu_valid = _as_cpu_bool(target_unlabeled_valid, num_classes)
        tgt_valid = _as_cpu_bool(target_valid, num_classes)
        src_counts = _as_cpu_count(source_counts, num_classes)
        tl_counts = _as_cpu_count(target_labeled_counts, num_classes)
        tu_counts = _as_cpu_count(target_unlabeled_counts, num_classes)
        tu_conf = _as_cpu_float(target_unlabeled_confidence)
        mix_counts = _as_cpu_count(source_mix_selected_counts, num_classes)
        mix_total = max(float(source_mix_total_count), 0.0)
        tgt_mix_counts = _as_cpu_count(target_mix_selected_counts, num_classes)
        tgt_mix_total = max(float(target_mix_total_count), 0.0)
        loss_contributions = loss_contributions or {}
        loss_stats = {}
        for branch in ("src", "tgt", "src_mix", "tgt_mix"):
            stats = loss_contributions.get(branch, {})
            loss_stats[branch] = {
                "sum": _as_cpu_count(stats.get("sum"), num_classes),
                "count": _as_cpu_count(stats.get("count"), num_classes),
            }

        rows = []
        default_count = 0
        valid_scores = []
        valid_weights = []
        valid_tu_conf = []
        for class_id in range(num_classes):
            score = (
                _safe_float(scores[class_id])
                if scores is not None and class_id < scores.shape[0]
                else math.nan
            )
            weight = (
                _safe_float(weights[class_id])
                if weights is not None and class_id < weights.shape[0]
                else math.nan
            )
            uses_default = not (
                bool(src_valid[class_id].item())
                and bool(tgt_valid[class_id].item())
            )
            if uses_default:
                default_count += 1
            else:
                valid_scores.append(score)
                valid_weights.append(weight)
            conf_mean = (
                _safe_float(tu_conf[class_id])
                if tu_conf is not None and class_id < tu_conf.shape[0]
                else math.nan
            )
            if bool(tu_valid[class_id].item()) and math.isfinite(conf_mean):
                valid_tu_conf.append(conf_mean)

            mix_count = _safe_float(mix_counts[class_id])
            mix_freq = mix_count / mix_total if mix_total > 0 else math.nan
            target_mix_count = _safe_float(tgt_mix_counts[class_id])
            target_mix_freq = (
                target_mix_count / tgt_mix_total
                if tgt_mix_total > 0 else math.nan
            )
            branch_loss_fields = {}
            for branch, stats in loss_stats.items():
                loss_sum = _safe_float(stats["sum"][class_id])
                loss_count = _safe_float(stats["count"][class_id])
                loss_mean = (
                    loss_sum / loss_count if loss_count > 0 else math.nan
                )
                branch_loss_fields.update({
                    f"{branch}_loss_sum": _format_number(loss_sum),
                    f"{branch}_loss_count": _format_number(loss_count),
                    f"{branch}_loss_mean": _format_number(loss_mean),
                })
            rows.append({
                "iter": str(int(iteration)),
                "class_id": str(class_id),
                "class_name": self._class_name(class_id),
                "src_valid": "1" if bool(src_valid[class_id].item()) else "0",
                "tl_valid": "1" if bool(tl_valid[class_id].item()) else "0",
                "tu_valid": "1" if bool(tu_valid[class_id].item()) else "0",
                "target_valid": "1" if bool(tgt_valid[class_id].item()) else "0",
                "src_count": _format_number(_safe_float(src_counts[class_id])),
                "tl_count": _format_number(_safe_float(tl_counts[class_id])),
                "tu_count": _format_number(_safe_float(tu_counts[class_id])),
                "tu_conf_mean": _format_number(conf_mean),
                "score_src_t": _format_number(score),
                "score_src_tl": _format_number(_pairwise_cosine(
                    src_proto, src_valid, tl_proto, tl_valid, class_id)),
                "score_src_tu": _format_number(_pairwise_cosine(
                    src_proto, src_valid, tu_proto, tu_valid, class_id)),
                "score_tl_tu": _format_number(_pairwise_cosine(
                    tl_proto, tl_valid, tu_proto, tu_valid, class_id)),
                "source_weight": _format_number(weight),
                "source_mix_select_count": _format_number(mix_count),
                "source_mix_select_freq": _format_number(mix_freq),
                "target_mix_select_count": _format_number(target_mix_count),
                "target_mix_select_freq": _format_number(target_mix_freq),
                **branch_loss_fields,
                "is_default_score": "1" if uses_default else "0",
            })

        self._write_rows(rows)

        summary = {
            "iter": int(iteration),
            "num_classes": int(num_classes),
            "source_coverage": _coverage(src_valid),
            "target_labeled_coverage": _coverage(tl_valid),
            "target_unlabeled_coverage": _coverage(tu_valid),
            "target_coverage": _coverage(tgt_valid),
            "default_score_class_count": int(default_count),
            "score_mean": float(sum(valid_scores) / len(valid_scores)) if valid_scores else math.nan,
            "score_min": float(min(valid_scores)) if valid_scores else math.nan,
            "score_max": float(max(valid_scores)) if valid_scores else math.nan,
            "source_weight_mean": float(sum(valid_weights) / len(valid_weights)) if valid_weights else math.nan,
            "target_unlabeled_confidence_mean": (
                float(sum(valid_tu_conf) / len(valid_tu_conf))
                if valid_tu_conf else math.nan
            ),
            "source_mix_total_count": float(mix_total),
            "target_mix_total_count": float(tgt_mix_total),
        }
        with (self.out_dir / f"iter_{int(iteration):06d}_summary.json").open("w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return summary
