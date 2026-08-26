"""Per-class diagnostics for target-calibrated source assistance.

This exporter is intentionally independent from feature-prototype banks. It is
used by target-deficit / class-route source-mix methods to answer whether a
method is actually selecting source classes according to target-domain need.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import torch


CLASS_TABLE_FIELDS = [
    "iter",
    "class_id",
    "class_name",
    "target_labeled_count",
    "target_unlabeled_count",
    "target_pseudo_conf_mean",
    "target_deficit_score",
    "route_score",
    "source_mix_score",
    "source_mix_select_count",
    "source_mix_select_freq",
    "target_loss_feedback",
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
    "tgt_mem_aux_loss_sum",
    "tgt_mem_aux_loss_count",
    "tgt_mem_aux_loss_mean",
]


def _as_cpu_float(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    return value.detach().float().cpu()


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


def _tensor_value(tensor: Optional[torch.Tensor], index: int) -> float:
    if tensor is None or index >= int(tensor.shape[0]):
        return math.nan
    return _safe_float(tensor[index])


class SourceAssistDiagnosticExporter:
    """Write source-assistance per-class diagnostics.

    Output files:
    - ``class_table.csv`` appends one row per class per exported iteration.
    - ``iter_XXXXXX_summary.json`` stores compact aggregate values.
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
        target_labeled_counts: Optional[torch.Tensor] = None,
        target_unlabeled_counts: Optional[torch.Tensor] = None,
        target_unlabeled_confidence: Optional[torch.Tensor] = None,
        target_deficit_scores: Optional[torch.Tensor] = None,
        route_scores: Optional[torch.Tensor] = None,
        source_mix_scores: Optional[torch.Tensor] = None,
        source_mix_selected_counts: Optional[torch.Tensor] = None,
        source_mix_total_count: float = 0.0,
        target_loss_feedback: Optional[torch.Tensor] = None,
        loss_contributions: Optional[dict[str, dict[str, torch.Tensor]]] = None,
    ) -> dict[str, float | int]:
        tensors = [
            target_labeled_counts,
            target_unlabeled_counts,
            target_unlabeled_confidence,
            target_deficit_scores,
            route_scores,
            source_mix_scores,
            source_mix_selected_counts,
            target_loss_feedback,
        ]
        lengths = []
        for tensor in tensors:
            if tensor is not None and tensor.numel() > 0:
                lengths.append(int(tensor.shape[0]))
        if self.class_names is not None:
            lengths.append(len(self.class_names))
        num_classes = max(lengths) if lengths else 0

        tl_counts = _as_cpu_count(target_labeled_counts, num_classes)
        tu_counts = _as_cpu_count(target_unlabeled_counts, num_classes)
        tu_conf = _as_cpu_float(target_unlabeled_confidence)
        deficit_scores = _as_cpu_float(target_deficit_scores)
        route_scores = _as_cpu_float(route_scores)
        source_mix_scores = _as_cpu_float(source_mix_scores)
        mix_counts = _as_cpu_count(source_mix_selected_counts, num_classes)
        mix_total = max(float(source_mix_total_count), 0.0)
        loss_feedback = _as_cpu_float(target_loss_feedback)
        loss_contributions = loss_contributions or {}
        loss_stats = {}
        for branch in ("src", "tgt", "src_mix", "tgt_mix", "tgt_mem_aux"):
            stats = loss_contributions.get(branch, {})
            loss_stats[branch] = {
                "sum": _as_cpu_count(stats.get("sum"), num_classes),
                "count": _as_cpu_count(stats.get("count"), num_classes),
            }

        rows = []
        valid_deficit = []
        valid_source_mix = []
        valid_conf = []
        for class_id in range(num_classes):
            mix_count = _safe_float(mix_counts[class_id])
            mix_freq = mix_count / mix_total if mix_total > 0 else math.nan
            deficit = _tensor_value(deficit_scores, class_id)
            mix_score = _tensor_value(source_mix_scores, class_id)
            conf = _tensor_value(tu_conf, class_id)
            if math.isfinite(deficit):
                valid_deficit.append(deficit)
            if math.isfinite(mix_score):
                valid_source_mix.append(mix_score)
            if math.isfinite(conf) and _safe_float(tu_counts[class_id]) > 0:
                valid_conf.append(conf)

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
                "target_labeled_count": _format_number(
                    _safe_float(tl_counts[class_id])),
                "target_unlabeled_count": _format_number(
                    _safe_float(tu_counts[class_id])),
                "target_pseudo_conf_mean": _format_number(conf),
                "target_deficit_score": _format_number(deficit),
                "route_score": _format_number(
                    _tensor_value(route_scores, class_id)),
                "source_mix_score": _format_number(mix_score),
                "source_mix_select_count": _format_number(mix_count),
                "source_mix_select_freq": _format_number(mix_freq),
                "target_loss_feedback": _format_number(
                    _tensor_value(loss_feedback, class_id)),
                **branch_loss_fields,
            })

        self._write_rows(rows)
        summary = {
            "iter": int(iteration),
            "num_classes": int(num_classes),
            "source_mix_total_count": float(mix_total),
            "target_deficit_score_mean": (
                float(sum(valid_deficit) / len(valid_deficit))
                if valid_deficit else math.nan),
            "source_mix_score_mean": (
                float(sum(valid_source_mix) / len(valid_source_mix))
                if valid_source_mix else math.nan),
            "target_pseudo_conf_mean": (
                float(sum(valid_conf) / len(valid_conf))
                if valid_conf else math.nan),
        }
        with (self.out_dir / f"iter_{int(iteration):06d}_summary.json").open(
            "w"
        ) as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return summary
