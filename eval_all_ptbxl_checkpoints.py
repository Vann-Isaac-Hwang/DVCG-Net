#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluator for PTB-XL BCE checkpoints produced by this project.

Usage example, run from the project root:

python eval_all_ptbxl_checkpoints.py \
  --weights_dir /path/to/PTBXL_BCE_Weights \
  --data_dir /path/to/PTBXL_Data/processed_sr100_freq10 \
  --threshold_mode val_calibrated \
  --kind best

Optional:
  --kind all             evaluate best + last + final_candidate groups
  --kind best,last       evaluate multiple kinds
  --include_incomplete   also evaluate groups that do not have all 5 folds, e.g. aff0p25 fold0 only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

# Import project code. This script is intended to be run from the project root.
from data_process.dataset import DualFreqECGDataset
from main_eval import (
    apply_thresholds,
    build_model,
    get_ptbxl_val_test_lists,
    run_inference,
    search_best_thresholds,
)
from utils.config_parser import load_config
from utils.registry import get_dataset_info


KNOWN_ABLATIONS = [
    "resnet_fusion_wo_cl",
    "old_dual_no_freq_split",
    "old_dual_wo_dgnn",
    "old_dual_wo_cl",
    "old_dual_lf_only",
    "old_dual_hf_only",
    "no_freq_split_aff",
    "resnet_fusion",
    "no_freq_split",
    "internal_fourier_graph1d",
    "internal_patch_transformer1d",
    "official_medformer",
    "official_patchtst",
    "official_itransformer",
    "official_fedformer",
    "official_fouriergnn",
    "official_medgnn",
    "fouriergnn1d",
    "transformer1d",
    "patchtst1d",
    "simclr1d",
    "resnet1d",
    "wo_supcon",
    "wo_edgedrop",
    "old_dual",
    "lf_only",
    "hf_only",
    "wo_dgnn",
    "wo_tcn",
    "wo_cl",
    "full",
]
KNOWN_ABLATIONS = sorted(KNOWN_ABLATIONS, key=len, reverse=True)

COMP_SHORT_TO_LONG = {
    "dom": "dominant",
    "res": "residual",
    "split": "split",
}


@dataclass(frozen=True)
class CheckpointInfo:
    kind: str                 # best / last / final_candidate
    dataset: str              # PTBXL
    loss: str                 # BCE
    spec: str                 # e.g. aff0p2_dom_no_freq_split_aff
    ablation: str             # e.g. no_freq_split_aff
    fold: int
    path: Path
    aff_top_ratio: float = 0.25
    aff_component: str = "dominant"
    ce_weight_mode: str = "balanced"
    aux_multilabel_lambda: float = 0.0
    aux_multilabel_head: str = "dual"

    @property
    def group_key(self) -> Tuple[str, str, str, float, str]:
        return (self.kind, self.loss, self.ablation, self.aff_top_ratio, self.aff_component)

    @property
    def display_name(self) -> str:
        if self.ablation == "no_freq_split_aff":
            ratio = str(self.aff_top_ratio).replace(".", "p")
            comp = {"dominant": "dom", "residual": "res", "split": "split"}[self.aff_component]
            return f"{self.kind}: {self.ablation}, aff{ratio}_{comp}"
        return f"{self.kind}: {self.ablation}"


def _parse_float_tag(text: str) -> float:
    # aff0p2 -> 0.2, aff0p25 -> 0.25
    return float(text.replace("p", "."))


def parse_spec(spec: str) -> Optional[Tuple[str, float, str]]:
    """Return (ablation, aff_top_ratio, aff_component)."""
    ablation = None
    for cand in KNOWN_ABLATIONS:
        if spec == cand or spec.endswith("_" + cand):
            ablation = cand
            break
    if ablation is None:
        return None

    aff_top_ratio = 0.25
    aff_component = "dominant"

    if ablation == "no_freq_split_aff":
        prefix = spec[: -len(ablation)].rstrip("_")
        # Expected prefix: aff0p2_dom / aff0p25_res / aff0p2_split
        m = re.search(r"aff(?P<ratio>\d+p?\d*)_(?P<comp>dom|res|split)$", prefix)
        if m:
            aff_top_ratio = _parse_float_tag(m.group("ratio"))
            aff_component = COMP_SHORT_TO_LONG[m.group("comp")]
        else:
            print(f"[WARN] Cannot parse AFF tag from spec={spec!r}; use defaults aff0p25_dominant")

    return ablation, aff_top_ratio, aff_component


def scan_checkpoints(weights_dir: Path) -> List[CheckpointInfo]:
    best_last_pat = re.compile(
        r"^(?P<kind>best|last)_dual_dgnn_(?P<dataset>ptbxl)_(?P<loss>bce)_(?P<spec>.+)_fold_(?P<fold>\d+)\.pth$",
        re.IGNORECASE,
    )
    final_pat = re.compile(
        r"^(?P<kind>final_candidate)_(?P<dataset>ptbxl)_(?P<loss>bce)_(?P<spec>.+)_fold(?P<fold>\d+)_.*\.pth$",
        re.IGNORECASE,
    )

    infos: List[CheckpointInfo] = []
    for path in sorted(weights_dir.glob("*.pth")):
        m = best_last_pat.match(path.name) or final_pat.match(path.name)
        if not m:
            continue
        parsed = parse_spec(m.group("spec"))
        if parsed is None:
            print(f"[WARN] Skip unrecognized checkpoint spec: {path.name}")
            continue
        ablation, aff_top_ratio, aff_component = parsed
        infos.append(
            CheckpointInfo(
                kind=m.group("kind").lower(),
                dataset=m.group("dataset").upper(),
                loss=m.group("loss").upper(),
                spec=m.group("spec"),
                ablation=ablation,
                fold=int(m.group("fold")),
                path=path,
                aff_top_ratio=aff_top_ratio,
                aff_component=aff_component,
            )
        )
    return infos


def resolve_ptbxl_csv_path(data_dir: str) -> str:
    ds_info = get_dataset_info("PTBXL")
    csv_path = ds_info.get("csv_path")
    if csv_path and os.path.exists(csv_path):
        return csv_path

    ptbxl_data_dir = os.path.dirname(data_dir.rstrip("/\\"))
    candidates = [
        os.path.join(ptbxl_data_dir, "ptbxl_database.csv"),
        os.path.join(
            ptbxl_data_dir,
            "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
            "ptbxl_database.csv",
        ),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Cannot locate PTB-XL ptbxl_database.csv; pass a correct --data_dir or set utils/registry.py paths.")


def make_model_args(info: CheckpointInfo, threshold_mode: str, fixed_threshold: float) -> SimpleNamespace:
    return SimpleNamespace(
        dataset="PTBXL",
        loss=info.loss,
        k_folds=5,
        start_fold=0,
        end_fold=None,
        task="multilabel",
        single_label_policy="max_scp_score",
        ce_weight_mode=info.ce_weight_mode,
        aux_multilabel_lambda=info.aux_multilabel_lambda,
        aux_multilabel_head=info.aux_multilabel_head,
        aff_top_ratio=info.aff_top_ratio,
        aff_component=info.aff_component,
        ablation=info.ablation,
        data_dir=None,
        threshold_mode=threshold_mode,
        fixed_threshold=fixed_threshold,
    )


def safe_metric(func, *args, default=np.nan, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        print(f"[WARN] Metric failed: {exc}")
        return default


METRIC_KEYS = [
    "macro_auroc",
    "micro_auroc",
    "weighted_auroc",
    "macro_auprc",
    "micro_auprc",
    "weighted_auprc",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "subset_accuracy",
]


def compute_multilabel_metrics(trues: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """Compute the multi-label metrics used throughout the paper experiments."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        trues, preds, average="macro", zero_division=0
    )
    return {
        "macro_auroc": safe_metric(roc_auc_score, trues, probs, average="macro"),
        "micro_auroc": safe_metric(roc_auc_score, trues, probs, average="micro"),
        "weighted_auroc": safe_metric(roc_auc_score, trues, probs, average="weighted"),
        "macro_auprc": safe_metric(average_precision_score, trues, probs, average="macro"),
        "micro_auprc": safe_metric(average_precision_score, trues, probs, average="micro"),
        "weighted_auprc": safe_metric(average_precision_score, trues, probs, average="weighted"),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "subset_accuracy": safe_metric(accuracy_score, trues, preds),
    }


def summarize_fold_metrics(fold_rows: List[Dict]) -> Dict[str, float]:
    """Return mean/std across evaluated folds, ignoring NaN values."""
    summary: Dict[str, float] = {}
    for key in METRIC_KEYS:
        vals = np.asarray([row.get(key, np.nan) for row in fold_rows], dtype=float)
        valid = vals[~np.isnan(vals)]
        if valid.size == 0:
            summary[f"{key}_mean"] = np.nan
            summary[f"{key}_std"] = np.nan
        else:
            summary[f"{key}_mean"] = float(np.mean(valid))
            summary[f"{key}_std"] = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
    return summary


def evaluate_group(
    group_infos: List[CheckpointInfo],
    data_dir: str,
    threshold_mode: str,
    fixed_threshold: float,
    batch_size: Optional[int],
    num_workers: int,
    device_override: Optional[str],
) -> Tuple[Dict, List[Dict], List[Dict]]:
    cfg = load_config()
    if batch_size is not None:
        cfg.train.batch_size = int(batch_size)

    device_name = device_override if device_override else (cfg.env.device if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    ds_info = get_dataset_info("PTBXL")
    num_classes = ds_info["num_classes"]
    classes = ds_info["class_names"]
    csv_path = resolve_ptbxl_csv_path(data_dir)

    group_infos = sorted(group_infos, key=lambda x: x.fold)
    first = group_infos[0]
    args_ns = make_model_args(first, threshold_mode=threshold_mode, fixed_threshold=fixed_threshold)

    all_probs: List[torch.Tensor] = []
    all_trues: List[torch.Tensor] = []
    all_preds: List[torch.Tensor] = []
    evaluated_folds: List[int] = []
    fold_rows: List[Dict] = []

    print("\n" + "=" * 88)
    print(f"Evaluating {first.display_name} | folds={[i.fold for i in group_infos]} | threshold={threshold_mode}")
    print("=" * 88)

    for info in group_infos:
        print(f"\n[Fold {info.fold}] {info.path.name}")
        val_files, test_files, single_label_map = get_ptbxl_val_test_lists(
            csv_path,
            data_dir,
            info.fold,
            task="multilabel",
            single_label_policy="max_scp_score",
        )
        val_dataset = DualFreqECGDataset(data_dir, file_list=val_files, label_mode="multilabel", single_label_map=single_label_map)
        test_dataset = DualFreqECGDataset(data_dir, file_list=test_files, label_mode="multilabel", single_label_map=single_label_map)

        val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False, num_workers=num_workers)

        model = build_model(args_ns, cfg, num_classes, device)
        checkpoint = torch.load(info.path, map_location=device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)

        if threshold_mode == "val_calibrated":
            val_probs, val_trues = run_inference(model, val_loader, device, task="multilabel")
            fold_thresholds = search_best_thresholds(val_probs, val_trues, classes, threshold_step=0.01)
        elif threshold_mode == "fixed":
            fold_thresholds = torch.full((num_classes,), fixed_threshold, device=device)
            print(f"Use fixed threshold = {fixed_threshold}")
        elif threshold_mode == "test_optimal":
            fold_thresholds = None
            print("[WARN] test_optimal uses test labels to choose thresholds; do not report it as a formal paper result.")
        else:
            raise ValueError(f"Unknown threshold_mode: {threshold_mode}")

        test_probs, test_trues = run_inference(model, test_loader, device, task="multilabel")
        if threshold_mode == "test_optimal":
            fold_thresholds = search_best_thresholds(test_probs, test_trues, classes, threshold_step=0.01)
        test_preds = apply_thresholds(test_probs, fold_thresholds)

        fold_probs_np = test_probs.detach().cpu().numpy()
        fold_trues_np = test_trues.detach().cpu().numpy()
        fold_preds_np = test_preds.detach().cpu().numpy()
        fold_metric_row = {
            "name": first.display_name,
            "kind": first.kind,
            "loss": first.loss,
            "ablation": first.ablation,
            "aff_top_ratio": first.aff_top_ratio if first.ablation == "no_freq_split_aff" else "",
            "aff_component": first.aff_component if first.ablation == "no_freq_split_aff" else "",
            "fold": info.fold,
            "n_samples": int(fold_trues_np.shape[0]),
            "threshold_mode": threshold_mode,
        }
        fold_metric_row.update(compute_multilabel_metrics(fold_trues_np, fold_preds_np, fold_probs_np))
        fold_rows.append(fold_metric_row)
        print(
            f"  Fold {info.fold} metrics: "
            f"Macro AUROC={fold_metric_row['macro_auroc']:.4f} | "
            f"Macro AUPRC={fold_metric_row['macro_auprc']:.4f} | "
            f"Macro F1={fold_metric_row['macro_f1']:.4f}"
        )

        all_probs.append(test_probs.detach().cpu())
        all_trues.append(test_trues.detach().cpu())
        all_preds.append(test_preds.detach().cpu())
        evaluated_folds.append(info.fold)

        del model, val_loader, test_loader, val_dataset, test_dataset
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    probs = torch.cat(all_probs, dim=0).numpy()
    trues = torch.cat(all_trues, dim=0).numpy()
    preds = torch.cat(all_preds, dim=0).numpy()

    row = {
        "name": first.display_name,
        "kind": first.kind,
        "loss": first.loss,
        "ablation": first.ablation,
        "aff_top_ratio": first.aff_top_ratio if first.ablation == "no_freq_split_aff" else "",
        "aff_component": first.aff_component if first.ablation == "no_freq_split_aff" else "",
        "folds": ",".join(map(str, evaluated_folds)),
        "n_folds": len(evaluated_folds),
        "n_samples": int(trues.shape[0]),
        "threshold_mode": threshold_mode,
    }
    row.update(compute_multilabel_metrics(trues, preds, probs))
    row.update(summarize_fold_metrics(fold_rows))

    per_class_rows: List[Dict] = []
    for c, cls_name in enumerate(classes):
        y_true = trues[:, c]
        y_prob = probs[:, c]
        y_pred = preds[:, c]
        pc_p, pc_r, pc_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        pc = {
            "name": first.display_name,
            "class": cls_name,
            "support": int(np.sum(y_true)),
            "precision": float(pc_p),
            "recall": float(pc_r),
            "f1": float(pc_f1),
            "auroc": np.nan,
            "auprc": np.nan,
        }
        if len(np.unique(y_true)) >= 2:
            pc["auroc"] = safe_metric(roc_auc_score, y_true, y_prob)
            pc["auprc"] = safe_metric(average_precision_score, y_true, y_prob)
        per_class_rows.append(pc)

    print("\nSummary:")
    print(
        f"  Pooled: Macro AUROC={row['macro_auroc']:.4f} | Macro AUPRC={row['macro_auprc']:.4f} | "
        f"Macro F1={row['macro_f1']:.4f} | P={row['macro_precision']:.4f} | R={row['macro_recall']:.4f}"
    )
    print(
        f"  5-fold: Macro AUROC={row['macro_auroc_mean']:.4f}±{row['macro_auroc_std']:.4f} | "
        f"Macro AUPRC={row['macro_auprc_mean']:.4f}±{row['macro_auprc_std']:.4f} | "
        f"Macro F1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f}"
    )
    return row, per_class_rows, fold_rows

def parse_kind_filter(kind_arg: str) -> set:
    kind_arg = kind_arg.strip().lower()
    if kind_arg == "all":
        return {"best", "last", "final_candidate"}
    return {k.strip() for k in kind_arg.split(",") if k.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch eval PTB-XL BCE checkpoints and save summary CSV.")
    parser.add_argument("--weights_dir", required=True, help="Directory containing PTBXL_BCE_Weights/*.pth")
    parser.add_argument("--data_dir", default=None, help="PTB-XL processed data dir; defaults to registry.py PTBXL default_data_dir")
    parser.add_argument("--threshold_mode", default="val_calibrated", choices=["val_calibrated", "fixed", "test_optimal"])
    parser.add_argument("--fixed_threshold", type=float, default=0.5)
    parser.add_argument("--kind", default="best", help="best, last, final_candidate, comma-separated, or all")
    parser.add_argument("--include_incomplete", action="store_true", help="Also evaluate groups with fewer than 5 folds.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config.yaml train.batch_size for eval.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument("--out_dir", default="./results/eval_all", help="Where to save summary CSV files.")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir).expanduser().resolve()
    if not weights_dir.exists():
        raise FileNotFoundError(weights_dir)

    ds_info = get_dataset_info("PTBXL")
    data_dir = args.data_dir or ds_info["default_data_dir"]

    all_infos = scan_checkpoints(weights_dir)
    kind_filter = parse_kind_filter(args.kind)
    infos = [i for i in all_infos if i.kind in kind_filter]
    if not infos:
        raise RuntimeError(f"No checkpoints matched kind={args.kind!r} in {weights_dir}")

    groups: Dict[Tuple[str, str, str, float, str], List[CheckpointInfo]] = {}
    for info in infos:
        groups.setdefault(info.group_key, []).append(info)

    # Sort: best 5-fold formal models first, then partial/final/last.
    group_items = sorted(
        groups.items(),
        key=lambda kv: (
            0 if kv[0][0] == "best" else 1 if kv[0][0] == "final_candidate" else 2,
            kv[0][2],
            kv[0][3],
            kv[0][4],
        ),
    )

    rows: List[Dict] = []
    per_class_rows: List[Dict] = []
    fold_result_rows: List[Dict] = []

    for key, group_infos in group_items:
        folds = sorted(i.fold for i in group_infos)
        if (not args.include_incomplete) and len(folds) < 5:
            print(f"[SKIP] incomplete group {group_infos[0].display_name}: folds={folds}. Use --include_incomplete to eval it.")
            continue
        try:
            row, pc_rows, fold_rows = evaluate_group(
                group_infos,
                data_dir=data_dir,
                threshold_mode=args.threshold_mode,
                fixed_threshold=args.fixed_threshold,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device_override=args.device,
            )
            rows.append(row)
            per_class_rows.extend(pc_rows)
            fold_result_rows.extend(fold_rows)
        except Exception as exc:
            print(f"[ERROR] Failed group {group_infos[0].display_name}: {exc}", file=sys.stderr)
            rows.append({
                "name": group_infos[0].display_name,
                "kind": group_infos[0].kind,
                "loss": group_infos[0].loss,
                "ablation": group_infos[0].ablation,
                "aff_top_ratio": group_infos[0].aff_top_ratio if group_infos[0].ablation == "no_freq_split_aff" else "",
                "aff_component": group_infos[0].aff_component if group_infos[0].ablation == "no_freq_split_aff" else "",
                "folds": ",".join(map(str, folds)),
                "n_folds": len(folds),
                "error": str(exc),
            })

    if not rows:
        print("No groups evaluated.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"ptbxl_bce_eval_summary_{ts}.csv"
    perclass_path = out_dir / f"ptbxl_bce_eval_perclass_{ts}.csv"
    folds_path = out_dir / f"ptbxl_bce_eval_folds_{ts}.csv"

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_path, index=False)
    if per_class_rows:
        pd.DataFrame(per_class_rows).to_csv(perclass_path, index=False)
    if fold_result_rows:
        pd.DataFrame(fold_result_rows).to_csv(folds_path, index=False)

    print("\n" + "#" * 88)
    print("Final summary")
    print("#" * 88)
    cols = [
        "name", "folds",
        "macro_auroc", "macro_auprc", "macro_f1",
        "macro_auroc_mean", "macro_auroc_std",
        "macro_auprc_mean", "macro_auprc_std",
        "macro_f1_mean", "macro_f1_std",
        "macro_precision", "macro_recall", "error"
    ]
    show_cols = [c for c in cols if c in summary_df.columns]
    with pd.option_context("display.max_colwidth", 60, "display.width", 180):
        print(summary_df[show_cols].to_string(index=False))
    print(f"\nSaved summary:   {summary_path}")
    if per_class_rows:
        print(f"Saved per-class: {perclass_path}")
    if fold_result_rows:
        print(f"Saved per-fold:  {folds_path}")


if __name__ == "__main__":
    main()
