#!/usr/bin/env python3
"""Append verified CPSC runner metrics to the experiment log."""

import argparse
import re
from datetime import date, datetime
from pathlib import Path


NUMBER = r"([0-9]+(?:\.[0-9]+)?)"


def capture(pattern, text, label, group=1):
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not find {label} in evaluation log.")
    return match.group(group)


def build_entry(args):
    run_dir = Path(args.run_dir)
    eval_log = run_dir / f"eval_{args.model}.log"
    train_log = run_dir / f"train_{args.model}.log"
    if not eval_log.is_file():
        raise FileNotFoundError(f"Evaluation log not found: {eval_log}")

    text = eval_log.read_text(encoding="utf-8", errors="replace")
    fold_ids = sorted(
        {int(value) for value in re.findall(r"正在推理 Fold\s+(\d+)", text)}
    )
    expected_ids = list(range(args.start_fold, args.end_fold))
    if fold_ids != expected_ids:
        raise ValueError(
            f"Expected evaluation folds {expected_ids}, found {fold_ids}; "
            "refusing to record an incomplete run."
        )

    fold_mean_label = "Five-fold mean" if len(fold_ids) == 5 else "Selected-fold mean"
    metrics = {
        f"{fold_mean_label} Macro AUROC": capture(
            rf"【Macro AUROC\s*】:\s*{NUMBER}", text, "fold-mean Macro AUROC"
        ),
        f"{fold_mean_label} Macro AUPRC": capture(
            rf"【Macro AUPRC\s*】:\s*{NUMBER}", text, "fold-mean Macro AUPRC"
        ),
        f"{fold_mean_label} Macro F1": capture(
            rf"【Macro Avg】:\s*P:\s*{NUMBER}\s*\|\s*R:\s*{NUMBER}\s*\|\s*F1:\s*{NUMBER}",
            text,
            "fold-mean Macro F1",
            group=3,
        ),
        "Pooled Macro AUROC": capture(
            rf"【Pooled Macro AUROC】:\s*{NUMBER}", text, "pooled Macro AUROC"
        ),
        "Pooled Macro AUPRC": capture(
            rf"【Pooled Macro AUPRC】:\s*{NUMBER}", text, "pooled Macro AUPRC"
        ),
        "Pooled Macro F1": capture(
            rf"【Pooled Macro Avg】:\s*P:\s*{NUMBER}\s*\|\s*R:\s*{NUMBER}\s*\|\s*F1:\s*{NUMBER}",
            text,
            "pooled Macro F1",
            group=3,
        ),
        "Micro AUROC": capture(
            rf"【Micro AUROC\s*】:\s*{NUMBER}", text, "Micro AUROC"
        ),
        "Micro AUPRC": capture(
            rf"【Micro AUPRC\s*】:\s*{NUMBER}", text, "Micro AUPRC"
        ),
    }

    roc_match = re.search(r"保存至:\s*(\S+)", text)
    roc_path = roc_match.group(1) if roc_match else "not reported"
    run_name = run_dir.name
    date_match = re.search(r"logs_cpsc2018_(\d{8})_", run_name)
    run_date = (
        datetime.strptime(date_match.group(1), "%Y%m%d").date()
        if date_match
        else date.today()
    )
    train_ref = train_log.as_posix()
    eval_ref = eval_log.as_posix()

    lines = [
        f"### CPSC2018 `{args.model}` (auto-recorded)",
        "",
        f"- Date: {run_date.isoformat()}",
        f"- Status: complete; evaluated folds {fold_ids}",
        (
            f"- Settings: epochs={args.epochs}, patience={args.patience}, "
            f"grad_clip={args.grad_clip}, batch_size={args.batch_size or 'config default'}"
        ),
        f"- Training log: `{train_ref}`",
        f"- Evaluation log: `{eval_ref}`",
        f"- ROC figure: `{roc_path}`",
        "- Metrics:",
    ]
    lines.extend(f"  - {name}: `{value}`" for name, value in metrics.items())
    lines.extend(
        [
            "",
            "  These values were extracted from the completed evaluation log.",
        ]
    )
    return "\n".join(lines), eval_ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--start-fold", type=int, required=True)
    parser.add_argument("--end-fold", type=int, required=True)
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--patience", required=True)
    parser.add_argument("--grad-clip", required=True)
    parser.add_argument("--batch-size", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    entry, eval_ref = build_entry(args)
    if args.dry_run:
        print(entry)
        return

    log_path = Path(__file__).resolve().parents[1] / "EXPERIMENT_LOG.md"
    existing = log_path.read_text(encoding="utf-8")
    if eval_ref in existing:
        print(f"Already recorded: {eval_ref}")
        return

    section_header = "## Automatically Captured Runs"
    if section_header not in existing:
        existing = existing.rstrip() + f"\n\n{section_header}\n\n"
    else:
        existing = existing.rstrip() + "\n\n"
    log_path.write_text(existing + entry + "\n", encoding="utf-8")
    print(f"Recorded experiment in {log_path}")


if __name__ == "__main__":
    main()
