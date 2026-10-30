#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast model-level smoke tests.

This script does not load ECG data. It checks whether each model can:
1) accept (x_hf, x_lf),
2) produce [B, num_classes] logits,
3) run BCEWithLogitsLoss backward without NaN/shape errors.

Use --list_models before writing experiments into a paper. It prints which
models are paper-safe generic baselines and which are internal controls only.
"""

import argparse
from types import SimpleNamespace

import torch
import torch.nn as nn

from models.model_factory import build_model_by_name
from models.baseline_registry import print_baseline_table, implemented_baseline_names
from models.official_baselines.utils import ExternalRepoError
from utils.config_parser import load_config
from utils.reproducibility import set_global_seed


def get_args():
    p = argparse.ArgumentParser(description="Smoke-test model forward/backward paths.")
    p.add_argument("--models", nargs="+", default=["resnet1d", "transformer1d"],
                   help="Ablation/model names to test. Use --list_models to inspect baseline provenance.")
    p.add_argument("--list_models", action="store_true", help="Print implemented baselines and paper-usage status, then exit.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--num_leads", type=int, default=12)
    p.add_argument("--seq_len", type=int, default=1000)
    p.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--aff_top_ratio", type=float, default=0.2)
    p.add_argument("--aff_component", default="dominant", choices=["dominant", "residual", "split"])
    return p.parse_args()


def main():
    args = get_args()
    if args.list_models:
        print_baseline_table()
        return

    set_global_seed(args.seed)
    cfg = load_config()
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    criterion = nn.BCEWithLogitsLoss()

    x_hf = torch.randn(args.batch_size, args.num_leads, args.seq_len, device=device)
    x_lf = torch.randn(args.batch_size, args.num_leads, args.seq_len, device=device)
    y = torch.randint(0, 2, (args.batch_size, args.num_classes), device=device).float()

    for name in args.models:
        print(f"\n[Smoke] {name}")
        model_args = SimpleNamespace(
            ablation=name,
            task="multilabel",
            aux_multilabel_lambda=0.0,
            aux_multilabel_head="dual",
            aff_top_ratio=args.aff_top_ratio,
            aff_component=args.aff_component,
        )
        try:
            model = build_model_by_name(model_args, cfg, args.num_classes, device)
        except ExternalRepoError as exc:
            print("  MISSING EXTERNAL SOURCE")
            print(str(exc))
            raise SystemExit(2)
        model.train()
        out = model(x_hf, x_lf)
        logits = out[0] if isinstance(out, tuple) else out
        if tuple(logits.shape) != (args.batch_size, args.num_classes):
            raise RuntimeError(f"{name}: expected logits {(args.batch_size, args.num_classes)}, got {tuple(logits.shape)}")
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: loss is not finite: {loss.item()}")
        loss.backward()
        print(f"  OK | logits={tuple(logits.shape)} | loss={loss.item():.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll requested smoke tests passed.")


if __name__ == "__main__":
    main()
