#!/usr/bin/env python3
"""Diagnose the iTransformer ECG classification adapter on real ECG batches.

This is deliberately a small overfit test, not a paper evaluation. A healthy
adapter should reduce BCE on a fixed subset and produce finite gradients before
any five-fold run is started.
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_process.dataset import DualFreqECGDataset
from models.model_factory import build_model_by_name
from models.official_baselines.utils import ExternalRepoError
from utils.config_parser import load_config
from utils.registry import get_dataset_info
from utils.reproducibility import set_global_seed


def get_args():
    parser = argparse.ArgumentParser(description="Real-data iTransformer overfit diagnostic")
    parser.add_argument("--dataset", choices=["PTBXL", "CPSC"], default="PTBXL")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_real_subset(args):
    info = get_dataset_info(args.dataset)
    data_dir = args.data_dir or info["default_data_dir"]

    if args.dataset == "CPSC":
        dataset = DualFreqECGDataset(data_dir)
        file_count = min(args.samples, len(dataset))
        dataset = torch.utils.data.Subset(dataset, range(file_count))
    else:
        file_list = sorted(
            name for name in os.listdir(data_dir)
            if name.endswith(".pt")
        )[: args.samples]
        dataset = DualFreqECGDataset(data_dir, file_list=file_list)

    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in {data_dir}")
    return info, dataset


def main():
    args = get_args()
    set_global_seed(args.seed, deterministic=True)
    cfg = load_config()
    info, dataset = build_real_subset(args)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    fixed_batch = next(iter(loader))
    x_hf, x_lf, y = (item.to(device) for item in fixed_batch[:3])
    y = y.float()

    model_args = SimpleNamespace(
        ablation="official_itransformer",
        task="multilabel",
        aux_multilabel_lambda=0.0,
        aux_multilabel_head="dual",
        aff_top_ratio=0.2,
        aff_component="dominant",
    )
    try:
        model = build_model_by_name(
            model_args,
            cfg,
            num_classes=info["num_classes"],
            device=device,
        )
    except ExternalRepoError as exc:
        raise SystemExit(
            "Official iTransformer source is missing. Install it under "
            "external_baselines/iTransformer or set "
            "ECG_BASELINE_ITRANSFORMER_ROOT.\n\n"
            f"{exc}"
        )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    losses = []

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x_hf, x_lf)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at step {step}: {grad_norm}")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(
                f"step={step + 1:03d} loss={losses[-1]:.6f} "
                f"grad_norm={float(grad_norm):.4f}"
            )

    print(
        f"dataset={args.dataset} samples={len(dataset)} device={device} "
        f"initial_loss={losses[0]:.6f} final_loss={losses[-1]:.6f} "
        f"reduced={losses[-1] < losses[0]}"
    )
    if losses[-1] >= losses[0]:
        raise SystemExit(
            "iTransformer did not reduce the fixed-batch loss. "
            "Inspect the adapter/source before running five-fold experiments."
        )


if __name__ == "__main__":
    main()
