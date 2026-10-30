"""Dataset split helpers shared by training and evaluation."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


IndexSplit = Tuple[np.ndarray, np.ndarray, np.ndarray]
FileSplit = Tuple[List[str], List[str], List[str], Dict[str, int]]


def build_cpsc_outer_splits(
    n_samples: int,
    n_splits: int = 5,
    seed: int = 42,
) -> List[IndexSplit]:
    """Build disjoint train/validation/test splits for CPSC2018.

    Fold k is the held-out test partition, the next partition cyclically is
    validation, and the remaining partitions are used for training.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if n_splits < 3:
        raise ValueError("CPSC outer evaluation requires at least 3 folds.")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    partitions = [test_idx for _, test_idx in kfold.split(np.arange(n_samples))]

    splits: List[IndexSplit] = []
    for fold in range(n_splits):
        test_idx = partitions[fold]
        val_idx = partitions[(fold + 1) % n_splits]
        train_idx = np.concatenate(
            [
                partitions[i]
                for i in range(n_splits)
                if i not in {fold, (fold + 1) % n_splits}
            ]
        )

        if (
            np.intersect1d(train_idx, val_idx).size
            or np.intersect1d(train_idx, test_idx).size
            or np.intersect1d(val_idx, test_idx).size
        ):
            raise RuntimeError(f"CPSC fold {fold} contains overlapping partitions.")

        splits.append((train_idx, val_idx, test_idx))

    return splits


def build_chapman_file_splits(
    metadata_path: str,
    processed_data_dir: str,
    n_splits: int = 5,
) -> List[FileSplit]:
    """Build Chapman splits from the persisted base-fold assignment."""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Chapman metadata not found: {metadata_path}")
    if not os.path.isdir(processed_data_dir):
        raise FileNotFoundError(
            f"Chapman processed data directory not found: {processed_data_dir}"
        )
    if n_splits < 3:
        raise ValueError("Chapman outer evaluation requires at least 3 folds.")

    df = pd.read_csv(metadata_path)
    required_columns = {"record_id", "class_index", "base_fold"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Chapman metadata is missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["record_id"] = df["record_id"].astype(str)
    df["class_index"] = df["class_index"].astype(int)
    df["base_fold"] = df["base_fold"].astype(int)
    if set(df["base_fold"].unique()) != set(range(n_splits)):
        raise ValueError(
            f"Chapman base_fold values must be {list(range(n_splits))}, "
            f"got {sorted(df['base_fold'].unique().tolist())}"
        )
    if df["record_id"].duplicated().any():
        raise ValueError("Chapman metadata contains duplicate record_id values.")

    available_files = {
        name for name in os.listdir(processed_data_dir) if name.endswith(".pt")
    }
    df["file_name"] = df["record_id"] + ".pt"
    missing_files = sorted(set(df["file_name"]) - available_files)
    if missing_files:
        raise FileNotFoundError(
            f"Chapman metadata references {len(missing_files)} missing .pt files; "
            f"first entries: {missing_files[:5]}"
        )

    label_map = dict(zip(df["file_name"], df["class_index"]))
    partitions = {
        fold: sorted(df.loc[df["base_fold"] == fold, "file_name"].tolist())
        for fold in range(n_splits)
    }
    splits: List[FileSplit] = []
    for fold in range(n_splits):
        test_fold = fold
        val_fold = (fold + 1) % n_splits
        train_files = sorted(
            file_name
            for partition_fold, names in partitions.items()
            if partition_fold not in {test_fold, val_fold}
            for file_name in names
        )
        val_files = partitions[val_fold]
        test_files = partitions[test_fold]
        train_set, val_set, test_set = (
            set(train_files),
            set(val_files),
            set(test_files),
        )
        if train_set & val_set or train_set & test_set or val_set & test_set:
            raise RuntimeError(f"Chapman fold {fold} contains overlapping files.")
        splits.append((train_files, val_files, test_files, label_map))
    return splits
