import os
import ast
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import wfdb
import torch
import numpy as np
import pandas as pd
from scipy import signal
from tqdm import tqdm

from utils.registry import get_dataset_info


SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def parse_scp_codes(x):
    """安全解析 PTB-XL 的 scp_codes 字段。"""
    if isinstance(x, dict):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return {}


def load_ptbxl_metadata(raw_dir: Path):
    """读取 ptbxl_database.csv 和 scp_statements.csv，并生成 5 大类标签。"""
    db_path = raw_dir / "ptbxl_database.csv"
    scp_path = raw_dir / "scp_statements.csv"

    if not db_path.exists():
        raise FileNotFoundError(f"找不到 ptbxl_database.csv: {db_path}")
    if not scp_path.exists():
        raise FileNotFoundError(f"找不到 scp_statements.csv: {scp_path}")

    df = pd.read_csv(db_path, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(parse_scp_codes)

    agg_df = pd.read_csv(scp_path, index_col=0)
    agg_df = agg_df[agg_df["diagnostic"] == 1]

    def aggregate_diagnostic(scp_dict):
        labels = []
        for code in scp_dict.keys():
            if code in agg_df.index:
                diagnostic_class = agg_df.loc[code, "diagnostic_class"]
                if diagnostic_class in SUPERCLASSES:
                    labels.append(diagnostic_class)
        return sorted(list(set(labels)))

    df["diagnostic_superclass"] = df["scp_codes"].apply(aggregate_diagnostic)

    # 只保留有 diagnostic superclass 的样本
    before = len(df)
    df = df[df["diagnostic_superclass"].apply(lambda x: len(x) > 0)].copy()
    after = len(df)

    print(f"PTB-XL total records in CSV: {before}")
    print(f"PTB-XL diagnostic records kept: {after}")
    print(f"Removed non-diagnostic / empty-label records: {before - after}")

    return df


def make_multihot(labels):
    """将 ['NORM', 'MI'] 转为 5 维 multi-hot 标签。"""
    y = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for label in labels:
        if label in SUPERCLASSES:
            y[SUPERCLASSES.index(label)] = 1.0
    return y


def read_wfdb_signal(raw_dir: Path, row, sampling_rate: int):
    """
    使用 PTB-XL 官方 CSV 中的 filename_lr / filename_hr 读取信号。
    注意：wfdb.rdsamp 传入的是不带 .hea / .dat 后缀的 record path。
    """
    if sampling_rate == 100:
        rel_path = row["filename_lr"]
    elif sampling_rate == 500:
        rel_path = row["filename_hr"]
    else:
        raise ValueError("sampling_rate 只能是 100 或 500。")

    record_path = raw_dir / rel_path

    signal_data, meta = wfdb.rdsamp(str(record_path))
    signal_data = np.asarray(signal_data, dtype=np.float32)

    # PTB-XL 正常是 [T, 12]
    if signal_data.ndim != 2:
        raise ValueError(f"信号维度异常: {record_path}, shape={signal_data.shape}")

    if signal_data.shape[1] != 12:
        if signal_data.shape[0] == 12:
            signal_data = signal_data.T
        else:
            raise ValueError(f"导联数不是 12: {record_path}, shape={signal_data.shape}")

    signal_data = np.nan_to_num(signal_data, nan=0.0, posinf=0.0, neginf=0.0)
    return signal_data, str(rel_path)


def butterworth_split(signal_data, fs: int, cutoff: float = 10.0, order: int = 4):
    """
    将 ECG 分为低频 LF 和高频残差 HF。
    输入 signal_data: [T, 12]
    输出 hf, lf: [12, T]
    """
    if cutoff <= 0:
        raise ValueError("cutoff 必须大于 0。")
    if cutoff >= fs / 2:
        raise ValueError(f"cutoff={cutoff} 必须小于 Nyquist 频率 fs/2={fs/2}。")

    normal_cutoff = cutoff / (0.5 * fs)
    sos = signal.butter(order, normal_cutoff, btype="lowpass", output="sos")

    lf = signal.sosfiltfilt(sos, signal_data, axis=0)
    hf = signal_data - lf

    # 转为模型习惯格式：[lead, time]
    hf = hf.T.astype(np.float32)
    lf = lf.T.astype(np.float32)
    return hf, lf


def fix_length(x, target_length: int):
    """
    可选长度对齐。
    x: [12, T]
    target_length <= 0 时不处理。
    """
    if target_length is None or target_length <= 0:
        return x

    current_len = x.shape[1]

    if current_len == target_length:
        return x

    if current_len > target_length:
        # 中心裁剪，避免总是只取开头
        start = (current_len - target_length) // 2
        return x[:, start:start + target_length]

    # 不足则尾部补 0
    pad_width = target_length - current_len
    return np.pad(x, ((0, 0), (0, pad_width)), mode="constant")


def preprocess_ptbxl(
    raw_dir,
    output_dir,
    sampling_rate=100,
    cutoff=10.0,
    filter_order=4,
    target_length=0,
    overwrite=False,
):
    raw_dir = Path(raw_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PTB-XL preprocessing started")
    print(f"Raw dir       : {raw_dir}")
    print(f"Output dir    : {output_dir}")
    print(f"Sampling rate : {sampling_rate} Hz")
    print(f"Cutoff        : {cutoff} Hz")
    print(f"Filter order  : {filter_order}")
    print(f"Target length : {target_length if target_length and target_length > 0 else 'keep original'}")
    print(f"Overwrite     : {overwrite}")
    print("=" * 80)

    df = load_ptbxl_metadata(raw_dir)

    filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    if filename_col not in df.columns:
        raise KeyError(f"CSV 中找不到 {filename_col} 字段。")

    manifest_rows = []
    failed_rows = []

    for ecg_id, row in tqdm(df.iterrows(), total=len(df), desc="Processing PTB-XL"):
        save_name = f"{int(ecg_id):05d}.pt"
        save_path = output_dir / save_name

        if save_path.exists() and not overwrite:
            manifest_rows.append({
                "ecg_id": int(ecg_id),
                "file": save_name,
                "status": "exists_skipped",
                "strat_fold": row.get("strat_fold", None),
                "patient_id": row.get("patient_id", None),
                "diagnostic_superclass": "|".join(row["diagnostic_superclass"]),
                "source_record": row[filename_col],
            })
            continue

        try:
            raw_signal, source_record = read_wfdb_signal(raw_dir, row, sampling_rate=sampling_rate)

            hf, lf = butterworth_split(
                raw_signal,
                fs=sampling_rate,
                cutoff=cutoff,
                order=filter_order,
            )

            hf = fix_length(hf, target_length)
            lf = fix_length(lf, target_length)

            label = make_multihot(row["diagnostic_superclass"])

            torch.save(
                {
                    "hf": torch.tensor(hf, dtype=torch.float32),
                    "lf": torch.tensor(lf, dtype=torch.float32),
                    "label": torch.tensor(label, dtype=torch.float32),

                    # 下面这些字段不影响当前 Dataset 读取，但方便以后审计和防泄露
                    "ecg_id": int(ecg_id),
                    "patient_id": int(row["patient_id"]) if "patient_id" in row and not pd.isna(row["patient_id"]) else -1,
                    "strat_fold": int(row["strat_fold"]) if "strat_fold" in row and not pd.isna(row["strat_fold"]) else -1,
                    "source_record": source_record,
                    "sampling_rate": sampling_rate,
                    "cutoff": float(cutoff),
                    "classes": SUPERCLASSES,
                },
                save_path,
            )

            manifest_rows.append({
                "ecg_id": int(ecg_id),
                "file": save_name,
                "status": "processed",
                "strat_fold": row.get("strat_fold", None),
                "patient_id": row.get("patient_id", None),
                "diagnostic_superclass": "|".join(row["diagnostic_superclass"]),
                "source_record": source_record,
            })

        except Exception as e:
            failed_rows.append({
                "ecg_id": int(ecg_id),
                "source_record": row.get(filename_col, ""),
                "error": repr(e),
            })

    manifest = pd.DataFrame(manifest_rows)
    failed = pd.DataFrame(failed_rows)

    manifest_path = output_dir / "ptbxl_manifest.csv"
    failed_path = output_dir / "ptbxl_failed.csv"

    manifest.to_csv(manifest_path, index=False)
    failed.to_csv(failed_path, index=False)

    print("\n" + "=" * 80)
    print("PTB-XL preprocessing finished")
    print(f"Expected diagnostic records : {len(df)}")
    print(f"Manifest rows               : {len(manifest)}")
    print(f"Failed records              : {len(failed)}")
    print(f"Saved manifest              : {manifest_path}")
    print(f"Saved failed log            : {failed_path}")

    processed_files = sorted([p for p in output_dir.glob("*.pt")])
    print(f"Actual .pt files in output  : {len(processed_files)}")
    print("=" * 80)

    if len(failed) > 0:
        print("\n⚠️ 有样本处理失败，请查看：")
        print(failed_path)


def main():
    parser = argparse.ArgumentParser(description="Robust PTB-XL preprocessing for DualFreqDGNN-ST")

    parser.add_argument(
        "--raw_dir",
        type=str,
        default=None,
        help="PTB-XL 原始目录，即包含 ptbxl_database.csv / records100 / records500 的目录。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出 processed 目录。",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=100,
        choices=[100, 500],
        help="使用 PTB-XL 的 records100 还是 records500。",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="低通滤波截止频率，默认 10 Hz。",
    )
    parser.add_argument(
        "--filter_order",
        type=int,
        default=4,
        help="Butterworth 滤波器阶数。",
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=0,
        help="是否强制统一长度；0 表示保持原始长度。records100 通常为 1000，records500 通常为 5000。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出文件已存在，是否覆盖。",
    )

    args = parser.parse_args()

    ds_info = get_dataset_info("PTBXL")

    raw_dir = args.raw_dir if args.raw_dir else ds_info["raw_data_dir"]

    if args.output_dir:
        output_dir = args.output_dir
    else:
        # 默认不要直接覆盖原 processed，防止把旧实验数据冲掉
        parent_dir = Path(ds_info["default_data_dir"]).parent
        output_dir = parent_dir / f"processed_sr{args.sampling_rate}_freq{int(args.cutoff)}"

    preprocess_ptbxl(
        raw_dir=raw_dir,
        output_dir=output_dir,
        sampling_rate=args.sampling_rate,
        cutoff=args.cutoff,
        filter_order=args.filter_order,
        target_length=args.target_length,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()