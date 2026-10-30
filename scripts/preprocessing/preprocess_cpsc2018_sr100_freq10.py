# preprocess_cpsc2018_sr100_freq10.py
# Robust CPSC2018 preprocessing for the ECG project.
# Output tensors are compatible with DualFreqECGDataset CPSC mode:
#   x_lf.pth, x_hf.pth, labels.pth, ages.pth, sexes.pth, metadata.csv
# Signals are resampled to 100 Hz and cropped/padded to 10 s (1000 points),
# then split into low/high frequency components at 10 Hz.

import argparse
from pathlib import Path
from math import gcd

import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import torch
import torch.nn.functional as F
from tqdm import tqdm


CLASS_NAMES = ["N", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
NAME_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
# Some CPSC files / papers use AFIB rather than AF.
NAME_TO_INDEX.update({"AFIB": NAME_TO_INDEX["AF"], "NORMAL": NAME_TO_INDEX["N"]})


def read_reference_csv(csv_path: Path) -> pd.DataFrame:
    """Support headered/headerless CSV/TXT reference files."""
    # Try automatic separator first.
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path)

    expected = {"Recording", "First_label", "Second_label", "Third_label"}
    if expected.issubset(set(df.columns)):
        return df

    # Common alternatives.
    lower_cols = {str(c).strip().lower(): c for c in df.columns}
    if "recording" in lower_cols and ("label" in lower_cols or "first_label" in lower_cols):
        rename = {lower_cols["recording"]: "Recording"}
        if "first_label" in lower_cols:
            rename[lower_cols["first_label"]] = "First_label"
        else:
            rename[lower_cols["label"]] = "First_label"
        if "second_label" in lower_cols:
            rename[lower_cols["second_label"]] = "Second_label"
        if "third_label" in lower_cols:
            rename[lower_cols["third_label"]] = "Third_label"
        return df.rename(columns=rename)

    # Try headerless file. Common format: Recording, First_label, Second_label, Third_label.
    df = pd.read_csv(csv_path, header=None, sep=None, engine="python")
    if df.shape[1] < 2:
        raise ValueError(f"REFERENCE file has too few columns: {csv_path}")

    cols = ["Recording", "First_label", "Second_label", "Third_label"]
    while len(cols) < df.shape[1]:
        cols.append(f"Extra_{len(cols)}")
    df.columns = cols[: df.shape[1]]

    # Drop accidental header row if present.
    if str(df.iloc[0]["Recording"]).strip().lower() in {"recording", "record", "filename", "file", "name"}:
        df = df.iloc[1:].reset_index(drop=True)
    return df


def extract_signal_age_sex(mat_path: Path):
    """Read CPSC .mat. Compatible with the ECG struct used by the existing project."""
    m = scipy.io.loadmat(str(mat_path))

    if "ECG" in m:
        ecg = m["ECG"]
        try:
            sex_raw = ecg[0][0][0][0]
            age_raw = ecg[0][0][1][0, 0]
            signal = ecg[0][0][2]
        except Exception as e:
            raise RuntimeError(f"Failed to parse ECG struct in {mat_path}: {e}")
    else:
        # Fallback: find the first numeric 2D array that looks like ECG.
        signal = None
        age_raw = 0
        sex_raw = "Unknown"
        for k, v in m.items():
            if k.startswith("__"):
                continue
            arr = np.asarray(v)
            if arr.ndim == 2 and 12 in arr.shape:
                signal = arr
                break
        if signal is None:
            raise RuntimeError(f"Cannot find ECG signal in {mat_path}. Keys={list(m.keys())}")

    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2:
        raise RuntimeError(f"Unexpected signal ndim={signal.ndim} in {mat_path}")

    # Make shape [12, T].
    if signal.shape[0] != 12 and signal.shape[1] == 12:
        signal = signal.T
    if signal.shape[0] != 12:
        raise RuntimeError(f"Signal is not 12-lead in {mat_path}, shape={signal.shape}")

    try:
        age = float(np.asarray(age_raw).squeeze())
        if not np.isfinite(age):
            age = 0.0
    except Exception:
        age = 0.0

    sex_str = str(np.asarray(sex_raw).squeeze()).strip().lower()
    sex = 1.0 if sex_str.startswith("male") or sex_str in {"m", "1"} else 0.0
    return signal, age, sex


def crop_or_pad(signal: np.ndarray, target_len: int) -> np.ndarray:
    signal = signal[:, :target_len]
    if signal.shape[1] < target_len:
        pad = target_len - signal.shape[1]
        signal_t = torch.tensor(signal, dtype=torch.float32)
        signal_t = F.pad(signal_t, (0, pad), mode="constant")
        return signal_t.numpy()
    return signal


def resample_signal(signal: np.ndarray, raw_fs: int, target_fs: int) -> np.ndarray:
    if raw_fs == target_fs:
        return signal.astype(np.float32, copy=False)
    g = gcd(int(raw_fs), int(target_fs))
    up = int(target_fs) // g
    down = int(raw_fs) // g
    return scipy.signal.resample_poly(signal, up=up, down=down, axis=-1).astype(np.float32)


def zscore_per_lead(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)
    return ((signal - mean) / (std + eps)).astype(np.float32)


def apply_filter(signal: np.ndarray, fs: int = 100, cutoff: float = 10.0):
    nyq = 0.5 * fs
    normal_cutoff = float(cutoff) / nyq
    if not (0 < normal_cutoff < 1):
        raise ValueError(f"cutoff must be between 0 and Nyquist. fs={fs}, cutoff={cutoff}")
    b_low, a_low = scipy.signal.butter(4, normal_cutoff, btype="lowpass", analog=False)
    b_high, a_high = scipy.signal.butter(4, normal_cutoff, btype="highpass", analog=False)
    lf = scipy.signal.filtfilt(b_low, a_low, signal, axis=-1).astype(np.float32)
    hf = scipy.signal.filtfilt(b_high, a_high, signal, axis=-1).astype(np.float32)
    return lf, hf


def label_to_index(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if s == "" or s.lower() in {"nan", "none", "0", "0.0"}:
        return None
    # Numeric labels in CPSC2018 are 1-based: 1..9.
    try:
        idx = int(float(s)) - 1
        if 0 <= idx < 9:
            return idx
    except Exception:
        pass
    # String label support: N, AF, I-AVB, ...
    s_up = s.upper().replace(" ", "")
    return NAME_TO_INDEX.get(s_up)


def build_label(row: pd.Series, num_classes: int = 9) -> torch.Tensor:
    y = torch.zeros(num_classes, dtype=torch.float32)
    label_cols = [c for c in ["First_label", "Second_label", "Third_label"] if c in row.index]
    # If no standard label columns, treat all columns except Recording as possible labels.
    if not label_cols:
        label_cols = [c for c in row.index if str(c).lower() not in {"recording", "record", "filename", "file", "name"}]
    for col in label_cols:
        idx = label_to_index(row[col])
        if idx is not None and 0 <= idx < num_classes:
            y[idx] = 1.0
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True, help="Directory containing CPSC2018 .mat files")
    ap.add_argument("--reference_csv", required=True, help="Path to REFERENCE.csv / txt")
    ap.add_argument("--out_dir", required=True, help="Output processed directory")
    ap.add_argument("--raw_fs", type=int, default=500)
    ap.add_argument("--target_fs", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--cutoff", type=float, default=10.0)
    ap.add_argument("--zscore", action="store_true", help="Apply per-lead z-score before filtering. Default off to stay close to existing code.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    csv_path = Path(args.reference_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_len = int(round(args.target_fs * args.seconds))
    df = read_reference_csv(csv_path)

    required_cols = {"Recording"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"REFERENCE file missing columns: {missing_cols}. Columns={list(df.columns)}")

    x_lf, x_hf, ages, sexes, labels, meta_rows = [], [], [], [], [], []
    missing_files = []
    label_empty = []

    print("CPSC2018 preprocessing")
    print(f"raw_dir       = {raw_dir}")
    print(f"reference_csv = {csv_path}")
    print(f"out_dir       = {out_dir}")
    print(f"raw_fs        = {args.raw_fs}")
    print(f"target_fs     = {args.target_fs}")
    print(f"target_len    = {target_len}")
    print(f"cutoff        = {args.cutoff} Hz")
    print(f"zscore        = {args.zscore}")
    print(f"records in csv= {len(df)}")
    print(f"columns       = {list(df.columns)}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="preprocess"):
        rec = str(row["Recording"]).strip()
        mat_path = raw_dir / f"{rec}.mat"
        if not mat_path.exists():
            missing_files.append(str(mat_path))
            continue

        signal, age, sex = extract_signal_age_sex(mat_path)
        signal = resample_signal(signal, raw_fs=args.raw_fs, target_fs=args.target_fs)
        signal = crop_or_pad(signal, target_len)
        if args.zscore:
            signal = zscore_per_lead(signal)
        lf, hf = apply_filter(signal, fs=args.target_fs, cutoff=args.cutoff)
        y = build_label(row, num_classes=9)
        if float(y.sum().item()) <= 0:
            label_empty.append(rec)
            continue

        x_lf.append(torch.tensor(lf).unsqueeze(0))
        x_hf.append(torch.tensor(hf).unsqueeze(0))
        ages.append(torch.tensor([age], dtype=torch.float32))
        sexes.append(torch.tensor([sex], dtype=torch.float32))
        labels.append(y.unsqueeze(0))
        meta_rows.append({"Recording": rec, "age": age, "sex": sex, "mat_path": str(mat_path)})

    if not x_lf:
        raise RuntimeError("No records were processed. Check raw_dir and REFERENCE file.")

    print(f"processed records = {len(x_lf)}")
    print(f"missing files     = {len(missing_files)}")
    print(f"empty labels      = {len(label_empty)}")
    if missing_files:
        (out_dir / "missing_files.txt").write_text("\n".join(missing_files), encoding="utf-8")
        print(f"missing file list saved: {out_dir / 'missing_files.txt'}")
    if label_empty:
        (out_dir / "empty_label_records.txt").write_text("\n".join(label_empty), encoding="utf-8")
        print(f"empty-label record list saved: {out_dir / 'empty_label_records.txt'}")

    x_lf_t = torch.cat(x_lf, dim=0)
    x_hf_t = torch.cat(x_hf, dim=0)
    ages_t = torch.cat(ages, dim=0).view(-1)
    sexes_t = torch.cat(sexes, dim=0).view(-1)
    labels_t = torch.cat(labels, dim=0)

    torch.save(x_lf_t, out_dir / "x_lf.pth")
    torch.save(x_hf_t, out_dir / "x_hf.pth")
    torch.save(ages_t, out_dir / "ages.pth")
    torch.save(sexes_t, out_dir / "sexes.pth")
    torch.save(labels_t, out_dir / "labels.pth")
    pd.DataFrame(meta_rows).to_csv(out_dir / "metadata.csv", index=False, encoding="utf-8-sig")

    print("Shapes:")
    print(f"  x_lf   {tuple(x_lf_t.shape)} {x_lf_t.dtype}")
    print(f"  x_hf   {tuple(x_hf_t.shape)} {x_hf_t.dtype}")
    print(f"  labels {tuple(labels_t.shape)} {labels_t.dtype}")
    print("Class counts:")
    counts = labels_t.sum(dim=0).cpu().numpy().astype(int).tolist()
    for name, count in zip(CLASS_NAMES, counts):
        print(f"  {name:<6} {count}")

    print("Saved:")
    for name in ["x_lf.pth", "x_hf.pth", "ages.pth", "sexes.pth", "labels.pth", "metadata.csv"]:
        p = out_dir / name
        print(f"  {name:<12} {p.stat().st_size / (1024**2):.2f} MB")
    print("OK")


if __name__ == "__main__":
    main()
