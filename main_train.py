import os
import sys
import argparse
import ast
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import logging
try:
    import wandb
except ImportError:
    class _DummyWandB:
        def init(self, *args, **kwargs):
            logging.warning("wandb 未安装：使用 DummyWandB，训练不会记录到 wandb。")
            return None
        def finish(self, *args, **kwargs):
            return None
    wandb = _DummyWandB()

from data_process.dataset import DualFreqECGDataset
from utils.losses import InfoNCELoss, MultiLabelFairLoss, MultiLabelSupConLoss
from utils.config_parser import load_config
from utils.registry import get_dataset_info
from utils.reproducibility import set_global_seed, seed_worker
from utils.splits import build_chapman_file_splits, build_cpsc_outer_splits
from models.model_factory import build_model_by_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PTBXL_CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
PTBXL_CLASS_TO_IDX = {c: i for i, c in enumerate(PTBXL_CLASSES)}
PTBXL_PRIORITY = ['MI', 'STTC', 'CD', 'HYP', 'NORM']

def str2bool(value):
    """Parse common CLI boolean values for Python 3.9-compatible argparse."""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")

def get_args():
    parser = argparse.ArgumentParser(description="DualFreqDGNN 统一训练框架")
    parser.add_argument('--dataset', type=str, required=True, choices=['CPSC', 'PTBXL', 'CHAPMAN', 'PTB'], help='目标数据集')
    parser.add_argument('--loss', type=str, required=True, choices=['BCE', 'MLFL', 'CE'], help='使用的分类损失函数；single_label 推荐 CE')
    parser.add_argument('--k_folds', type=int, default=5, help='交叉验证的折数')
    parser.add_argument('--start_fold', type=int, default=0, help='从第几个 fold 开始训练，方便断点续跑')
    parser.add_argument('--end_fold', type=int, default=None, help='训练到第几个 fold 之前停止；默认等于 k_folds')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖 config.yaml 中的 train.num_epochs')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience；<=0 表示关闭')
    parser.add_argument('--min_delta', type=float, default=0.0, help='验证损失至少降低多少才算 improved')
    parser.add_argument('--lr', type=float, default=None, help='覆盖 config.yaml 中的 train.learning_rate；官方 baseline 建议先试 1e-4')
    parser.add_argument('--weight_decay', type=float, default=None, help='覆盖 config.yaml 中的 train.weight_decay')
    parser.add_argument('--batch_size', type=int, default=None, help='覆盖 config.yaml 中的 train.batch_size')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='梯度裁剪阈值；>0 时启用 clip_grad_norm_，FEDformer 可用 1.0 稳定训练')
    parser.add_argument('--deterministic', type=str2bool, default=True, help='是否强制 cuDNN 确定性算法。true 更利于严格复现；false 通常更快')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader 工作进程数；Windows 服务器建议从 4 开始')
    parser.add_argument('--save_last_every', type=int, default=1, help='每隔多少个 epoch 保存完整 last checkpoint；0 表示不保存，建议 Ours 用 5')
    parser.add_argument('--lightweight_best', type=str2bool, default=False, help='true 时 best checkpoint 只保存模型和必要元数据，不保存 Adam 状态，显著减少写盘')
    parser.add_argument('--disable_tqdm', action='store_true', help='关闭逐 batch 进度条；日志重定向时建议启用')
    parser.add_argument(
        '--resume_checkpoint',
        type=str,
        default=None,
        help='从指定 checkpoint 续训；支持单个 .pth 文件、目录，或包含 {fold} 的模板路径',
    )
    parser.add_argument('--data_dir', type=str, default=None, help='强制覆盖的数据路径 (可选)')
    parser.add_argument('--task', type=str, default='multilabel', choices=['multilabel', 'single_label'], help='multilabel: BCE/MLFL 多标签；single_label: CE 单标签多分类')
    parser.add_argument('--single_label_policy', type=str, default='max_scp_score', choices=['max_scp_score', 'unique', 'priority'], help='PTB-XL 多标签转单标签策略：max_scp_score=按 scp_codes 诊断分数最高；unique=仅保留唯一 superclass 样本；priority=按疾病优先级选主标签')
    parser.add_argument('--ce_weight_mode', type=str, default='balanced', choices=['none', 'balanced', 'effective'], help='single_label + CE 的类别权重策略：balanced=按类别频次反比加权；effective=Class-Balanced effective number；none=普通 CE')
    parser.add_argument('--ce_effective_beta', type=float, default=0.9999, help='ce_weight_mode=effective 时的 beta，一般取 0.999 或 0.9999')
    parser.add_argument('--aux_multilabel_lambda', type=float, default=0.0, help='single_label 模式下的多标签辅助 BCE 权重；0 表示关闭')
    parser.add_argument('--aux_multilabel_head', type=str, default='dual', choices=['dual', 'shared'], help='multi-label auxiliary 的头部形式：dual=独立辅助头；shared=复用主分类 logits')
    parser.add_argument('--aff_top_ratio', type=float, default=0.25, help='no_freq_split_aff 的自适应 FFT top 频率比例，建议先试 0.2 或 0.25')
    parser.add_argument('--aff_component', type=str, default='dominant', choices=['dominant', 'residual', 'split'], help='no_freq_split_aff 的频谱视图：dominant=主导频率视图；residual=残差频率视图；split=两路分别用 dominant/residual')

    parser.add_argument('--ablation', type=str, default='full', 
                        choices=['full', 'wo_supcon', 'wo_cl', 'wo_tcn', 'lf_only', 'hf_only', 'no_freq_split', 'no_freq_split_aff', 'wo_dgnn', 'wo_edgedrop', 'resnet1d', 'resnet_fusion', 'resnet_fusion_wo_cl', 'old_dual', 'old_dual_wo_cl', 'old_dual_lf_only', 'old_dual_hf_only', 'old_dual_no_freq_split', 'old_dual_wo_dgnn', 'simclr1d', 'transformer1d', 'internal_fourier_graph1d', 'internal_patch_transformer1d', 'official_medformer', 'official_patchtst', 'official_itransformer', 'official_fedformer', 'official_medgnn', 'official_fouriergnn', 'tgllnet', 'fouriergnn1d', 'patchtst1d'],
                        help='消融实验模式或基线模型。论文正式 baseline 请优先使用 official_* 外部源码适配器；tgllnet 为论文复现版；fouriergnn1d/patchtst1d 仅为旧别名，不是官方方法。')
    return parser.parse_args()

def resolve_ptbxl_csv_path(ds_info, data_dir):
    csv_path = ds_info.get("csv_path", None)

    if csv_path is not None and os.path.exists(csv_path):
        return csv_path

    ptbxl_data_dir = os.path.dirname(data_dir.rstrip('/\\'))
    candidates = [
        os.path.join(ptbxl_data_dir, "ptbxl_database.csv"),
        os.path.join(
            ptbxl_data_dir,
            "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
            "ptbxl_database.csv"
        ),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError("找不到 PTB-XL 的 ptbxl_database.csv，请检查 registry.py 或 --data_dir。")


def load_ptbxl_diagnostic_df(csv_path, single_label_policy=None):
    """
    读取 PTB-XL CSV，并过滤出具有 5 大诊断 superclass 的样本。

    single_label_policy:
      None            : 保留原 multi-label 样本。
      max_scp_score   : 按 scp_codes 中诊断语句分数聚合后选择最高 superclass。
      unique          : 仅保留唯一 diagnostic superclass 的样本。
      priority        : 多标签样本按 MI > STTC > CD > HYP > NORM 选择主标签。

    说明：初稿4.20明确是 single-label + cross-entropy，但没有公开说明 PTB-XL
    多标签样本如何归并；这里默认 max_scp_score，作为最接近“主诊断”的透明规则。
    """
    df = pd.read_csv(csv_path)

    scp_path = os.path.join(os.path.dirname(csv_path), "scp_statements.csv")
    if not os.path.exists(scp_path):
        raise FileNotFoundError(f"找不到 scp_statements.csv: {scp_path}")

    agg_df = pd.read_csv(scp_path, index_col=0)
    agg_df = agg_df[agg_df["diagnostic"] == 1]
    superclasses = set(PTBXL_CLASSES)

    def get_superclass_scores(scp_codes_str):
        try:
            scp_codes = ast.literal_eval(scp_codes_str)
        except Exception:
            return {}

        scores = {}
        for code, score in scp_codes.items():
            if code in agg_df.index:
                cls = agg_df.loc[code, "diagnostic_class"]
                if cls in superclasses:
                    try:
                        score = float(score)
                    except Exception:
                        score = 1.0
                    # 同一 superclass 可能包含多个 scp code，这里求和作为主诊断强度。
                    scores[cls] = scores.get(cls, 0.0) + score
        return scores

    def choose_single_label(scores):
        labels = sorted(scores.keys())
        if len(labels) == 0:
            return None

        if single_label_policy == 'unique':
            return labels[0] if len(labels) == 1 else None

        if single_label_policy == 'priority':
            for cls in PTBXL_PRIORITY:
                if cls in scores:
                    return cls
            return labels[0]

        # default: max_scp_score
        max_score = max(scores.values())
        candidates = [cls for cls, s in scores.items() if s == max_score]
        if len(candidates) == 1:
            return candidates[0]
        for cls in PTBXL_PRIORITY:
            if cls in candidates:
                return cls
        return sorted(candidates)[0]

    df["diagnostic_superclass_scores_eval"] = df["scp_codes"].apply(get_superclass_scores)
    df["diagnostic_superclass_eval"] = df["diagnostic_superclass_scores_eval"].apply(lambda d: sorted(d.keys()))

    before = len(df)
    df = df[df["diagnostic_superclass_eval"].apply(lambda x: len(x) > 0)].copy()

    if single_label_policy is not None:
        df["single_label_name_eval"] = df["diagnostic_superclass_scores_eval"].apply(choose_single_label)
        before_sl = len(df)
        df = df[df["single_label_name_eval"].notna()].copy()
        df["single_label_eval"] = df["single_label_name_eval"].map(PTBXL_CLASS_TO_IDX).astype(int)
        print(
            f"PTB-XL single-label policy={single_label_policy}: "
            f"kept={len(df)}, removed_multi_or_empty={before_sl - len(df)}"
        )

    after = len(df)
    print(f"PTB-XL diagnostic filter: total={before}, kept={after}, removed={before - after}")
    return df

def ecg_ids_to_files(ecg_ids):
    return {f"{int(ecg_id):05d}.pt" for ecg_id in ecg_ids}


def match_files(expected_files, processed_data_dir, split_name, fold_idx):
    all_exist_files = {
        f for f in os.listdir(processed_data_dir)
        if f.endswith(".pt")
    }

    matched_files = sorted(expected_files.intersection(all_exist_files))
    missing_files = sorted(expected_files - all_exist_files)

    print(
        f"PTB-XL Fold {fold_idx} [{split_name}]: "
        f"expected={len(expected_files)}, matched={len(matched_files)}, missing={len(missing_files)}"
    )

    if len(missing_files) > 0:
        print(f"  First missing files: {missing_files[:10]}")

    if len(matched_files) == 0:
        raise RuntimeError(f"PTB-XL Fold {fold_idx} [{split_name}] 没有匹配到任何 processed 文件。")

    return matched_files


def get_ptbxl_train_val_lists(csv_path, processed_data_dir, fold_idx, task="multilabel", single_label_policy="max_scp_score"):
    """
    与 main_eval.py 保持一致。

    fold 0 test: 1,2 ; val: 3,4 ; train: 5,6,7,8,9,10
    fold 1 test: 3,4 ; val: 5,6 ; train: 1,2,7,8,9,10
    fold 2 test: 5,6 ; val: 7,8 ; train: 1,2,3,4,9,10
    fold 3 test: 7,8 ; val: 9,10; train: 1,2,3,4,5,6
    fold 4 test: 9,10; val: 1,2 ; train: 3,4,5,6,7,8
    """
    df = load_ptbxl_diagnostic_df(csv_path, single_label_policy if task == "single_label" else None)

    test_pair_idx = fold_idx
    val_pair_idx = (fold_idx + 1) % 5

    test_folds = [test_pair_idx * 2 + 1, test_pair_idx * 2 + 2]
    val_folds = [val_pair_idx * 2 + 1, val_pair_idx * 2 + 2]
    train_folds = [
        f for f in range(1, 11)
        if f not in test_folds and f not in val_folds
    ]

    train_df = df[df["strat_fold"].isin(train_folds)].copy()
    val_df = df[df["strat_fold"].isin(val_folds)].copy()

    train_expected = ecg_ids_to_files(train_df["ecg_id"].tolist())
    val_expected = ecg_ids_to_files(val_df["ecg_id"].tolist())

    train_files = match_files(train_expected, processed_data_dir, "train", fold_idx)
    val_files = match_files(val_expected, processed_data_dir, "val", fold_idx)

    print(
        f"PTB-XL Fold {fold_idx} split summary: "
        f"train_folds={train_folds}, val_folds={val_folds}, test_folds={test_folds}"
    )

    label_map = None
    if task == "single_label":
        label_map = {f"{int(row.ecg_id):05d}.pt": int(row.single_label_eval) for row in df.itertuples()}

    return train_files, val_files, label_map

def get_rng_state():
    """
    保存随机数状态，便于后续真正断点续训时尽量复现训练过程。
    """
    rng_state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        rng_state["torch_cuda_random_state_all"] = torch.cuda.get_rng_state_all()

    return rng_state


def build_checkpoint(
    epoch,
    fold,
    model,
    optimizer,
    scheduler,
    best_val_loss,
    avg_val_loss,
    args,
    cfg,
    num_classes,
    train_size,
    val_size,
    include_optimizer=True,
    loader_generator=None,
):
    """
    构造完整 checkpoint。
    epoch 保存的是已经完成的 epoch 数，即 epoch + 1。
    """
    checkpoint_data = {
        "epoch": epoch + 1,
        "fold": fold,

        "model_state_dict": model.state_dict(),

        "best_val_loss": best_val_loss,
        "current_val_loss": avg_val_loss,

        "num_classes": num_classes,
        "train_size": train_size,
        "val_size": val_size,

        "args": vars(args),
        "config": {
            "device": cfg.env.device,
            "seed": cfg.env.seed,
            "batch_size": cfg.train.batch_size,
            "learning_rate": cfg.train.learning_rate,
            "num_epochs": cfg.train.num_epochs,
            "weight_decay": cfg.train.weight_decay,
            "hidden_dim": cfg.model.hidden_dim,
            "num_nodes": cfg.model.num_nodes,
            "edge_dropout": cfg.model.edge_dropout,
            "loss_alpha": cfg.loss.alpha,
            "temperature": cfg.loss.temperature,
        },

    }

    if include_optimizer:
        checkpoint_data["optimizer_state_dict"] = optimizer.state_dict()
        checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()
        checkpoint_data["rng_state"] = get_rng_state()
        if loader_generator is not None:
            checkpoint_data["loader_generator_state"] = loader_generator.get_state()

    return checkpoint_data


def resolve_resume_checkpoint(resume_spec, checkpoint_dir, dataset, loss_tag, ablation, fold):
    """Resolve a fold-specific resume path without silently mixing experiments."""
    if not resume_spec:
        return None

    spec = os.path.expanduser(str(resume_spec))
    if "{fold}" in spec:
        spec = spec.format(fold=fold)
    elif os.path.isdir(spec):
        spec = os.path.join(
            spec,
            f"last_dual_dgnn_{dataset.lower()}_{loss_tag}_{ablation}_fold_{fold}.pth",
        )
    elif os.path.isfile(spec):
        pass
    else:
        # A directory may not exist yet, but a path relative to the checkpoint
        # directory is still useful for short command lines.
        candidate = os.path.join(checkpoint_dir, spec)
        if os.path.isfile(candidate):
            spec = candidate

    if not os.path.isfile(spec):
        raise FileNotFoundError(f"Resume checkpoint not found for fold {fold}: {spec}")
    return spec


def restore_rng_state(checkpoint):
    """Restore all available random states from a resumable checkpoint."""
    rng_state = checkpoint.get("rng_state")
    if not rng_state:
        return

    if "python_random_state" in rng_state:
        random.setstate(rng_state["python_random_state"])
    if "numpy_random_state" in rng_state:
        np.random.set_state(rng_state["numpy_random_state"])
    if "torch_random_state" in rng_state:
        torch.set_rng_state(rng_state["torch_random_state"])
    if torch.cuda.is_available() and "torch_cuda_random_state_all" in rng_state:
        torch.cuda.set_rng_state_all(rng_state["torch_cuda_random_state_all"])


def validate_resume_checkpoint(checkpoint, args, fold, num_classes, train_size, val_size):
    """Reject checkpoints from a different fold/task before loading model state."""
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Resume checkpoint must be a project checkpoint containing model_state_dict.")

    saved_fold = checkpoint.get("fold")
    if saved_fold is not None and int(saved_fold) != int(fold):
        raise ValueError(f"Checkpoint fold={saved_fold} does not match requested fold={fold}.")

    saved_num_classes = checkpoint.get("num_classes")
    if saved_num_classes is not None and int(saved_num_classes) != int(num_classes):
        raise ValueError(
            f"Checkpoint num_classes={saved_num_classes} does not match current num_classes={num_classes}."
        )

    for key, current in {
        "train_size": train_size,
        "val_size": val_size,
    }.items():
        saved = checkpoint.get(key)
        if saved is not None and int(saved) != int(current):
            raise ValueError(f"Checkpoint {key}={saved} does not match current {key}={current}.")

    saved_args = checkpoint.get("args") or {}
    for key in ("dataset", "loss", "ablation", "task"):
        saved = saved_args.get(key)
        current = getattr(args, key)
        if saved is not None and str(saved) != str(current):
            raise ValueError(
                f"Checkpoint {key}={saved!r} does not match current {key}={current!r}."
            )


def collect_single_label_counts(dataset, num_classes):
    """
    从训练集统计 single-label 类别数量，用于加权 CrossEntropy。
    优先使用 DualFreqECGDataset.file_list + single_label_map，避免反复 torch.load。
    兼容 Subset 或其他 Dataset。
    """
    labels = []

    if hasattr(dataset, "file_list") and hasattr(dataset, "single_label_map"):
        for fname in dataset.file_list:
            if fname in dataset.single_label_map:
                labels.append(int(dataset.single_label_map[fname]))
        if len(labels) == len(dataset):
            return np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)

    # fallback：逐样本读取 label，速度慢一些，但只在初始化 loss 时执行一次。
    for i in range(len(dataset)):
        y = dataset[i][2]
        if torch.is_tensor(y):
            y = int(y.view(-1)[0].item())
        else:
            y = int(y)
        labels.append(y)

    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)


def build_ce_class_weights(dataset, num_classes, device, mode="balanced", beta=0.9999):
    """
    构建 single-label CrossEntropy 的类别权重。

    balanced:
        weight_c = N / (K * count_c)，再归一化到均值为 1。
    effective:
        Cui et al. Class-Balanced Loss 的 effective number 形式：
        weight_c = (1 - beta) / (1 - beta ** count_c)，再归一化到均值为 1。
    none:
        返回 None，相当于普通 CrossEntropy。
    """
    counts = collect_single_label_counts(dataset, num_classes).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)

    if mode == "none":
        print("Single-label CE class counts:", counts.astype(int).tolist())
        print("Single-label CE weights: None (普通 CrossEntropy)")
        return None, counts.astype(int)

    if mode == "balanced":
        weights = safe_counts.sum() / (num_classes * safe_counts)
    elif mode == "effective":
        beta = float(beta)
        weights = (1.0 - beta) / (1.0 - np.power(beta, safe_counts))
    else:
        raise ValueError(f"Unsupported ce_weight_mode: {mode}")

    # 归一化到均值为 1，避免 loss 尺度过大影响日志判断。
    weights = weights / np.mean(weights)

    print("Single-label CE class counts:", counts.astype(int).tolist())
    print(f"Single-label CE weight mode: {mode}")
    print("Single-label CE weights:", np.round(weights, 4).tolist())

    return torch.tensor(weights, dtype=torch.float32, device=device), counts.astype(int)

def main():
    args = get_args()
    cfg = load_config()
    set_global_seed(int(cfg.env.seed), deterministic=bool(args.deterministic))
    if args.epochs is not None:
        cfg.train.num_epochs = int(args.epochs)
    if args.lr is not None:
        cfg.train.learning_rate = float(args.lr)
    if args.weight_decay is not None:
        cfg.train.weight_decay = float(args.weight_decay)
    if args.batch_size is not None:
        if int(args.batch_size) <= 0:
            raise ValueError("--batch_size must be a positive integer.")
        cfg.train.batch_size = int(args.batch_size)
    
    ds_info = get_dataset_info(args.dataset)
    if args.dataset == "CHAPMAN":
        if args.task != "single_label":
            raise ValueError("CHAPMAN is a single-label task; use --task single_label.")
        if args.loss != "CE":
            raise ValueError("CHAPMAN uses CrossEntropyLoss; use --loss CE.")
    num_classes = ds_info['num_classes']
    data_dir = args.data_dir if args.data_dir else ds_info['default_data_dir']
    
    device = torch.device(cfg.env.device if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 启动训练 | 数据集: {args.dataset} | 算法: {args.loss} | 模式: {args.ablation}")
    
    wandb_config = {**vars(args), "loss_alpha": cfg.loss.alpha, "lr": cfg.train.learning_rate, "batch_size": cfg.train.batch_size}
    wandb.init(project="Dual_ECG_Project", name=f"{args.dataset}_{args.loss}_{args.ablation}", config=wandb_config)

    if args.dataset == "PTBXL":
        csv_path = resolve_ptbxl_csv_path(ds_info, data_dir)
        dataset = None
        folds_data = None
        chapman_splits = None
    elif args.dataset == "CHAPMAN":
        chapman_splits = build_chapman_file_splits(
            metadata_path=ds_info["folds_path"],
            processed_data_dir=data_dir,
            n_splits=args.k_folds,
        )
        dataset = None
        folds_data = None
    else:
        dataset = DualFreqECGDataset(data_dir)
        folds_data = build_cpsc_outer_splits(
            n_samples=len(dataset),
            n_splits=args.k_folds,
            seed=int(cfg.env.seed),
        )
        chapman_splits = None
    
    checkpoint_dir = os.path.join(cfg.env.checkpoint_dir, f"{args.dataset}_{args.loss}_Weights")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    num_folds = 5 if args.dataset in {"PTBXL", "CHAPMAN"} else args.k_folds
    start_fold = max(0, int(args.start_fold))
    end_fold = num_folds if args.end_fold is None else min(num_folds, int(args.end_fold))
    if start_fold >= end_fold:
        raise ValueError(f"Invalid fold range: start_fold={start_fold}, end_fold={end_fold}, num_folds={num_folds}")
    logging.info(f"📌 Fold range: {start_fold} -> {end_fold - 1} / total={num_folds}")

    for fold in range(start_fold, end_fold):
        # Fold-specific seed makes resumed runs (--start_fold) reproduce the same
        # model initialization and shuffled order as a full 0..4 run.
        fold_seed = int(cfg.env.seed) + int(fold)
        set_global_seed(fold_seed, deterministic=bool(args.deterministic))

        if args.dataset == "PTBXL":
            train_files, val_files, single_label_map = get_ptbxl_train_val_lists(csv_path, data_dir, fold, task=args.task, single_label_policy=args.single_label_policy)
            train_sub = DualFreqECGDataset(data_dir, file_list=train_files, label_mode=args.task, single_label_map=single_label_map)
            val_sub = DualFreqECGDataset(data_dir, file_list=val_files, label_mode=args.task, single_label_map=single_label_map)
        elif args.dataset == "CHAPMAN":
            train_files, val_files, _, single_label_map = chapman_splits[fold]
            train_sub = DualFreqECGDataset(
                data_dir,
                file_list=train_files,
                label_mode="single_label",
                single_label_map=single_label_map,
            )
            val_sub = DualFreqECGDataset(
                data_dir,
                file_list=val_files,
                label_mode="single_label",
                single_label_map=single_label_map,
            )
        else:
            train_idx, val_idx, _ = folds_data[fold]
            train_sub = Subset(dataset, train_idx)
            val_sub = Subset(dataset, val_idx)
        
        loader_generator = torch.Generator()
        loader_generator.manual_seed(fold_seed)

        num_workers = max(0, int(args.num_workers))
        pin_memory = device.type == 'cuda'
        loader_kwargs = {
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'worker_init_fn': seed_worker,
        }
        if num_workers > 0:
            # Keep workers alive across epochs and overlap small-file loading with GPU work.
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2,
            )

        train_loader = DataLoader(
            train_sub,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            generator=loader_generator,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_sub,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        non_blocking = pin_memory
        
        print(f"🧠 初始化模型结构 (模式: {args.ablation})...")
        model = build_model_by_name(args, cfg, num_classes=num_classes, device=device)
        
        criterion_cl = InfoNCELoss(temperature=cfg.loss.temperature).to(device)
        if args.task == 'single_label':
            if args.loss != 'CE':
                print(f"⚠️ single_label 模式下自动使用 CrossEntropyLoss，当前 --loss={args.loss} 仅用于 checkpoint 命名。建议正式运行用 --loss CE。")
            ce_weights, ce_counts = build_ce_class_weights(
                train_sub,
                num_classes=num_classes,
                device=device,
                mode=args.ce_weight_mode,
                beta=args.ce_effective_beta,
            )
            criterion_cls = nn.CrossEntropyLoss(weight=ce_weights).to(device)
        elif args.loss == 'BCE':
            criterion_cls = nn.BCEWithLogitsLoss().to(device) 
        elif args.loss == 'MLFL':
            criterion_cls = MultiLabelFairLoss(num_classes=num_classes, alpha=cfg.loss.alpha).to(device)
        else:
            raise ValueError('multi-label 模式仅支持 BCE 或 MLFL')
            
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs)
        
        loss_tag = args.loss.lower()
        if args.task == 'single_label' and args.loss == 'CE' and args.ce_weight_mode != 'none':
            loss_tag = f"{loss_tag}_{args.ce_weight_mode}"
        if args.task == 'single_label' and args.aux_multilabel_lambda > 0:
            aux_tag = str(args.aux_multilabel_lambda).replace('.', 'p')
            aux_prefix = 'auxdual' if args.aux_multilabel_head == 'dual' else 'auxml'
            loss_tag = f"{loss_tag}_{aux_prefix}{aux_tag}"
        if args.ablation == 'no_freq_split_aff':
            aff_tag = str(args.aff_top_ratio).replace('.', 'p')
            comp_tag = {'dominant': 'dom', 'residual': 'res', 'split': 'split'}[args.aff_component]
            loss_tag = f"{loss_tag}_aff{aff_tag}_{comp_tag}"

        best_save_name = f'best_dual_dgnn_{args.dataset.lower()}_{loss_tag}_{args.ablation}_fold_{fold}.pth'
        last_save_name = f'last_dual_dgnn_{args.dataset.lower()}_{loss_tag}_{args.ablation}_fold_{fold}.pth'

        best_save_path = os.path.join(checkpoint_dir, best_save_name)
        last_save_path = os.path.join(checkpoint_dir, last_save_name)

        start_epoch = 0
        best_val_loss = float('inf')
        epochs_since_improve = 0

        resume_path = resolve_resume_checkpoint(
            args.resume_checkpoint,
            checkpoint_dir=checkpoint_dir,
            dataset=args.dataset,
            loss_tag=loss_tag,
            ablation=args.ablation,
            fold=fold,
        )
        if resume_path is not None:
            logging.info(f"↻ Fold {fold}: loading resume checkpoint {resume_path}")
            resume_checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            validate_resume_checkpoint(
                resume_checkpoint,
                args=args,
                fold=fold,
                num_classes=num_classes,
                train_size=len(train_sub),
                val_size=len(val_sub),
            )
            model.load_state_dict(resume_checkpoint["model_state_dict"])

            if "optimizer_state_dict" in resume_checkpoint:
                optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            else:
                logging.warning(
                    "Resume checkpoint has no optimizer state; continuing with a fresh optimizer."
                )

            if "scheduler_state_dict" in resume_checkpoint:
                scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            else:
                logging.warning(
                    "Resume checkpoint has no scheduler state; continuing with a fresh scheduler."
                )

            start_epoch = max(0, int(resume_checkpoint.get("epoch", 0)))
            best_val_loss = float(resume_checkpoint.get("best_val_loss", float("inf")))
            epochs_since_improve = int(resume_checkpoint.get("epochs_since_improve", 0))

            if "rng_state" in resume_checkpoint:
                restore_rng_state(resume_checkpoint)
            if "loader_generator_state" in resume_checkpoint:
                loader_generator.set_state(resume_checkpoint["loader_generator_state"])

            if start_epoch >= cfg.train.num_epochs:
                logging.info(
                    f"Fold {fold}: checkpoint already reached epoch {start_epoch}; "
                    f"nothing to resume for num_epochs={cfg.train.num_epochs}."
                )
                continue
        
        criterion_supcon = MultiLabelSupConLoss(temperature=0.1).to(device)
        criterion_aux_bce = nn.BCEWithLogitsLoss().to(device)
        
        for epoch in range(start_epoch, cfg.train.num_epochs):
            model.train()
            total_loss = 0.0
            
            # The runner may mirror output through tee, so stderr is not
            # necessarily a TTY even when live progress is requested.
            show_progress = not args.disable_tqdm
            progress_bar = tqdm(
                train_loader,
                desc=f"Fold {fold} - Epoch {epoch+1}/{cfg.train.num_epochs}",
                leave=False,
                mininterval=1.0,
                disable=not show_progress,
            )
            for batch in progress_bar:
                x_hf = batch[0].to(device, non_blocking=non_blocking)
                x_lf = batch[1].to(device, non_blocking=non_blocking)
                y = batch[2].to(device, non_blocking=non_blocking)

                y_multi = None
                if args.task == 'single_label' and args.aux_multilabel_lambda > 0 and len(batch) > 5:
                    y_multi = batch[5].to(device, non_blocking=non_blocking).float()

                if args.task == 'single_label':
                    y = y.long().view(-1)
                else:
                    y = y.float()
                optimizer.zero_grad(set_to_none=True)
                outputs = model(x_hf, x_lf)
                
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                    p_hf = outputs[1] if len(outputs) > 1 else None
                    p_lf = outputs[2] if len(outputs) > 2 else None
                    aux_logits = outputs[3] if len(outputs) > 3 else None
                else:
                    logits = outputs
                    p_hf, p_lf = None, None
                    aux_logits = None
                
                # 🌟 核心逻辑分离：SimCLR 模式下，丢弃有监督分类，完全无监督！
                if args.ablation == 'simclr1d':
                    loss_cls = torch.tensor(0.0).to(device)
                    loss_supcon = torch.tensor(0.0).to(device)
                    loss_cl = criterion_cl(p_hf, p_lf)  # 只做无监督特征拉扯
                    loss = loss_cl
                else:
                    cls_out = criterion_cls(logits, y)
                    loss_cls = cls_out[0] if isinstance(cls_out, tuple) else cls_out

                    loss_aux = torch.tensor(0.0, device=device)
                    if args.task == 'single_label' and args.aux_multilabel_lambda > 0 and y_multi is not None:
                        # dual-head: use independent auxiliary logits when available;
                        # shared mode / baselines: fall back to the main logits.
                        aux_source = aux_logits if aux_logits is not None else logits
                        loss_aux = criterion_aux_bce(aux_source, y_multi)
                    
                    if args.ablation not in ['resnet1d', 'wo_cl', 'resnet_fusion_wo_cl', 'lf_only', 'hf_only', 'no_freq_split', 'old_dual_wo_cl', 'old_dual_lf_only', 'old_dual_hf_only', 'old_dual_no_freq_split'] and p_hf is not None and p_lf is not None:
                        loss_cl = criterion_cl(p_hf, p_lf)
                        loss_supcon = criterion_supcon(p_lf, y)
                    else:
                        loss_cl = torch.tensor(0.0).to(device)
                        loss_supcon = torch.tensor(0.0).to(device)
                        
                    no_cl_modes = [
                        'wo_cl',
                        'resnet1d',
                        'resnet_fusion_wo_cl',
                        'lf_only',
                        'hf_only',
                        'no_freq_split',
                        'old_dual_wo_cl',
                        'old_dual_lf_only',
                        'old_dual_hf_only',
                        'old_dual_no_freq_split',
                    ]

                    lambda_supcon = 0.1 if args.ablation not in ['wo_supcon'] + no_cl_modes else 0.0
                    lambda_cl = cfg.loss.alpha if args.ablation not in no_cl_modes else 0.0
                    loss = (
                        loss_cls
                        + args.aux_multilabel_lambda * loss_aux
                        + lambda_cl * loss_cl
                        + lambda_supcon * loss_supcon
                    )
                
                if not torch.isfinite(loss):
                    print(
                        f"⚠️ Fold {fold} Epoch {epoch+1}: non-finite training loss "
                        f"({loss.detach().item()}); skip this batch."
                    )
                    continue

                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
                optimizer.step()
                total_loss += loss.item()
                
                if show_progress:
                    postfix_dict = {'Tot_L': f"{loss.item():.3f}"}
                    if args.ablation != 'simclr1d':
                        postfix_dict['Cls_L'] = f"{loss_cls.item():.3f}"
                    if args.task == 'single_label' and args.aux_multilabel_lambda > 0 and args.ablation != 'simclr1d':
                        postfix_dict['Aux_L'] = f"{loss_aux.item():.3f}"
                    progress_bar.set_postfix(postfix_dict)
                
            scheduler.step()
            
            # 验证阶段 (SimCLR无监督预训练不关心分类准确率，直接保存最后几个epoch或Loss最低的)
            model.eval()
            val_loss = 0.0
            valid_val_batches = 0
            skipped_val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    x_hf = batch[0].to(device, non_blocking=non_blocking)
                    x_lf = batch[1].to(device, non_blocking=non_blocking)
                    y = batch[2].to(device, non_blocking=non_blocking)

                    y_multi = None
                    if args.task == 'single_label' and args.aux_multilabel_lambda > 0 and len(batch) > 5:
                        y_multi = batch[5].to(device, non_blocking=non_blocking).float()

                    if args.task == 'single_label':
                        y = y.long().view(-1)
                    else:
                        y = y.float()
                    outputs = model(x_hf, x_lf)
                    if args.ablation == 'simclr1d':
                        # 无监督验证
                        v_loss = criterion_cl(outputs[1], outputs[2])
                        val_loss += v_loss.item()
                    else:
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                            aux_logits = outputs[3] if len(outputs) > 3 else None
                        else:
                            logits = outputs
                            aux_logits = None
                        v_out = criterion_cls(logits, y)
                        v_loss = v_out[0] if isinstance(v_out, tuple) else v_out
                        if args.task == 'single_label' and args.aux_multilabel_lambda > 0 and y_multi is not None:
                            aux_source = aux_logits if aux_logits is not None else logits
                            v_aux = criterion_aux_bce(aux_source, y_multi)
                            v_loss = v_loss + args.aux_multilabel_lambda * v_aux

                    if torch.isfinite(v_loss):
                        val_loss += v_loss.item()
                        valid_val_batches += 1
                    else:
                        skipped_val_batches += 1

            if skipped_val_batches > 0:
                print(
                    f"⚠️ Fold {fold} Epoch {epoch+1}: skipped {skipped_val_batches} "
                    f"non-finite validation batches."
                )
            avg_val_loss = val_loss / valid_val_batches if valid_val_batches > 0 else float('inf')

            improved = avg_val_loss < (best_val_loss - args.min_delta)
            next_epochs_since_improve = 0 if improved else (epochs_since_improve + 1)
            will_early_stop = (
                (not improved)
                and args.patience
                and args.patience > 0
                and next_epochs_since_improve >= args.patience
            )

            # Save a full resumable checkpoint less frequently to avoid writing ~600 MB every epoch.
            save_last_every = max(0, int(args.save_last_every))
            should_save_last = save_last_every > 0 and (
                (epoch + 1) % save_last_every == 0
                or (epoch + 1) == cfg.train.num_epochs
                or will_early_stop
            )

            # Save best model on validation improvement. Optional lightweight mode omits Adam/RNG states.
            if improved:
                best_val_loss = avg_val_loss
                epochs_since_improve = 0

                best_checkpoint_data = build_checkpoint(
                    epoch=epoch,
                    fold=fold,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_val_loss=best_val_loss,
                    avg_val_loss=avg_val_loss,
                    args=args,
                    cfg=cfg,
                    num_classes=num_classes,
                    train_size=len(train_sub),
                    val_size=len(val_sub),
                    include_optimizer=not bool(args.lightweight_best),
                    loader_generator=loader_generator,
                )

                torch.save(best_checkpoint_data, best_save_path)

                print(
                    f"✅ Fold {fold} Epoch {epoch+1}: "
                    f"train_loss={total_loss / max(1, len(train_loader)):.6f}, "
                    f"val_loss improved to {avg_val_loss:.6f}, saved best checkpoint."
                )
            else:
                epochs_since_improve = next_epochs_since_improve
                print(
                    f"Fold {fold} Epoch {epoch+1}: "
                    f"train_loss={total_loss / max(1, len(train_loader)):.6f}, "
                    f"val_loss={avg_val_loss:.6f}, best={best_val_loss:.6f}."
                )
                if will_early_stop:
                    logging.info(
                        f"⏹️ Fold {fold} early stopped at epoch {epoch+1}. "
                        f"Best val_loss={best_val_loss:.6f}, patience={args.patience}."
                    )

            # Save after best_val_loss/early-stopping state has been updated, so
            # a resumed run continues with the exact validation bookkeeping.
            if should_save_last:
                last_checkpoint_data = build_checkpoint(
                    epoch=epoch,
                    fold=fold,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_val_loss=best_val_loss,
                    avg_val_loss=avg_val_loss,
                    args=args,
                    cfg=cfg,
                    num_classes=num_classes,
                    train_size=len(train_sub),
                    val_size=len(val_sub),
                    include_optimizer=True,
                    loader_generator=loader_generator,
                )
                last_checkpoint_data["epochs_since_improve"] = epochs_since_improve
                torch.save(last_checkpoint_data, last_save_path)

            if will_early_stop:
                break
                
        logging.info(f"✅ Fold {fold} 训练完成. Best val_loss={best_val_loss:.6f}.")
    wandb.finish()

if __name__ == '__main__':
    main()
