# main_eval.py
import os
import argparse
import ast
import torch
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_curve, auc, roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import logging

from data_process.dataset import DualFreqECGDataset
# from models.physics_dual_model import PhysicsDualFreqDGNN as DualFreqDGNN # 使用微分方程版本
from utils.config_parser import load_config
from utils.registry import get_dataset_info
from utils.splits import build_chapman_file_splits, build_cpsc_outer_splits
from models.model_factory import build_model_by_name

logging.basicConfig(level=logging.INFO, format='%(message)s')

PTBXL_CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
PTBXL_CLASS_TO_IDX = {c: i for i, c in enumerate(PTBXL_CLASSES)}
PTBXL_PRIORITY = ['MI', 'STTC', 'CD', 'HYP', 'NORM']

def get_args():
    parser = argparse.ArgumentParser(description="DualFreqDGNN 统一评估框架")
    parser.add_argument('--dataset', type=str, required=True, choices=['CPSC', 'PTBXL', 'CHAPMAN', 'PTB'])
    parser.add_argument('--loss', type=str, required=True, choices=['BCE', 'MLFL', 'CE'])
    parser.add_argument('--k_folds', type=int, default=5, help='评估的总 fold 数')
    parser.add_argument('--start_fold', type=int, default=0, help='从第几个 fold 开始评估')
    parser.add_argument('--end_fold', type=int, default=None, help='评估到第几个 fold 之前停止；默认等于 k_folds')


    parser.add_argument('--task', type=str, default='multilabel', choices=['multilabel', 'single_label'], help='评估任务类型；single_label 会使用 softmax/argmax 与 multiclass AUROC')

    parser.add_argument('--single_label_policy', type=str, default='max_scp_score', choices=['max_scp_score', 'unique', 'priority'], help='PTB-XL 多标签转单标签策略，需要和训练保持一致')
    parser.add_argument('--ce_weight_mode', type=str, default='balanced', choices=['none', 'balanced', 'effective'], help='single_label + CE 评估时用于定位 checkpoint 文件名，需要和训练保持一致')
    parser.add_argument('--aux_multilabel_lambda', type=float, default=0.0, help='仅用于匹配训练时的 multi-label auxiliary checkpoint 命名；评估本身不使用辅助 BCE')
    parser.add_argument('--aux_multilabel_head', type=str, default='dual', choices=['dual', 'shared'], help='仅用于匹配训练时的辅助头形式和 checkpoint 命名')
    parser.add_argument('--aff_top_ratio', type=float, default=0.25, help='仅用于匹配 no_freq_split_aff 训练时的 checkpoint 命名，并构建同结构模型')
    parser.add_argument('--aff_component', type=str, default='dominant', choices=['dominant', 'residual', 'split'], help='仅用于匹配 no_freq_split_aff 训练时的频谱视图形式')

    parser.add_argument(
        '--ablation',
        type=str,
        default='full',
        help='需要和训练时的 --ablation 保持一致，例如 full / wo_cl / wo_supcon'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='可选：强制指定 processed 数据路径'
    )

    parser.add_argument(
        '--threshold_mode',
        type=str,
        default='val_calibrated',
        choices=['val_calibrated', 'test_optimal', 'fixed'],
        help=(
            'val_calibrated: 用验证集找阈值，再测试集评估，正式论文推荐；'
            'test_optimal: 在测试集上找最优阈值，仅用于debug；'
            'fixed: 固定0.5阈值'
        )
    )

    parser.add_argument(
        '--fixed_threshold',
        type=float,
        default=0.5,
        help='threshold_mode=fixed 时使用的固定阈值'
    )

    return parser.parse_args()

def load_ptbxl_diagnostic_df(csv_path, single_label_policy=None):
    """
    读取 PTB-XL CSV，并过滤出具有 5 大诊断 superclass 的样本。
    single_label_policy 不为 None 时，将 PTB-XL 多诊断样本归并为单标签。
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
    """
    和新版 ptbxl_preprocess.py 保持一致：
    ecg_id=1 -> 00001.pt
    """
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
        f"expected diagnostic records={len(expected_files)}, "
        f"matched processed files={len(matched_files)}, "
        f"missing={len(missing_files)}"
    )

    if len(missing_files) > 0:
        print(f"  First missing files: {missing_files[:10]}")

    return matched_files


def get_ptbxl_val_test_lists(csv_path, processed_data_dir, fold_idx, task="multilabel", single_label_policy="max_scp_score"):
    """
    5-fold 设计：
    PTB-XL 官方 strat_fold 有 1~10。
    我们把两个 strat_fold 合成一个外层 fold：

    fold 0 test: strat_fold 1,2
    fold 1 test: strat_fold 3,4
    fold 2 test: strat_fold 5,6
    fold 3 test: strat_fold 7,8
    fold 4 test: strat_fold 9,10

    validation fold 使用下一个 pair：
    fold 0 val: strat_fold 3,4
    fold 1 val: strat_fold 5,6
    fold 2 val: strat_fold 7,8
    fold 3 val: strat_fold 9,10
    fold 4 val: strat_fold 1,2

    注意：main_train.py 后面也应该使用同样划分，否则评估仍然不严谨。
    """
    df = load_ptbxl_diagnostic_df(csv_path, single_label_policy if task == "single_label" else None)

    test_pair_idx = fold_idx
    val_pair_idx = (fold_idx + 1) % 5

    test_folds = [test_pair_idx * 2 + 1, test_pair_idx * 2 + 2]
    val_folds = [val_pair_idx * 2 + 1, val_pair_idx * 2 + 2]

    test_df = df[df["strat_fold"].isin(test_folds)].copy()
    val_df = df[df["strat_fold"].isin(val_folds)].copy()

    test_expected = ecg_ids_to_files(test_df["ecg_id"].tolist())
    val_expected = ecg_ids_to_files(val_df["ecg_id"].tolist())

    val_files = match_files(val_expected, processed_data_dir, "val", fold_idx)
    test_files = match_files(test_expected, processed_data_dir, "test", fold_idx)

    label_map = None
    if task == "single_label":
        label_map = {f"{int(row.ecg_id):05d}.pt": int(row.single_label_eval) for row in df.itertuples()}

    return val_files, test_files, label_map

def run_inference(model, dataloader, device, task="multilabel"):
    """
    task=multilabel: 返回 sigmoid 概率 [N,C] 与 multi-hot 标签 [N,C]
    task=single_label: 返回 softmax 概率 [N,C] 与 long 标签 [N]
    """
    probs_list = []
    trues_list = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Infer", leave=False):
            x_hf = batch[0].to(device)
            x_lf = batch[1].to(device)
            y = batch[2].to(device)

            outputs = model(x_hf, x_lf)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            if task == "single_label":
                probs_list.append(torch.softmax(logits, dim=1))
                trues_list.append(y.long().view(-1))
            else:
                probs_list.append(torch.sigmoid(logits))
                trues_list.append(y.float())

    probs = torch.cat(probs_list, dim=0)
    trues = torch.cat(trues_list, dim=0)
    return probs, trues

def search_best_thresholds(probs, trues, classes, threshold_step=0.01):
    """
    在验证集上为每个类别独立搜索最佳 F1 阈值。
    """
    device = probs.device
    num_classes = probs.shape[1]

    thresholds = torch.arange(
        threshold_step,
        1.0,
        threshold_step,
        device=device
    )

    best_thresholds = torch.zeros(num_classes, device=device)

    print("\n⚡ 正在验证集上执行动态阈值校准...")
    for c in range(num_classes):
        best_th = torch.tensor(0.5, device=device)
        best_f1_c = torch.tensor(0.0, device=device)

        probs_c = probs[:, c]
        trues_c = trues[:, c]

        for th in thresholds:
            preds_c = (probs_c > th).float()

            tp = (preds_c * trues_c).sum()
            fp = (preds_c * (1.0 - trues_c)).sum()
            fn = ((1.0 - preds_c) * trues_c).sum()

            f1_temp = 2 * tp / (2 * tp + fp + fn + 1e-8)

            if f1_temp > best_f1_c:
                best_f1_c = f1_temp
                best_th = th

        best_thresholds[c] = best_th
        print(
            f"  🎯 [{classes[c]:<4}]: "
            f"val最佳阈值 = {best_th.item():.2f} -> val F1 = {best_f1_c.item():.4f}"
        )

    return best_thresholds


def apply_thresholds(probs, thresholds):
    """Apply fixed, validation-calibrated class-wise thresholds."""
    return (probs > thresholds.view(1, -1)).float()

def build_model(args, cfg, num_classes, device):
    """Backward-compatible wrapper around the shared model factory."""
    return build_model_by_name(args, cfg, num_classes=num_classes, device=device)

def main():
    args = get_args()
    cfg = load_config()
    ds_info = get_dataset_info(args.dataset)
    if args.dataset == "CHAPMAN" and args.task != "single_label":
        raise ValueError("CHAPMAN is a single-label task; use --task single_label.")
    num_classes = ds_info['num_classes']
    classes = ds_info['class_names']
    data_dir = args.data_dir if args.data_dir else ds_info["default_data_dir"]
    
    device = torch.device(cfg.env.device if torch.cuda.is_available() else "cpu")
    logging.info(f"🚀 启动 OOF 评估 | {args.dataset} | {args.loss}")
    
    if args.dataset == 'PTBXL':
        dataset_info = get_dataset_info(args.dataset)
        csv_path = dataset_info.get('csv_path', None)
        
        # 🚀 修复：如果注册表里没写，我们就根据 data_dir 自动反推它的位置！
        # data_dir 通常是 .../PTBXL_Data/processed，它的上一级就是 CSV 所在处
        if csv_path is None or not os.path.exists(csv_path):
            ptbxl_data_dir = os.path.dirname(data_dir.rstrip('/\\'))
            candidates = [
                os.path.join(ptbxl_data_dir, 'ptbxl_database.csv'),
                os.path.join(
                    ptbxl_data_dir,
                    'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1',
                    'ptbxl_database.csv'
                ),
            ]

            csv_path = None
            for p in candidates:
                if os.path.exists(p):
                    csv_path = p
                    break

        if csv_path is None or not os.path.exists(csv_path):
            raise FileNotFoundError("找不到 PTB-XL 的标签文件，请检查 registry.py 里的 csv_path。")
        
    elif args.dataset == 'CHAPMAN':
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
    
    gpu_all_probs = []
    gpu_all_trues = []
    gpu_all_preds = []
    fold_metric_rows = []

    num_folds = 5 if args.dataset in {'PTBXL', 'CHAPMAN'} else args.k_folds
    start_fold = max(0, int(args.start_fold))
    end_fold = num_folds if args.end_fold is None else min(num_folds, int(args.end_fold))
    if start_fold >= end_fold:
        raise ValueError(f"Invalid fold range: start_fold={start_fold}, end_fold={end_fold}, num_folds={num_folds}")

    for fold in range(start_fold, end_fold):
        print(f"\n🔍 正在推理 Fold {fold}...")

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

        model_path = os.path.join(
            checkpoint_dir,
            f"best_dual_dgnn_{args.dataset.lower()}_{loss_tag}_{args.ablation}_fold_{fold}.pth"
        )

        if not os.path.exists(model_path):
            logging.warning(f"⚠️ 找不到权重文件: {model_path}，跳过此折。")
            continue

        # ========= 构建验证集和测试集 =========
        if args.dataset == "PTBXL":
            val_files, test_files, single_label_map = get_ptbxl_val_test_lists(csv_path, data_dir, fold, task=args.task, single_label_policy=args.single_label_policy)

            val_dataset = DualFreqECGDataset(data_dir, file_list=val_files, label_mode=args.task, single_label_map=single_label_map)
            test_dataset = DualFreqECGDataset(data_dir, file_list=test_files, label_mode=args.task, single_label_map=single_label_map)

        elif args.dataset == "CHAPMAN":
            _, val_files, test_files, single_label_map = chapman_splits[fold]
            val_dataset = DualFreqECGDataset(
                data_dir,
                file_list=val_files,
                label_mode="single_label",
                single_label_map=single_label_map,
            )
            test_dataset = DualFreqECGDataset(
                data_dir,
                file_list=test_files,
                label_mode="single_label",
                single_label_map=single_label_map,
            )

        else:
            _, val_idx, test_idx = folds_data[fold]
            val_dataset = Subset(dataset, val_idx)
            test_dataset = Subset(dataset, test_idx)

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=0
        )

        # ========= 加载模型 =========
        m = build_model(args, cfg, num_classes, device)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if "model_state_dict" in checkpoint:
            m.load_state_dict(checkpoint["model_state_dict"])
        else:
            m.load_state_dict(checkpoint)

        if args.task == 'single_label':
            print("📌 single_label 模式：使用 softmax + argmax，不进行阈值搜索。")
            test_probs, test_trues = run_inference(m, test_loader, device, task=args.task)
            test_preds = torch.argmax(test_probs, dim=1)
        else:
            # ========= validation threshold calibration =========
            if args.threshold_mode == "val_calibrated":
                val_probs, val_trues = run_inference(m, val_loader, device, task=args.task)
                fold_thresholds = search_best_thresholds(
                    val_probs,
                    val_trues,
                    classes,
                    threshold_step=0.01
                )

            elif args.threshold_mode == "fixed":
                fold_thresholds = torch.full(
                    (num_classes,),
                    args.fixed_threshold,
                    device=device
                )
                print(f"📌 使用固定阈值: {args.fixed_threshold}")

            elif args.threshold_mode == "test_optimal":
                fold_thresholds = None
                print("⚠️ 使用 test_optimal 模式：将在测试集上搜索阈值，仅用于debug，不能写入论文。")

            else:
                raise ValueError(f"未知 threshold_mode: {args.threshold_mode}")

            # ========= 测试集推理 =========
            test_probs, test_trues = run_inference(m, test_loader, device, task=args.task)

            if args.threshold_mode == "test_optimal":
                fold_thresholds = search_best_thresholds(
                    test_probs,
                    test_trues,
                    classes,
                    threshold_step=0.01
                )

            test_preds = apply_thresholds(test_probs, fold_thresholds)

        gpu_all_probs.append(test_probs)
        gpu_all_trues.append(test_trues)
        gpu_all_preds.append(test_preds)

        fold_trues_np = test_trues.cpu().numpy()
        fold_preds_np = test_preds.cpu().numpy()
        fold_probs_np = test_probs.cpu().numpy()
        if args.task == "multilabel":
            fold_metric_rows.append(
                {
                    "macro_auroc": roc_auc_score(
                        fold_trues_np, fold_probs_np, average="macro"
                    ),
                    "macro_auprc": average_precision_score(
                        fold_trues_np, fold_probs_np, average="macro"
                    ),
                    "macro_precision": precision_recall_fscore_support(
                        fold_trues_np,
                        fold_preds_np,
                        average="macro",
                        zero_division=0,
                    )[0],
                    "macro_recall": precision_recall_fscore_support(
                        fold_trues_np,
                        fold_preds_np,
                        average="macro",
                        zero_division=0,
                    )[1],
                    "macro_f1": precision_recall_fscore_support(
                        fold_trues_np,
                        fold_preds_np,
                        average="macro",
                        zero_division=0,
                    )[2],
                }
            )
        else:
            try:
                fold_auc = roc_auc_score(
                    fold_trues_np,
                    fold_probs_np,
                    average="macro",
                    multi_class="ovr",
                )
            except ValueError:
                fold_auc = float("nan")
            fold_precision, fold_recall, fold_f1, _ = precision_recall_fscore_support(
                fold_trues_np,
                fold_preds_np,
                average="macro",
                zero_division=0,
            )
            fold_metric_rows.append(
                {
                    "macro_auroc": fold_auc,
                    "accuracy": accuracy_score(fold_trues_np, fold_preds_np),
                    "macro_precision": fold_precision,
                    "macro_recall": fold_recall,
                    "macro_f1": fold_f1,
                }
            )

    if not gpu_all_probs:
        print("❌ 没有任何 fold 被成功评估，请检查 checkpoint 路径。")
        return

    all_probs = torch.cat(gpu_all_probs, dim=0)
    all_trues = torch.cat(gpu_all_trues, dim=0)
    all_preds = torch.cat(gpu_all_preds, dim=0)
            
    # 提取回 CPU 交给 sklearn 打印详细报告和画图
    all_trues_np = all_trues.cpu().numpy()
    all_preds_np = all_preds.cpu().numpy()
    all_probs_np = all_probs.cpu().numpy()
    
    if args.task == 'single_label':
        print("\n📈 Single-label AUROC 指标:")
        try:
            macro_auc = roc_auc_score(all_trues_np, all_probs_np, average="macro", multi_class="ovr")
            weighted_auc = roc_auc_score(all_trues_np, all_probs_np, average="weighted", multi_class="ovr")
            print(f"【Macro AUROC 】: {macro_auc:.4f}")
            print(f"【Weighted AUROC】: {weighted_auc:.4f}")
        except ValueError as e:
            print(f"⚠️ AUROC 计算失败: {e}")

        print("\n📊 Per-class AUROC:")
        for i, cls_name in enumerate(classes):
            y_true_c = (all_trues_np == i).astype(int)
            y_prob_c = all_probs_np[:, i]
            if len(np.unique(y_true_c)) < 2:
                print(f"  {cls_name:<5}: AUROC = nan  缺少正/负样本")
            else:
                auc_c = roc_auc_score(y_true_c, y_prob_c)
                print(f"  {cls_name:<5}: AUROC = {auc_c:.4f}")

        print("\n" + "🌟"*20)
        print(f"🏆 {args.dataset} Single-label 最终评估报告 ({args.loss})")
        print("🌟"*20)
        acc = accuracy_score(all_trues_np, all_preds_np)
        precision, recall, f1, _ = precision_recall_fscore_support(all_trues_np, all_preds_np, average='macro', zero_division=0)
        valid_fold_auc = [
            row["macro_auroc"] for row in fold_metric_rows
            if np.isfinite(row["macro_auroc"])
        ]
        fold_mean_auc = float(np.mean(valid_fold_auc)) if valid_fold_auc else float("nan")
        fold_mean_acc = float(np.mean([row["accuracy"] for row in fold_metric_rows]))
        fold_mean_precision = float(np.mean([row["macro_precision"] for row in fold_metric_rows]))
        fold_mean_recall = float(np.mean([row["macro_recall"] for row in fold_metric_rows]))
        fold_mean_f1 = float(np.mean([row["macro_f1"] for row in fold_metric_rows]))
        print("\nFive-fold mean (paper protocol):")
        print(f"【Accuracy 】: {fold_mean_acc:.4f}")
        print(f"【Macro AUROC 】: {fold_mean_auc:.4f}")
        print(f"【Macro Avg】: P: {fold_mean_precision:.4f} | R: {fold_mean_recall:.4f} | F1: {fold_mean_f1:.4f}")
        print("\nPooled result (supplementary):")
        print(f"【Accuracy 】: {acc:.4f}")
        print(f"【Macro Avg】: P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}")
        print(classification_report(all_trues_np, all_preds_np, target_names=classes, zero_division=0))
    else:
        # ==========================================
        # AUROC / AUPRC evaluation for multi-label ECG diagnosis
        # ==========================================
        print("\n📈 AUROC / AUPRC 指标:")

        try:
            pooled_macro_auc = roc_auc_score(all_trues_np, all_probs_np, average="macro")
            micro_auc = roc_auc_score(all_trues_np, all_probs_np, average="micro")
            weighted_auc = roc_auc_score(all_trues_np, all_probs_np, average="weighted")

            pooled_macro_auprc = average_precision_score(
                all_trues_np, all_probs_np, average="macro"
            )
            micro_auprc = average_precision_score(all_trues_np, all_probs_np, average="micro")
            weighted_auprc = average_precision_score(all_trues_np, all_probs_np, average="weighted")

            fold_mean_auc = float(np.mean([row["macro_auroc"] for row in fold_metric_rows]))
            fold_mean_auprc = float(np.mean([row["macro_auprc"] for row in fold_metric_rows]))
            fold_mean_precision = float(
                np.mean([row["macro_precision"] for row in fold_metric_rows])
            )
            fold_mean_recall = float(
                np.mean([row["macro_recall"] for row in fold_metric_rows])
            )
            fold_mean_f1 = float(np.mean([row["macro_f1"] for row in fold_metric_rows]))

            print("Five-fold mean (paper protocol):")
            print(f"【Macro AUROC 】: {fold_mean_auc:.4f}")
            print(f"【Macro AUPRC 】: {fold_mean_auprc:.4f}")
            print(f"【Macro Avg】: P: {fold_mean_precision:.4f} | R: {fold_mean_recall:.4f} | F1: {fold_mean_f1:.4f}")
            print(f"【Pooled Macro AUROC】: {pooled_macro_auc:.4f}")
            print(f"【Micro AUROC 】: {micro_auc:.4f}")
            print(f"【Weighted AUROC】: {weighted_auc:.4f}")
            print(f"【Pooled Macro AUPRC】: {pooled_macro_auprc:.4f}")
            print(f"【Micro AUPRC 】: {micro_auprc:.4f}")
            print(f"【Weighted AUPRC】: {weighted_auprc:.4f}")

        except ValueError as e:
            print(f"⚠️ AUROC/AUPRC 计算失败: {e}")
        
        print("\n📊 Per-class AUROC / AUPRC:")

        for i, cls_name in enumerate(classes):
            y_true_c = all_trues_np[:, i]
            y_prob_c = all_probs_np[:, i]

            if len(np.unique(y_true_c)) < 2:
                print(f"  {cls_name:<5}: AUROC = nan | AUPRC = nan  缺少正/负样本")
            else:
                auc_c = roc_auc_score(y_true_c, y_prob_c)
                auprc_c = average_precision_score(y_true_c, y_prob_c)
                print(f"  {cls_name:<5}: AUROC = {auc_c:.4f} | AUPRC = {auprc_c:.4f}")
                
        print("\n" + "🌟"*20)
        print(f"🏆 {args.dataset} 最终评估报告 ({args.loss})")
        print("🌟"*20)
        pooled_precision, pooled_recall, pooled_f1, _ = precision_recall_fscore_support(
            all_trues_np,
            all_preds_np,
            average='macro',
            zero_division=0,
        )
        print(
            f"【Pooled Macro Avg】: P: {pooled_precision:.4f} | "
            f"R: {pooled_recall:.4f} | F1: {pooled_f1:.4f}"
        )
        print(classification_report(all_trues_np, all_preds_np, target_names=classes, zero_division=0))

    # --- 统一绘制保存 ROC ---
    os.makedirs(cfg.env.results_dir, exist_ok=True)
    plt.figure(figsize=(10, 8), dpi=300)
    colors = plt.get_cmap('tab10')
    for c in range(num_classes):
        if args.task == 'single_label':
            y_true_c = (all_trues_np == c).astype(int)
        else:
            y_true_c = all_trues_np[:, c]
        if len(np.unique(y_true_c)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_c, all_probs_np[:, c])
        color = 'darkred' if args.dataset == 'PTB' and c == 1 else colors(c)
        if args.dataset == 'PTB' and c == 0:
            continue 
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{classes[c]} (AUC = {auc(fpr, tpr):.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6)
    plt.xlim([0.0, 1.0]), plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(f'ROC Curve - {args.dataset} ({args.loss})', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(cfg.env.results_dir, f'ROC_{args.dataset}_{args.loss}_{args.task}_{timestamp}.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\n✅ 【ROC图】已保存至: {save_path}")

if __name__ == '__main__':
    main()
