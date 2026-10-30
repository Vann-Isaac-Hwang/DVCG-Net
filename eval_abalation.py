import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from models.dual_model import DualFreqDGNN
from models.physics_dual_model import PhysicsDualFreqDGNN
from data_process.dataset import DualFreqECGDataset
from utils.registry import get_dataset_info
from utils.config_parser import load_config
import warnings
warnings.filterwarnings('ignore') # 忽略 sklearn 的一些除零警告

def evaluate_model(model, dataloader, device):
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            x_hf, x_lf, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            logits, _, _ = model(x_hf, x_lf)
            probs = torch.sigmoid(logits)
            
            all_probs.append(probs.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    # 🚀 开启与您 5 折实验一模一样的动态阈值寻优！
    best_thresholds = np.zeros(5)
    best_f1s = np.zeros(5)
    
    for i in range(5):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.95, 0.05):
            preds = (all_probs[:, i] > t).astype(int)
            f1 = f1_score(all_targets[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[i] = best_t
        best_f1s[i] = best_f1
        
    # 用寻找到的最佳阈值统一计算
    all_preds = np.zeros_like(all_probs)
    for i in range(5):
        all_preds[:, i] = (all_probs[:, i] > best_thresholds[i]).astype(int)

    auc = roc_auc_score(all_targets, all_probs, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    acc = accuracy_score(all_targets, all_preds) 
    
    return auc, f1, acc

def main():
    cfg = load_config()
    ds_info = get_dataset_info('PTBXL')
    device = torch.device(cfg.env.device if torch.cuda.is_available() else "cpu")
    
    # 动态获取真实路径
    base_processed_dir = os.path.abspath(ds_info['default_data_dir']) 
    base_parent_dir = os.path.dirname(base_processed_dir.rstrip('/\\'))
    dir_freq5 = os.path.join(base_parent_dir, "processed_freq5")
    dir_freq20 = os.path.join(base_parent_dir, "processed_freq20")

    # 我们要评估的 4 个消融任务矩阵
    ablation_tasks = [
        ("Freq_5Hz",         dir_freq5,          0.0),
        ("Freq_10Hz",        base_processed_dir,         0.0),
        ("Freq_20Hz",        dir_freq20,         0.0),
        ("Physics_20Hz",     dir_freq20,         0.5),
        ("Graph_EdgeDP_0.5", base_processed_dir, 0.5),
        ("Graph_EdgeDP_1.0", base_processed_dir, 1.0),
    ]

    print(f"{'='*60}")
    print(f"🚀 开始批量评估消融模型 (在 Fold 0 验证集上)")
    print(f"{'='*60}")
    print(f"{'实验名称':<20} | {'AUC':<8} | {'Macro F1':<8} | {'Accuracy':<8}")
    print(f"{'-'*60}")

    results = []

    for name, data_dir, edge_dp in ablation_tasks:
        pth_path = os.path.join("./ablations_workspace/checkpoints", f"ablation_{name}.pth")
        
        if not os.path.exists(pth_path):
            print(f"{name:<20} | ❌ 模型权重缺失，跳过！")
            continue

        # 1. 挂载对应的数据集
        dataset = DualFreqECGDataset(data_dir)
        
        # 2. 严格切分出 Fold 0 的验证集 (保持与训练时绝对一致的 seed)
        kf = KFold(n_splits=5, shuffle=True, random_state=cfg.env.seed)
        _, val_idx = next(kf.split(range(len(dataset)))) 
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64, shuffle=False)

        # 3. 初始化对应的模型结构 (智能判定肉体)
        if "Physics" in name:
            # 如果是物理战神，就用带有 ODE/PDE 的新架构
            model = PhysicsDualFreqDGNN(
                num_nodes=cfg.model.num_nodes, 
                hidden_dim=cfg.model.hidden_dim, 
                num_classes=ds_info['num_classes'],
                edge_dropout=edge_dp
            ).to(device)
        else:
            # 如果是以前的消融实验，就用老的纯数学架构
            model = DualFreqDGNN(
                num_nodes=cfg.model.num_nodes, 
                hidden_dim=cfg.model.hidden_dim, 
                num_classes=ds_info['num_classes'],
                edge_dropout=edge_dp
            ).to(device)

        # 4. 加载权重并评估
        model.load_state_dict(torch.load(pth_path, map_location=device))
        auc, f1, acc = evaluate_model(model, val_loader, device)
        
        results.append({"Experiment": name, "AUC": auc, "F1": f1, "Accuracy": acc})
        print(f"{name:<20} | {auc:.4f}   | {f1:.4f}     | {acc:.4f}")

    print(f"{'='*60}")
    print("✅ 评估完成！您可以将上述数据填入论文表格中。")

if __name__ == '__main__':
    main()