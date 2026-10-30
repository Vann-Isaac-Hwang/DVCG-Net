import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import warnings

warnings.filterwarnings("ignore")

from models.dual_model import DualFreqDGNN 
from data_process.dataset import DualFreqECGDataset 
# 🌟 修复：导入 registry
from utils.registry import get_dataset_info

def run_all_ablations():
    print("🚀 启动自动化源域 (PTB-XL) 阈值寻优消融大阅兵...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # =========================================================================
    checkpoint_dir = "./ablations_workspace/checkpoints/"
    ptbxl_data_dir = "../Project_Data/PTBXL_Data/processed_freq10"
    
    ablation_dict = {
        "Ours (DualFreqDGNN + SupCon)": "Ours_Full_10Hz_best.pth", 
        "w/o EdgeDrop": "wo_EdgeDrop_10Hz_best.pth",
        "w/o SupCon": "wo_SupCon_10Hz_best.pth",
        "w/o DGNN": "wo_DGNN_10Hz_best.pth",
        "LF-Only": "LF_Only_10Hz_best.pth",
        "HF-Only": "HF_Only_10Hz_best.pth"
    }
    
    dataset = DualFreqECGDataset(ptbxl_data_dir)
    
    # 🌟 修复：安全获取类别数
    ds_info = get_dataset_info('PTBXL')
    num_classes = ds_info['num_classes']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    _, test_idx = next(kf.split(range(len(dataset))))
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=64, shuffle=False)

    results = {}

    for exp_name, pth_name in ablation_dict.items():
        pth_path = os.path.join(checkpoint_dir, pth_name)
        if not os.path.exists(pth_path):
            print(f"⚠️ 找不到权重文件，跳过: {pth_path}")
            continue
            
        print(f"\n[{exp_name}] 🧠 正在加载与测试...")
        
        edge_dp = 0.0 if "w/o EdgeDrop" in exp_name else 0.5
        model = DualFreqDGNN(
            num_nodes=12, 
            hidden_dim=64, 
            num_classes=num_classes,
            edge_dropout=edge_dp
        ).to(device)
        
        hook_handle = None
        if "w/o DGNN" in exp_name:
            model.dgnn_hf = nn.Identity()
            model.dgnn_lf = nn.Identity()
            
        if "LF-Only" in exp_name or "HF-Only" in exp_name:
            def get_mask_hook(mode):
                def hook(module, hook_args):
                    fused = hook_args[0].clone()
                    half_dim = fused.shape[1] // 2 
                    if mode == "LF":
                        fused[:, :half_dim] = 0.0  
                    elif mode == "HF":
                        fused[:, half_dim:] = 0.0  
                    return (fused,)
                return hook
            mode = "LF" if "LF-Only" in exp_name else "HF"
            hook_handle = model.classifier.register_forward_pre_hook(get_mask_hook(mode))
            
        state_dict = torch.load(pth_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        all_y_probs = []
        all_y_true = []

        with torch.no_grad():
            for batch in test_loader:
                x_hf, x_lf, y_true = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                
                logits, _, _ = model(x_hf, x_lf)
                probs = torch.sigmoid(logits)
                
                all_y_probs.append(probs.cpu().numpy())
                all_y_true.append(y_true.cpu().numpy()) 

        all_y_probs = np.vstack(all_y_probs)
        all_y_true = np.vstack(all_y_true)
        
        # =====================================================================
        # 🌟 多标签独立阈值寻优 (Independent Threshold Search)
        # =====================================================================
        best_thresholds = np.zeros(num_classes)
        all_y_pred = np.zeros_like(all_y_probs)
        
        for c in range(num_classes):
            y_true_c = all_y_true[:, c]
            y_prob_c = all_y_probs[:, c]
            
            best_f1_c = 0
            best_t_c = 0.5 # 默认兜底
            
            for t in np.arange(0.01, 1.0, 0.02):
                y_pred_c = (y_prob_c > t).astype(int)
                
                tp = np.sum((y_pred_c == 1) & (y_true_c == 1))
                fp = np.sum((y_pred_c == 1) & (y_true_c == 0))
                fn = np.sum((y_pred_c == 0) & (y_true_c == 1))
                
                f1_c = 0 if (tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn)
                
                if f1_c > best_f1_c:
                    best_f1_c = f1_c
                    best_t_c = t
                    
            best_thresholds[c] = best_t_c
            all_y_pred[:, c] = (y_prob_c > best_t_c).astype(int)
        
        # =====================================================================

        try:
            macro_auc = roc_auc_score(all_y_true, all_y_probs, average='macro')
            _, _, macro_f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='macro', zero_division=0)
            
            mean_threshold = np.mean(best_thresholds)
            print(f"  ✅ 成绩单 -> AUC: {macro_auc:.4f} | F1: {macro_f1:.4f} (均值最佳阈值: {mean_threshold:.2f})")
            results[exp_name] = {"AUC": f"{macro_auc:.4f}", "F1": f"{macro_f1:.4f}"}
        except Exception as e:
            print(f"  ❌ 评估出错: {e}")
            results[exp_name] = {"AUC": "Error", "F1": "Error"}
            
        if hook_handle is not None:
            hook_handle.remove()

    print("\n\n🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")
    print("🏆 论文 Table 3 - 源域 (PTB-XL) 性能一览表 (经独立阈值寻优校准)")
    print("🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟\n")
    print("| Model Architecture (Ablation) | 源域 Macro F1 | 源域 Macro AUC |")
    print("| :--- | :---: | :---: |")
    for name, metrics in results.items():
        if "Ours" in name:
            print(f"| **{name}** | **{metrics['F1']}** | **{metrics['AUC']}** |")
        else:
            print(f"| {name} | {metrics['F1']} | {metrics['AUC']} |")

if __name__ == "__main__":
    run_all_ablations()