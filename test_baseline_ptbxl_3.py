import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import warnings

warnings.filterwarnings("ignore")

from data_process.dataset import DualFreqECGDataset 
from utils.registry import get_dataset_info
# 引入基线模型
from models.baselines import Transformer1D, FourierGNN1D, Mamba1D, PatchTST1D
from models.dual_model import DualFreqDGNN  # 请根据您实际的文件名和类名修改

# =====================================================================
# 辅助函数：统一获取预测结果，保持代码整洁
# =====================================================================
# =====================================================================
# 辅助函数：统一获取预测结果，保持代码整洁
# =====================================================================
def get_predictions(model, dataloader, device):
    all_probs = []
    all_true = []
    with torch.no_grad():
        for batch in dataloader:
            x_hf, x_lf, y_true = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            outputs = model(x_hf, x_lf)
            
            # 🌟 修复核心：兼容返回 Tuple 的高级模型 (如 Ours 带有特征/图输出)
            if isinstance(outputs, tuple):
                logits = outputs[0]  # 取出第一项分类 logits
            else:
                logits = outputs
                
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_true.append(y_true.cpu().numpy())
    return np.vstack(all_probs), np.vstack(all_true)

def run_rigorous_evaluation():
    print("🚀 启动顶刊级严格评估: 基于 Hold-out 校准集的独立阈值寻优...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # =========================================================================
    # ⚠️ 请核对您的权重路径和数据路径
    checkpoint_dir = "./checkpoints/PTBXL_BCE_Weights/" 
    ptbxl_data_dir = "../Project_Data/PTBXL_Data/processed_freq10" 
    
    models_to_test = {
        "Transformer (Medformer)": {"file": "best_dual_dgnn_ptbxl_bce_transformer1d_fold_0.pth", "class": Transformer1D},
        "FourierGNN": {"file": "best_dual_dgnn_ptbxl_bce_fouriergnn1d_fold_0.pth", "class": FourierGNN1D},
        "PatchTST": {"file": "best_dual_dgnn_ptbxl_bce_patchtst1d_fold_0.pth", "class": PatchTST1D},
        "Ours (DualFreqDGNN)": {"file": "best_dual_dgnn_ptbxl_mlfl_fold_0.pth", "class": DualFreqDGNN}  # 您可以把自己的模型也加进来一起测
    }
    # =========================================================================
    
    dataset = DualFreqECGDataset(ptbxl_data_dir)
    ds_info = get_dataset_info('PTBXL')
    num_classes = ds_info['num_classes']
    
    # 1. 还原 5 折交叉验证的划分（保证与您训练时一致）
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_idx_full, test_idx = next(kf.split(range(len(dataset))))
    
    # 🌟 核心严格逻辑：从 80% 的训练集中，再划出 10% 作为“校准集(Calibration Set)”
    # 这个校准集绝对没有在最终测试集中出现过！
    train_idx, calib_idx = train_test_split(train_idx_full, test_size=0.1, random_state=42)
    
    calib_loader = DataLoader(Subset(dataset, calib_idx), batch_size=64, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=64, shuffle=False)

    results = {}

    for exp_name, model_info in models_to_test.items():
        pth_path = os.path.join(checkpoint_dir, model_info["file"])
        if not os.path.exists(pth_path):
            print(f"⚠️ 找不到权重文件，跳过: {pth_path}")
            continue
            
        print(f"\n==================================================")
        print(f"🧠 正在严格评估: [{exp_name}]")
        print(f"==================================================")
        
        model = model_info["class"](num_classes=num_classes).to(device)
        
        state_dict = torch.load(pth_path, map_location=device)
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            res = model.load_state_dict(state_dict['model_state_dict'], strict=False)
        else:
            res = model.load_state_dict(state_dict, strict=False)
            
        print(f"      🔍 [诊断] 权重加载未匹配层数: {len(res.missing_keys)}")
        model.eval()

        # ---------------------------------------------------------
        # Step A: 在校准集 (Calibration Set) 上寻找最佳阈值
        # ---------------------------------------------------------
        print("      ⚙️ Step 1: 正在 Calibration Set 上校准阈值 (防止数据泄露)...")
        calib_probs, calib_true = get_predictions(model, calib_loader, device)
        best_thresholds = np.zeros(num_classes)
        
        for c in range(num_classes):
            y_true_c = calib_true[:, c]
            y_prob_c = calib_probs[:, c]
            best_f1_c = 0
            best_t_c = 0.5 
            
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
        
        mean_t = np.mean(best_thresholds)
        print(f"      ✅ 校准完成! 均值阈值: {mean_t:.2f} | 阈值分布: {np.round(best_thresholds, 2)}")

        # ---------------------------------------------------------
        # Step B: 盲测！将找好的阈值应用于未见过的测试集 (Test Set)
        # ---------------------------------------------------------
        print("      📈 Step 2: 正在 Test Set 上进行盲测计算最终成绩...")
        test_probs, test_true = get_predictions(model, test_loader, device)
        
        test_preds = np.zeros_like(test_probs)
        for c in range(num_classes):
            test_preds[:, c] = (test_probs[:, c] > best_thresholds[c]).astype(int)
            
        try:
            macro_auc = roc_auc_score(test_true, test_probs, average='macro')
            _, _, macro_f1, _ = precision_recall_fscore_support(test_true, test_preds, average='macro', zero_division=0)
            
            print(f"  🏆 最终严谨成绩 -> AUC: {macro_auc:.4f} | F1: {macro_f1:.4f}")
            results[exp_name] = {"AUC": f"{macro_auc:.4f}", "F1": f"{macro_f1:.4f}"}
        except Exception as e:
            print(f"  ❌ 评估出错: {e}")
            results[exp_name] = {"AUC": "Error", "F1": "Error"}

    print("\n\n🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")
    print("🏆 论文 Table 1 - 源域 (PTB-XL) 合法评估一览表 (无数据泄露)")
    print("🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟\n")
    print("| Model Architecture | 源域 Macro F1 | 源域 Macro AUC |")
    print("| :--- | :---: | :---: |")
    for name, metrics in results.items():
        print(f"| {name} | {metrics['F1']} | {metrics['AUC']} |")
    print(f"| **Ours (DualFreqDGNN)** | **待填入** | **待填入** |")

if __name__ == "__main__":
    run_rigorous_evaluation()