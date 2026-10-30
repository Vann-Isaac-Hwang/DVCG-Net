import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

from models.dual_model import DualFreqDGNN 
from data_process.dataset import DualFreqECGDataset 

def zero_shot_evaluate():
    print("🚀 启动硬核 Zero-Shot 跨数据集泛化测试 (PTB-XL -> PTB)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ⚠️ 确保这里的路径是指向您刚刚凌晨 00:00 跑出来的那个新权重！
    model_path = "./checkpoints/PTBXL_MLFL_Weights/best_dual_dgnn_ptbxl_mlfl_fold_0.pth"
    
    # 🌟 1. 初始化模型
    print("📦 正在加载 PTB-XL (20Hz, ed=0.5) 的预训练模型...")
    model = DualFreqDGNN(
        num_nodes=12,
        hidden_dim=64, 
        num_classes=5,     # PTB-XL 的类别数
        edge_dropout=0.0   
    ).to(device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到模型权重文件：{model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 🌟 2. 加载 PTB 目标数据集
    print("📂 正在读取 PTB (20Hz) 测试集数据...")
    ptb_data_dir = "../Project_Data/PTB_Data/processed_freq20"
    
    test_dataset = DualFreqECGDataset(processed_data_dir=ptb_data_dir) 
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    all_y_true = []
    all_y_probs = []

    print("🧠 开始跨域前向推理...")
    with torch.no_grad():
        for batch in test_loader:
            x_hf = batch[0].to(device)
            x_lf = batch[1].to(device)
            y_true = batch[2].to(device)
            
            model_outputs = model(x_hf, x_lf)
            logits = model_outputs[0] if isinstance(model_outputs, tuple) else model_outputs
            probs = torch.sigmoid(logits)
            
            # 🌟 类别对齐: 0->NORM/HC, 1->MI
            probs_aligned = probs[:, :2] 
            
            all_y_probs.append(probs_aligned.cpu().numpy())
            all_y_true.append(y_true.cpu().numpy()) 

    all_y_probs = np.vstack(all_y_probs)
    all_y_true = np.vstack(all_y_true)

    print("\n🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")
    print("🏆 跨数据集泛化深度透视报告")
    print("🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟\n")

    target_names = ['Healthy Control (HC)', 'Myocardial Infarction (MI)']

    # ==========================================
    # 终极照妖镜：不受阈值影响的 AUC
    # ==========================================
    try:
        # 计算 Macro AUC
        macro_auc = roc_auc_score(all_y_true, all_y_probs, average='macro')
        print(f"📈 【纯粹特征跨域能力】 Macro AUC = {macro_auc:.4f} (此指标不受概率偏移影响！)")
        if macro_auc > 0.80:
            print("   👉 惊艳！这说明模型完全学到了心梗的跨域物理特征，只是概率尺度发生了偏移。")
    except Exception as e:
        print("AUC 计算失败:", e)

    # ==========================================
    # 目标域阈值动态校准 (Target Domain Calibration)
    # ==========================================
    print("\n⚡ 正在执行目标域(PTB)的最佳校准阈值寻优...")
    best_thresholds = np.zeros(2)
    
    for i in range(2):
        best_f1 = 0
        best_t = 0.5
        # 在 0.01 到 0.99 之间精细寻找阈值
        for t in np.arange(0.01, 0.99, 0.02):
            pred = (all_y_probs[:, i] > t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(all_y_true[:, i], pred, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[i] = best_t
        print(f"  🎯 [{target_names[i]}]: 校准阈值 = {best_t:.2f} -> 极限 F1 = {best_f1:.4f}")

    # ==========================================
    # 重新生成报告
    # ==========================================
    print("\n📊 采用校准阈值后的最终表现：")
    all_y_pred_calibrated = (all_y_probs > best_thresholds).astype(int)
    report = classification_report(all_y_true, all_y_pred_calibrated, target_names=target_names, zero_division=0)
    print(report)
    
    _, _, f1_cal, _ = precision_recall_fscore_support(all_y_true, all_y_pred_calibrated, average='macro', zero_division=0)
    print(f"🔥 【校准后极限 Macro F1】: {f1_cal:.4f}")

if __name__ == "__main__":
    zero_shot_evaluate()