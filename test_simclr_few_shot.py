import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

from models.baselines import SimCLR1D 
from data_process.dataset import DualFreqECGDataset 

def run_simclr_few_shot():
    print("🚀 启动临床真实场景：SimCLR-1D (无监督对比) 10% Few-Shot 微调...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # =========================================================================
    # ⚠️ 请确保这个名字和刚刚生成的权重文件完全一致！
    model_path = "./checkpoints/PTBXL_MLFL_Weights/best_dual_dgnn_ptbxl_mlfl_simclr1d_fold_0.pth" 
    ptb_data_dir = "../Project_Data/PTB_Data/processed_freq20"
    train_size_ratio = 0.10
    finetune_epochs = 50
    learning_rate = 2e-3
    # =========================================================================

    print("📦 加载 PTB-XL 预训练的 SimCLR-1D 无监督权重...")
    model = SimCLR1D(num_classes=5, num_channels=12).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    print("🔒 冻结特征提取层，仅开放分类器...")
    for param in model.parameters():
        param.requires_grad = False
        
    # 重置分类器为 2 分类 (健康 vs 心梗)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2).to(device)
    for param in model.classifier.parameters():
        param.requires_grad = True

    print("🔍 扫描数据集以进行分层抽样...")
    dataset = DualFreqECGDataset(processed_data_dir=ptb_data_dir)
    all_labels = []
    hc_count, mi_count = 0, 0
    for i in range(len(dataset)):
        y = dataset[i][2]
        if y[0] == 1:
            all_labels.append(0)
            hc_count += 1
        else:
            all_labels.append(1)
            mi_count += 1
            
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(
        indices, train_size=train_size_ratio, random_state=42, stratify=all_labels
    )

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=64, shuffle=False)

    pos_weight = torch.tensor([
        (len(dataset) - hc_count) / (hc_count + 1e-5),  
        (len(dataset) - mi_count) / (mi_count + 1e-5)   
    ]).to(device)

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print("🔥 开始进行 50 Epochs 的线性探测 (微调)...")
    model.train()
    for epoch in range(finetune_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x_hf, x_lf, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            y = y[:, :2].float()
            
            optimizer.zero_grad()
            # 🌟 使用合并后的信号提取特征进行评估，保证与 ResNet1D 公平对比
            logits, _ = model.forward_one_view(x_hf + x_lf)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    print("\n🧠 在 90% 未见过的 PTB 数据上进行最终推理...")
    model.eval()
    all_y_true, all_y_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_hf, x_lf, y_true = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            logits, _ = model.forward_one_view(x_hf + x_lf)
            probs = torch.sigmoid(logits)
            
            all_y_probs.append(probs.cpu().numpy())
            all_y_true.append(y_true[:, :2].cpu().numpy()) 

    all_y_probs = np.vstack(all_y_probs)
    all_y_true = np.vstack(all_y_true)
    all_y_pred = (all_y_probs > 0.5).astype(int)

    print("\n🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")
    print("🏆 SimCLR-1D 基线 10% Few-Shot 报告")
    print("🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟\n")
    
    target_names = ['Healthy Control (HC)', 'Myocardial Infarction (MI)']
    print(classification_report(all_y_true, all_y_pred, target_names=target_names, zero_division=0))
    
    macro_auc = roc_auc_score(all_y_true, all_y_probs, average='macro')
    print(f"📈 【SimCLR-1D 微调后真实 AUC】: {macro_auc:.4f}")

if __name__ == "__main__":
    run_simclr_few_shot()