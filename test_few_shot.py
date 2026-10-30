import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

from models.dual_model import DualFreqDGNN 
from data_process.dataset import DualFreqECGDataset 

def few_shot_finetune():
    print("🚀 启动临床真实场景：Few-Shot 小样本微调 (Linear Probing)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # =========================================================================
    # model_path = "./checkpoints/PTBXL_MLFL_Weights/best_dual_dgnn_ptbxl_mlfl_fold_0.pth"
    # ⚠️ 注意这里的文件名变成了带 wo_supcon 后缀的
    # model_path = "./checkpoints/PTBXL_MLFL_Weights/best_dual_dgnn_ptbxl_mlfl_wo_supcon_fold_0.pth"
    # ⚠️ 注意后缀变成了 lf_only
    # model_path = "./checkpoints/PTBXL_MLFL_Weights/best_dual_dgnn_ptbxl_mlfl_lf_only_fold_0.pth"
    # ⚠️ 注意后缀变成了 hf_only
    # model_path = "./checkpoints/PTBXL_MLFL_Weights/best_dual_dgnn_ptbxl_mlfl_hf_only_fold_0.pth"
    # ⚠️ 注意后缀变成了 wo_dgnn
    # model_path = "./checkpoints/PTBXL_MLFL_Weights/best_dual_dgnn_ptbxl_mlfl_wo_dgnn_fold_0.pth"
    # ⚠️ 注意后缀变成了 wo_edgedrop
    model_path = "./checkpoints/PTBXL_MLFL_Weights/best_dual_dgnn_ptbxl_mlfl_wo_edgedrop_fold_0.pth"

    ptb_data_dir = "../Project_Data/PTB_Data/processed_freq20"
    
    few_shot_ratio = 0.10 
    finetune_epochs = 50   # 增加 Epoch，因为加了正则化，需要更多时间收敛
    learning_rate = 2e-3
    # =========================================================================
    
    print("📦 加载 PTB-XL 预训练权重...")
    model = DualFreqDGNN(num_nodes=12, hidden_dim=64, num_classes=5, edge_dropout=0.0).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print("🔒 冻结特征提取层，仅开放分类器...")
    for param in model.parameters():
        param.requires_grad = False
    
    # 重置最后一层
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2).to(device)
    for param in model.classifier.parameters():
        param.requires_grad = True

    print("🔍 扫描数据集以进行分层抽样 (Stratified Sampling)...")
    full_dataset = DualFreqECGDataset(processed_data_dir=ptb_data_dir)
    
    # 提取标签用于分层抽样和计算权重
    all_labels = []
    hc_count, mi_count = 0, 0
    for i in range(len(full_dataset)):
        y = full_dataset[i][2] # y: [HC, MI, 0, 0, 0]
        # 判断是纯 HC 还是 纯 MI
        if y[0] == 1:
            all_labels.append(0)
            hc_count += 1
        else:
            all_labels.append(1)
            mi_count += 1
            
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    
    # ✅ 修复 1：分层抽样 stratify=all_labels，保证微调集里一定有 HC！
    train_idx, test_idx = train_test_split(
        indices, test_size=(1 - few_shot_ratio), random_state=42, stratify=all_labels
    )
    
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=32, shuffle=True)
    test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=64, shuffle=False)
    
    # ✅ 修复 2：计算 BCE Pos Weight，赋予少数类 HC 巨大的权重
    total_train = len(train_idx)
    # 计算训练集里的正负样本比例
    pos_weight = torch.tensor([
        (dataset_size - hc_count) / (hc_count + 1e-5),  # HC 的权重 (极大)
        (dataset_size - mi_count) / (mi_count + 1e-5)   # MI 的权重 (较小)
    ]).to(device)
    
    print(f"⚖️ 自动计算类别权重 (BCE Pos Weight): {pos_weight.cpu().numpy()}")

    # ✅ 修复 3：加入 weight_decay (L2 正则化) 防止在几十个样本上死记硬背
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 

    print(f"🔥 开始进行 {finetune_epochs} Epochs 的线性探测 (微调)...")
    model.train()
    for epoch in range(finetune_epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x_hf, x_lf, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            y = y[:, :2].float() 
            
            optimizer.zero_grad()
            outputs = model(x_hf, x_lf)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1}/{finetune_epochs} | Loss: {epoch_loss/len(train_loader):.4f}")

    print("\n🧠 在 90% 未见过的 PTB 数据上进行最终推理...")
    model.eval()
    all_y_true, all_y_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_hf, x_lf, y_true = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            outputs = model(x_hf, x_lf)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.sigmoid(logits)
            
            all_y_probs.append(probs.cpu().numpy())
            all_y_true.append(y_true[:, :2].cpu().numpy()) 

    all_y_probs = np.vstack(all_y_probs)
    all_y_true = np.vstack(all_y_true)

    all_y_pred = (all_y_probs > 0.5).astype(int)

    print("\n🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")
    print(f"🏆 Few-Shot 跨数据集微调报告 (仅用 {few_shot_ratio*100}% 目标数据)")
    print("🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟\n")
    
    target_names = ['Healthy Control (HC)', 'Myocardial Infarction (MI)']
    report = classification_report(all_y_true, all_y_pred, target_names=target_names, zero_division=0)
    print(report)
    
    try:
        macro_auc = roc_auc_score(all_y_true, all_y_probs, average='macro')
        print(f"📈 【微调后真实 AUC】: {macro_auc:.4f}")
    except:
        pass
        
    _, _, f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='macro', zero_division=0)
    print(f"🔥 【Few-Shot 终极 Macro F1】: {f1:.4f}")

if __name__ == "__main__":
    few_shot_finetune()