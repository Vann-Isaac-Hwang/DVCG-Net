import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# 引入您刚才改好的基线模型
from models.baselines import Transformer1D, FourierGNN1D, Mamba1D

def run_baselines_few_shot():
    print("🚀 启动基线模型 (Baselines) 跨域泛化极限测试 (Full Fine-Tuning模式)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # =========================================================================
    # ⚠️【请核对以下路径】⚠️
    # 1. 您跑完 main_train.py 后，这三个基线模型保存的最佳权重文件夹
    checkpoint_dir = "./checkpoints/PTBXL_BCE_Weights/"  
    
    # 2. 目标域 (PTB) 10Hz 预处理数据的文件夹
    ptb_data_dir = "../Project_Data/PTB_Data/processed" 
    
    # 3. 填入基线模型跑出来的权重文件名
    models_to_test = {
        "Transformer (Medformer)": {"file": "best_dual_dgnn_ptbxl_bce_transformer1d_fold_0.pth", "class": Transformer1D},
        "FourierGNN": {"file": "best_dual_dgnn_ptbxl_bce_fouriergnn1d_fold_0.pth", "class": FourierGNN1D},
        # "Mamba-1D": {"file": "best_mamba1d_ptbxl_bce_fold_0.pth", "class": Mamba1D} # 等跑了Mamba再打开
    }
    
    ratio = 0.10
    finetune_epochs = 50
    # 🌟 核心修改 1：全参数微调时，学习率必须调低，防止灾难性遗忘
    learning_rate = 1e-4
    # =========================================================================

    from data_process.dataset import DualFreqECGDataset
    full_dataset = DualFreqECGDataset(processed_data_dir=ptb_data_dir)
    
    all_labels = []
    hc_count, mi_count = 0, 0
    for i in range(len(full_dataset)):
        y = full_dataset[i][2] 
        if y[0] == 1:
            all_labels.append(0)
            hc_count += 1
        else:
            all_labels.append(1)
            mi_count += 1
            
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    
    results = {}

    for model_name, model_info in models_to_test.items():
        weight_path = os.path.join(checkpoint_dir, model_info["file"])
        if not os.path.exists(weight_path):
            print(f"⚠️ 跳过: 找不到权重 [{weight_path}]")
            continue
            
        print(f"\n==================================================")
        print(f"🧠 正在跨域评估基线: {model_name}")
        print(f"==================================================")
        
        # 1. 实例化对应的基线模型
        model = model_info["class"](num_classes=5).to(device)
        checkpoint = torch.load(weight_path, map_location=device)
        
        # 🌟 核心修改 2：加入严格的权重诊断日志
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            res = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            res = model.load_state_dict(checkpoint, strict=False)
            
        print(f"      🔍 [诊断日志] 权重加载未匹配层数: {len(res.missing_keys)}")
        if len(res.missing_keys) > 10:
            print(f"      ⚠️ 警告: 未匹配层数过多，可能权重格式有误或未成功加载！")
        
        # 2. 找到最后一层全连接层并替换为二分类 (PTB HC vs MI)
        if hasattr(model, 'mlp_head'): # Transformer
            in_features = model.mlp_head[-1].in_features
            model.mlp_head[-1] = nn.Linear(in_features, 2).to(device)
        elif hasattr(model, 'fc'): # FourierGNN
            # 🌟 修复：检查 fc 是不是 Sequential，如果是，取最后一层的 in_features
            if isinstance(model.fc, nn.Sequential):
                in_features = model.fc[-1].in_features
                model.fc[-1] = nn.Linear(in_features, 2).to(device)
            else:
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, 2).to(device)
        elif hasattr(model, 'classifier'): # Mamba
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, 2).to(device)

        # 🌟 核心修改 3：给予基线最公平的待遇，解冻所有参数！
        for param in model.parameters():
            param.requires_grad = True

        train_idx, test_idx = train_test_split(indices, test_size=(1 - ratio), random_state=42, stratify=all_labels)
        train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=32, shuffle=True)
        test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=64, shuffle=False)
        
        pos_weight = torch.tensor([(dataset_size - hc_count) / (hc_count + 1e-5), (dataset_size - mi_count) / (mi_count + 1e-5)]).to(device)
        
        # 🌟 核心修改 4：优化器必须接管整个模型的所有参数
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 

        model.train()
        for epoch in range(finetune_epochs):
            for batch in train_loader:
                x_hf, x_lf, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                y = y[:, :2].float()
                
                optimizer.zero_grad()
                outputs = model(x_hf, x_lf) 
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

        model.eval()
        all_y_true, all_y_probs = [], []
        with torch.no_grad():
            for batch in test_loader:
                x_hf, x_lf, y_true = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                outputs = model(x_hf, x_lf)
                probs = torch.sigmoid(outputs)
                all_y_probs.append(probs.cpu().numpy())
                all_y_true.append(y_true[:, :2].cpu().numpy()) 

        all_y_probs = np.vstack(all_y_probs)
        all_y_true = np.vstack(all_y_true)
        
        # 计算 AUC
        macro_auc = roc_auc_score(all_y_true, all_y_probs, average='macro')
        
        # 计算 F1-Score
        all_y_pred = (all_y_probs > 0.5).astype(int)
        _, _, macro_f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='macro', zero_division=0)
        
        results[model_name] = {"AUC": macro_auc, "F1": macro_f1}
        print(f"      ✅ {model_name} 跨域 10% 微调 -> AUC: {macro_auc:.4f} | F1: {macro_f1:.4f}")

    # 打印最终的大表
    print("\n\n🏆 论文 Table 2 基线跨域性能大比拼 🏆")
    print("| Model Architecture | 🚀 目标域跨域 (PTB 10%) Macro AUC | 🚀 目标域跨域 (PTB 10%) Macro F1 |")
    print("| :--- | :---: | :---: |")
    for name, metrics in results.items():
        print(f"| {name} | {metrics['AUC']:.4f} | {metrics['F1']:.4f} |")
    
    # 填入您自己的最佳成绩
    print(f"| **Ours (DualFreqDGNN)** | **0.9307** | **0.7903** | (用于对比参照)")

if __name__ == "__main__":
    run_baselines_few_shot()