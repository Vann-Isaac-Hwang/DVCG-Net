import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# 请确保您的网络结构导入正确
from models.dual_model import DualFreqDGNN 
from data_process.dataset import DualFreqECGDataset 

def run_few_shot_10hz_evaluation():
    print("🚀 启动 10Hz 跨域泛化极限测试 (Few-Shot Fine-tuning)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # =========================================================================
    # ⚠️【请核对以下路径】⚠️
    # 1. 您的 10Hz 模型权重所在的文件夹 (如果是当前目录，请写 "./")
    checkpoint_dir = "./ablations_workspace/checkpoints/"  
    
    # 2. 目标域 (PTB) 10Hz 预处理数据的文件夹
    ptb_data_dir = "../Project_Data/PTB_Data/processed" 
    
    # 3. 将您跑出的 10Hz 消融权重名称填入字典中
    # 只要存在的文件就会被自动加载测试，不存在的会自动跳过
    models_to_test = {
        "Ours (DualFreqDGNN + SupCon)": "Ours_Full_10Hz_best.pth", 
        "w/o EdgeDrop": "wo_EdgeDrop_10Hz_best.pth",  # 如果有的话
        "w/o SupCon": "wo_SupCon_10Hz_best.pth",      # 如果有的话
        "w/o DGNN": "wo_DGNN_10Hz_best.pth",
        "LF-Only": "LF_Only_10Hz_best.pth",
        "HF-Only": "HF_Only_10Hz_best.pth"
    }
    
    # 极少样本测试比例：
    # 10% 的结果用于填 Table 3 消融大表；所有比例用于画 Few-Shot 折线图
    ratios = [0.02, 0.05, 0.10, 0.20]
    finetune_epochs = 50  # 冻结骨干后的分类器微调轮数
    learning_rate = 2e-3
    # =========================================================================

    print(f"🔍 扫描目标域 (PTB) 10Hz 数据集以进行分层抽样: {ptb_data_dir}")
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
    
    # 记录结果的大字典
    results = {model_name: {"AUC": [], "F1": []} for model_name in models_to_test.keys()}

    for model_name, weight_file in models_to_test.items():
        weight_path = os.path.join(checkpoint_dir, weight_file)
        if not os.path.exists(weight_path):
            print(f"⚠️ 跳过: 找不到权重 [{weight_file}]")
            continue
            
        print(f"\n==================================================")
        print(f"🧠 正在评估模型: {model_name}")
        print(f"==================================================")
        
        for ratio in ratios:
            print(f"  👉 跨域微调数据量: {ratio*100:.0f}% (约 {int(dataset_size * ratio)} 个样本)")
            
            # 1. 每次循环重新加载干净的预训练权重 (防止被上个ratio污染)
            # 注：如果您 LF/HF 的消融使用了不同的网络 Class，请在这里进行 if 判断实例化
            model = DualFreqDGNN(num_nodes=12, hidden_dim=64, num_classes=5, edge_dropout=0.0).to(device)
            checkpoint = torch.load(weight_path, map_location=device)
            
            # 兼容保存了完整 dict 或者只保存了 state_dict 的情况
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            # 2. 冻结骨干网络，仅重置并放开最后一层分类器 (Linear Probing / Fine-tuning)
            for param in model.parameters():
                param.requires_grad = False
                
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, 2).to(device) # PTB 是二分类 (HC vs MI)
            for param in model.classifier.parameters():
                param.requires_grad = True

            # 3. 极少样本严格分层切分
            train_idx, test_idx = train_test_split(
                indices, test_size=(1 - ratio), random_state=42, stratify=all_labels
            )
            
            train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=32, shuffle=True)
            test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=64, shuffle=False)
            
            # 自动计算类别不平衡权重 (Pos Weight)
            pos_weight = torch.tensor([
                (dataset_size - hc_count) / (hc_count + 1e-5),  
                (dataset_size - mi_count) / (mi_count + 1e-5)   
            ]).to(device)
            
            optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay=1e-2)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 

            # 4. 极少样本快速微调
            model.train()
            for epoch in range(finetune_epochs):
                for batch in train_loader:
                    x_hf, x_lf, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    y = y[:, :2].float() # 截取前两个类
                    
                    optimizer.zero_grad()
                    outputs = model(x_hf, x_lf)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    loss = criterion(logits, y)
                    loss.backward()
                    optimizer.step()

            # 5. 在庞大的未知跨域测试集上进行最终阅卷
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
            
            # 记录成绩
            try:
                macro_auc = roc_auc_score(all_y_true, all_y_probs, average='macro')
                _, _, macro_f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='macro', zero_division=0)
                
                results[model_name]["AUC"].append(macro_auc)
                results[model_name]["F1"].append(macro_f1)
                print(f"      ✅ 结果 -> AUC: {macro_auc:.4f} | F1: {macro_f1:.4f}")
            except Exception as e:
                print(f"      ❌ 计算出错: {e}")
                results[model_name]["AUC"].append(0)
                results[model_name]["F1"].append(0)

    # 打印最终 Markdown 大表，直接用于粘贴到论文
    print("\n\n🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")
    print("🏆 论文极少样本折线图数据 (Few-Shot Ratio Curve)")
    print("🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟\n")
    
    ratio_str = " | ".join([f"{r*100:.0f}%" for r in ratios])
    print(f"| Model / Finetune Ratio | {ratio_str} |")
    print("| :--- " + "| :---: " * len(ratios) + "|")
    
    for model_name, metrics in results.items():
        if len(metrics["AUC"]) > 0:
            auc_str = " | ".join([f"{val:.4f}" for val in metrics["AUC"]])
            print(f"| **{model_name} (AUC)** | {auc_str} |")
            
    print("\n\n(提示: 论文 Table 3 的跨域泛化列，请直接提取上方 10% 列的数据填入！)")

if __name__ == "__main__":
    run_few_shot_10hz_evaluation()