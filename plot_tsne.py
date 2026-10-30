import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")

# ========== 导入项目核心模块 ==========
from models.dual_model import DualFreqDGNN 
from data_process.dataset import DualFreqECGDataset 
# ============================================

def get_tsne_plot():
    print("🚀 启动特征空间诊断：见证 SupCon 对比学习的语义对齐...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # =========================================================================
    # ⚠️ 确保这里的路径是指向您刚刚跑出包含 SupCon Loss 的最佳权重
    model_path = "./checkpoints/PTBXL_MLFL_Weights/best_dual_dgnn_ptbxl_mlfl_fold_0.pth" 
    ptbxl_data_dir = "../Project_Data/PTBXL_Data/processed_freq20" # 源域数据路径
    ptb_data_dir = "../Project_Data/PTB_Data/processed_freq20"   # 目标域数据路径
    
    num_samples_per_domain = 400 
    # =========================================================================

    # 1. 初始化模型
    print("📦 加载包含 SupCon 知识的预训练模型...")
    model = DualFreqDGNN(num_nodes=12, hidden_dim=64, num_classes=5, edge_dropout=0.0).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 3. 准备数据收集容器
    all_features = []
    all_labels = []   # 0: NORM/HC, 1: MI
    all_domains = []  # 0: PTB-XL (源域), 1: PTB (目标域)

    def process_loader(loader, domain_label, domain_name):
        print(f"📂 正在提取 {domain_name} 的特征...")
        count = 0
        with torch.no_grad():
            for batch in loader:
                if count >= num_samples_per_domain:
                    break
                
                x_hf = batch[0].to(device)
                x_lf = batch[1].to(device)
                y_true = batch[2]
                
                # 🌟 核心魔法：我们不需要 Hook 了！
                # 直接获取被 SupCon 强力约束过的低频特征 p_lf
                logits, p_hf, p_lf = model(x_hf, x_lf)
                
                # 提取低频特征向量 (Batch_size, 768)
                feats = p_lf.detach().cpu().numpy() 
                
                # 筛选纯 NORM 或纯 MI
                for i in range(len(y_true)):
                    if y_true[i, 0] == 1 and y_true[i, 1] == 0:   # 纯 NORM/HC
                        all_features.append(feats[i])
                        all_labels.append("NORM/HC")
                        all_domains.append(domain_name)
                        count += 1
                    elif y_true[i, 0] == 0 and y_true[i, 1] == 1: # 纯 MI
                        all_features.append(feats[i])
                        all_labels.append("MI")
                        all_domains.append(domain_name)
                        count += 1
                        
                    if count >= num_samples_per_domain:
                        break

    # 4. 加载并提取特征
    ptbxl_dataset = DualFreqECGDataset(processed_data_dir=ptbxl_data_dir)
    ptb_dataset = DualFreqECGDataset(processed_data_dir=ptb_data_dir)
    
    ptbxl_loader = DataLoader(ptbxl_dataset, batch_size=64, shuffle=True)
    ptb_loader = DataLoader(ptb_dataset, batch_size=64, shuffle=True)

    process_loader(ptbxl_loader, 0, "PTB-XL (Source)")
    process_loader(ptb_loader, 1, "PTB (Target)")

    # 5. 执行 t-SNE 降维
    print("🌀 正在执行 t-SNE 高维空间压缩...")
    all_features = np.array(all_features)
    
    if len(all_features.shape) > 2:
        all_features = all_features.reshape(all_features.shape[0], -1)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(all_features)

    # 6. 绘图
    print("🎨 正在绘制特征分布图...")
    plt.figure(figsize=(10, 8))
    
    sns.scatterplot(
        x=features_2d[:, 0], 
        y=features_2d[:, 1],
        hue=all_labels,      
        style=all_domains,   
        palette={'NORM/HC': '#2ca02c', 'MI': '#d62728'},
        s=80, alpha=0.8
    )
    
    plt.title('t-SNE Space: Semantic Alignment by Cross-Domain SupCon', fontsize=14, pad=15)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = './tsne_supcon_aligned.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存至: {save_path}")

if __name__ == "__main__":
    get_tsne_plot()