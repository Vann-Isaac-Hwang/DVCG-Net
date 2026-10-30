import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import datetime

from models.dual_model import DualFreqDGNN
from utils.config_parser import load_config

# ================= ⚙️ 顶层控制台 =================
# 我们锁定为您今晚跑出 0.90 高分的 PTBXL 满血版
TARGET_DATASET = 'PTBXL'
TARGET_LOSS = 'MLFL'
# ===============================================

def extract_adj_matrix(model_path, device, num_classes):
    """抽取满血模型的高低频邻接矩阵"""
    model = DualFreqDGNN(num_nodes=12, hidden_dim=64, num_classes=num_classes, edge_dropout=0.0).to(device)
    
    # 🌟 核心修复：强制兼容加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    with torch.no_grad():
        # 提取高频图的节点嵌入，并计算注意力邻接矩阵
        hf_emb1, hf_emb2 = model.dgnn_hf[0].node_emb1, model.dgnn_hf[0].node_emb2
        adj_hf = F.softmax(F.relu(torch.mm(hf_emb1, hf_emb2)), dim=1).cpu().numpy()
        
        # 提取低频图的节点嵌入，并计算注意力邻接矩阵
        lf_emb1, lf_emb2 = model.dgnn_lf[0].node_emb1, model.dgnn_lf[0].node_emb2
        adj_lf = F.softmax(F.relu(torch.mm(lf_emb1, lf_emb2)), dim=1).cpu().numpy()
        
    return adj_hf, adj_lf

def visualize_graphs():
    cfg = load_config()
    device = torch.device(cfg.env.device if torch.cuda.is_available() else "cpu")
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    target_weight_dir = os.path.join(cfg.env.checkpoint_dir, "PTBXL_MLFL_Weights")
    
    print(f"\n" + "="*50)
    print(f"🚀 正在提取医学可解释性拓扑图 (Interpretability)")
    print(f"📂 目标权重目录: {target_weight_dir}")
    print("="*50)

    # 直接锁定今天跑的最强满血版权重！
    filename = 'best_dual_dgnn_ptbxl_mlfl_fold_0.pth'
    model_path = os.path.join(target_weight_dir, filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到满血版权重: {model_path}，请确认文件名！")
                
    print(f"  -> 成功加载满血版模型...")
    adj_hf, adj_lf = extract_adj_matrix(model_path, device, num_classes=5)
            
    main_title = f"12-Lead ECG Spatial Topology (Ours)\nDataset: {TARGET_DATASET} | Method: {TARGET_LOSS} + SupCon"

    # 🎨 开始绘制热力图
    print("\n🎨 正在绘制高分辨率网络热力图...")
    os.makedirs(cfg.env.results_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})
    
    # 高频子图
    sns.heatmap(adj_hf, cmap='Reds', xticklabels=leads, yticklabels=leads, ax=axes[0], cbar_kws={'label': 'Connection Strength'})
    axes[0].set_title('High-Frequency (HF) Graph\n(Focus: QRS & Rapid Morphologies)', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Source Lead', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Target Lead', fontsize=12, fontweight='bold')
    
    # 低频子图
    sns.heatmap(adj_lf, cmap='Blues', xticklabels=leads, yticklabels=leads, ax=axes[1], cbar_kws={'label': 'Connection Strength'})
    axes[1].set_title('Low-Frequency (LF) Graph\n(Focus: P/T Waves & Slow Morphologies)', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('Source Lead', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Target Lead', fontsize=12, fontweight='bold')
    
    plt.suptitle(main_title, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 🛡️ 保存图片
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f'GraphTopology_{TARGET_DATASET}_{TARGET_LOSS}_single_{timestamp}.png'
    save_path = os.path.join(cfg.env.results_dir, image_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 大功告成！【动态图卷积拓扑图】已安全保存至: {save_path}")
    print(f"👉 快去文件夹里双击打开它看看吧！")

if __name__ == '__main__':
    visualize_graphs()