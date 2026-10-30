# main_ablation.py
import os
import argparse
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# 导入您项目内的自定义模块
from data_process.dataset import DualFreqECGDataset
from models.dual_model import DualFreqDGNN
from utils.config_parser import load_config
from utils.registry import get_dataset_info

# 🌟 修复: 完整导入三种核心 Loss
from utils.losses import MultiLabelFairLoss, InfoNCELoss, MultiLabelSupConLoss

logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_args():
    parser = argparse.ArgumentParser(description="双频图神经网络 - 消融实验专用引擎")
    parser.add_argument('--exp_name', type=str, required=True, help="本次消融实验的名称")
    parser.add_argument('--dataset', type=str, default='PTBXL')
    parser.add_argument('--data_dir', type=str, required=True, help="指定数据集路径")
    parser.add_argument('--edge_dropout', type=float, default=0.0, help="图拓扑边丢弃率")
    parser.add_argument('--alpha', type=float, default=0.2, help="MLFL 的 alpha 参数")
    
    # 🌟 新增的消融控制开关
    parser.add_argument('--no_supcon', action='store_true', help="禁用 SupCon 对比学习")
    parser.add_argument('--no_dgnn', action='store_true', help="禁用 DGNN 图网络")
    parser.add_argument('--freq_mode', type=str, choices=['Full', 'LF', 'HF'], default='Full', help="频段模式")
    
    return parser.parse_args()

def main():
    args = get_args()
    logging.info(f"========== 开始消融实验: {args.exp_name} ==========")
    
    # 1. 加载配置与设备
    cfg = load_config()
    device = torch.device(cfg.env.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.env.seed)

    # 2. 数据集加载
    ds_info = get_dataset_info(args.dataset)
    dataset = DualFreqECGDataset(args.data_dir)
    
    # ⚠️ 【防泄露警告】
    # 下方使用的是随机划分，仅为了消融实验能快速跑通对比。
    # 真实跨域或严谨 5-Fold 请务必使用 scripts/maintenance/fix_ptb_leakage.py 中的患者级划分逻辑。
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=cfg.env.seed)
    train_idx, val_idx = next(kf.split(range(len(dataset)))) 
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=cfg.train.batch_size, shuffle=False)

    # 3. 初始化模型
    model = DualFreqDGNN(
        num_nodes=cfg.model.num_nodes, 
        hidden_dim=cfg.model.hidden_dim, 
        num_classes=ds_info['num_classes'],
        edge_dropout=args.edge_dropout
    ).to(device)

    # ==========================================
    # 🔪 动态实施消融手术
    # ==========================================
    if args.no_dgnn:
        logging.info("🔪 [消融] 已切除 DGNN (替换为 Identity)")
        model.dgnn_hf = nn.Identity()
        model.dgnn_lf = nn.Identity()

    if args.freq_mode in ['LF', 'HF']:
        logging.info(f"🔪 [消融] 已开启单频段模式: {args.freq_mode}-Only")
        # 使用 Hook 在分类器前向传播前强制清零另一半特征
        def get_mask_hook(mode):
            def hook(module, hook_args):
                fused = hook_args[0].clone()
                # 假设隐藏层维度拼接后是 1536 (768 + 768)
                half_dim = fused.shape[1] // 2 
                # 假设拼接顺序为 torch.cat([z_hf, z_lf], dim=-1)
                if mode == "LF":
                    fused[:, :half_dim] = 0.0  # 高频清零
                elif mode == "HF":
                    fused[:, half_dim:] = 0.0  # 低频清零
                return (fused,)
            return hook
        model.classifier.register_forward_pre_hook(get_mask_hook(args.freq_mode))

    # 4. 优化器与损失函数
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    
    # 💡 [修复]: 传入 num_classes
    criterion = MultiLabelFairLoss(num_classes=ds_info['num_classes'], alpha=args.alpha)
    
    # 💡 [修复]: 实例化无监督跨频对比和有监督病理语义对比 (带安全回退机制)
    temp_cl = getattr(cfg.loss, 'temperature', 0.1) if hasattr(cfg, 'loss') else 0.1
    criterion_cl = InfoNCELoss(temperature=temp_cl).to(device)
    criterion_supcon = MultiLabelSupConLoss(temperature=0.1).to(device)

    epochs = cfg.train.num_epochs
    best_val_loss = float('inf')

    # 5. 训练主循环
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}", leave=False):
            x_hf, x_lf, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            
            optimizer.zero_grad()
            logits, z_hf, z_lf = model(x_hf, x_lf)
            
            # 计算基础的分类 Loss
            loss_out = criterion(logits, y)
            base_loss = loss_out[0] if isinstance(loss_out, tuple) else loss_out
            
            # ==========================================
            # 💡 [已修复 Bug] 正确的分支 Loss 计算与完美融合
            # ==========================================
            if args.no_supcon:
                # 🔪 剥离对比学习 (wo_SupCon)：只保留基础分类损失
                loss = base_loss
            else:
                # 🚀 满血版 (Ours_Full)：三位一体联合优化
                loss_cl = criterion_cl(z_hf, z_lf)          # 1. 无监督：跨频段特征对齐
                loss_supcon = criterion_supcon(z_lf, y)     # 2. 有监督：基于标签的病理语义聚类
                
                lambda_cl = getattr(cfg.loss, 'alpha', 0.1) if hasattr(cfg, 'loss') else 0.1
                lambda_supcon = 0.1
                
                # 总损失 = 基础分类 + 无监督跨频对比 + 有监督语义对比
                loss = base_loss + lambda_cl * loss_cl + lambda_supcon * loss_supcon
            
            # 直接反向传播 (不再有被错误覆盖的隐患)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
            
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{epochs}", leave=False):
                x_hf, x_lf, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                logits, _, _ = model(x_hf, x_lf)
                
                loss_out = criterion(logits, y)
                val_loss += loss_out[0].item() if isinstance(loss_out, tuple) else loss_out.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 实时打印每轮日志
        logging.info(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('ablations_workspace/checkpoints', exist_ok=True)
            save_path = f"ablations_workspace/checkpoints/{args.exp_name}_best.pth"
            torch.save(model.state_dict(), save_path)
            logging.info(f"🌟 新的最佳模型已保存 -> {save_path}")

    logging.info(f"========== {args.exp_name} 训练完成！==========")

if __name__ == '__main__':
    main()
