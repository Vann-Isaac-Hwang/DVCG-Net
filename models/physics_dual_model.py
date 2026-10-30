import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ECGBackbone

# ==========================================
# 0. 动态图发生器 (保持不变)
# ==========================================
class DynamicGraphBuilder(nn.Module):
    def __init__(self, num_nodes=12, edge_dropout=0.5):
        super().__init__()
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, 10))
        self.node_emb2 = nn.Parameter(torch.randn(10, num_nodes))
        self.dropout = nn.Dropout(p=edge_dropout)

    def forward(self):
        raw_adj = F.relu(torch.mm(self.node_emb1, self.node_emb2))
        mean_val = raw_adj.mean()
        mask_value = torch.full_like(raw_adj, -1e9) 
        sparse_adj = torch.where(raw_adj < mean_val * 0.5, mask_value, raw_adj)
        adj = F.softmax(sparse_adj, dim=1)
        return self.dropout(adj)

# ==========================================
# 1. 离散展开版 ODE：高频阻尼弹簧 (极度稳定)
# ==========================================
class DiscreteHF_Oscillator(nn.Module):
    def __init__(self, num_nodes, hidden_dim, edge_dropout, steps=4, dt=0.25):
        super().__init__()
        self.graph_builder = DynamicGraphBuilder(num_nodes, edge_dropout)
        self.k_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(0.1)) # 阻尼系数
        self.steps = steps # 演化步数
        self.dt = dt       # 时间步长
        self.bn = nn.BatchNorm1d(hidden_dim)
        # 🚀 零初始化魔法：初始状态下，弹簧力为 0！
        nn.init.zeros_(self.k_conv.weight)
        nn.init.zeros_(self.k_conv.bias)

    def forward(self, x):
        adj = self.graph_builder()
        v = torch.zeros_like(x) # 初始速度为0
        
        # 🚀 显式欧拉展开：把连续积分变成一个稳定的 4 层 for 循环
        for _ in range(self.steps):
            # 1. 计算图拉普拉斯力 (Ax - x)
            Ax = torch.einsum('ncv,vw->ncw', x, adj)
            laplacian = Ax - x
            force = torch.tanh(self.k_conv(laplacian)) # Tanh 防爆炸
            
            # 2. 计算速度和位置的更新 (微分方程的离散化)
            gamma_val = F.softplus(self.gamma)
            v_next = v + self.dt * (force - gamma_val * v)
            x_next = x + self.dt * v
            
            # 3. 步进
            v = v_next
            x = x_next
            
        return F.relu(self.bn(x))

# ==========================================
# 2. 离散展开版 PDE：低频热扩散 (极度稳定)
# ==========================================
class DiscreteLF_Diffusion(nn.Module):
    def __init__(self, num_nodes, hidden_dim, edge_dropout, steps=4, dt=0.25):
        super().__init__()
        self.graph_builder = DynamicGraphBuilder(num_nodes, edge_dropout)
        self.c_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.steps = steps
        self.dt = dt
        self.bn = nn.BatchNorm1d(hidden_dim)
        # 🚀 零初始化魔法：初始状态下，扩散力为 0！
        nn.init.zeros_(self.c_conv.weight)
        nn.init.zeros_(self.c_conv.bias)

    def forward(self, x):
        adj = self.graph_builder()
        
        # 🚀 热传导方程的显式展开
        for _ in range(self.steps):
            Ax = torch.einsum('ncv,vw->ncw', x, adj)
            laplacian = Ax - x
            dx = torch.tanh(self.c_conv(laplacian))
            # x_t+1 = x_t + dt * dx/dt
            x = x + self.dt * dx
            
        return F.relu(self.bn(x))

# ==========================================
# 3. 终极主网络 (无需更换引用方式)
# ==========================================
class PhysicsDualFreqDGNN(nn.Module):
    def __init__(self, num_nodes=12, hidden_dim=64, num_classes=5, edge_dropout=0.5):
        super(PhysicsDualFreqDGNN, self).__init__()
        
        self.backbone_hf = nn.ModuleList([ECGBackbone(hidden_dim) for _ in range(num_nodes)])
        self.backbone_lf = nn.ModuleList([ECGBackbone(hidden_dim) for _ in range(num_nodes)])
        
        # 🌟 替换为离散物理块
        self.physics_hf = nn.Sequential(
            DiscreteHF_Oscillator(num_nodes, hidden_dim, edge_dropout),
            DiscreteHF_Oscillator(num_nodes, hidden_dim, edge_dropout)
        )
        self.physics_lf = nn.Sequential(
            DiscreteLF_Diffusion(num_nodes, hidden_dim, edge_dropout),
            DiscreteLF_Diffusion(num_nodes, hidden_dim, edge_dropout)
        )
        
        self.proj_head = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.gate_net = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes) # 注意这里会自动拿到 ds_info['num_classes']
        )

    def forward(self, x_hf, x_lf):
        batch_size = x_hf.size(0)
        
        hf_feats, lf_feats = [], []
        for i in range(12):
            hf_feats.append(self.backbone_hf[i](x_hf[:, i:i+1, :].repeat(1, 12, 1)))
            lf_feats.append(self.backbone_lf[i](x_lf[:, i:i+1, :].repeat(1, 12, 1)))
        
        f_hf = torch.stack(hf_feats, dim=2).transpose(1, 2)
        f_lf = torch.stack(lf_feats, dim=2).transpose(1, 2)
        
        raw_hf_flat = f_hf.transpose(1, 2).reshape(batch_size, -1)
        raw_lf_flat = f_lf.transpose(1, 2).reshape(batch_size, -1)
        
        # 🚀 核心手术！把投影头接在原始 CNN 特征上，免受物理层初期的梯度毒害！
        p_hf = self.proj_head(raw_hf_flat)
        p_lf = self.proj_head(raw_lf_flat)
        
        # 物理演化开始
        z_hf = self.physics_hf(f_hf.transpose(1, 2)) 
        z_lf = self.physics_lf(f_lf.transpose(1, 2))
        
        z_hf_flat = z_hf.reshape(batch_size, -1)
        z_lf_flat = z_lf.reshape(batch_size, -1)
        
        fused_pre = torch.cat([z_hf_flat, z_lf_flat], dim=1)
        gate = torch.sigmoid(self.gate_net(fused_pre) / 2.0) 
        
        # 门控融合
        z_fused = gate * z_hf_flat + (1 - gate) * z_lf_flat
        
        # 残差保底
        z_final = z_fused + raw_hf_flat + raw_lf_flat
        
        logits = self.classifier(z_final)
        
        return logits, p_hf, p_lf