import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ECGBackbone
from .baselines import BasicBlock1D

# ==========================================
# 0. 保持时序维度的主干网络包装器
# ==========================================
class TemporalBackboneWrapper(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.bb = ECGBackbone(hidden_dim)
        # 拦截 ECGBackbone 的特征，layer3 输出通道是 256
        self.dim_reduce = nn.Conv1d(256, hidden_dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        # 逐层提取特征，跳过最后的 AdaptiveAvgPool1d，从而保留时间维度
        x = self.bb.stem(x)
        x = self.bb.layer1(x)
        x = self.bb.layer2(x)
        x = self.bb.layer3(x)
        # x 维度: [Batch, 256, Time]
        x = F.relu(self.bn(self.dim_reduce(x)))
        return x # 降维至: [Batch, hidden_dim, Time]

# ==========================================
# 1. 时间卷积网络模块 (TCN)
# ==========================================
class TCN_Module(nn.Module):
    def __init__(self, channels, dilation=2):
        super().__init__()
        # 保证输入输出长度一致 (Padding = dilation * (kernel_size - 1) / 2)
        padding = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        # x 维度: [Batch, Channels, Nodes, Time]
        B, C, V, T = x.shape
        # 折叠 Batch 和 Nodes 维度，使得 Conv1d 专门作用于 Time 维度
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * V, C, T)
        out = F.relu(self.bn(self.conv(x_reshaped)))
        out = out.reshape(B, V, C, T).permute(0, 2, 1, 3)
        return out + x # 经典的残差连接

# ==========================================
# 2. 动态图卷积模块 (DGNN - 升级支持时间维度)
# ==========================================
class DGNN_Module(nn.Module):
    def __init__(self, num_nodes=12, hidden_dim=64, edge_dropout=0.5):
        super().__init__()
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, 10))
        self.node_emb2 = nn.Parameter(torch.randn(10, num_nodes))
        
        # 升级为 Conv2d，兼容 4D 时空张量
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.dropout = nn.Dropout(p=edge_dropout)

    def forward(self, x):
        # x 维度: [Batch, Channels, Nodes, Time]
        raw_adj = F.relu(torch.mm(self.node_emb1, self.node_emb2))
        mean_val = raw_adj.mean()
        mask_value = torch.full_like(raw_adj, -1e9)
        sparse_adj = torch.where(raw_adj < mean_val * 0.5, mask_value, raw_adj)
        adj = self.dropout(F.softmax(sparse_adj, dim=1))
        
        # 在空间节点维度 (v, w) 进行消息传递，保留通道 (c) 和时间 (t) 不受影响
        x_gcn = torch.einsum('ncvt,vw->ncwt', x, adj)
        x_out = F.relu(self.bn(self.conv(x_gcn)))
        return x_out

# ==========================================
# 3. 终极时空块 (Spatiotemporal Block - 支持顺序一键切换)
# ==========================================
class ST_DGNN_Block(nn.Module):
    def __init__(
        self,
        num_nodes=12,
        hidden_dim=64,
        edge_dropout=0.5,
        tcn_order='tcn_first',
        use_tcn=True,
        use_dgnn=True
    ):
        super().__init__()
        self.tcn_order = tcn_order
        self.use_tcn = use_tcn
        self.use_dgnn = use_dgnn

        self.tcn = TCN_Module(hidden_dim, dilation=2)
        self.dgnn = DGNN_Module(num_nodes, hidden_dim, edge_dropout)

    def forward(self, x):
        if self.use_tcn and self.use_dgnn:
            if self.tcn_order == 'tcn_first':
                return self.dgnn(self.tcn(x))
            else:
                return self.tcn(self.dgnn(x))

        elif self.use_tcn and not self.use_dgnn:
            return self.tcn(x)

        elif not self.use_tcn and self.use_dgnn:
            return self.dgnn(x)

        else:
            return x

class ResNetGlobalEncoder(nn.Module):
    """
    Raw ECG global encoder.
    Input: x_raw = x_hf + x_lf, shape [B, 12, T]
    Output: global morphology feature, shape [B, 512]
    """
    def __init__(self, num_channels=12):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock1D, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        features = out.view(out.size(0), -1)
        return features


# ==========================================
# 3.5 自适应 FFT 视图模块（Raw-preserving AFF View）
# ==========================================
class AdaptiveFourierView(nn.Module):
    """
    Adaptive Fourier view inspired by DeMa's Adaptive Fourier Filter.

    Input:  x_raw, shape [B, C, T]
    Output: adaptive spectral view, shape [B, C, T]

    It performs rFFT along the temporal axis, ranks frequency bins by the
    sample-wise averaged amplitude across leads, keeps the top-ratio dominant
    bins, and then reconstructs a time-domain view with irFFT.

    component:
        dominant: return top-energy frequency reconstruction.
        residual: return complementary reconstruction.
        split:    caller may use dominant/residual separately via return_pair=True.
    """
    def __init__(self, top_ratio=0.25, component='dominant', keep_dc=True):
        super().__init__()
        self.top_ratio = float(top_ratio)
        self.component = component
        self.keep_dc = keep_dc
        if not (0.0 < self.top_ratio < 1.0):
            raise ValueError(f"top_ratio should be in (0, 1), got {top_ratio}")
        if self.component not in ['dominant', 'residual', 'split']:
            raise ValueError(f"Unsupported AFF component: {component}")

    def _build_mask(self, spectrum):
        # spectrum: [B, C, F]
        B, C, Freq = spectrum.shape
        amp = spectrum.abs().mean(dim=1)  # [B, F]

        # Exclude DC from ranking to avoid selecting baseline offset only.
        if Freq <= 1:
            return torch.ones(B, Freq, device=spectrum.device, dtype=spectrum.real.dtype)

        k = max(1, int(round((Freq - 1) * self.top_ratio)))
        k = min(k, Freq - 1)
        idx = torch.topk(amp[:, 1:], k=k, dim=-1).indices + 1

        mask = torch.zeros(B, Freq, device=spectrum.device, dtype=spectrum.real.dtype)
        mask.scatter_(1, idx, 1.0)
        if self.keep_dc:
            mask[:, 0] = 1.0
        return mask

    def forward(self, x_raw, return_pair=False):
        T = x_raw.size(-1)
        X = torch.fft.rfft(x_raw.float(), dim=-1, norm='ortho')
        mask = self._build_mask(X).unsqueeze(1)  # [B, 1, F]

        # Compute only the reconstruction that is actually requested. The previous
        # implementation always ran two irFFT operations even in dominant/residual mode.
        if return_pair or self.component == 'split':
            x_dom = torch.fft.irfft(
                X * mask, n=T, dim=-1, norm='ortho'
            ).to(dtype=x_raw.dtype)
            x_res = torch.fft.irfft(
                X * (1.0 - mask), n=T, dim=-1, norm='ortho'
            ).to(dtype=x_raw.dtype)
            return x_dom, x_res

        if self.component == 'residual':
            return torch.fft.irfft(
                X * (1.0 - mask), n=T, dim=-1, norm='ortho'
            ).to(dtype=x_raw.dtype)

        return torch.fft.irfft(
            X * mask, n=T, dim=-1, norm='ortho'
        ).to(dtype=x_raw.dtype)

# ==========================================
# 4. 双频时空多视角核心大模型
# ==========================================
class DualFreqDGNN(nn.Module):
    # 🌟 核心控制：在这里切换 tcn_order='tcn_first' 或 'dgnn_first'
    def __init__(
        self,
        num_nodes=12,
        hidden_dim=64,
        num_classes=5,
        edge_dropout=0.5,
        tcn_order='tcn_first',
        ablation='full',
        use_resnet_fusion=False,
        use_aux_head=False,
        aff_top_ratio=0.25,
        aff_component='dominant'
    ):
        super(DualFreqDGNN, self).__init__()
        
        self.ablation = ablation
        self.use_resnet_fusion = use_resnet_fusion
        self.use_aux_head = use_aux_head
        self.aff_top_ratio = float(aff_top_ratio)
        self.aff_component = aff_component
        self.aff_view = AdaptiveFourierView(top_ratio=self.aff_top_ratio, component=self.aff_component)
        
        use_tcn = ablation not in ['wo_tcn']
        use_dgnn = ablation not in ['wo_dgnn']

        # 1. 高低频双路骨干特征提取 (使用新包装器，保留时间维度)
        self.backbone_hf = nn.ModuleList([TemporalBackboneWrapper(hidden_dim) for _ in range(num_nodes)])
        self.backbone_lf = nn.ModuleList([TemporalBackboneWrapper(hidden_dim) for _ in range(num_nodes)])
        
        # 2. 时空图卷积 (ST-DGNN)
        self.dgnn_hf = nn.Sequential(
            ST_DGNN_Block(
                num_nodes,
                hidden_dim,
                edge_dropout,
                tcn_order,
                use_tcn=use_tcn,
                use_dgnn=use_dgnn
            ),
            ST_DGNN_Block(
                num_nodes,
                hidden_dim,
                edge_dropout,
                tcn_order,
                use_tcn=use_tcn,
                use_dgnn=use_dgnn
            )
        )

        self.dgnn_lf = nn.Sequential(
            ST_DGNN_Block(
                num_nodes,
                hidden_dim,
                edge_dropout,
                tcn_order,
                use_tcn=use_tcn,
                use_dgnn=use_dgnn
            ),
            ST_DGNN_Block(
                num_nodes,
                hidden_dim,
                edge_dropout,
                tcn_order,
                use_tcn=use_tcn,
                use_dgnn=use_dgnn
            )
        )
        
        # 3. 对比学习投影头
        self.proj_head = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        if self.use_resnet_fusion:
            self.raw_encoder = ResNetGlobalEncoder(num_channels=num_nodes)
            fusion_dim = num_nodes * hidden_dim * 2 + 512
        else:
            fusion_dim = num_nodes * hidden_dim * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Optional auxiliary multi-label head for single-label training.
        # It has its own parameters, so CE softmax logits and auxiliary BCE logits do not conflict.
        self.aux_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        ) if self.use_aux_head else None

    def forward(self, x_hf, x_lf):
        batch_size = x_hf.size(0)

        # ========= 频率分支消融（当前 TCN-ST-DGNN 版本） =========
        # no_freq_split: 不做高低频分解，把 raw ECG 同时送入两路，保留参数量一致。
        # lf_only / hf_only: 输入层和特征层双重清零，避免被置零分支通过 BN/bias 产生可学习常量特征。
        if self.ablation == 'no_freq_split':
            x_raw = x_hf + x_lf
            x_hf = x_raw
            x_lf = x_raw
        elif self.ablation == 'no_freq_split_aff':
            # Raw-preserving Adaptive FFT view:
            # branch-1 keeps complete ECG morphology; branch-2 receives an adaptive spectral view.
            x_raw = x_hf + x_lf
            if self.aff_component == 'split':
                x_dom, x_res = self.aff_view(x_raw, return_pair=True)
                x_hf = x_dom
                x_lf = x_res
            else:
                x_hf = x_raw
                x_lf = self.aff_view(x_raw)
        elif self.ablation == 'lf_only':
            x_hf = torch.zeros_like(x_hf)
        elif self.ablation == 'hf_only':
            x_lf = torch.zeros_like(x_lf)
        
        # 第一步：逐导联提取时序特征
        hf_feats, lf_feats = [], []
        for i in range(12):
            hf_feats.append(self.backbone_hf[i](x_hf[:, i:i+1, :].repeat(1, 12, 1)))
            lf_feats.append(self.backbone_lf[i](x_lf[:, i:i+1, :].repeat(1, 12, 1)))
        
        # 组装为完整的时空张量: [Batch, Channels, Nodes, Time]
        # f_hf 维度: [Batch, 64, 12, Time]
        f_hf = torch.stack(hf_feats, dim=2)
        f_lf = torch.stack(lf_feats, dim=2)
        
        # 第二步：时空建模交互 (TCN + DGNN)
        z_hf = self.dgnn_hf(f_hf) # [Batch, 64, 12, Time]
        z_lf = self.dgnn_lf(f_lf)

        # 第三步：时间维度池化 (全导联全局时间平均池化)
        z_hf_pooled = z_hf.mean(dim=-1) # [Batch, 64, 12]
        z_lf_pooled = z_lf.mean(dim=-1)
        
        # 第四步：展平双频特征以输入分类器
        z_hf_flat = z_hf_pooled.reshape(batch_size, -1) # [Batch, 768]
        z_lf_flat = z_lf_pooled.reshape(batch_size, -1)

        if self.ablation == 'lf_only':
            z_hf_flat = torch.zeros_like(z_hf_flat)
        elif self.ablation == 'hf_only':
            z_lf_flat = torch.zeros_like(z_lf_flat)
        
        # 第五步：融合与预测
        fusion_features = [z_hf_flat, z_lf_flat]

        if self.use_resnet_fusion:
            x_raw = x_hf + x_lf
            z_raw = self.raw_encoder(x_raw)
            fusion_features.append(z_raw)

        fused = torch.cat(fusion_features, dim=1)
        logits = self.classifier(fused)

        if self.use_aux_head and self.aux_classifier is not None:
            aux_logits = self.aux_classifier(fused)
            return logits, z_hf_flat, z_lf_flat, aux_logits
        
        return logits, z_hf_flat, z_lf_flat

# ==========================================
# 5. 旧版 DualFreqDGNN 复现模块（5.19 comparison 版本核心结构）
# ==========================================
class OldDGNN_Block(nn.Module):
    """
    旧版动态图卷积模块：输入 [B, hidden_dim, num_nodes]，不保留时间维度。
    用于复现早期 DualFreqDGNN 架构，便于和当前 TCN-ST-DGNN 严格同协议对比。
    """
    def __init__(self, num_nodes=12, hidden_dim=64, edge_dropout=0.1):
        super(OldDGNN_Block, self).__init__()
        self.node_emb1 = nn.Parameter(torch.randn(num_nodes, 10))
        self.node_emb2 = nn.Parameter(torch.randn(10, num_nodes))
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=edge_dropout)

    def forward(self, x):
        # x: [B, C, V]
        raw_adj = F.relu(torch.mm(self.node_emb1, self.node_emb2))
        mean_val = raw_adj.mean()
        mask_value = torch.full_like(raw_adj, -1e9)
        sparse_adj = torch.where(raw_adj < mean_val * 0.5, mask_value, raw_adj)
        adj = F.softmax(sparse_adj, dim=1)
        adj = self.dropout(adj)
        x_gcn = torch.einsum('ncv,vw->ncw', x, adj)
        x_out = F.relu(self.bn(self.conv(x_gcn)))
        return x_out


class OldDualFreqDGNN(nn.Module):
    """
    旧版双频图网络复现：
    - 每个导联单独 ECGBackbone 编码，直接池化成 hidden_dim；
    - 高频/低频各自经过 2 层旧版 DGNN；
    - 展平 HF/LF 导联节点特征后拼接分类。

    支持 ablation:
    old_dual: 旧版完整结构；
    old_dual_wo_cl: 结构不变，但训练脚本会关闭 InfoNCE/SupCon；
    old_dual_lf_only / old_dual_hf_only: 单频分支消融；
    old_dual_no_freq_split: 不分频，raw ECG 同时送入 HF/LF 两路；
    old_dual_wo_dgnn: 跳过旧版 DGNN，仅用逐导联 backbone 特征。
    """
    def __init__(self, num_nodes=12, hidden_dim=64, num_classes=5, edge_dropout=0.1, ablation='old_dual', use_aux_head=False):
        super(OldDualFreqDGNN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.ablation = ablation
        self.use_aux_head = use_aux_head

        self.backbone_hf = nn.ModuleList([ECGBackbone(hidden_dim) for _ in range(num_nodes)])
        self.backbone_lf = nn.ModuleList([ECGBackbone(hidden_dim) for _ in range(num_nodes)])

        self.dgnn_hf = nn.Sequential(
            OldDGNN_Block(num_nodes, hidden_dim, edge_dropout),
            OldDGNN_Block(num_nodes, hidden_dim, edge_dropout),
        )
        self.dgnn_lf = nn.Sequential(
            OldDGNN_Block(num_nodes, hidden_dim, edge_dropout),
            OldDGNN_Block(num_nodes, hidden_dim, edge_dropout),
        )

        self.proj_head = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self.aux_classifier = nn.Sequential(
            nn.Linear(num_nodes * hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        ) if self.use_aux_head else None

    def forward(self, x_hf, x_lf):
        batch_size = x_hf.size(0)

        if self.ablation == 'old_dual_no_freq_split':
            x_raw = x_hf + x_lf
            x_hf = x_raw
            x_lf = x_raw
        elif self.ablation == 'old_dual_lf_only':
            x_hf = torch.zeros_like(x_hf)
        elif self.ablation == 'old_dual_hf_only':
            x_lf = torch.zeros_like(x_lf)

        hf_feats, lf_feats = [], []
        for i in range(self.num_nodes):
            # 旧版实现：单导联复制为 12 通道后送入 ECGBackbone，保持与 5.19 代码一致。
            hf_feats.append(self.backbone_hf[i](x_hf[:, i:i + 1, :].repeat(1, self.num_nodes, 1)))
            lf_feats.append(self.backbone_lf[i](x_lf[:, i:i + 1, :].repeat(1, self.num_nodes, 1)))

        # [B, 12, hidden_dim]
        f_hf = torch.stack(hf_feats, dim=2).transpose(1, 2)
        f_lf = torch.stack(lf_feats, dim=2).transpose(1, 2)

        # 转为 [B, hidden_dim, 12]
        hf_nodes = f_hf.transpose(1, 2)
        lf_nodes = f_lf.transpose(1, 2)

        if self.ablation == 'old_dual_wo_dgnn':
            z_hf = hf_nodes
            z_lf = lf_nodes
        else:
            z_hf = self.dgnn_hf(hf_nodes)
            z_lf = self.dgnn_lf(lf_nodes)

        z_hf_flat = z_hf.reshape(batch_size, -1)
        z_lf_flat = z_lf.reshape(batch_size, -1)

        if self.ablation == 'old_dual_lf_only':
            z_hf_flat = torch.zeros_like(z_hf_flat)
        elif self.ablation == 'old_dual_hf_only':
            z_lf_flat = torch.zeros_like(z_lf_flat)

        fused = torch.cat([z_hf_flat, z_lf_flat], dim=1)
        logits = self.classifier(fused)
        if self.use_aux_head and self.aux_classifier is not None:
            aux_logits = self.aux_classifier(fused)
            return logits, z_hf_flat, z_lf_flat, aux_logits
        return logits, z_hf_flat, z_lf_flat
