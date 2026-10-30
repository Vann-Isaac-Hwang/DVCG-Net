import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    """
    标准的项目内 ResNet-1D 基线模型。
    为了兼容当前的 DataLoader，forward 函数接受 (x_hf, x_lf) 并将其还原为原始信号喂入网络。
    """
    def __init__(self, num_classes=5, num_channels=12):
        super(ResNet1D, self).__init__()
        self.in_planes = 64

        # 初始卷积层
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 残差块 (模拟 ResNet-18 的精简结构)
        self.layer1 = self._make_layer(BasicBlock1D, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # 投影头 (用于对比学习的特征空间)
        self.proj_head = nn.Sequential(
            nn.Linear(512 * BasicBlock1D.expansion, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        # 分类器
        self.classifier = nn.Linear(512 * BasicBlock1D.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x_hf, x_lf):
        # 🌟 兼容性黑科技：在模型内部把高低频加回来，还原成原始 ECG 信号！
        x = x_hf + x_lf 
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        features = out.view(out.size(0), -1) # 骨干特征 (512维)

        proj = self.proj_head(features)      # 用于 SupCon 对比学习的投影特征
        logits = self.classifier(features)   # 用于分类的 Logits

        return logits, proj

class SimCLR1D(nn.Module):
    """
    经典的无监督对比学习基线 (SimCLR-1D)。
    将高频(HF)和低频(LF)作为心电图的两个增强视图(Augmented Views)输入。
    """
    def __init__(self, num_classes=5, num_channels=12):
        super(SimCLR1D, self).__init__()
        self.in_planes = 64
        # 共享的编码器
        self.conv1 = nn.Conv1d(num_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock1D, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1D, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # 对比学习的非线性投影头
        self.proj_head = nn.Sequential(
            nn.Linear(512 * BasicBlock1D.expansion, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
        # 预训练阶段不需要分类器，但为了与测试脚本兼容，提供一个占位符
        self.classifier = nn.Linear(512 * BasicBlock1D.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_one_view(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        features = out.view(out.size(0), -1)
        proj = self.proj_head(features)
        logits = self.classifier(features)
        return logits, proj

    def forward(self, x_hf, x_lf):
        # 🌟 核心：将高频和低频分别通过共享编码器，得到两组特征
        logits_hf, proj_hf = self.forward_one_view(x_hf)
        logits_lf, proj_lf = self.forward_one_view(x_lf)
        
        # 返回假 logits，以及两个视图的投影特征 (交给 InfoNCELoss 进行推拉)
        return logits_hf, proj_hf, proj_lf

# =====================================================================
# 1. Transformer-1D (自适应长度版)
# =====================================================================
class Transformer1D(nn.Module):
    def __init__(self, num_leads=12, seq_len=1000, patch_size=50, num_classes=5, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        self.patch_proj = nn.Linear(num_leads * patch_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # 初始一个默认长度的位置编码，后续在 forward 里动态插值
        num_patches = seq_len // patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x_hf, x_lf):
        x = x_hf + x_lf  
        B, C, L = x.shape
        
        x = x.unfold(2, self.patch_size, self.patch_size).contiguous()
        x = x.view(B, C, -1, self.patch_size).transpose(1, 2)
        x = x.reshape(B, -1, C * self.patch_size) 
        x = self.patch_proj(x) # (B, P, d_model)
        
        P = x.size(1)
        
        # 🌟 核心修复：如果实际 Patch 数与初始位置编码不同，进行 1D 线性插值自适应
        pos_embed = self.pos_embed[:, 1:, :] # 取出除了 CLS 之外的位置编码 (1, P_init, d_model)
        if pos_embed.size(1) != P:
            pos_embed = pos_embed.transpose(1, 2) # (1, d_model, P_init)
            pos_embed = F.interpolate(pos_embed, size=P, mode='linear', align_corners=False)
            pos_embed = pos_embed.transpose(1, 2) # (1, P, d_model)
            
        cls_embed = self.pos_embed[:, :1, :]
        dynamic_pos_embed = torch.cat((cls_embed, pos_embed), dim=1) # (1, P+1, d_model)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        
        x = x + dynamic_pos_embed # 完美匹配维度！
        
        x = self.transformer(x)
        cls_out = x[:, 0] 
        return self.mlp_head(cls_out)


# =====================================================================
# 2. Internal Fourier-Graph Control (NOT official FourierGNN)
# =====================================================================
class FourierGNN1D(nn.Module):
    def __init__(self, num_leads=12, seq_len=1000, num_classes=5):
        super().__init__()
        self.num_leads = num_leads
        # 频域图邻接矩阵 (学习导联间的相干性)
        self.complex_weight = nn.Parameter(torch.randn(num_leads, num_leads, dtype=torch.cfloat))
        
        # 🌟 修复核心：引入 1D 卷积特征提取器，替代粗暴的全局均值池化
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(num_leads * 2, 64, kernel_size=15, stride=2, padding=7), # 拼接原信号与图信号
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 使用 AdaptiveMaxPool 防止正负波相互抵消，专门捕获波峰异常
            nn.AdaptiveMaxPool1d(1) 
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_hf, x_lf):
        x = x_hf + x_lf
        B, C, L = x.shape
        
        # 1. 傅里叶变换到频域，做图卷积
        x_freq = torch.fft.rfft(x, dim=-1) 
        out_freq = torch.einsum('bcf,cd->bdf', x_freq, self.complex_weight)
        x_graph = torch.fft.irfft(out_freq, n=L, dim=-1) # (B, 12, L)
        
        # 2. 结合原始时域特征与图卷积增强特征
        combined = torch.cat([x, x_graph], dim=1) # (B, 24, L)
        
        # 3. 深度特征提取 (不再抵消波形)
        features = self.feature_extractor(combined).squeeze(-1) # (B, 128)
        
        return self.fc(features)



# =====================================================================
# 3. Internal Patch-Transformer Control (NOT official PatchTST)
# =====================================================================
import torch
import torch.nn as nn

class PatchTST1D(nn.Module):
    def __init__(self, num_leads=12, seq_len=1000, patch_size=50, num_classes=5, d_model=64, nhead=8, num_layers=3):
        super().__init__()
        self.patch_size = patch_size
        self.num_leads = num_leads
        self.d_model = d_model
        
        # 🌟 修复核心：必须加入通道独立的实例归一化 (InstanceNorm)
        # 否则 ECG 的电压尖峰会直接摧毁 Transformer 的注意力矩阵
        self.norm = nn.InstanceNorm1d(num_leads, affine=True)
        
        # 针对单通道的 Patch 投影
        self.patch_proj = nn.Linear(patch_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 初始位置编码
        num_patches = seq_len // patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头：汇聚所有通道的特征
        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_leads * d_model, 128),
            nn.BatchNorm1d(128), # 🌟 新增：稳压输出
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_hf, x_lf):
        x = x_hf + x_lf  
        
        # 🌟 第一步：应用独立通道归一化，把信号压制在健康范围内
        x = self.norm(x)
        
        B, C, L = x.shape
        # 切片: (B, C, num_patches, patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size).contiguous()
        P = x.size(2)
        
        # 把 Batch 和 Channel 维度合并，每个通道独立过 Transformer
        x = x.view(B * C, P, self.patch_size) 
        x = self.patch_proj(x) # (B*C, P, d_model)
        
        # 自适应位置编码插值
        pos_embed = self.pos_embed[:, 1:, :] 
        if pos_embed.size(1) != P:
            pos_embed = pos_embed.transpose(1, 2) 
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=P, mode='linear', align_corners=False)
            pos_embed = pos_embed.transpose(1, 2) 
            
        cls_embed = self.pos_embed[:, :1, :]
        dynamic_pos_embed = torch.cat((cls_embed, pos_embed), dim=1) 
        
        cls_tokens = self.cls_token.expand(B * C, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) 
        x = x + dynamic_pos_embed 
        
        # 通过 Transformer (B*C, P+1, d_model)
        x = self.transformer(x)
        
        # 提取 CLS token: (B*C, d_model)
        cls_out = x[:, 0] 
        
        # 拆解回 Batch 和 Channel 维度
        out = cls_out.view(B, self.num_leads, self.d_model)
        return self.mlp_head(out)
# Backward-compatible class aliases with honest names for paper hygiene.
# These classes are project-local internal controls, not official reproductions
# of FourierGNN or PatchTST. Use model names internal_fourier_graph1d and
# internal_patch_transformer1d in new experiments when you only need these
# internal controls.
InternalFourierGraph1D = FourierGNN1D
InternalPatchTransformer1D = PatchTST1D
