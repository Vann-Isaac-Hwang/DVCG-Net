import torch
import torch.nn as nn

class BasicBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1):
        super(BasicBlock1d, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 如果输入和输出的通道数或序列长度变了，需要一个 1x1 卷积来调整 shortcut 的维度
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # 正宗的残差连接
        return self.relu(out)

class ECGBackbone(nn.Module):
    def __init__(self, hidden_dim=64):
        super(ECGBackbone, self).__init__()
        # 1. 初始大卷积层：快速降低采样率 (15000 -> 3000 -> 1500)
        self.stem = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 2. 深度残差层 (提取高级语义特征)
        self.layer1 = self._make_layer(32, 64, num_blocks=2, stride=2)   # 1500 -> 750
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)  # 750 -> 375
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2) # 375 -> 188

        # 3. 池化与降维
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, hidden_dim)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock1d(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1d(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: [Batch, 12, 15000]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).squeeze(-1) # [Batch, 256]
        return self.fc(x) # [Batch, hidden_dim]