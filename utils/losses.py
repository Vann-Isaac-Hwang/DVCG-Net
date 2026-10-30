import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, proj_hf, proj_lf):
        """
        计算跨视图（高低频）的对比损失
        proj_hf, proj_lf: 经过投影头降维后的特征 [Batch, hidden_dim]
        """
        # 1. L2 归一化，将特征映射到单位球面上
        proj_hf = F.normalize(proj_hf, dim=1)
        proj_lf = F.normalize(proj_lf, dim=1)
        
        batch_size = proj_hf.size(0)
        
        # 2. 计算相似度矩阵 (Batch x Batch)
        # 对角线元素为同一样本的高低频相似度（正样本），其余为负样本
        sim_matrix = torch.matmul(proj_hf, proj_lf.T) / self.temperature
        
        # 3. 构造伪标签：对角线索引 0, 1, 2... 
        labels = torch.arange(batch_size, dtype=torch.long, device=proj_hf.device)
        
        # 4. 计算双向的 CrossEntropy Loss (HF找LF, 以及 LF找HF)
        loss_hf_to_lf = F.cross_entropy(sim_matrix, labels)
        loss_lf_to_hf = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_hf_to_lf + loss_lf_to_hf) / 2.0

# 专治长尾分布的多标签 Focal Loss
class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        """
        alpha: 控制正负样本比例的权重 (默认 0.25)
        gamma: 聚焦参数，越大越逼迫模型关注分类困难的样本 (默认 2.0)
        pos_weight: 类别权重，用于缓解类别间的不平衡
        """
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # 1. 计算基础的 BCE Loss (保留了我们的类别平滑权重)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        
        # 2. 计算 pt (模型对真实类别的预测概率)
        # 根据推导: CE(pt) = -log(pt)  => pt = exp(-CE)
        pt = torch.exp(-bce_loss)
        
        # 3. 施加 Focal 动态缩放系数
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()

# 【核心新增】：导师要求的“代价敏感 + Focal Loss”完美融合版
class CostSensitiveFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        """
        gamma: Focal Loss 的困难样本聚焦系数 (挖掘难例)
        pos_weight: Cost-Sensitive Learning (CSL) 的代价敏感权重矩阵 (解决类别不平衡)
        """
        super(CostSensitiveFocalLoss, self).__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        # 1. 基础 BCE，并天然融入 CSL 权重矩阵 (基于数据先验分布的惩罚)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        
        # 2. 计算模型对真实标签的预测概率 p_t
        probs = torch.sigmoid(logits)
        # 如果 target=1, p_t = probs; 如果 target=0, p_t = 1-probs
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 3. 计算 Focal 动态缩放因子: (1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # 4. 融合：CSL代价矩阵 * 困难度缩放 * BCE
        loss = modulating_factor * bce_loss
        
        return loss.mean()

class MultiLabelFairLoss(nn.Module):
    """
    专为 ECG 多标签任务定制的公平性损失函数 (融合了导师的 STFairBench 思想)
    """
    def __init__(self, num_classes, alpha=0.5, temperature=1.0, base_loss_weight=1.0):
        super(MultiLabelFairLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # 公平性惩罚权重 (控制拉平各项分数的力度)
        self.temperature = temperature
        self.base_loss_weight = base_loss_weight

    def forward(self, logits, labels):
        # 1. 基础多标签交叉熵 (Base BCE Loss)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='mean')
        
        # 2. 计算每个类别的独立概率 (注意：多标签必须用 Sigmoid)
        probs = torch.sigmoid(logits / self.temperature)
        
        soft_recalls = []
        class_counts = []
        
        # 3. 动态计算每个类别的软召回率 (Soft Recall)
        for c in range(self.num_classes):
            # 在多标签中，找类别c为正例的样本
            mask = (labels[:, c] == 1.0) 
            class_count = mask.sum().float()
            class_counts.append(class_count)
            
            if class_count > 0:
                # 只计算真正患有该病的样本，预测概率的均值
                soft_recall = probs[mask, c].mean()
            else:
                # 当前 batch 没这个病，采用全局平滑防止除零
                soft_recall = probs[:, c].mean()
                
            soft_recalls.append(soft_recall)
            
        soft_recalls = torch.stack(soft_recalls)
        # 错误率 = 1 - 召回率
        error_rates = 1.0 - soft_recalls
        class_counts = torch.tensor(class_counts, device=logits.device)
        
        # 4. 根据当前 batch 的样本量计算动态权重 (样本越少，权重越大)
        valid_mask = (class_counts > 0)
        if valid_mask.any():
            avg_count = class_counts[valid_mask].mean()
            weights = torch.where(class_counts > 0, avg_count / class_counts, torch.ones_like(class_counts))
            weights = weights / (weights.mean() + 1e-8)
        else:
            weights = torch.ones_like(soft_recalls)
            
        # 5. 加权错误率
        weighted_errors = weights * error_rates
        
        # 6. 【核心灵魂】：公平性惩罚 (标准差)
        mw_error = weighted_errors.mean() # 平均错误率
        # sw_error 就是各项错误率的标准差，它越大说明模型偏科越严重！
        sw_error = torch.sqrt(((weighted_errors - mw_error) ** 2).mean() + 1e-8) 
        
        # 7. 总损失 = BCE基础损失 + 平均加权错误率 + (alpha * 偏科惩罚)
        fair_loss = mw_error + self.alpha * sw_error
        total_loss = self.base_loss_weight * bce_loss + fair_loss
        
        # 记录指标，供您在 tqdm 或 log 中查看
        metrics = {
            'bce_loss': bce_loss.item(),
            'fair_loss': fair_loss.item(),
            'sw_error(偏科程度)': sw_error.item(),
        }
        
        return total_loss, metrics

class MultiLabelSupConLoss(nn.Module):
    """
    针对多标签 ECG 数据的有监督对比损失
    目标：拉近患有相同疾病（正样本）的特征，推开不同疾病的特征
    """
    def __init__(self, temperature=0.1):
        super(MultiLabelSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [Batch_size, Hidden_dim] (例如，低频分支输出的特征向量)
        labels: [Batch_size, Num_classes] (多标签 Multi-hot, 如 [0, 1, 0, 0, 1])
        """
        device = features.device
        batch_size = features.shape[0]

        # 1. 特征 L2 归一化 (投影到单位超球面上，这是计算余弦相似度的前提)
        features = F.normalize(features, p=2, dim=1)

        # 2. 计算特征之间的余弦相似度矩阵 [Batch, Batch]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 3. 构建正样本掩码；同时支持 multi-hot 和 single-label 标签。
        if labels.ndim == 1:
            mask = labels.view(-1, 1).eq(labels.view(1, -1)).float()
        else:
            label_sim = torch.matmul(labels, labels.T)
            mask = (label_sim > 0).float()
        
        # 排除自己和自己的对比 (对角线清零)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # 4. 计算 Contrastive Loss (InfoNCE 变体)
        # 为了数值稳定性，减去每行的最大值
        sim_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - sim_max.detach()
        
        # 计算分母 (所有负样本和正样本的指数和)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # 仅对正样本对求均值 (如果一个样本在 batch 内没有正样本，分母可能为 0，加 1e-8 防御)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # 损失值
        loss = -mean_log_prob_pos.mean()
        
        return loss
