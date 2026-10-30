# data_process/dataset.py
import os
import torch
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader

class DualFreqECGDataset(Dataset):
    def __init__(self, processed_data_dir, file_list=None, label_mode="multilabel", single_label_map=None):
        """
        processed_data_dir: 存放数据的文件夹路径 (支持 CPSC, PTB-XL, PTB)
        """
        super().__init__()
        self.processed_data_dir = processed_data_dir
        self.label_mode = label_mode
        self.single_label_map = single_label_map or {}
        
        # 🤖 智能探测模式
        if os.path.exists(os.path.join(processed_data_dir, 'x_lf.pth')):
            # ==========================================
            # 模式 A: CPSC (原始的大文件加载模式)
            # ==========================================
            self.mode = 'cpsc'
            print(f"正在将张量加载到内存中 (CPSC 模式, 来自 {processed_data_dir})...")
            self.x_lf = torch.load(os.path.join(processed_data_dir, 'x_lf.pth'), weights_only=True)
            self.x_hf = torch.load(os.path.join(processed_data_dir, 'x_hf.pth'), weights_only=True)
            self.labels = torch.load(os.path.join(processed_data_dir, 'labels.pth'), weights_only=True)
            self.ages = torch.load(os.path.join(processed_data_dir, 'ages.pth'), weights_only=True).unsqueeze(1).float()
            self.sexes = torch.load(os.path.join(processed_data_dir, 'sexes.pth'), weights_only=True).unsqueeze(1).float()
            self.length = self.x_lf.shape[0]
            print(f" [OK] CPSC 加载完毕！总样本数: {self.length}")
            
            # CPSC 暂不需要患者映射
            self.patient_ids = None 
            
        else:
            # ==========================================
            # 模式 B: PTB / PTB-XL / Chapman (独立 .pt 文件动态加载模式)
            # ==========================================
            self.mode = 'ptbxl'
            if file_list is not None:
                self.file_list = file_list
            else:
                self.file_list = sorted([f for f in os.listdir(processed_data_dir) if f.endswith('.pt')])
            self.length = len(self.file_list)
            
            # 🚀 漏洞修复：尝试加载防泄露的患者映射表 (针对 PTB)
            self.patient_ids = None
            mapping_path = os.path.join(processed_data_dir, 'ptb_patient_mapping.csv')
            
            if os.path.exists(mapping_path):
                df_map = pd.read_csv(mapping_path)
                
                # 1. 构建智能模糊匹配字典
                # 把官方的 's0010_re' 变成纯粹的 '0010' 作为钥匙
                mapping_dict = {}
                for index, row in df_map.iterrows():
                    official_id = str(row['ecg_id']) 
                    patient_id = str(row['patient_id']) 
                    
                    match = re.search(r'\d+', official_id)
                    if match:
                        core_num = match.group()
                        mapping_dict[core_num] = patient_id
                        
                # 2. 对本地文件进行脱壳匹配
                self.patient_ids = []
                success_count = 0
                for f in self.file_list:
                    ecg_id = f.replace('.pt', '') # 例如 'S0010'
                    
                    # 提取本地 ID 中的数字部分 ('S0010' -> '0010')
                    match = re.search(r'\d+', ecg_id)
                    if match:
                        local_core_num = match.group()
                        patient_id = mapping_dict.get(local_core_num, ecg_id)
                        if local_core_num in mapping_dict:
                            success_count += 1
                    else:
                        patient_id = ecg_id
                        
                    self.patient_ids.append(patient_id)
                    
                print(f"🛡️ PTB 患者防泄露映射成功率: {success_count} / {len(self.file_list)}")

    def __getitem__(self, index):
        if self.mode == 'cpsc':
            # label_multi 始终保留原始 multi-hot 标签，用于 single-label 训练时的辅助 BCE。
            label_multi = self.labels[index].float()
            label = label_multi
            if self.label_mode == 'single_label':
                label = torch.argmax(label_multi).long() if label_multi.ndim > 0 else label_multi.long()
            return (
                self.x_hf[index],
                self.x_lf[index],
                label,
                self.ages[index],
                self.sexes[index],
                label_multi,
            )
        else:
            # PTB / PTB-XL 模式下动态读取
            file_path = os.path.join(self.processed_data_dir, self.file_list[index])
            data = torch.load(file_path, weights_only=True)
            
            # label_multi 保留文件中的原始标签；single-label 数据也兼容此字段。
            # single_label 模式下，label 用于 CE 主任务；label_multi 用于可选辅助 BCE。
            label_multi = data['label'].float()
            label = label_multi
            if self.label_mode == 'single_label':
                fname = self.file_list[index]
                if fname in self.single_label_map:
                    label = torch.tensor(int(self.single_label_map[fname]), dtype=torch.long)
                else:
                    # 兜底：如果没有从 CSV 构造 single-label，则从 multi-hot 里取 argmax。
                    label = torch.argmax(label_multi).long() if label_multi.ndim > 0 else label_multi.long()
            # 返回 6 个元素；前 5 个保持兼容，第 6 个是原始 multi-label。
            return (
                data['hf'],
                data['lf'],
                label,
                torch.tensor([0.0]),
                torch.tensor([0.0]),
                label_multi,
            )

    def __len__(self):
        return self.length

# 保留旧的辅助函数防止其他脚本报错
def get_dataloaders(processed_data_dir, batch_size=16, train_ratio=0.8):
    pass
