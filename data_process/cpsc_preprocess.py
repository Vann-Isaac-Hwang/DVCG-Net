import os
import torch
import scipy.io
import scipy.signal
import pandas as pd
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils.config_parser import load_config

def apply_filter(signal, fs=500, cutoff=10):
    """
    巴特沃斯滤波器：将信号解耦为低频 (LF) 和高频 (HF)
    signal shape: (12, 15000)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    b_low, a_low = scipy.signal.butter(4, normal_cutoff, btype='lowpass', analog=False)
    b_high, a_high = scipy.signal.butter(4, normal_cutoff, btype='highpass', analog=False)
    
    # filtfilt 保证零相位失真
    lf_signal = scipy.signal.filtfilt(b_low, a_low, signal, axis=-1)
    hf_signal = scipy.signal.filtfilt(b_high, a_high, signal, axis=-1)
    
    return torch.tensor(lf_signal.copy(), dtype=torch.float32), \
           torch.tensor(hf_signal.copy(), dtype=torch.float32)

def process_single_record(mat_path, df_row, cutoff=10):
    """处理单条医疗记录"""
    data_entry = scipy.io.loadmat(mat_path)
    
    # 1. 提取波形并统一长度至 15000
    signal = torch.tensor(data_entry['ECG'][0][0][2][:, :15000], dtype=torch.float32)
    if signal.shape[1] < 15000:
        n_pad = 15000 - signal.shape[1]
        signal = F.pad(signal, (0, n_pad), mode='constant')
        
    # 2. 提取年龄和性别
    age = torch.tensor(data_entry['ECG'][0][0][1][0, 0], dtype=torch.float32)
    sex_str = data_entry['ECG'][0][0][0][0]
    sex = torch.tensor(1 if sex_str == 'Male' else 0, dtype=torch.float32)
    
    # 3. 高低频解耦
    lf_signal, hf_signal = apply_filter(signal.numpy(), cutoff=cutoff)
    
    # 4. 生成 9 分类 One-hot 标签
    label = torch.zeros(9, dtype=torch.float32)
    for col in ['First_label', 'Second_label', 'Third_label']:
        val = df_row[col]
        if pd.notna(val) and str(val).strip() != '':
            idx = int(float(val)) - 1  # 类别 1-9 转为索引 0-8
            if 0 <= idx < 9:
                label[idx] = 1.0
                
    return lf_signal, hf_signal, age, sex, label

def main():
    cfg = load_config()
    print("🚀 启动数据预处理 (读取配置文件路径)...")
    
    # 直接从 config.yaml 读取路径
    data_dir = cfg.env.raw_data_dir  # 原始 .mat 文件夹路径
    csv_path = cfg.env.csv_path      # 原始 REFERENCE.csv 路径
    save_dir = cfg.env.data_dir      # 处理后张量的保存路径
    
    os.makedirs(save_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    lf_list, hf_list, age_list, sex_list, label_list = [], [], [], [], []
    
    print("开始读取并进行频域解耦 (这可能需要几分钟到十几分钟)...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        mat_path = os.path.join(data_dir, f"{row['Recording']}.mat")
        
        if os.path.exists(mat_path):
            lf_sig, hf_sig, age, sex, label = process_single_record(mat_path, row)
            
            lf_list.append(lf_sig.unsqueeze(0))
            hf_list.append(hf_sig.unsqueeze(0))
            age_list.append(age.unsqueeze(0))
            sex_list.append(sex.unsqueeze(0))
            label_list.append(label.unsqueeze(0))
        else:
            print(f" [Notice] 警告: 找不到文件 {mat_path}")
            
    # 将 List 拼装成巨型 Tensor
    print("正在拼接张量并保存到硬盘...")
    torch.save(torch.cat(lf_list, dim=0), os.path.join(save_dir, 'x_lf.pth'))
    torch.save(torch.cat(hf_list, dim=0), os.path.join(save_dir, 'x_hf.pth'))
    torch.save(torch.cat(age_list, dim=0), os.path.join(save_dir, 'ages.pth'))
    torch.save(torch.cat(sex_list, dim=0), os.path.join(save_dir, 'sexes.pth'))
    torch.save(torch.cat(label_list, dim=0), os.path.join(save_dir, 'labels.pth'))
    
    print(" [OK] 数据预处理全部完成！")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="CPSC 2018 预处理")
    parser.add_argument('--cutoff', type=int, default=10, help='分频截止频率 (Hz)')
    args = parser.parse_args()

    cfg = load_config()
    data_dir = cfg.env.raw_data_dir  
    csv_path = cfg.env.csv_path      
    
    # 🚀 动态生成输出路径：原本是 ./data/cpsc_processed，现在动态加上 _freq20
    base_save_dir = os.path.dirname(cfg.env.data_dir.rstrip('/\\'))
    save_dir = os.path.join(base_save_dir, f'processed_freq{args.cutoff}')
    
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    lf_list, hf_list, age_list, sex_list, label_list = [], [], [], [], []
    
    print(f"🚀 开始 CPSC 频域解耦 (截断频率: {args.cutoff}Hz)...")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        mat_path = os.path.join(data_dir, f"{row['Recording']}.mat")
        if os.path.exists(mat_path):
            # 🚀 传入 args.cutoff
            lf_sig, hf_sig, age, sex, label = process_single_record(mat_path, row, cutoff=args.cutoff)
            
            lf_list.append(lf_sig.unsqueeze(0))
            hf_list.append(hf_sig.unsqueeze(0))
            age_list.append(age.unsqueeze(0))
            sex_list.append(sex.unsqueeze(0))
            label_list.append(label.unsqueeze(0))
    
    print(f"拼接并保存到硬盘: {save_dir}...")
    torch.save(torch.cat(lf_list), os.path.join(save_dir, 'x_lf.pth'))
    torch.save(torch.cat(hf_list), os.path.join(save_dir, 'x_hf.pth'))
    torch.save(torch.cat(age_list), os.path.join(save_dir, 'ages.pth'))
    torch.save(torch.cat(sex_list), os.path.join(save_dir, 'sexes.pth'))
    torch.save(torch.cat(label_list), os.path.join(save_dir, 'labels.pth'))
    print("✅ CPSC 处理完成！")