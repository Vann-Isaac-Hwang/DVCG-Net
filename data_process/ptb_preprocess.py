import os
import wfdb
import torch
import numpy as np
from scipy import signal
from tqdm import tqdm
from utils.config_parser import load_config

def apply_butterworth(data, cutoff=10, fs=500):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
    
    # 🚀 FIX: Ensure data is contiguous before filtering
    data = data.copy() 
    
    lf_data = signal.filtfilt(b, a, data, axis=1) 
    hf_data = data - lf_data
    return hf_data, lf_data

def process_ptb_cinc2020(raw_data_dir, output_dir, cutoff=10):
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 开始扫描 CinC 2020 PTB 数据库...")
    all_records = []
    
    for record in os.listdir(raw_data_dir):
        if record.endswith('.hea'):
            all_records.append(os.path.join(raw_data_dir, record[:-4]))
                
    print(f"找到 {len(all_records)} 条记录，开始解密 SNOMED-CT 标签并转换张量...")
    
    count_mi = 0
    count_hc = 0
    target_len = 15000
    
    for record_path in tqdm(all_records, desc="Processing ECG"):
        try:
            rec = wfdb.rdrecord(record_path)
            
            # 🚨 核心破解：识别 SNOMED-CT 医疗编码！
            comments_str = " ".join(rec.comments)
            
            # 164865005 是心肌梗死 (Myocardial Infarction) 的国际编码
            if "164865005" in comments_str:
                label_vec = np.array([0.0, 1.0], dtype=np.float32) # [HC=0, MI=1]
                count_mi += 1
            # 426783006 是健康/正常窦性心律 (Normal sinus rhythm) 的国际编码
            elif "426783006" in comments_str:
                label_vec = np.array([1.0, 0.0], dtype=np.float32) # [HC=1, MI=0]
                count_hc += 1
            else:
                # 其他疾病编码，跳过
                continue
                
            # 提取 12 导联信号
            raw_signal = rec.p_signal.T 
            raw_signal = np.nan_to_num(raw_signal)
            
            # 🚨 智能降频
            if rec.fs == 1000:
                resampled_signal = raw_signal[:, ::2].copy() # 🚀 FIX: Add .copy() here
            elif rec.fs == 500:
                resampled_signal = raw_signal.copy() # 🚀 FIX: Add .copy() here
            else:
                target_points = int(raw_signal.shape[1] * (500 / rec.fs))
                resampled_signal = signal.resample(raw_signal, target_points, axis=1).copy() # 🚀 FIX: Add .copy() here
            
            # 双频解耦
            hf_data, lf_data = apply_butterworth(resampled_signal, cutoff=cutoff, fs=500)
            
            # 对齐长度
            if hf_data.shape[1] > target_len:
                hf_data = hf_data[:, :target_len]
                lf_data = lf_data[:, :target_len]
            elif hf_data.shape[1] < target_len:
                pad_width = target_len - hf_data.shape[1]
                hf_data = np.pad(hf_data, ((0, 0), (0, pad_width)), mode='constant')
                lf_data = np.pad(lf_data, ((0, 0), (0, pad_width)), mode='constant')
                
            # 保存张量
            ecg_id = os.path.basename(record_path)
            # 🚀 FIX: Add .copy() before converting to tensor, just to be absolutely safe
            torch.save({
                'hf': torch.tensor(hf_data.copy(), dtype=torch.float32), 
                'lf': torch.tensor(lf_data.copy(), dtype=torch.float32), 
                'label': torch.tensor(label_vec, dtype=torch.float32)
            }, os.path.join(output_dir, f"{ecg_id}.pt"))

        except Exception as e:
            print(f"处理 {record_path} 时出错: {e}")

    print(f"\n✅ 数据处理彻底完成！")
    print(f"📊 心梗 (MI) 样本数: {count_mi}")
    print(f"📊 健康 (HC) 样本数: {count_hc}")
    print(f"存放在 {output_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="PTB 预处理")
    parser.add_argument('--cutoff', type=int, default=10, help='分频截止频率 (Hz)')
    args = parser.parse_args()

    cfg = load_config()
    raw_data_dir = cfg.env.ptb_raw_dir
    
    # 🚀 动态生成输出路径
    base_dir = os.path.dirname(cfg.env.ptb_processed_dir.rstrip('/\\'))
    output_dir = os.path.join(base_dir, f'processed_freq{args.cutoff}')
    
    print(f"🚀 PTB 预处理启动 (截断频率: {args.cutoff}Hz) -> 输出到 {output_dir}")
    process_ptb_cinc2020(raw_data_dir, output_dir, cutoff=args.cutoff)