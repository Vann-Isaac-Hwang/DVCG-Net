import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
import glob

def plot_ecg_spectrum(record_path, dataset_name, lead_name='II', ax=None):
    """
    读取单条ECG记录，计算并绘制其指定导联的功率谱密度 (PSD)
    """
    try:
        # 读取心电数据
        record = wfdb.rdrecord(record_path)
        signals = record.p_signal
        fs = record.fs
        sig_names = record.sig_name
        
        # 寻找对应的导联索引 (通常选择 Lead II 或 V5)
        if lead_name in sig_names:
            lead_idx = sig_names.index(lead_name)
        else:
            lead_idx = 1 # 默认退化为第二导联
            lead_name = sig_names[lead_idx]
            
        signal = signals[:, lead_idx]
        
        # 减去均值，去除直流(DC)分量 (0Hz)
        signal = signal - np.mean(signal)
        
        # 使用 Welch 方法计算功率谱密度 (PSD)
        # nperseg 设置为 fs*2 表示窗口长度为 2 秒，频率分辨率为 0.5 Hz
        frequencies, psd = welch(signal, fs, nperseg=int(fs*2))
        
        # 归一化 PSD 以便不同数据集之间进行对比
        psd = psd / np.max(psd)
        
        # 绘图
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
            
        ax.plot(frequencies, psd, color='#1f77b4', linewidth=1.5, label=f'PSD ({lead_name})')
        
        # 绘制分频点参考线
        ax.axvline(x=10, color='r', linestyle='--', alpha=0.7, label='10 Hz Cut-off')
        ax.axvline(x=20, color='g', linestyle='-.', alpha=0.7, label='20 Hz Cut-off')
        
        # 标注和美化
        ax.set_title(f'{dataset_name} Spectrum Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Normalized Power', fontsize=12)
        ax.set_xlim(0, 50) # 绝大部分ECG能量集中在 50Hz 以内
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')
        
    except Exception as e:
        print(f"Error processing {record_path}: {e}")

def main():
    # 设定数据集路径 (请根据您的实际挂载路径补全 PTB-XL 和 CPSC 的路径)
    paths = {
        'PTB Diagnostic': '../Project_Data/PTB_Data/',
        'PTB-XL': '../Project_Data/PTBXL_Data/', # [TODO: 替换为您的真实路径]
        'CPSC 2018': '../Project_Data/CPSC_Data/' # [TODO: 替换为您的真实路径]
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.subplots_adjust(hspace=0.4)
    
    for ax, (dataset_name, folder_path) in zip(axes, paths.items()):
        # 在每个文件夹中寻找第一个 .hea 文件作为代表样本
        search_pattern = os.path.join(folder_path, '*.hea')
        files = glob.glob(search_pattern)
        
        if files:
            # 去掉 .hea 后缀，供 wfdb 读取
            sample_record = files[0][:-4] 
            print(f"Plotting {dataset_name} using sample: {os.path.basename(sample_record)}")
            plot_ecg_spectrum(sample_record, dataset_name, lead_name='II', ax=ax)
        else:
            ax.text(0.5, 0.5, f'No .hea files found in {folder_path}', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='red', fontsize=12)
            ax.set_title(f'{dataset_name} - Not Found')

    plt.suptitle("Cross-Dataset ECG Frequency Spectrum Distribution", fontsize=18, y=0.92, fontweight='bold')
    # plt.savefig('Spectrum_Verification.png', dpi=300, bbox_inches='tight') # 随时可以保存给论文用
    plt.show()

if __name__ == '__main__':
    main()