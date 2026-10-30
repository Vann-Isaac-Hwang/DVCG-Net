# fix_ptb_leakage.py
import os
import urllib.request
import pandas as pd
from utils.registry import get_dataset_info

def generate_mapping():
    # 🚀 修复：从最新的注册表 (Registry) 中获取 PTB 的正确路径
    ptb_info = get_dataset_info('PTB')
    ptb_processed_dir = ptb_info['default_data_dir']
    
    os.makedirs(ptb_processed_dir, exist_ok=True)
    
    url = "https://physionet.org/files/ptbdb/1.0.0/RECORDS"
    print("🌐 正在从 PhysioNet 官方下载 PTB 原始患者映射结构...")
    
    try:
        response = urllib.request.urlopen(url)
        records = response.read().decode('utf-8').splitlines()
        
        mapping = []
        for line in records:
            if '/' in line:
                patient_id, record_id = line.split('/')
                mapping.append({'ecg_id': record_id, 'patient_id': patient_id})
                
        df = pd.DataFrame(mapping)
        out_path = os.path.join(ptb_processed_dir, 'ptb_patient_mapping.csv')
        df.to_csv(out_path, index=False)
        print(f"✅ 成功！已生成防泄露映射表: {out_path}")
        print(f"📊 共有 {len(df)} 条记录，归属于 {df['patient_id'].nunique()} 位真实患者。")
        
    except Exception as e:
        print(f"❌ 下载或解析失败: {e}")
        print("如果您在离线服务器，请尝试根据 年龄+性别 组合生成 Pseudo-Patient-ID。")

if __name__ == '__main__':
    generate_mapping()