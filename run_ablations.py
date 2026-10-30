# run_ablations.py
import os
import subprocess
import sys

sys.path.append(os.path.abspath(os.getcwd()))
from utils.registry import get_dataset_info

WORKSPACE = "./ablations_workspace"
os.makedirs(os.path.join(WORKSPACE, "logs"), exist_ok=True)

# ==========================================
# 🌟 动态寻址与 10Hz 数据准备
# ==========================================
ds_info = get_dataset_info('PTBXL')
base_processed_dir = os.path.abspath(ds_info['default_data_dir']) 
base_parent_dir = os.path.dirname(base_processed_dir.rstrip('/\\'))

# 指定 10Hz 数据输出目录
dir_freq10 = os.path.join(base_parent_dir, "processed_freq10")

env = os.environ.copy()
env["PYTHONPATH"] = os.path.abspath(os.getcwd())

# 步骤 1：自动预处理 10Hz 数据 (如果还没有的话)
if not os.path.exists(dir_freq10):
    print(f"⚙️ 正在生成 10Hz 预处理数据 -> {dir_freq10}")
    subprocess.run(["python", "data_process/ptbxl_preprocess.py", "--cutoff", "10", "--output_dir", dir_freq10], env=env)
else:
    print(f"✅ 10Hz 数据已存在: {dir_freq10}")

# ==========================================
# 🌟 定义我们要跑的 6 个核心消融任务
# ==========================================
# 格式: (实验名称, 预处理数据目录, edge_dropout, 是否使用SupCon, 是否使用DGNN, 频段限制)
ablation_tasks = [
    # 1. 满血版
    ("Ours_Full", dir_freq10, 0.5, True, True, "Full"),
    # 2. 剥离对比学习
    ("wo_SupCon", dir_freq10, 0.5, False, True, "Full"),
    # 3. 剥离动态图网络
    ("wo_DGNN", dir_freq10, 0.5, True, False, "Full"),
    # 4. 剥离边丢弃
    ("wo_EdgeDrop", dir_freq10, 0.0, True, True, "Full"),
    # 5. 纯低频
    ("LF_Only", dir_freq10, 0.5, True, True, "LF"),
    # 6. 纯高频
    ("HF_Only", dir_freq10, 0.5, True, True, "HF")
]

print(f"\n🚀 启动 PTBXL 域内 10Hz 消融流水线...")
for name, data_dir, edge_dp, use_supcon, use_dgnn, freq_mode in ablation_tasks:
    print(f"\n" + "="*60)
    print(f"🔥 正在运行任务: [{name}] | Edge_DP: {edge_dp} | SupCon: {use_supcon} | DGNN: {use_dgnn} | Freq: {freq_mode}")
    print("="*60)
    
    log_file = os.path.join(WORKSPACE, "logs", f"{name}_10Hz.log")
    
    # 构建传给 main_ablation.py 的命令行参数
    cmd = [
        "python", "main_ablation.py", 
        "--exp_name", f"{name}_10Hz", 
        "--data_dir", data_dir, 
        "--edge_dropout", str(edge_dp), 
        "--alpha", "0.2" # 默认 MLFL alpha
    ]
    
    # 传递标志位（需要修改 main_ablation.py 接收这些参数）
    if not use_supcon: cmd.append("--no_supcon")
    if not use_dgnn: cmd.append("--no_dgnn")
    if freq_mode != "Full":
        cmd.extend(["--freq_mode", freq_mode])

    with open(log_file, "w", encoding="utf-8") as f:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        process.wait()