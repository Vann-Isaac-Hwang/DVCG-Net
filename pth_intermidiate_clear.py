import torch
import os
import glob

# 找到您权重文件夹下的所有 .pth 文件
weights_dir = "checkpoints/CPSC_MLFL_Weights"
files = glob.glob(os.path.join(weights_dir, "*.pth"))

print("开始为模型权重瘦身...")
for f in files:
    # 1. 加载 621MB 的大字典
    checkpoint = torch.load(f, map_location='cpu')
    
    # 2. 如果它是一个字典且包含 model_state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 3. 剥离出干净的模型权重
        clean_model = checkpoint['model_state_dict']
        
        # 4. 覆盖原文件（此时只保存 200MB 的模型权重）
        torch.save(clean_model, f)
        print(f"✅ {os.path.basename(f)} 瘦身成功！")
        
print("全部清理完毕！现在您可以省出几个 G 的硬盘空间了。")