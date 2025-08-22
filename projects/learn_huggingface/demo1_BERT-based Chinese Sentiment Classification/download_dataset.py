# 首先确保安装必要的库
# pip install datasets

import os
from datasets import load_dataset

# 创建目标文件夹
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# 下载ChnSentiCorp数据集
print("正在下载ChnSentiCorp数据集...")
dataset = load_dataset("lansinuote/ChnSentiCorp")

# 保存数据集到本地
dataset_path = os.path.join(data_dir, "ChnSentiCorp")
dataset.save_to_disk(dataset_path)

print(f"数据集已成功下载到: {dataset_path}")

# 查看数据集信息
print("\n数据集信息:")
print(dataset)

# 查看训练集的前几个样本
print("\n训练集前3个样本:")
for i in range(3):
    print(f"样本 {i+1}: {dataset['train'][i]}")