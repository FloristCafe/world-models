import os
# 强行将 Hugging Face 的下载节点劫持到国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

# 下载 RLVR-World 官方的微调数据集 (请确保安装了 datasets 包: pip install datasets)
print("正在通过镜像源下载数据...")
dataset = load_dataset("thuml/bytesized32-world-model-sft")

# 打印训练集的前 5 条数据
print("\n--- 前 5 条数据样本 ---")
for i in range(5):
    print(f"样本 {i+1}:")
    print(dataset['train'][i])
    print("-" * 30)