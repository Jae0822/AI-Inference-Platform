import os

from modelscope import snapshot_download

# 1. 明确指定把模型下载到我们的大数据盘，绝对不挤爆系统盘
output_dir = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"

print("🚀 正在启动 ModelScope 内网加速引擎...")
print("📂 大模型将安全存入数据盘仓库:", output_dir)

# 2. 唤醒搬运工，开始抓取阿里最强 7B 开源模型
model_dir = snapshot_download(
    "Qwen/Qwen2.5-7B-Instruct",
    cache_dir=output_dir
)

print("\n✅ [大功告成] 15GB 模型文件已完美下载！")
print("📍 实际物理存储路径为:", model_dir)
