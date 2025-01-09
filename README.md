# 项目名称
结合专家网络与重参数化技术的增益模型

# 项目简介
受到一种基于专家混合的多值处理网络（M3TN）[8]的启发，并且借鉴了显示特征交互感知网络（EFIN）的先进理念，本文提出了一种新的模型，即多门专家混合网络和显式建模提升效应模块。这种模型通过结合多种专家网络的优势，不仅提高了模型的效率，还显著提升了预测效果。为了验证模型的有效性，我们在criteo数据集上进行了实验，并与现有的多种建模方法进行了比较。实验结果表明，本文提出的模型不仅在训练数据上表现出了优越性，而且在实际应用中也展现出了高效性和准确性。

## 使用方法
以下是一个基本使用示例：

```bash
CUDA_VISBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 12466 --use_env tune_m3tn.py
```

## 数据集下载地址
```bash
https://www.kaggle.com/datasets/mrkmakr/criteo-dataset?resource=download
```

## 项目结构
```bash
project/
├── datax/         # 数据集
├── tables/        # 实验结果
├── models/        # 模型文件
├── utils/         # 数据处理
└── README.md      # 项目说明文档
```
