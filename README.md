# 点云 VPCC 坐标优化后处理任务

## 文件说明

- `train.py`: 模型训练主程序
- `evaluate.py`: 模型评估脚本
- `network.py`: 神经网络模型结构定义
- `preprocessing.py`: 数据预处理相关函数
- `utils.py`: 通用工具函数集合
- `config.py`: 存储所有常量配置（如文件路径、模型参数等）

## 使用方法

- 在 ⁠`constants.py` 中设置数据集相关路径和参数
- 进行 `python preprocessing.py` 数据分块预处理
- 训练模型：`python train.py`
- 评估模型：`python evaluate.py`
