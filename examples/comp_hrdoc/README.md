# comp_hrdoc - Detect-Order-Construct 文档结构重构

基于论文 [Detect-Order-Construct](https://arxiv.org/html/2401.11874v2) 的实现，
使用 LayoutXLM 作为多模态基座模型。

## 目录结构

```
comp_hrdoc/
├── models/                     # 可学习的网络结构
│   ├── __init__.py
│   ├── backbone.py             # LayoutXLM 基座封装
│   ├── heads.py                # 各任务预测头
│   │                           # - DetectHead: 逻辑角色分类 + 区域内阅读顺序
│   │                           # - OrderHead: 区域间阅读顺序
│   │                           # - ConstructHead: 目录结构 (父子/兄弟关系)
│   ├── modules/                # 可复用组件
│   │   ├── __init__.py
│   │   ├── attention.py        # 注意力机制
│   │   ├── pooling.py          # 特征池化 (RoIAlign, Mean pooling)
│   │   └── position.py         # 位置编码 (旋转位置编码等)
│   └── build.py                # 模型构建工厂
│
├── tasks/                      # 任务定义层
│   ├── __init__.py
│   ├── base.py                 # Task 基类接口
│   ├── detect.py               # Detect 任务: 逻辑角色 + 区域内顺序
│   ├── order.py                # Order 任务: 区域间阅读顺序
│   └── construct.py            # Construct 任务: 目录结构提取
│
├── engines/                    # 训练/推理/评估引擎
│   ├── __init__.py
│   ├── trainer.py              # 训练器 (支持多任务联合训练)
│   ├── predictor.py            # 推理管线
│   └── evaluator.py            # 评估管线
│
├── data/                       # 数据处理
│   ├── __init__.py
│   ├── dataset.py              # 数据集定义
│   ├── transforms.py           # 数据增强/预处理
│   └── collator.py             # Batch 组织
│
├── metrics/                    # 评估指标
│   ├── __init__.py
│   ├── detect_metrics.py       # Detect 指标 (分类 F1, 顺序准确率)
│   ├── order_metrics.py        # Order 指标 (阅读顺序准确率)
│   └── construct_metrics.py    # Construct 指标 (TEDS, 树编辑距离)
│
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── config.py               # 配置管理
│   ├── logging.py              # 日志工具
│   └── io.py                   # 文件读写
│
├── configs/                    # 配置文件
│   ├── base.yaml               # 基础配置
│   ├── detect.yaml             # Detect 任务配置
│   ├── order.yaml              # Order 任务配置
│   ├── construct.yaml          # Construct 任务配置
│   └── joint.yaml              # 联合训练配置
│
└── scripts/                    # 入口脚本
    ├── train.py                # 训练入口
    ├── predict.py              # 推理入口
    └── evaluate.py             # 评估入口
```

## 论文方法概述

### Detect 模块
- 多模态特征提取 (视觉 + 文本 + 布局)
- 逻辑角色分类 (title, section, para, etc.)
- 区域内阅读顺序关系预测

### Order 模块
- 三层 Transformer 编码器
- 区域间阅读顺序关系预测
- 关系类型分类 (文本区域 vs 图形区域)

### Construct 模块
- 带旋转位置编码的 Transformer
- 父子关系预测
- 兄弟关系预测
- 树插入算法构建目录树

## 使用方法

```bash
# 训练
python examples/comp_hrdoc/scripts/train.py --config configs/joint.yaml

# 推理
python examples/comp_hrdoc/scripts/predict.py --model_path <checkpoint> --input <pdf>

# 评估
python examples/comp_hrdoc/scripts/evaluate.py --model_path <checkpoint> --dataset hrds
```
