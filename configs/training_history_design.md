# 训练历史记录设计方案

## 背景

模型可能经历多阶段、多数据集的训练流程，例如：
- 先在 HRDS 上训练 10000 步
- 再在 HRDH 上微调 5000 步
- 最后在业务数据上微调 2000 步

需要记录完整的训练谱系，方便追溯模型来源和对比不同训练策略的效果。

## 方案选择

**采用方案 2 + 3 组合**：

- **方案 2**：在每个 stage 目录下生成独立的 `training_info.json`
- **方案 3**：通过 `ExperimentManager` 统一管理写入，避免手动维护

### 为什么不用方案 1（嵌入 checkpoint）

1. 强耦合训练框架，导出 ONNX/TorchScript 时会丢失
2. 多份 checkpoint 容易信息不一致
3. 不方便聚合查询所有模型的训练历史

## 目录结构

```
artifact/
├── exp_20251210_201220/
│   ├── config.yml                    # 实验配置（已有）
│   ├── stage1_hrds/
│   │   ├── checkpoint-10000/
│   │   └── training_info.json        # 新增
│   ├── stage1_hrdh/
│   │   ├── checkpoint-5000/
│   │   └── training_info.json
│   ├── features_hrds/
│   │   └── training_info.json
│   ├── parent_finder_hrds/
│   │   ├── best_model.pt
│   │   └── training_info.json
│   └── relation_classifier_hrds/
│       ├── best_model.pt
│       └── training_info.json
```

## training_info.json 结构

```json
{
  "model_name": "layoutxlm_token_classifier",
  "stage": "stage1",
  "final_checkpoint": "checkpoint-10000",
  "base_model": "microsoft/layoutxlm-base",
  "created_at": "2025-12-11T10:30:00",
  "last_updated": "2025-12-11T15:20:00",
  "total_steps": 15000,
  "last_dataset": "hrdh",
  "training_history": [
    {
      "dataset": "hrds",
      "steps": 10000,
      "epochs": null,
      "metrics": {
        "f1": 0.85,
        "precision": 0.84,
        "recall": 0.86
      },
      "hyperparams": {
        "lr": 2e-5,
        "batch_size": 3,
        "max_steps": 10000
      },
      "start_time": "2025-12-11T10:30:00",
      "end_time": "2025-12-11T12:00:00"
    },
    {
      "dataset": "hrdh",
      "steps": 5000,
      "epochs": null,
      "metrics": {
        "f1": 0.82
      },
      "hyperparams": {
        "lr": 2e-5,
        "batch_size": 3,
        "max_steps": 5000
      },
      "start_time": "2025-12-11T14:00:00",
      "end_time": "2025-12-11T15:20:00"
    }
  ]
}
```

## 字段说明

### 顶层字段

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| model_name | string | 是 | 模型名称，如 `layoutxlm_token_classifier`, `parent_finder`, `relation_classifier` |
| stage | string | 是 | 训练阶段：`stage1`, `stage2`, `stage3`, `stage4` |
| final_checkpoint | string | 是 | 当前最优 checkpoint 名称 |
| base_model | string | 是 | 初始模型来源（预训练模型路径或上一阶段 checkpoint） |
| created_at | string | 是 | 首次创建时间 (ISO 8601) |
| last_updated | string | 是 | 最后更新时间 (ISO 8601) |
| total_steps | int | 是 | 累计训练步数（跨所有数据集） |
| last_dataset | string | 是 | 最后训练使用的数据集 |
| training_history | array | 是 | 训练历史记录列表 |

### training_history 元素字段

| 字段 | 类型 | 必须 | 说明 |
|------|------|------|------|
| dataset | string | 是 | 数据集名称：`hrds`, `hrdh`, `my_biz` 等 |
| steps | int | 是 | 本次训练步数 |
| epochs | int/null | 否 | 本次训练轮数（step-based 训练可为 null） |
| metrics | object | 是 | 验证集指标 |
| hyperparams | object | 否 | 训练超参数 |
| start_time | string | 是 | 开始时间 (ISO 8601) |
| end_time | string | 是 | 结束时间 (ISO 8601) |

### metrics 常用字段

- Stage 1 (Token Classification): `f1`, `precision`, `recall`, `accuracy`
- Stage 2 (Feature Extraction): `num_documents`, `num_chunks`
- Stage 3 (ParentFinder): `accuracy`, `val_loss`
- Stage 4 (RelationClassifier): `macro_f1`, `accuracy`, `per_class_f1`

### hyperparams 常用字段

- `lr`: 学习率
- `batch_size`: 批次大小
- `max_steps` / `num_epochs`: 训练长度
- `optimizer`: 优化器名称
- `scheduler`: 学习率调度器
- 任务相关：`neg_ratio`, `max_lines_limit` 等

## 实现计划

### 1. 扩展 ExperimentManager

在 `examples/stage/util/experiment_manager.py` 中添加：

```python
def append_training_history(
    self,
    exp: Optional[str],
    stage: str,
    dataset: str,
    model_name: str,
    steps: int,
    metrics: Dict[str, float],
    base_model: str,
    final_checkpoint: str,
    hyperparams: Optional[Dict] = None,
    epochs: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
):
    """
    追加训练历史记录到 training_info.json

    如果文件不存在则创建，存在则追加新的训练记录。
    """
    pass

def get_training_info(self, exp: Optional[str], stage: str, dataset: str) -> Dict:
    """
    读取 training_info.json
    """
    pass
```

### 2. 修改训练脚本

在每个 stage 脚本的训练完成处调用：

```python
# train_stage1.py 示例
if result.returncode == 0:
    exp_manager.append_training_history(
        exp=args.exp,
        stage="stage1",
        dataset=args.dataset,
        model_name="layoutxlm_token_classifier",
        steps=train_cfg.max_steps,
        metrics={"f1": best_f1},  # 从训练结果获取
        base_model=model_path,
        final_checkpoint=best_checkpoint,
        hyperparams={
            "lr": train_cfg.learning_rate,
            "batch_size": train_cfg.per_device_train_batch_size,
        },
        start_time=start_time,
        end_time=datetime.now().isoformat(),
    )
```

### 3. 需要修改的文件

1. `examples/stage/util/experiment_manager.py` - 添加 `append_training_history()` 方法
2. `examples/stage/scripts/train_stage1.py` - 训练完成时调用
3. `examples/stage/scripts/train_stage2.py` - 特征提取完成时调用
4. `examples/stage/scripts/train_stage3.py` - 训练完成时调用
5. `examples/stage/scripts/train_stage4.py` - 训练完成时调用

## 使用场景

### 查看模型训练历史

```bash
cat artifact/exp_xxx/stage1_hrds/training_info.json | jq
```

### 聚合所有模型信息

```bash
find artifact -name "training_info.json" -exec cat {} \; | jq -s '.'
```

### 在代码中读取

```python
from util.experiment_manager import get_experiment_manager

exp_manager = get_experiment_manager(config)
info = exp_manager.get_training_info(exp=None, stage="stage1", dataset="hrds")
print(f"Total steps: {info['total_steps']}")
print(f"Training history: {info['training_history']}")
```

## 后续扩展

1. **Git commit 记录**：记录训练时的代码版本
2. **环境信息**：GPU 型号、CUDA 版本等
3. **模型对比脚本**：自动对比不同训练策略的效果
4. **Web UI**：可视化展示所有模型的训练谱系
