# Claude Code 项目规范

## 最重要的原则：复用优先

**在修改或新增任何代码前，必须先检查是否有可复用的现有实现。**

comp_hrdoc 是独立的项目，这个目录中的代码也遵循下方的规则

### 强制检查清单

修改代码前，按以下顺序检查：

1. **tasks/** - 是否有相关的 Task 类（loss/decode/metrics）
   - `semantic_cls.py` - 分类任务的 decode 逻辑
   - `parent_finding.py` - 父节点查找的 decode 逻辑
   - `relation_cls.py` - 关系分类的 decode 逻辑

2. **models/** - 是否有相关的模型组件
   - `modules/` - line_pooling, attention 等可复用组件

3. **metrics/** - 是否有相关的指标计算
   - `examples/comp_hrdoc/metrics/` - 分类、TEDS 等指标

4. **engines/** - 是否有相关的运行逻辑
   - `predictor.py` - 应该调用 tasks/ 中的 decode，不能自己实现

### 绝对禁止

- 在 engines/ 中重写 tasks/ 已有的 decode 逻辑
- 在推理代码中重写训练代码已有的逻辑
- 在多处重复实现相同的功能

### 复用调用链

```
训练 (train_joint.py)
  └── JointModel.forward()
        ├── self.line_pooling ──────┐
        ├── self.cls_head ──────────┼── 共享模块
        ├── self.stage3             │
        └── self.stage4             │
                                    │
推理/评估 (predictor.py)            │
  └── Predictor.predict()           │
        ├── SemanticClassificationTask.decode() ─┤
        ├── ParentFindingTask.decode() ──────────┤ 复用 tasks/
        └── model.line_pooling ──────────────────┘
```

### 目录职责

| 目录 | 职责 | 不放 |
|------|------|------|
| models/ | 网络结构、可复用组件 | 训练循环、指标 |
| tasks/ | loss/decode/metrics 逻辑 | 网络层实现 |
| engines/ | 训练/推理/评估循环 | 任务损失细节 |
| metrics/ | 纯指标计算 | 模型推理 |
| data/ | 数据加载、预处理 | 模型逻辑 |
| scripts/ | 极薄入口脚本 | 业务逻辑 |

## 修改代码的正确流程

1. 先搜索是否有现成实现：`Grep` 搜索相关函数名
2. 如果有，直接导入复用
3. 如果没有，在正确的目录创建，然后复用
4. 永远不要在调用方直接实现业务逻辑

## 详细规范

参见：`examples/项目结构说明.md`
