# Comp_HRDoc 重构计划 - 删除 use-stage-features 模式

## 目标
简化代码，只保留 train-stage1 联合训练模式，删除 use-stage-features 冻结模式。

## 预期收益
- ✅ 减少 40% 代码量（~800 行）
- ✅ 统一训练/推理/评估接口
- ✅ 符合端到端训练理念
- ✅ 更容易维护和调试

---

## 修改清单

### 1. StageFeatureExtractor (utils/stage_feature_extractor.py)

#### 删除
- ❌ `extract_features()` 的 `@torch.no_grad()` 装饰器
- ❌ `extract_features_with_grad()` 方法
- ❌ `_extract_features_impl()` 的 `no_grad` 参数

#### 简化后
```python
class StageFeatureExtractor:
    def extract_features(self, ...):
        """统一的特征提取方法

        - 训练时：model.train() → 自动有梯度
        - 推理时：model.eval() + with torch.no_grad() → 无梯度
        """
        hidden_states = self.model.encode_with_micro_batch(...)
        text_hidden = hidden_states[:, :seq_len, :]

        # Line Pooling
        if is_page_level:
            line_features, line_mask = self._aggregate_page_level(...)
        else:
            line_features, line_mask = self._aggregate_document_level(...)

        # Line Enhancer
        if self.model.line_enhancer is not None:
            line_features = self.model.line_enhancer(line_features, line_mask)

        return line_features, line_mask
```

---

### 2. train_doc.py

#### 删除参数
- ❌ `--use-stage-features`
- ❌ `--train-stage1` (改为默认行为)

#### 删除代码块
- ❌ `if args.use_stage_features:` 分支（~300 行）
- ❌ `save_construct_model()` 函数
- ❌ `train_epoch_with_stage_features()` 的 `train_backbone` 参数

#### 简化后
```python
# 1. 初始化（统一）
stage_feature_extractor = StageFeatureExtractor(
    checkpoint_path=args.stage_checkpoint,
)
stage_feature_extractor.set_train_mode(freeze_visual=args.freeze_visual)

# 2. 训练（统一）
def train_epoch(model, dataloader, feature_extractor, optimizer):
    feature_extractor.model.train()
    model.train()

    for batch in dataloader:
        # 自动有梯度
        line_features, line_mask = feature_extractor.extract_features(...)

        # Stage1 cls loss (可选)
        cls_loss = compute_cls_loss_if_needed(...)

        # Construct forward
        outputs = model(region_features=line_features, ...)

        # Total loss
        loss = outputs["loss"] + cls_loss
        loss.backward()
        optimizer.step()

# 3. 评估（统一）
def evaluate(model, dataloader, feature_extractor):
    feature_extractor.model.eval()
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            line_features, line_mask = feature_extractor.extract_features(...)
            outputs = model(region_features=line_features, ...)
            # 计算指标...

# 4. 保存（统一）
def save_model(construct_model, save_path, feature_extractor, tokenizer):
    # 保存 backbone
    feature_extractor.model.backbone.save_pretrained("stage1/")
    # 保存 cls_head
    torch.save(feature_extractor.model.cls_head.state_dict(), "cls_head.pt")
    # 保存 line_enhancer
    if feature_extractor.model.line_enhancer is not None:
        torch.save(feature_extractor.model.line_enhancer.state_dict(), "line_enhancer.pt")
    # 保存 Construct
    torch.save(construct_model.state_dict(), "pytorch_model.bin")
```

---

### 3. engines/predictor.py

#### 无需修改
推理代码已经使用标准的 `extract_features()` + `with torch.no_grad()`

---

### 4. api/infer_service.py

#### 无需修改
API 代码也是使用标准的 `extract_features()` + model.eval()

---

## 实施步骤

### Phase 1: StageFeatureExtractor 简化
1. 删除 `@torch.no_grad()` 装饰器
2. 删除 `extract_features_with_grad()` 方法
3. 简化 `_extract_features_impl()` 去掉 `no_grad` 参数
4. 更新 JointModel.encode_with_micro_batch() 去掉 `no_grad` 参数

### Phase 2: train_doc.py 简化
1. 删除 `--use-stage-features` 参数和相关代码
2. 删除 `--train-stage1` 参数（改为默认）
3. 合并 `train_epoch_with_stage_features` 去掉 `train_backbone` 参数
4. 删除 `save_construct_model` 函数
5. 统一优化器配置逻辑

### Phase 3: 验证
1. 运行训练（确保梯度正常）
2. 运行评估（确保无梯度）
3. 运行推理（确保输出正确）
4. 启动 API 服务（确保服务正常）

---

## 风险控制

### 兼容性
- ✅ 旧的训练脚本需要删除 `--train-stage1` 参数
- ✅ 旧的训练脚本需要删除 `--use-stage-features` 参数

### 回滚
- 如果出问题，可以通过 git revert 回退

---

## 完成标准

- [x] 设计思路.md 更新完成
- [ ] StageFeatureExtractor 简化完成
- [ ] train_doc.py 简化完成
- [ ] 训练/评估/推理/API 功能验证通过
- [ ] 代码量减少 ~800 行
