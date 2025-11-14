# 论文参数对齐说明

HRDoc论文训练参数的完整对齐情况。

---

## 📊 论文 vs 当前配置对比

### HRDoc-Simple

| 参数 | 论文值 | cloud配置 | 状态 |
|------|--------|----------|------|
| **Training Steps** | 30,000 | **30,000** | ✅ 完全一致 |
| **Batch Size** | 3 (page-level) | **3** | ✅ 完全一致 |
| **Gradient Accumulation** | 推测=1 | **1** | ✅ 对齐 |
| **Training Time** | ~4.5小时 | ~4.5小时 | ✅ 预期一致 |
| **Hardware** | V100-24G | - | - |
| **Learning Rate** | 未公开 | **5e-5** | ⚠️ LayoutLMv2标准值 |
| **Warmup Ratio** | 未公开 | **0.1** | ⚠️ HF常用值 |
| **Weight Decay** | 未公开 | **0.01** | ⚠️ BERT/LayoutLM标准 |
| **LR Scheduler** | 未公开 | **linear** | ⚠️ HF默认 |
| **Optimizer** | 未公开 | **AdamW** | ⚠️ HF默认 |
| **FP16** | 推测使用 | **True** | ✅ 对齐硬件环境 |

### HRDoc-Hard

| 参数 | 论文值 | cloud_hard配置 | 状态 |
|------|--------|---------------|------|
| **Training Steps** | 40,000 | **40,000** | ✅ 完全一致 |
| **Batch Size** | 3 (page-level) | **3** | ✅ 完全一致 |
| **Training Time** | ~6小时 | ~6小时 | ✅ 预期一致 |
| 其他参数同Simple | - | 同上 | - |

---

## 📄 论文原文引用

> "We trained LayoutLMv2 on the HRDoc-Simple dataset with a **batch size of 3 (page-level) for 30,000 steps**, the training stage costs about **4.5 hours** on single NVIDIA V100-24G GPU."
>
> — [HRDoc GitHub README](https://github.com/jfma-USTC/HRDoc)

> "We trained LayoutLMv2 on the HRDoc-Hard dataset with a **batch size of 3 (page-level) for 40,000 steps**, the training stage costs about **6 hours** on single NVIDIA V100-24G GPU."
>
> — [HRDoc GitHub README](https://github.com/jfma-USTC/HRDoc)

---

## ⚙️ 未公开参数的处理策略

论文和官方仓库**未公开**以下关键超参数：

- Learning Rate
- Warmup策略
- Weight Decay
- LR Scheduler类型
- 优化器选择

### 解决方案

基于以下依据填补：

1. **LayoutLMv2官方示例**
2. **layoutlmft项目默认配置**
3. **BERT/Transformer微调最佳实践**
4. **HuggingFace Trainer默认值**

具体选择：

```python
learning_rate = 5e-5        # LayoutLM系列常用finetune lr
warmup_ratio = 0.1          # 前10% steps做warmup（3000步）
weight_decay = 0.01         # BERT标准值
lr_scheduler_type = "linear" # HF Trainer默认
optimizer = "AdamW"         # HF Trainer默认
```

### 参考来源

- [Hugging Face TrainingArguments文档](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
- [LayoutLMv2论文](https://arxiv.org/abs/2012.14740)
- [layoutlmft示例](https://github.com/microsoft/unilm/tree/master/layoutlmft)

---

## 🔬 验证方法

训练完成后，检查以下文件确认配置正确：

### 1. `trainer_state.json`

```bash
cat ./output/hrdoc_simple_full/trainer_state.json | grep -E "max_steps|global_step"
```

应输出：
```json
"max_steps": 30000,
"global_step": 30000  // 或接近30000
```

### 2. `training_args.bin`

```python
import torch
args = torch.load('./output/hrdoc_simple_full/training_args.bin')
print(f"batch_size: {args.per_device_train_batch_size}")  # 应为 3
print(f"learning_rate: {args.learning_rate}")            # 应为 5e-5
print(f"warmup_ratio: {args.warmup_ratio}")              # 应为 0.1
print(f"weight_decay: {args.weight_decay}")              # 应为 0.01
```

---

## 📈 预期训练曲线

基于论文和实践经验：

### HRDoc-Simple (30k步)

| Metric | 预期值 | 备注 |
|--------|-------|------|
| Final F1 | ~98% | 论文表格中Simple数据集F1 |
| Training Loss | 下降至<0.1 | 正常收敛 |
| Eval F1 (峰值) | 98-99% | 可能在后期略有波动 |

### HRDoc-Hard (40k步)

| Metric | 预期值 | 备注 |
|--------|-------|------|
| Final F1 | ~95% | 论文表格中Hard数据集F1 |
| Training Loss | 下降至<0.2 | Hard数据集难度更高 |
| Eval F1 (峰值) | 95-97% | 波动可能更明显 |

---

## 🚀 启动命令

### 使用配置文件

```bash
# HRDoc-Simple (30k步)
python train.py --config configs/cloud_config.json

# HRDoc-Hard (40k步)
python train.py --config configs/cloud_hard_config.json
```

### 使用便捷脚本

```bash
# HRDoc-Simple
./train_hrdoc_official.sh simple

# HRDoc-Hard
./train_hrdoc_official.sh hard
```

### 直接命令行

```bash
# HRDoc-Simple
python examples/run_hrdoc.py \
  --model_name_or_path microsoft/layoutlmv2-base-uncased \
  --output_dir ./output/hrdoc_simple_full \
  --do_train --do_eval \
  --max_steps 30000 \
  --per_device_train_batch_size 3 \
  --per_device_eval_batch_size 8 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --logging_steps 100 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --evaluation_strategy steps \
  --save_total_limit 3 \
  --fp16 \
  --overwrite_output_dir

# HRDoc-Hard: 只需改 max_steps 和 output_dir
```

---

## ⚠️ 注意事项

### 显存要求

- **推荐配置**: 20GB+ GPU显存 (V100/A100)
- **最低配置**: 16GB (可能需要调整batch_size)

如果显存不足：

```bash
# 方案1: 减小batch_size + 增加梯度累积
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2    # 有效batch=2×2=4 (接近论文的3)

# 方案2: 关闭FP16（不推荐，会更慢）
# 移除 --fp16 参数
```

### 训练中断恢复

如果训练中断，可以从checkpoint恢复：

```bash
# 系统会自动检测最新checkpoint
python examples/run_hrdoc.py \
  --output_dir ./output/hrdoc_simple_full \
  ...（其他参数同上）
  # 移除 --overwrite_output_dir
```

---

## 📊 与50步测试配置的对比

| 参数 | 测试配置 | 论文配置 | 差距 |
|------|---------|---------|------|
| max_steps | 50 | 30,000 | **600倍** |
| 训练时长 | ~8分钟 | ~4.5小时 | **34倍** |
| batch_size | 未知 | 3 | - |
| F1性能 | 低（训练不充分） | ~98% | 显著差距 |

**结论**: 之前的50步配置仅用于代码调试，**不能**用于正式实验或论文复现。

---

## 🔗 参考资料

- **HRDoc论文**: [arXiv:2303.13839](https://arxiv.org/abs/2303.13839)
- **HRDoc GitHub**: [jfma-USTC/HRDoc](https://github.com/jfma-USTC/HRDoc)
- **LayoutLMv2论文**: [arXiv:2012.14740](https://arxiv.org/abs/2012.14740)
- **HuggingFace Trainer文档**: [transformers.Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
