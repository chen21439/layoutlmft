"""测试 HRDocLayoutXLMCollator"""
import os
import sys

# 尝试使用本地缓存
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"

from examples.comp_hrdoc.data.hrdoc_loader import HRDocDataset, HRDocLayoutXLMCollator
from transformers import AutoTokenizer

# 尝试多个模型路径
model_paths = [
    "microsoft/layoutxlm-base",
    "/home/ubuntu/.cache/huggingface/hub/models--microsoft--layoutxlm-base",
]

tokenizer = None
for path in model_paths:
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        print(f"Loaded tokenizer from: {path}")
        break
    except Exception as e:
        print(f"Failed to load from {path}: {e}")

if tokenizer is None:
    # 最后尝试联网下载
    print("Trying to download from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")

dataset = HRDocDataset(
    data_dir="/data/LLM_group/layoutlmft/data/HRDS",
    dataset_name="hrds",
    split="train",
    covmatch="doc_covmatch_0.8"
)
print(f"Dataset size: {len(dataset)}")

collator = HRDocLayoutXLMCollator(tokenizer, max_length=512)
batch = [dataset[i] for i in range(2)]
output = collator(batch)

print(f"\nCollator output:")
print(f"  input_ids: {output['input_ids'].shape}")
print(f"  line_ids: {output['line_ids'].shape}")
print(f"  parent_ids: {output['parent_ids'].shape}")
print(f"  sibling_labels: {output['sibling_labels'].shape}")
print(f"  line_mask: {output['line_mask'].shape}")

mask = output['line_mask']
parent = output['parent_ids']
print(f"\nValid lines: {mask.sum(dim=1).tolist()}")
print(f"Parent IDs [0, :15]: {parent[0, :15].tolist()}")

# 检查 parent_ids 中的值是否在有效范围内
for b in range(2):
    valid_lines = mask[b].nonzero().squeeze(-1).tolist()
    max_valid = max(valid_lines) if valid_lines else 0
    parent_vals = parent[b, :max_valid+1].tolist()
    invalid = [p for p in parent_vals if p > max_valid and p != -1]
    if invalid:
        print(f"WARNING: Sample {b} has invalid parent values: {invalid}")
    else:
        print(f"Sample {b}: All parent values valid (max line={max_valid})")

print(f"\nSibling pairs: {(output['sibling_labels'] > 0).sum().item() // 2}")
