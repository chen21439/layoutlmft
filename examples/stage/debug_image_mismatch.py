#!/usr/bin/env python
"""
调试脚本：检查 tender 数据集中 image 数量和 chunks 数量是否匹配
"""
import os
import sys
import socket

# 根据主机名选择路径
hostname = socket.gethostname()
if hostname in ["i-2vc905bm", "ubuntu"]:  # 89_server
    BASE_DIR = "/home/ubuntu/code/layoutlmft"
    DATA_DIR = "/home/ubuntu/data/Tender"
    MODEL_PATH = "microsoft/layoutxlm-base"  # 使用 HuggingFace 模型
else:  # 本地
    BASE_DIR = "/root/code/layoutlmft"
    DATA_DIR = "/root/data/Tender"
    MODEL_PATH = "/root/models/layoutxlm-base"

# 添加项目路径
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "examples/stage"))

os.environ["HRDOC_DATA_DIR"] = DATA_DIR

from transformers import AutoTokenizer
from data.hrdoc_data_loader import HRDocDataLoader, HRDocDataLoaderConfig

def main():
    print("=" * 60)
    print("调试：检查 tender 数据集 image/chunks 数量匹配")
    print(f"主机: {hostname}")
    print("=" * 60)

    # 加载 tokenizer
    model_path = MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 配置 - 使用 max_pages_per_doc=10 拆分
    config = HRDocDataLoaderConfig(
        data_dir=DATA_DIR,
        max_length=512,
        label_all_tokens=True,
        document_level=True,  # 文档级别模式
        max_pages_per_doc=10,  # 拆分为最多 10 页
    )

    # 创建 DataLoader
    loader = HRDocDataLoader(
        tokenizer=tokenizer,
        config=config,
        dataset_name="tender",
        include_line_info=True,
    )

    # 准备数据集
    print("\n[1] 加载原始数据...")
    loader.load_raw_datasets()

    print("\n[2] 准备文档级别数据集...")
    loader.prepare_datasets()

    # 获取训练集
    train_dataset = loader.get_train_dataset()

    print(f"\n[3] 训练集文档数: {len(train_dataset)}")
    print("\n[4] 检查每个文档的 chunks 和 images 数量:")
    print("-" * 60)

    mismatches = []
    for idx, doc in enumerate(train_dataset):
        doc_name = doc.get("document_name", f"doc_{idx}")
        chunks = doc.get("chunks", [])
        num_chunks = len(chunks)

        # 统计有 image 的 chunk 数量
        num_images = sum(1 for c in chunks if c.get("image") is not None)

        status = "✓" if num_chunks == num_images else "✗ MISMATCH"
        print(f"  [{idx}] {doc_name}: chunks={num_chunks}, images={num_images} {status}")

        if num_chunks != num_images:
            mismatches.append({
                "idx": idx,
                "doc_name": doc_name,
                "num_chunks": num_chunks,
                "num_images": num_images,
            })

            # 打印详细信息
            print(f"       详细检查:")
            for ci, chunk in enumerate(chunks):
                has_image = chunk.get("image") is not None
                page_num = chunk.get("page_number", "?")
                print(f"         chunk[{ci}]: page={page_num}, has_image={has_image}")

    print("-" * 60)

    if mismatches:
        print(f"\n[!] 发现 {len(mismatches)} 个文档存在 image/chunks 数量不匹配:")
        for m in mismatches:
            print(f"    - {m['doc_name']}: {m['num_chunks']} chunks, {m['num_images']} images")
    else:
        print("\n[✓] 所有文档的 image 和 chunks 数量都匹配")

    # 测试 collator
    print("\n[5] 测试 collator...")
    from joint_data_collator import HRDocDocumentLevelCollator

    collator = HRDocDocumentLevelCollator(
        tokenizer=tokenizer,
        max_length=512,
    )

    # 取第一个文档测试
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        batch = collator([sample])

        input_ids_shape = batch["input_ids"].shape
        num_batch_chunks = input_ids_shape[0]

        if "image" in batch:
            images = batch["image"]
            num_batch_images = len(images) if isinstance(images, list) else images.shape[0]

            print(f"    input_ids shape: {input_ids_shape}")
            print(f"    num_chunks: {num_batch_chunks}")
            print(f"    num_images: {num_batch_images}")

            if num_batch_chunks != num_batch_images:
                print(f"    [!] MISMATCH: chunks={num_batch_chunks}, images={num_batch_images}")
            else:
                print(f"    [✓] 匹配")
        else:
            print(f"    [!] batch 中没有 image 字段")

if __name__ == "__main__":
    main()
