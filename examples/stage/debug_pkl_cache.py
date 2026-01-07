#!/usr/bin/env python
"""
调试脚本：直接检查 pkl 缓存文件中 image 和 chunks 的数量
"""
import os
import sys
import glob
import pickle
import socket

# 根据主机名选择路径
hostname = socket.gethostname()
if hostname in ["i-2vc905bm", "ubuntu"]:  # 89_server
    CACHE_DIR = "/home/ubuntu/data/Tender/.cache"
else:  # 本地
    CACHE_DIR = "/root/data/Tender/.cache"

def main():
    print("=" * 60)
    print("调试：检查 pkl 缓存文件中 image/chunks 数量")
    print(f"主机: {hostname}")
    print(f"缓存目录: {CACHE_DIR}")
    print("=" * 60)

    # 查找所有 pkl 文件
    pkl_pattern = os.path.join(CACHE_DIR, "**/*.pkl")
    pkl_files = glob.glob(pkl_pattern, recursive=True)

    # 只看 document_level 相关的缓存
    doc_level_pkls = [f for f in pkl_files if "document_level" in f or "doc_" in f.lower()]

    print(f"\n找到 {len(pkl_files)} 个 pkl 文件")
    print(f"其中 document_level 相关: {len(doc_level_pkls)} 个")

    if not pkl_files:
        print("\n没有找到 pkl 文件，列出缓存目录内容:")
        for root, dirs, files in os.walk(CACHE_DIR):
            level = root.replace(CACHE_DIR, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files[:10]:  # 只显示前10个文件
                print(f'{subindent}{file}')
            if len(files) > 10:
                print(f'{subindent}... 还有 {len(files)-10} 个文件')
        return

    # 检查每个 pkl 文件
    print("\n" + "-" * 60)

    for pkl_path in sorted(pkl_files)[:20]:  # 只检查前20个
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)

            filename = os.path.basename(pkl_path)

            if isinstance(data, list):
                # 文档列表
                print(f"\n[{filename}] - 包含 {len(data)} 个文档")
                for i, doc in enumerate(data[:5]):  # 只显示前5个
                    if isinstance(doc, dict):
                        doc_name = doc.get('document_name', f'doc_{i}')
                        chunks = doc.get('chunks', [])
                        num_chunks = len(chunks)

                        # 统计有 image 的 chunk 数量
                        num_images = 0
                        for c in chunks:
                            if isinstance(c, dict) and c.get('image') is not None:
                                num_images += 1

                        status = "✓" if num_chunks == num_images else f"✗ MISMATCH (diff={num_chunks-num_images})"
                        print(f"    [{i}] {doc_name}: chunks={num_chunks}, images={num_images} {status}")

                        if num_chunks != num_images:
                            # 详细检查
                            for ci, chunk in enumerate(chunks[:5]):
                                has_img = chunk.get('image') is not None if isinstance(chunk, dict) else False
                                page = chunk.get('page_number', '?') if isinstance(chunk, dict) else '?'
                                print(f"        chunk[{ci}]: page={page}, has_image={has_img}")
                            if len(chunks) > 5:
                                print(f"        ... 还有 {len(chunks)-5} 个 chunks")

                if len(data) > 5:
                    print(f"    ... 还有 {len(data)-5} 个文档")

            elif isinstance(data, dict):
                # 单个文档或其他结构
                print(f"\n[{filename}] - dict with keys: {list(data.keys())[:10]}")

                # 检查是否有 chunks
                if 'chunks' in data:
                    chunks = data['chunks']
                    num_chunks = len(chunks) if isinstance(chunks, list) else 0
                    num_images = sum(1 for c in chunks if isinstance(c, dict) and c.get('image') is not None)
                    print(f"    chunks={num_chunks}, images={num_images}")

            else:
                print(f"\n[{filename}] - type: {type(data)}")

        except Exception as e:
            print(f"\n[{os.path.basename(pkl_path)}] - Error: {e}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
