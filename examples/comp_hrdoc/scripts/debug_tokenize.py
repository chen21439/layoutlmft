"""调试脚本：检查 LayoutXLM tokenizer 对中文层级词的分词

用法：python examples/comp_hrdoc/scripts/debug_tokenize.py
"""

from transformers import AutoTokenizer


def analyze_tokenization(tokenizer, texts):
    """分析文本的分词结果"""
    print("=" * 60)
    print("Tokenization Analysis")
    print("=" * 60)

    for text in texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        print(f"\n文本: {text}")
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Token数: {len(tokens)}")

        # 检查关键字符是否独立
        key_chars = ['册', '章', '节', '一', '二', '三']
        for char in key_chars:
            if char in text:
                # 检查是否有独立的 token 包含该字符
                matching = [t for t in tokens if char in t]
                print(f"  '{char}' 出现在 tokens: {matching}")


def main():
    # 加载 LayoutXLM tokenizer
    model_path = "microsoft/layoutxlm-base"
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 测试文本
    test_texts = [
        "第一册",
        "第二册",
        "第一册 招标编号需求编制及格式",
        "第二册 招标编号需求编制及格式",
        "第一章",
        "第二章",
        "第一节",
        "第二节",
        "第一章 总则",
        "第二章 招标文件",
        "第一节 投标须知",
        "第三节 评标办法",
    ]

    analyze_tokenization(tokenizer, test_texts)

    # 对比 "一" vs "二" 的 token
    print("\n" + "=" * 60)
    print("数字字符 Token 对比")
    print("=" * 60)

    nums = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
    for num in nums:
        tokens = tokenizer.tokenize(num)
        ids = tokenizer.encode(num, add_special_tokens=False)
        print(f"  '{num}': tokens={tokens}, ids={ids}")

    # 单独检查 "册" "章" "节"
    print("\n" + "=" * 60)
    print("层级关键字 Token 对比")
    print("=" * 60)

    units = ['册', '章', '节', '编', '部', '条']
    for unit in units:
        tokens = tokenizer.tokenize(unit)
        ids = tokenizer.encode(unit, add_special_tokens=False)
        print(f"  '{unit}': tokens={tokens}, ids={ids}")


if __name__ == "__main__":
    main()
