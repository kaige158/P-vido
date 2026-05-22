"""
语气词插入：在文本中按配置概率插入口语语气词。

用法：
    python scripts/insert_fillers.py --input output/text/ --output output/text_with_fillers/
"""

import argparse
import logging
import random
import re
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_sentences(text: str) -> list[str]:
    """按中文标点分句，保留标点。"""
    # 在 。！？；， 后切分
    pattern = re.compile(r"(.*?[。！？；，：])")
    parts = pattern.findall(text)
    if not parts:
        return [text]
    # 收集剩余部分
    remaining = pattern.sub("", text).strip()
    if remaining:
        parts.append(remaining)
    return parts


def is_headline(sentence: str) -> bool:
    """简单判断是否为标题（通常较短、无句号结尾）。"""
    stripped = sentence.strip()
    if len(stripped) <= 30 and not stripped.endswith("。"):
        return True
    return False


def is_code_block(sentence: str) -> bool:
    """简单判断是否为代码块内容（高密度英文符号）。"""
    stripped = sentence.strip()
    if len(stripped) == 0:
        return True
    # 含代码特征
    code_indicators = ["{", "}", "def ", "import ", "class ", "function ", "=>"]
    if any(ind in stripped for ind in code_indicators):
        return True
    return False


def is_list_item(sentence: str) -> bool:
    """判断是否为列表项。"""
    stripped = sentence.strip()
    if re.match(r"^[\s]*[-*\d.]+\s", stripped):
        return True
    return False


def should_insert(config: dict, sentence: str) -> bool:
    """判断当前句是否应该插入语气词。"""
    fillers_cfg = config.get("fillers", {})
    if not fillers_cfg.get("enabled", False):
        return False

    # 检查避免场景
    avoid_in = fillers_cfg.get("avoid_in", [])
    if "headline" in avoid_in and is_headline(sentence):
        return False
    if "code_block" in avoid_in and is_code_block(sentence):
        return False
    if "list_item" in avoid_in and is_list_item(sentence):
        return False

    rate = fillers_cfg.get("rate", 0.06)
    return random.random() < rate


def insert_fillers(text: str, config: dict) -> str:
    """对文本分段，在适用句子后插入语气词。"""
    fillers_cfg = config.get("fillers", {})
    words = fillers_cfg.get("words", ["嗯", "啊"])
    max_per_para = fillers_cfg.get("max_per_paragraph", 3)

    paragraphs = text.split("\n")
    result_parts = []

    for para in paragraphs:
        if para.strip() == "":
            result_parts.append(para)
            continue

        sentences = extract_sentences(para)
        filler_count = 0
        new_sentences = []

        for sent in sentences:
            new_sentences.append(sent)
            if filler_count < max_per_para and should_insert(config, sent):
                word = random.choice(words)
                new_sentences.append(word)
                filler_count += 1

        result_parts.append("".join(new_sentences))

    return "\n".join(result_parts)


def main():
    parser = argparse.ArgumentParser(description="语气词插入")
    parser.add_argument("--input", default="output/text/", help="纯文本目录或文件")
    parser.add_argument("--output", default="output/text_with_fillers/", help="输出目录")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，可复现")
    parser.add_argument("--dry-run", action="store_true", help="预览模式，只打印不写文件")
    args = parser.parse_args()

    random.seed(args.seed)
    config = load_config(args.config)
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if args.dry_run:
        logger.info("=== 预览模式 ===")

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.rglob("*.txt"))

    for f in files:
        text = f.read_text(encoding="utf-8")
        new_text = insert_fillers(text, config)

        if args.dry_run:
            logger.info("--- %s ---", f.name)
            print(new_text[:500])
            continue

        out_path = output_dir / f.relative_to(input_path.parent)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(new_text, encoding="utf-8")
        filler_count = len(re.findall("|".join(config["fillers"]["words"]), new_text))
        logger.info("%s → %s (插入 %d 处)", f.name, out_path.name, filler_count)


if __name__ == "__main__":
    main()
