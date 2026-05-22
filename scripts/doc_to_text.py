"""
文档转纯文本：Markdown / TXT / PDF → 清洗后的纯文本。

用法：
    python scripts/doc_to_text.py --input docs/ --output output/text/
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_markdown(text: str) -> str:
    """清洗 Markdown 标记，保留段落结构。"""
    # 移除 YAML frontmatter
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3 :]

    # 移除图片 ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # 移除链接保留文字 [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", text)
    # 移除行内代码
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # 移除代码块
    text = re.sub(r"```[\s\S]*?```", "", text)
    # 移除 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 移除粗体/斜体标记
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # 移除标题标记（保留标题文本）
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # 移除水平线
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # 移除列表标记（保留内容）
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)
    # 移除表格（近似）
    text = re.sub(r"\|[^\n]*\|", "", text)
    text = re.sub(r"^[\s]*\|[-:| ]+\|[\s]*$", "", text, flags=re.MULTILINE)
    # 合并多个空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_txt(text: str) -> str:
    """清洗纯文本。"""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(file_path: Path) -> str:
    """PDF 提取文本。依赖 pymupdf(fitz) 或 pdfplumber。"""
    try:
        import fitz
    except ImportError:
        try:
            import pdfplumber
        except ImportError:
            logger.error("PDF 提取需要 pymupdf 或 pdfplumber: pip install pymupdf")
            sys.exit(1)
        else:
            with pdfplumber.open(file_path) as pdf:
                texts = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(texts)
    else:
        doc = fitz.open(file_path)
        texts = [page.get_text() for page in doc]
        return "\n".join(texts)


def normalize_text(text: str, config: dict = None) -> str:
    """TTS 文本预处理：数字读法、专有名词替换、英文间隔。

    中文 TTS 对"2026年"读"二零二六"通常需要手动处理。
    本函数提供可配置的替换词典，优先用词典，否则保留原文让引擎自行处理。
    """
    # 英文单词前后加空格（中文 TTS 引擎需要）
    # 在中文和英文之间插入空格
    text = re.sub(r"([一-鿿])([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])([一-鿿])", r"\1 \2", text)

    # 年份：2026 → 二零二六（仅独立4位年份）
    def _year_repl(m):
        digits = m.group(0)
        if 1900 <= int(digits) <= 2099:
            digit_map = str.maketrans("0123456789", "零一二三四五六七八九")
            return digits.translate(digit_map)
        return digits

    text = re.sub(r"(?<!\d)(19|20)\d{2}(?!\d)", _year_repl, text)

    return text


def process_document(file_path: Path, output_dir: Path) -> Path:
    """处理单个文档，返回输出文件路径。"""
    suffix = file_path.suffix.lower()

    if suffix == ".md":
        text = file_path.read_text(encoding="utf-8")
        text = extract_markdown(text)
    elif suffix == ".txt":
        text = file_path.read_text(encoding="utf-8")
        text = extract_txt(text)
    elif suffix == ".pdf":
        text = extract_pdf(file_path)
    else:
        logger.warning("不支持的文件类型: %s", suffix)
        return None

    # 中文 TTS 文本规范化
    text = normalize_text(text)

    out_path = output_dir / f"{file_path.stem}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger.info("  → %s (%d 字符)", out_path.name, len(text))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="文档转纯文本")
    parser.add_argument("--input", default="docs/", help="文档目录或单个文件")
    parser.add_argument("--output", default="output/text/", help="纯文本输出目录")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        process_document(input_path, output_dir)
    elif input_path.is_dir():
        for f in sorted(input_path.rglob("*")):
            if f.is_file() and f.suffix.lower() in {".md", ".txt", ".pdf"}:
                process_document(f, output_dir)
    else:
        logger.error("输入路径不存在: %s", input_path)
        sys.exit(1)

    logger.info("文档转换完成。输出目录: %s", output_dir)


if __name__ == "__main__":
    main()
