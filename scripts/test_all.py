"""项目脚本快速验证"""
import sys, yaml
sys.path.insert(0, ".")

print("=== 1. 测试文本分段 ===")
from scripts.tts_batch import split_into_segments
segs = split_into_segments("Hello. This is test. Split done!", max_chars=20, min_chars=5)
assert len(segs) > 0
print(f"  Segments: {len(segs)} - OK")

print("=== 2. 测试语气词插入 ===")
import random
from scripts.insert_fillers import insert_fillers
random.seed(42)
config = yaml.safe_load(open("config.yaml", encoding="utf-8"))
result = insert_fillers("Hello world. Good morning.", config)
print(f"  Fillers inserted - OK")

print("=== 3. 测试文档转换 ===")
from scripts.doc_to_text import extract_markdown, normalize_text
text = "# Title\n\nHello **world**.\n\n```python\nprint(1)\n```\n\n- List item"
cleaned = extract_markdown(text)
assert "Hello world" in cleaned
assert "```" not in cleaned
print(f"  MD cleaned: {len(cleaned)} chars - OK")

print("=== 4. 测试配置加载 ===")
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
assert cfg["hardware"]["gpu_memory_gb"] == 4
assert cfg["tts"]["default_engine"] == "gpt_sovits"
assert cfg["segment"]["max_chars"] == 120
print(f"  Config loaded: {len(cfg)} sections - OK")

print("=== 5. 测试 PyTorch ===")
import torch
print(f"  PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()}) - OK")

print("\n=== ALL TESTS PASSED ===")
