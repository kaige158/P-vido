"""CUDA 验证脚本"""
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"VRAM: {props.total_memory / 1024**3:.1f} GB")
    x = torch.randn(2, 3).cuda()
    print(f"GPU tensor test: {x.device} - OK!")
    print("")
    print("=== CUDA 已可用，TTS 推理可正常进行 ===")
else:
    print("CUDA 不可用，仅 CPU 推理")
