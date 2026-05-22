#!/bin/bash
# CUDA PyTorch 安装脚本
# 用法：在 Git Bash 中运行 bash scripts/install_cuda_torch.sh

set -e

echo "=== CUDA PyTorch 安装 ==="
echo ""
echo "方式1（推荐）：浏览器下载 wheel 后本地安装"
echo "  1. 浏览器打开: https://download.pytorch.org/whl/cu124"
echo "  2. 下载: torch-2.6.0+cu124-cp310-cp310-win_amd64.whl"
echo "  3. 运行: pip install <下载路径>/torch-2.6.0+cu124-cp310-cp310-win_amd64.whl"
echo "  4. 运行: pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
echo ""
echo "方式2：清华大学镜像（推荐国内用户）"
echo "  pip install torch torchvision torchaudio --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch/whl/cu124"
echo ""
echo "方式3：直接 pip（可能很慢）"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
echo ""
echo "=== 当前 PyTorch 状态 ==="
python -c "import torch; print('版本:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
