#!/bin/bash
# ============================================================
# VideoKit Colab 一键安装脚本
# 在Colab中运行: !bash <(curl -s 脚本地址)
# 或者直接复制到单元格中执行
# ============================================================

set -e

echo "========================================"
echo "  VideoKit Colab 一键安装"
echo "========================================"

# 工作目录
WORK_DIR="/content/videokit_workspace"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo ""
echo "[1/5] 安装系统依赖..."
apt-get update -qq
apt-get install -y -qq ffmpeg curl wget git > /dev/null 2>&1
echo "  完成"

echo ""
echo "[2/5] 安装Python依赖..."
pip install -q gradio==5.44.1 gradio-rangeslider==0.0.8
pip install -q numpy==2.2.1 onnx==1.21.0
pip install -q opencv-python==4.13.0.92
pip install -q tqdm==4.67.3 scipy==1.17.1
pip install -q onnxruntime-gpu==1.24.4
echo "  完成"

echo ""
echo "[3/5] 创建目录结构..."
mkdir -p .assets/models
mkdir -p temp
mkdir -p output
echo "  完成"

echo ""
echo "[4/5] 环境变量..."
export OMP_NUM_THREADS=1
export GRADIO_ANALYTICS_ENABLED=0
echo "  完成"

echo ""
echo "========================================"
echo "  安装完成！"
echo "========================================"
echo ""
echo "工作目录: $WORK_DIR"
echo ""
echo "下一步："
echo "  1. 将 videokit 文件夹和 videokit.py 上传到此目录"
echo "  2. 运行: python videokit.py run --execution-providers cuda"
echo ""
