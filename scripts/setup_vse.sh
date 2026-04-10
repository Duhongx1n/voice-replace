#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# VSE（Video-subtitle-extractor）安装脚本
#
# 使用方法：
#   bash scripts/setup_vse.sh
#
# 此脚本会：
# 1. 将 VSE 仓库 clone 到 third_party/video-subtitle-extractor
# 2. 安装 PaddlePaddle、PaddleOCR 等 Python 依赖

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VSE_DIR="$PROJECT_DIR/third_party/video-subtitle-extractor"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Video-subtitle-extractor (VSE) 安装脚本               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "项目目录: $PROJECT_DIR"
echo "VSE 目录: $VSE_DIR"
echo ""

# 步骤 1: Clone VSE 仓库
if [ -d "$VSE_DIR/backend" ]; then
    echo "✅ VSE 仓库已存在，跳过 clone"
else
    echo "📥 正在 clone VSE 仓库..."
    mkdir -p "$PROJECT_DIR/third_party"

    # 尝试多个镜像源
    CLONE_SUCCESS=false

    # 尝试 GitHub 直连
    echo "  尝试 GitHub 直连..."
    if git clone --depth 1 https://github.com/YaoFANGUK/video-subtitle-extractor.git "$VSE_DIR" 2>/dev/null; then
        CLONE_SUCCESS=true
        echo "  ✅ GitHub 直连成功"
    fi

    # 尝试 ghproxy 镜像
    if [ "$CLONE_SUCCESS" = false ]; then
        echo "  尝试 ghproxy 镜像..."
        if git clone --depth 1 https://ghproxy.com/https://github.com/YaoFANGUK/video-subtitle-extractor.git "$VSE_DIR" 2>/dev/null; then
            CLONE_SUCCESS=true
            echo "  ✅ ghproxy 镜像成功"
        fi
    fi

    # 尝试 gitclone 镜像
    if [ "$CLONE_SUCCESS" = false ]; then
        echo "  尝试 gitclone 镜像..."
        if git clone --depth 1 https://gitclone.com/github.com/YaoFANGUK/video-subtitle-extractor.git "$VSE_DIR" 2>/dev/null; then
            CLONE_SUCCESS=true
            echo "  ✅ gitclone 镜像成功"
        fi
    fi

    if [ "$CLONE_SUCCESS" = false ]; then
        echo "❌ 所有 clone 方式均失败，请手动 clone："
        echo "   git clone https://github.com/YaoFANGUK/video-subtitle-extractor.git $VSE_DIR"
        exit 1
    fi
fi

# 步骤 2: 安装 Python 依赖
echo ""
echo "📦 正在安装 Python 依赖..."

# 检测是否在虚拟环境中
if [ -n "$VIRTUAL_ENV" ]; then
    PIP="pip"
elif [ -d "$PROJECT_DIR/.venv" ]; then
    PIP="$PROJECT_DIR/.venv/bin/pip"
    echo "  使用虚拟环境: $PROJECT_DIR/.venv"
else
    PIP="pip3"
fi

# 安装 PaddlePaddle（CPU 版本，如需 GPU 版本请手动安装）
echo "  安装 PaddlePaddle..."
$PIP install paddlepaddle 2>&1 | tail -3

# 安装 PaddleOCR
echo "  安装 PaddleOCR..."
$PIP install paddleocr 2>&1 | tail -3

# 安装 OpenCV
echo "  安装 OpenCV..."
$PIP install opencv-python-headless 2>&1 | tail -3

# 步骤 3: 验证安装
echo ""
echo "🔍 验证安装..."

PYTHON="python3"
if [ -d "$PROJECT_DIR/.venv" ]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python3"
fi

$PYTHON -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')

# 检查 VSE
import os
vse_ok = os.path.isfile('$VSE_DIR/backend/main.py')
print(f'  VSE 仓库: {\"✅\" if vse_ok else \"❌\"}')

# 检查 PaddleOCR
try:
    from paddleocr import PaddleOCR
    print('  PaddleOCR: ✅')
except ImportError:
    print('  PaddleOCR: ❌')

# 检查 OpenCV
try:
    import cv2
    print(f'  OpenCV: ✅ (v{cv2.__version__})')
except ImportError:
    print('  OpenCV: ❌')

# 检查 PaddlePaddle
try:
    import paddle
    print(f'  PaddlePaddle: ✅ (v{paddle.__version__})')
except ImportError:
    print('  PaddlePaddle: ❌')
"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   安装完成！使用方法：                                   ║"
echo "║                                                          ║"
echo "║   python -m voice_replace --input video.mp4 \\           ║"
echo "║       --output_dir output --remove_subtitle              ║"
echo "║                                                          ║"
echo "║   指定 VSE 模式：                                        ║"
echo "║   python -m voice_replace --input video.mp4 \\           ║"
echo "║       --output_dir output --remove_subtitle \\           ║"
echo "║       --subtitle_mode vse                                ║"
echo "╚══════════════════════════════════════════════════════════╝"
