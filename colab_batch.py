#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VideoKit Colab 批处理模式脚本
适合在Colab中进行批量处理，无需打开Web UI
"""

import os
import sys
import argparse

# ===== 配置区域 =====
WORK_DIR = "/content/videokit_workspace"
SOURCE_PATHS = ["/content/drive/MyDrive/source.jpg"]  # 源文件路径
TARGET_PATH = "/content/drive/MyDrive/target.mp4"      # 目标文件路径
OUTPUT_PATH = "/content/drive/MyDrive/output.mp4"      # 输出路径

PROCESSORS = ["style_transfer"]  # 使用的处理器
EXECUTION_PROVIDERS = ["cuda"]  # 执行提供者
EXECUTION_DEVICE_IDS = [0]       # 设备ID


def main():
    # 设置工作目录
    os.chdir(WORK_DIR)
    sys.path.insert(0, WORK_DIR)
    
    # 环境变量
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # 导入videokit
    from videokit import core
    
    # 构建参数
    args = [
        "videokit.py",
        "headless-run",
        "--source-paths", *SOURCE_PATHS,
        "--target-path", TARGET_PATH,
        "--output-path", OUTPUT_PATH,
        "--processors", *PROCESSORS,
        "--execution-providers", *EXECUTION_PROVIDERS,
        "--execution-device-ids", *map(str, EXECUTION_DEVICE_IDS),
        "--video-memory-strategy", "strict",
        "--output-video-quality", "80",
        "--output-video-encoder", "libx264",
        "--log-level", "info"
    ]
    
    print("执行命令:")
    print(" ".join(args))
    print()
    
    # 执行
    sys.argv = args
    core.cli()


if __name__ == "__main__":
    main()
