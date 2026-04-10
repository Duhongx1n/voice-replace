#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频语音替换工具 — 安装配置。"""

from __future__ import annotations

from setuptools import find_packages, setup

setup(
    name="voice-replace",
    version="1.0.0",
    description="视频语音替换工具 — 替换视频中的语言内容，保持原音色不变",
    author="duhongxin",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "openai-whisper>=20231117",
        "qwen-tts>=0.1.0",
        "torch>=2.0.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "demucs": ["demucs>=4.0.0"],
        "edge": ["edge-tts>=6.1.0"],
        "llm": ["openai>=1.0.0"],
        "all": ["demucs>=4.0.0", "edge-tts>=6.1.0", "openai>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "voice-replace=voice_replace.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)
