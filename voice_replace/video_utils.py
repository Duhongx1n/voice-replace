# -*- coding: utf-8 -*-
"""视频工具模块 — 封装 FFmpeg / FFprobe 常用操作。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def check_ffmpeg() -> list:
    """
    检查 FFmpeg 和 FFprobe 是否可用。

    :return: 缺失工具的错误信息列表，空列表表示全部可用
    """
    missing = []
    if not shutil.which("ffmpeg"):
        missing.append("FFmpeg 未安装。安装方法: brew install ffmpeg")
    if not shutil.which("ffprobe"):
        missing.append("ffprobe 未安装。安装方法: brew install ffmpeg")
    return missing


def get_duration(filepath: str) -> float:
    """
    用 ffprobe 获取媒体文件时长（秒）。

    :param filepath: 媒体文件路径
    :return: 时长（秒），失败返回 0.0
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                filepath,
            ],
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (ValueError, FileNotFoundError):
        return 0.0


def extract_audio(
    input_path: str,
    output_path: str,
    sample_rate: int = 16000,
    max_duration: Optional[float] = None,
) -> str:
    """
    从视频/音频文件中提取音频。

    :param input_path: 输入文件路径
    :param output_path: 输出 WAV 文件路径
    :param sample_rate: 采样率（默认 16000）
    :param max_duration: 最大处理时长（秒），None 表示不限制
    :return: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
    ]
    if max_duration is not None:
        cmd.extend(["-t", str(max_duration)])
    cmd.extend([
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path,
    ])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[错误] FFmpeg 提取音频失败:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    return output_path


def extract_audio_segment(
    audio_path: str,
    start: float,
    end: float,
    output_path: str,
    sample_rate: int = 24000,
) -> str:
    """
    截取音频片段。

    :param audio_path: 源音频路径
    :param start: 开始时间（秒）
    :param end: 结束时间（秒）
    :param output_path: 输出路径
    :param sample_rate: 采样率
    :return: 输出文件路径
    """
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", str(start),
        "-t", str(duration),
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[错误] 截取音频片段失败: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return output_path


def replace_audio_track(
    input_video: str,
    new_audio: str,
    output_path: str,
) -> str:
    """
    用 FFmpeg 将新音频替换到原视频中。

    :param input_video: 原视频路径
    :param new_audio: 新音频路径
    :param output_path: 输出视频路径
    :return: 输出视频路径
    """
    video_duration = get_duration(input_video)
    audio_duration = get_duration(new_audio)

    print(f"  原视频时长: {video_duration:.2f}s")
    print(f"  新音频时长: {audio_duration:.2f}s")

    if abs(video_duration - audio_duration) > 1.0:
        print(f"  ⚠️ 时长差异: {abs(video_duration - audio_duration):.2f}s")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_video,
        "-i", new_audio,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]

    print("  正在合并...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[错误] FFmpeg 合并失败:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    final_duration = get_duration(output_path)
    print(f"  最终视频时长: {final_duration:.2f}s")
    return output_path


def is_video_file(filepath: str) -> bool:
    """
    检测文件是否为视频（是否包含视频流）。

    :param filepath: 文件路径
    :return: 是否为视频文件
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "csv=p=0",
                filepath,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return (
            result.returncode == 0
            and result.stdout.strip() not in ("", "0")
        )
    except Exception:
        ext = os.path.splitext(filepath)[1].lower()
        return ext in (
            ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"
        )
