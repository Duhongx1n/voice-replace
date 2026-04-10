# -*- coding: utf-8 -*-
"""音频工具模块 — 静音生成、拼接、变速等通用音频操作。"""

from __future__ import annotations

import os
import subprocess
import sys
import wave
from typing import List, Tuple


def generate_silence_wav(
    duration_sec: float,
    sample_rate: int,
    output_path: str,
) -> str:
    """
    生成指定时长的静音 WAV 文件。

    :param duration_sec: 静音时长（秒）
    :param sample_rate: 采样率
    :param output_path: 输出文件路径
    :return: 输出文件路径
    """
    samples = int(duration_sec * sample_rate)
    with wave.open(output_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * samples)
    return output_path


def concatenate_audio_files(
    audio_files: List[str],
    output_path: str,
    sample_rate: int = 24000,
) -> str:
    """
    使用 FFmpeg concat 协议拼接多个音频文件。

    :param audio_files: 音频文件路径列表
    :param output_path: 输出文件路径
    :param sample_rate: 采样率
    :return: 输出文件路径
    """
    if not audio_files:
        return output_path

    list_path = output_path + ".filelist.txt"
    with open(list_path, "w") as f:
        for af in audio_files:
            abs_path = os.path.abspath(af)
            f.write(f"file '{abs_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[错误] 音频拼接失败: {result.stderr}", file=sys.stderr)

    # 清理临时文件列表
    if os.path.exists(list_path):
        os.remove(list_path)

    return output_path


def speed_up_audio(input_path: str, speed_factor: float) -> float:
    """
    对音频文件进行变速处理（保持音调不变），原地替换。

    :param input_path: 音频文件路径
    :param speed_factor: 加速倍率（如 1.2 表示加速 20%）
    :return: 变速后的时长（秒）
    """
    from voice_replace.video_utils import get_duration

    if speed_factor <= 1.0:
        return get_duration(input_path)

    tmp_path = input_path + ".tmp.wav"

    # ffmpeg atempo 支持范围 [0.5, 100.0]，对大于 2.0 的需要链式
    atempo_filters = []
    remaining = speed_factor
    while remaining > 2.0:
        atempo_filters.append("atempo=2.0")
        remaining /= 2.0
    atempo_filters.append(f"atempo={remaining:.4f}")
    filter_str = ",".join(atempo_filters)

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-filter:a", filter_str,
                "-loglevel", "error",
                tmp_path,
            ],
            check=True,
            capture_output=True,
        )
        os.replace(tmp_path, input_path)
        return get_duration(input_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"    [警告] 变速失败: {e}")
        return get_duration(input_path)


def trim_leading_noise(audio_data, sr: int, text: str) -> Tuple:
    """
    检测并裁剪音频开头的杂音/无关音节。

    Qwen3-TTS 经常在音频开头产生短促噪声或参考音频泄漏的无关音节。
    策略：用短时能量分析检测开头区域，找到第一个稳定语音起始点，
    裁掉之前的孤立噪声突刺。保护性裁剪：最多裁掉 0.8 秒。

    :param audio_data: numpy 音频数组
    :param sr: 采样率
    :param text: 原始文本（用于判断预期内容长度）
    :return: (trimmed_audio, trimmed_ms) 裁剪后的音频和裁掉的毫秒数
    """
    import numpy as np

    total_samples = len(audio_data)
    total_duration = total_samples / sr

    # 太短的音频不处理
    if total_duration < 0.5:
        return audio_data, 0

    # 分析区间：只检测前 1.2 秒
    analyze_sec = min(1.2, total_duration * 0.5)
    analyze_samples = int(analyze_sec * sr)
    segment = audio_data[:analyze_samples].astype(np.float64)

    # 计算短时能量（帧级别）
    frame_ms = 10
    frame_size = int(sr * frame_ms / 1000)
    hop = frame_size

    frame_energies = []
    for i in range(0, len(segment) - frame_size, hop):
        frame = segment[i:i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        frame_energies.append(rms)

    if len(frame_energies) < 5:
        return audio_data, 0

    frame_energies = np.array(frame_energies)

    # 用音频中段的能量作为基准
    mid_start = int(total_samples * 0.3)
    mid_end = int(total_samples * 0.7)
    mid_segment = audio_data[mid_start:mid_end].astype(np.float64)
    if len(mid_segment) > sr * 0.2:
        mid_rms = np.sqrt(np.mean(mid_segment ** 2))
    else:
        mid_rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))

    voice_threshold = mid_rms * 0.15

    # 找到第一个稳定语音段的起始帧
    min_stable_frames = 5
    stable_start_frame = None

    for i in range(len(frame_energies) - min_stable_frames):
        window = frame_energies[i:i + min_stable_frames]
        voice_ratio = np.mean(window >= voice_threshold)
        if voice_ratio >= 0.7:
            stable_start_frame = i
            break

    if stable_start_frame is None or stable_start_frame == 0:
        return audio_data, 0

    pre_frames = frame_energies[:stable_start_frame]
    noise_frames = np.sum(pre_frames >= voice_threshold)

    if noise_frames == 0:
        silent_ms = stable_start_frame * frame_ms
        if silent_ms > 200:
            trim_frames = max(0, stable_start_frame - 5)
            trim_samples = trim_frames * hop
            trimmed = audio_data[trim_samples:]
            return trimmed, trim_frames * frame_ms
        return audio_data, 0

    # 前导区域有能量脉冲 → 很可能是杂音
    safe_margin_frames = 3
    cut_frame = max(0, stable_start_frame - safe_margin_frames)
    cut_sample = cut_frame * hop

    max_cut_samples = int(0.8 * sr)
    if cut_sample > max_cut_samples:
        cut_sample = max_cut_samples

    if cut_sample <= 0:
        return audio_data, 0

    trimmed = audio_data[cut_sample:]
    trimmed_ms = int(cut_sample / sr * 1000)

    return trimmed, trimmed_ms


def suppress_transient_noise(
    input_path: str,
    output_path: str,
    sample_rate: int = 24000,
) -> str:
    """
    抑制背景音轨中的瞬态杂音（如观众笑声、掌声等）。

    策略：使用 FFmpeg 的 anlmdn（非局部均值降噪）和 highpass/lowpass
    滤波器组合，保留持续性的背景音乐（BGM），抑制短促的瞬态声音。

    处理链：
    1. highpass: 去除极低频隆隆声
    2. anlmdn: 非局部均值降噪，抑制瞬态噪声
    3. dynaudnorm: 动态音频归一化，平滑音量突变

    :param input_path: 输入背景音轨路径
    :param output_path: 输出处理后的背景音轨路径
    :param sample_rate: 采样率
    :return: 输出文件路径
    """
    # 使用 FFmpeg 滤波器链抑制瞬态噪声
    # anlmdn: 非局部均值降噪，s=7 表示搜索窗口大小，p=0.002 表示降噪强度
    # highpass: 去除 80Hz 以下的低频隆隆声
    # dynaudnorm: 动态归一化，平滑音量突变（笑声通常是突然变大的）
    filter_chain = (
        "highpass=f=80,"
        "anlmdn=s=7:p=0.002:r=0.002:m=15,"
        "dynaudnorm=f=150:g=15:p=0.7:m=10:r=0.9:n=1"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", filter_chain,
        "-ar", str(sample_rate),
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-loglevel", "error",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [警告] 瞬态噪声抑制失败: {result.stderr}")
        # 降级：只做简单的低通滤波
        fallback_cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", "highpass=f=80,lowpass=f=8000",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            output_path,
        ]
        result = subprocess.run(fallback_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [警告] 降级滤波也失败，直接复制原文件")
            import shutil
            shutil.copy2(input_path, output_path)

    return output_path


def mix_bgm_with_speech(
    speech_path: str,
    bgm_path: str,
    output_path: str,
    bgm_volume: float = 0.15,
    sample_rate: int = 24000,
) -> str:
    """
    将背景音（BGM）混合到新语音中。

    使用 FFmpeg 的 amix 滤波器将两个音轨混合，
    BGM 音量通过 volume 参数控制。

    :param speech_path: 新语音文件路径
    :param bgm_path: 背景音文件路径
    :param output_path: 输出混合后的音频路径
    :param bgm_volume: BGM 音量比例（0.0~1.0，默认 0.15 即 15%）
    :param sample_rate: 采样率
    :return: 输出文件路径
    """
    # 使用 FFmpeg 混合两个音轨
    # 先对 BGM 调整音量，再与语音混合
    filter_complex = (
        f"[1:a]volume={bgm_volume:.3f}[bgm];"
        f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", speech_path,
        "-i", bgm_path,
        "-filter_complex", filter_complex,
        "-ar", str(sample_rate),
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-loglevel", "error",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [警告] BGM 混合失败: {result.stderr}")
        print("  将使用纯语音（无背景音）")
        import shutil
        shutil.copy2(speech_path, output_path)

    return output_path
