# -*- coding: utf-8 -*-
"""音色提取模块 — 使用 Demucs 分离人声，筛选最佳参考片段。"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from voice_replace.audio_utils import suppress_transient_noise
from voice_replace.video_utils import extract_audio, extract_audio_segment, get_duration


# 只处理输入音频的前 N 秒（避免长视频全量处理）
MAX_PROCESS_DURATION = 180  # 3 分钟

# 参考片段时长范围（秒）
REF_MIN_DURATION = 5.0
REF_MAX_DURATION = 10.0

# 理想时长（评分偏好）
REF_IDEAL_DURATION = 7.0


def check_demucs() -> list:
    """
    检查 Demucs 是否可用。

    :return: 缺失依赖的错误信息列表
    """
    missing = []
    try:
        import demucs  # noqa: F401
    except ImportError:
        missing.append("demucs 未安装。安装方法: pip install demucs")
    return missing


def separate_vocals(audio_path: str, output_dir: str) -> str:
    """
    使用 Demucs 分离人声与背景。

    :param audio_path: 输入音频路径
    :param output_dir: 输出目录
    :return: 纯净人声文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    print("  使用 Demucs 分离人声...")

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "--out", output_dir,
        audio_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  [警告] htdemucs 失败，尝试 mdx_extra...")
        cmd[cmd.index("htdemucs")] = "mdx_extra"
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[错误] Demucs 分离失败:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)

    # 查找人声输出文件
    vocals_candidates = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if (
                f.lower().endswith(".wav")
                and "vocals" in f.lower()
                and "no_vocals" not in f.lower()
            ):
                vocals_candidates.append(os.path.join(root, f))

    if not vocals_candidates:
        print("[错误] Demucs 未生成人声输出。", file=sys.stderr)
        sys.exit(1)

    vocals_path = max(vocals_candidates, key=os.path.getmtime)
    print(f"  人声文件: {vocals_path}")

    # 查找背景音（no_vocals）输出文件
    no_vocals_candidates = []
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if (
                f.lower().endswith(".wav")
                and "no_vocals" in f.lower()
            ):
                no_vocals_candidates.append(os.path.join(root, f))

    no_vocals_path = None
    if no_vocals_candidates:
        no_vocals_path = max(no_vocals_candidates, key=os.path.getmtime)
        print(f"  背景音文件: {no_vocals_path}")
    else:
        print("  [警告] 未找到背景音（no_vocals）文件")

    # 转换为 TTS 标准格式（24kHz 单声道 16位）
    clean_vocals_path = os.path.join(output_dir, "clean_vocals.wav")
    convert_cmd = [
        "ffmpeg", "-y",
        "-i", vocals_path,
        "-ar", "24000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        clean_vocals_path,
    ]
    subprocess.run(convert_cmd, capture_output=True, text=True, check=True)

    print(f"  纯净人声: {clean_vocals_path}")

    # 处理背景音：转换格式 + 抑制观众笑声等瞬态杂音
    clean_bgm_path = None
    if no_vocals_path:
        raw_bgm_path = os.path.join(output_dir, "raw_bgm.wav")
        convert_bgm_cmd = [
            "ffmpeg", "-y",
            "-i", no_vocals_path,
            "-ar", "24000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            raw_bgm_path,
        ]
        subprocess.run(convert_bgm_cmd, capture_output=True, text=True, check=True)

        clean_bgm_path = os.path.join(output_dir, "clean_bgm.wav")
        print("  处理背景音（抑制观众笑声等瞬态杂音）...")
        suppress_transient_noise(raw_bgm_path, clean_bgm_path)
        print(f"  处理后背景音: {clean_bgm_path}")

    return clean_vocals_path, clean_bgm_path


def transcribe_with_whisper(audio_path: str) -> List[Dict]:
    """
    使用 Whisper 转录音频，返回带时间戳的句子列表。

    :param audio_path: 音频文件路径
    :return: 句子列表，每个元素包含 text, start, end, duration
    """
    import whisper

    print("  使用 Whisper 转录人声...")

    model = whisper.load_model("base")
    result = model.transcribe(
        audio_path,
        language="zh",
        word_timestamps=False,
    )

    segments = []
    for seg in result.get("segments", []):
        text = seg["text"].strip()
        start = seg["start"]
        end = seg["end"]
        duration = end - start

        if text and duration > 0:
            segments.append({
                "text": text,
                "start": start,
                "end": end,
                "duration": duration,
            })

    print(f"  Whisper 转录完成: {len(segments)} 个片段")
    return segments


def score_segment(
    segment: Dict,
    audio_path: str,
    sample_rate: int = 24000,
) -> float:
    """
    为候选参考片段打分（0~100）。

    综合评估：时长适中、语句完整、声音清晰、位置偏好。

    :param segment: 片段信息字典
    :param audio_path: 音频文件路径
    :param sample_rate: 采样率
    :return: 评分（0~100）
    """
    import numpy as np
    import wave

    score = 0.0
    duration = segment["duration"]
    text = segment["text"]

    # 1. 时长评分（满分 30 分）
    dur_diff = abs(duration - REF_IDEAL_DURATION)
    if dur_diff < 1.0:
        score += 30
    elif dur_diff < 2.0:
        score += 25
    elif dur_diff < 3.0:
        score += 15
    else:
        score += 5

    # 2. 语句完整性（满分 20 分）
    if re.search(r'[。！？!?]$', text):
        score += 20
    elif re.search(r'[，,；;：:]$', text):
        score += 8
    else:
        score += 12

    # 文字长度合理（20~80字最佳）
    text_len = len(text)
    if 20 <= text_len <= 80:
        score += 10
    elif 10 <= text_len <= 120:
        score += 5

    # 3. 声音质量（满分 30 分）
    try:
        start_sample = int(segment["start"] * sample_rate)
        end_sample = int(segment["end"] * sample_rate)

        with wave.open(audio_path, "rb") as wf:
            wf.readframes(start_sample)
            n_frames = end_sample - start_sample
            raw = wf.readframes(n_frames)

        audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)

        if len(audio_data) > 0:
            rms = np.sqrt(np.mean(audio_data ** 2))
            rms_norm = min(rms / 3000, 1.0)

            if 0.1 < rms_norm < 0.8:
                score += 20
            elif 0.05 < rms_norm <= 0.1:
                score += 10
            else:
                score += 5

            frame_size = int(sample_rate * 0.025)
            frames_rms = []
            for i in range(0, len(audio_data) - frame_size, frame_size):
                frame = audio_data[i:i + frame_size]
                frames_rms.append(np.sqrt(np.mean(frame ** 2)))

            if frames_rms:
                silence_ratio = np.mean(np.array(frames_rms) < 100)
                if silence_ratio < 0.1:
                    score += 10
                elif silence_ratio < 0.3:
                    score += 5
    except Exception:
        score += 10

    # 4. 位置偏好（满分 10 分）
    if segment["start"] < 60:
        score += 10
    elif segment["start"] < 120:
        score += 7
    else:
        score += 3

    return score


def select_best_segment(
    segments: List[Dict],
    audio_path: str,
) -> Optional[Dict]:
    """
    从 Whisper 转录结果中筛选最佳参考片段。

    策略：先找 5~10 秒单句 → 合并相邻短句 → 放宽条件。

    :param segments: Whisper 转录的片段列表
    :param audio_path: 音频文件路径
    :return: 最佳片段字典，或 None
    """
    candidates = []

    # 方案 A：直接找 5~10 秒的单句
    for seg in segments:
        if REF_MIN_DURATION <= seg["duration"] <= REF_MAX_DURATION:
            seg["score"] = score_segment(seg, audio_path)
            candidates.append(seg)

    # 方案 B：合并相邻短句到 5~10 秒
    if len(candidates) < 3:
        print("  单句候选不足，尝试合并相邻句子...")
        for i in range(len(segments)):
            merged_text = segments[i]["text"]
            merged_start = segments[i]["start"]
            merged_end = segments[i]["end"]

            for j in range(i + 1, min(i + 4, len(segments))):
                gap = segments[j]["start"] - merged_end
                if gap > 1.0:
                    break

                merged_text += segments[j]["text"]
                merged_end = segments[j]["end"]
                merged_duration = merged_end - merged_start

                if REF_MIN_DURATION <= merged_duration <= REF_MAX_DURATION:
                    merged_seg = {
                        "text": merged_text,
                        "start": merged_start,
                        "end": merged_end,
                        "duration": merged_duration,
                        "merged": True,
                    }
                    merged_seg["score"] = score_segment(merged_seg, audio_path)
                    candidates.append(merged_seg)
                    break
                elif merged_duration > REF_MAX_DURATION:
                    break

    # 方案 C：放宽条件
    if not candidates:
        print("  放宽时长限制，选择最接近 7s 的片段...")
        for seg in segments:
            if seg["duration"] >= 3.0:
                seg["score"] = score_segment(seg, audio_path)
                candidates.append(seg)

    if not candidates:
        print("[警告] 未找到任何合适的参考片段", file=sys.stderr)
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # 打印 top 3 候选
    print(f"\n  候选片段（Top {min(3, len(candidates))}）：")
    for i, c in enumerate(candidates[:3]):
        merged_tag = " [合并]" if c.get("merged") else ""
        print(
            f"    #{i + 1} 评分={c['score']:.0f} "
            f"时长={c['duration']:.1f}s "
            f"时间={c['start']:.1f}-{c['end']:.1f}s"
            f"{merged_tag}"
        )
        print(f"        文本: {c['text'][:60]}...")

    return candidates[0]


def extract_reference_voice(
    input_video: str,
    output_dir: str,
    max_duration: float = MAX_PROCESS_DURATION,
) -> str:
    """
    完整的音色提取流程：提取音频 → Demucs 分离 → Whisper 转录 → 筛选最佳片段。

    :param input_video: 输入视频路径
    :param output_dir: 输出目录
    :param max_duration: 最大处理时长（秒）
    :return: reference_voice.wav 文件路径
    """
    voice_dir = os.path.join(output_dir, "step2_voice")
    os.makedirs(voice_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("🎙️ 步骤 2/5：提取原视频音色参考片段")
    print("=" * 60)

    # 步骤 1a: 提取音频（仅前 N 分钟）
    audio_path = os.path.join(voice_dir, "extracted_audio.wav")
    print(f"  [1/4] 提取音频（最多 {max_duration}s）: {input_video}")
    extract_audio(
        input_video, audio_path,
        sample_rate=24000,
        max_duration=max_duration,
    )
    duration = get_duration(audio_path)
    print(f"  音频已提取: {audio_path}（{duration:.1f}s）")

    # 步骤 1b: 分离人声
    print(f"  [2/4] 分离人声...")
    demucs_output_dir = os.path.join(voice_dir, "demucs_output")
    clean_vocals, clean_bgm = separate_vocals(audio_path, demucs_output_dir)

    # 步骤 1c: Whisper 转录
    print(f"  [3/4] Whisper 转录...")
    segments = transcribe_with_whisper(clean_vocals)

    if not segments:
        print("[错误] Whisper 未识别出任何语音内容", file=sys.stderr)
        sys.exit(1)

    # 步骤 1d: 筛选最佳参考片段
    print(f"  [4/4] 筛选最佳参考片段...")
    best = select_best_segment(segments, clean_vocals)

    if best is None:
        print("[错误] 无法找到合适的参考片段", file=sys.stderr)
        sys.exit(1)

    # 截取参考片段
    ref_voice_path = os.path.join(voice_dir, "reference_voice.wav")
    extract_audio_segment(
        clean_vocals, best["start"], best["end"], ref_voice_path,
    )

    ref_text = best["text"]
    ref_text_path = os.path.join(voice_dir, "reference_text.txt")
    with open(ref_text_path, "w", encoding="utf-8") as f:
        f.write(ref_text)

    # 保存详细信息
    ref_info = {
        "text": ref_text,
        "start": best["start"],
        "end": best["end"],
        "duration": best["duration"],
        "score": best["score"],
        "merged": best.get("merged", False),
        "total_segments": len(segments),
        "audio_file": ref_voice_path,
        "text_file": ref_text_path,
        "bgm_file": clean_bgm,
    }
    ref_info_path = os.path.join(voice_dir, "reference_info.json")
    with open(ref_info_path, "w", encoding="utf-8") as f:
        json.dump(ref_info, f, ensure_ascii=False, indent=2)

    duration = get_duration(ref_voice_path)
    print(f"\n[步骤 2 完成] 最佳参考片段已提取")
    print(f"  参考音频: {ref_voice_path}")
    print(f"  参考文本: {ref_text}")
    print(f"  时长: {duration:.2f}秒")
    print(f"  评分: {best['score']:.0f}/100")

    return ref_voice_path
