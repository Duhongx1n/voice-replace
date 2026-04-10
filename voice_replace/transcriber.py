# -*- coding: utf-8 -*-
"""语音识别模块 — 封装 OpenAI Whisper，提取语音文字和时间戳。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from voice_replace.vad import refine_whisper_timestamps
from voice_replace.video_utils import extract_audio


def check_whisper() -> list:
    """
    检查 Whisper 是否可用。

    :return: 缺失依赖的错误信息列表
    """
    missing = []
    try:
        import whisper  # noqa: F401
    except ImportError:
        missing.append("openai-whisper 未安装。安装方法: pip install openai-whisper")
    return missing


def transcribe(
    audio_path: str,
    model_name: str = "base",
    language: Optional[str] = "zh",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    使用 Whisper 转写音频文件。

    :param audio_path: 音频文件路径
    :param model_name: Whisper 模型名称（tiny/base/small/medium/large）
    :param language: 语言代码（如 zh、en），None 表示自动检测
    :param device: 推理设备（cpu/cuda），None 表示自动选择
    :return: Whisper 转写结果字典
    """
    import whisper

    print(f"  加载 Whisper 模型: {model_name}")
    model = whisper.load_model(model_name, device=device)

    print(f"  转写中...")
    result = model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=False,
        verbose=False,
    )
    return result


def normalize_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    标准化 Whisper 的 segment 列表。

    :param segments: Whisper 原始 segments
    :return: 标准化后的 segments 列表
    """
    normalized = []
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        normalized.append({
            "id": int(segment.get("id", len(normalized))),
            "start": float(segment["start"]),
            "end": float(segment["end"]),
            "text": text,
        })
    return normalized


def seconds_to_srt_time(seconds: float) -> str:
    """
    将秒数转换为 SRT 时间格式。

    :param seconds: 秒数
    :return: SRT 格式时间字符串（HH:MM:SS,mmm）
    """
    total_milliseconds = int(round(seconds * 1000))
    hours = total_milliseconds // 3_600_000
    remainder = total_milliseconds % 3_600_000
    minutes = remainder // 60_000
    remainder %= 60_000
    secs = remainder // 1000
    milliseconds = remainder % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_outputs(
    output_dir: str,
    input_file: str,
    audio_file: str,
    model_name: str,
    language: Optional[str],
    text: str,
    segments: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    将转写结果写入多种格式文件。

    :param output_dir: 输出目录
    :param input_file: 输入文件名
    :param audio_file: 音频文件名
    :param model_name: 模型名称
    :param language: 语言代码
    :param text: 完整文本
    :param segments: 标准化后的 segments
    :return: 输出文件路径字典
    """
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(output_dir, "transcript.txt")
    srt_path = os.path.join(output_dir, "subtitles.srt")
    json_path = os.path.join(output_dir, "transcript.json")

    # 写入纯文本
    Path(txt_path).write_text(text.strip() + "\n", encoding="utf-8")

    # 写入 SRT 字幕
    srt_lines = []
    for index, segment in enumerate(segments, start=1):
        start = seconds_to_srt_time(float(segment["start"]))
        end = seconds_to_srt_time(float(segment["end"]))
        seg_text = str(segment.get("text", "")).strip()
        srt_lines.extend([str(index), f"{start} --> {end}", seg_text, ""])
    Path(srt_path).write_text("\n".join(srt_lines), encoding="utf-8")

    # 写入 JSON
    payload = {
        "input_file": os.path.basename(input_file),
        "audio_file": os.path.basename(audio_file),
        "model": model_name,
        "language": language,
        "text": text.strip(),
        "segments": segments,
    }
    Path(json_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "txt": txt_path,
        "srt": srt_path,
        "json": json_path,
    }


def extract_transcript(
    input_video: str,
    output_dir: str,
    language: str = "zh",
    model: str = "base",
    audio_rate: int = 16000,
    use_vad: bool = True,
) -> str:
    """
    完整的视频转写流程：提取音频 → Whisper 转写 → VAD 修正 → 输出文件。

    :param input_video: 输入视频路径
    :param output_dir: 输出目录
    :param language: 语言代码
    :param model: Whisper 模型名称
    :param audio_rate: 音频采样率
    :param use_vad: 是否使用 VAD 修正 Whisper 时间戳
    :return: transcript.txt 文件路径
    """
    transcribe_dir = os.path.join(output_dir, "step1_transcribe")
    os.makedirs(transcribe_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("📝 步骤 1/5：提取视频语音文字")
    print("=" * 60)

    # 提取音频
    audio_path = os.path.join(transcribe_dir, "audio.wav")
    print(f"  [1/4] 提取音频: {input_video}")
    extract_audio(input_video, audio_path, sample_rate=audio_rate)

    # Whisper 转写
    print(f"  [2/4] Whisper 转写中，模型: {model}")
    result = transcribe(audio_path, model_name=model, language=language)

    text = str(result.get("text", "")).strip()
    segments = normalize_segments(list(result.get("segments", [])))
    detected_language = language or result.get("language")

    # VAD 修正时间戳
    if use_vad and segments:
        print(f"  [3/4] VAD 修正时间戳...")
        try:
            segments = refine_whisper_timestamps(
                whisper_segments=segments,
                audio_path=audio_path,
                sample_rate=audio_rate,
            )
        except Exception as e:
            print(f"  [VAD] 修正失败，保持原始时间戳: {e}")
    else:
        print(f"  [3/4] 跳过 VAD 修正")

    # 写入输出文件
    print(f"  [4/4] 写入输出文件: {transcribe_dir}")
    paths = write_outputs(
        transcribe_dir,
        input_video,
        audio_path,
        model,
        detected_language,
        text,
        segments,
    )

    print(f"\n[步骤 1 完成] 转写文本: {paths['txt']}")
    print(f"  识别到 {len(segments)} 个语句片段")
    return paths["txt"]
