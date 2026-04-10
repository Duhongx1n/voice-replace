# -*- coding: utf-8 -*-
"""
VAD（语音活动检测）模块 — 使用 Silero VAD 修正 Whisper 时间戳。

Whisper 的 segment 级别时间戳存在以下问题：
1. 相邻 segment 的 end/start 无缝衔接，吞掉了实际的非语音间隔
2. 非语音内容（铃声、音效等）被错误地分配到相邻 segment 的时间范围中

本模块通过 Silero VAD 检测真正的语音活动区间，修正每个 segment 的
start/end 时间，暴露出被 Whisper 隐藏的非语音间隔。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


def _load_silero_vad():
    """
    加载 Silero VAD 模型。

    通过 torch.hub 加载，不需要额外安装依赖。

    :return: (model, utils) 元组
    """
    import torch

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
        skip_validation=True,
    )
    return model, utils


def detect_speech_segments(
    audio_path: str,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
) -> List[Dict[str, float]]:
    """
    使用 Silero VAD 检测音频中的语音活动区间。

    :param audio_path: 音频文件路径（WAV 格式）
    :param sample_rate: 采样率（Silero VAD 支持 8000 或 16000）
    :param threshold: 语音检测阈值（0~1，越高越严格）
    :param min_speech_duration_ms: 最短语音段时长（毫秒）
    :param min_silence_duration_ms: 最短静音段时长（毫秒）
    :param speech_pad_ms: 语音段前后的填充时长（毫秒）
    :return: 语音区间列表，每个元素包含 start 和 end（秒）
    """
    import torch
    import torchaudio

    model, utils = _load_silero_vad()
    (
        get_speech_timestamps,
        save_audio,
        read_audio,
        VADIterator,
        collect_chunks,
    ) = utils

    # 读取音频
    wav = read_audio(audio_path, sampling_rate=sample_rate)

    # 获取语音时间戳（以采样点为单位）
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=threshold,
        sampling_rate=sample_rate,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=False,
    )

    # 转换为秒
    segments = []
    for ts in speech_timestamps:
        segments.append({
            "start": ts["start"] / sample_rate,
            "end": ts["end"] / sample_rate,
        })

    return segments


def refine_whisper_timestamps(
    whisper_segments: List[Dict[str, Any]],
    audio_path: str,
    sample_rate: int = 16000,
    vad_threshold: float = 0.5,
    tolerance: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    使用 VAD 结果修正 Whisper 的 segment 时间戳。

    修正策略：
    1. 用 Silero VAD 检测整段音频的语音活动区间
    2. 对每个 Whisper segment，在其时间范围附近查找 VAD 语音区间
    3. 用 VAD 检测到的实际语音边界替换 Whisper 的粗粒度时间戳
    4. 这样就能暴露出被 Whisper 隐藏的非语音间隔（铃声、音效等）

    :param whisper_segments: Whisper 转写的 segments 列表
    :param audio_path: 原始音频文件路径
    :param sample_rate: VAD 使用的采样率
    :param vad_threshold: VAD 检测阈值
    :param tolerance: 时间容差（秒），VAD 边界与 Whisper 边界的最大偏差
    :return: 修正后的 segments 列表
    """
    if not whisper_segments:
        return whisper_segments

    print("  [VAD] 正在检测语音活动区间...")
    vad_segments = detect_speech_segments(
        audio_path,
        sample_rate=sample_rate,
        threshold=vad_threshold,
    )
    print(f"  [VAD] 检测到 {len(vad_segments)} 个语音活动区间")

    if not vad_segments:
        print("  [VAD] 未检测到语音活动，保持原始时间戳")
        return whisper_segments

    # 打印 VAD 检测结果概览
    for i, vs in enumerate(vad_segments[:10]):
        print(f"    VAD 区间 {i}: [{vs['start']:.2f}s - {vs['end']:.2f}s]")
    if len(vad_segments) > 10:
        print(f"    ... 共 {len(vad_segments)} 个区间")

    # 修正每个 Whisper segment 的时间戳
    refined = []
    corrections = 0

    for seg in whisper_segments:
        w_start = float(seg["start"])
        w_end = float(seg["end"])
        w_mid = (w_start + w_end) / 2.0

        # 找到与当前 Whisper segment 重叠的所有 VAD 区间
        overlapping_vad = []
        for vs in vad_segments:
            # 计算重叠
            overlap_start = max(w_start, vs["start"])
            overlap_end = min(w_end, vs["end"])
            if overlap_end > overlap_start:
                overlapping_vad.append(vs)

        if not overlapping_vad:
            # 没有重叠的 VAD 区间，尝试找最近的
            nearest = _find_nearest_vad(w_mid, vad_segments)
            if nearest and abs(nearest["start"] - w_start) < tolerance * 3:
                overlapping_vad = [nearest]

        if overlapping_vad:
            # 如果有多个不连续的 VAD 区间，选择与 segment 中心最近的主要区间
            # 这避免了跨越非语音间隔（如铃声）的错误合并
            primary_vad = _select_primary_vad(overlapping_vad, w_mid)
            vad_start = primary_vad["start"]
            vad_end = primary_vad["end"]

            new_start = w_start
            new_end = w_end

            # 修正 start：如果 VAD 检测到的语音开始时间比 Whisper 晚，
            # 说明 Whisper 的 start 包含了非语音内容
            if vad_start > w_start + tolerance:
                new_start = vad_start
                corrections += 1

            # 修正 end：如果 VAD 检测到的语音结束时间比 Whisper 早，
            # 说明 Whisper 的 end 包含了非语音内容
            if vad_end < w_end - tolerance:
                new_end = vad_end
                corrections += 1

            # 安全检查：修正后的区间不能太短
            if new_end - new_start < 0.2:
                new_start = w_start
                new_end = w_end

            refined.append({
                **seg,
                "start": new_start,
                "end": new_end,
                "original_start": w_start,
                "original_end": w_end,
                "vad_refined": (new_start != w_start or new_end != w_end),
            })
        else:
            # 没有找到匹配的 VAD 区间，保持原始时间戳
            refined.append({
                **seg,
                "original_start": w_start,
                "original_end": w_end,
                "vad_refined": False,
            })

    # 后处理：确保相邻 segment 不重叠
    for i in range(1, len(refined)):
        if refined[i]["start"] < refined[i - 1]["end"]:
            # 取中间值作为分界点
            mid = (refined[i - 1]["end"] + refined[i]["start"]) / 2.0
            refined[i - 1]["end"] = mid
            refined[i]["start"] = mid

    # 统计修正情况
    refined_count = sum(1 for r in refined if r.get("vad_refined", False))
    print(f"  [VAD] 修正了 {refined_count}/{len(refined)} 个 segment 的时间戳")

    # 打印修正详情（前 20 个有变化的）
    shown = 0
    for i, r in enumerate(refined):
        if r.get("vad_refined") and shown < 20:
            orig_s = r.get("original_start", r["start"])
            orig_e = r.get("original_end", r["end"])
            print(
                f"    segment {i}: "
                f"[{orig_s:.2f}s-{orig_e:.2f}s] → "
                f"[{r['start']:.2f}s-{r['end']:.2f}s] "
                f"'{r.get('text', '')[:20]}...'"
            )
            shown += 1

    return refined


def _select_primary_vad(
    overlapping_vad: List[Dict[str, float]],
    segment_mid: float,
) -> Dict[str, float]:
    """
    从多个重叠的 VAD 区间中选择主要区间。

    当一个 Whisper segment 跨越了多个不连续的 VAD 区间时（中间有非语音间隔，
    如铃声、音效），简单地取 min(start)/max(end) 会把间隔"吞掉"。

    策略：
    1. 如果只有一个 VAD 区间，直接返回
    2. 如果有多个区间且它们之间有显著间隙（>0.5s），选择与 segment 中心
       重叠最多的那个区间
    3. 如果多个区间之间间隙很小（<=0.5s），合并它们

    :param overlapping_vad: 与 Whisper segment 重叠的 VAD 区间列表
    :param segment_mid: Whisper segment 的中心时间点
    :return: 选中的主要 VAD 区间
    """
    if len(overlapping_vad) <= 1:
        return overlapping_vad[0]

    # 按 start 排序
    sorted_vad = sorted(overlapping_vad, key=lambda x: x["start"])

    # 检查相邻 VAD 区间之间是否有显著间隙
    has_significant_gap = False
    for i in range(len(sorted_vad) - 1):
        gap = sorted_vad[i + 1]["start"] - sorted_vad[i]["end"]
        if gap > 0.5:
            has_significant_gap = True
            break

    if not has_significant_gap:
        # 间隙很小，合并所有区间
        return {
            "start": sorted_vad[0]["start"],
            "end": sorted_vad[-1]["end"],
        }

    # 有显著间隙，选择与 segment 中心最近的区间
    best = None
    best_score = float("-inf")
    for vs in sorted_vad:
        # 计算区间与 segment 中心的重叠程度
        # 优先选择包含中心点的区间，其次选择最近的
        if vs["start"] <= segment_mid <= vs["end"]:
            return vs
        # 用区间长度和距离的综合评分
        dist = min(abs(segment_mid - vs["start"]), abs(segment_mid - vs["end"]))
        duration = vs["end"] - vs["start"]
        score = duration - dist
        if score > best_score:
            best_score = score
            best = vs

    return best


def _find_nearest_vad(
    time_point: float,
    vad_segments: List[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    """
    找到距离指定时间点最近的 VAD 语音区间。

    :param time_point: 时间点（秒）
    :param vad_segments: VAD 语音区间列表
    :return: 最近的 VAD 区间，或 None
    """
    if not vad_segments:
        return None

    best = None
    best_dist = float("inf")

    for vs in vad_segments:
        # 计算时间点到区间的距离
        if vs["start"] <= time_point <= vs["end"]:
            return vs
        dist = min(abs(time_point - vs["start"]), abs(time_point - vs["end"]))
        if dist < best_dist:
            best_dist = dist
            best = vs

    return best
