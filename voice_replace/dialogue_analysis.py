# -*- coding: utf-8 -*-
"""
对话结构分析模块 — 分析原视频的对话结构、角色切换和对话节奏。

本模块从 timeline.py 中提取出来，避免 timeline.py 与 text_adapter.py
之间的循环导入问题。
"""

from __future__ import annotations

from typing import Dict, List


def segments_as_slots(segments: List[Dict]) -> List[Dict]:
    """
    将 Whisper segments 转为时间槽列表。

    每个 segment 就是一个独立的时间槽，用户的每行台词对应一个槽。
    这样可以精确对齐每句台词在原视频中出现的时间点。

    :param segments: Whisper segments 列表
    :return: 时间槽列表
    """
    if not segments:
        return []
    return [
        {
            "start": seg["start"],
            "end": seg["end"],
            "original_text": seg.get("text", "").strip(),
        }
        for seg in segments
    ]


def analyze_dialogue_structure(slots: List[Dict]) -> List[Dict]:
    """
    分析原视频的对话结构，推断角色切换和对话节奏。

    通过相邻句子之间的停顿时长、句子长度变化等特征，
    推断可能的角色切换点和对话模式（问答、独白、评价等）。

    :param slots: 时间槽列表
    :return: 带有对话结构标注的时间槽列表
    """
    if not slots:
        return slots

    # 标注每句的特征
    annotated = []
    for i, s in enumerate(slots):
        duration = s["end"] - s["start"]
        char_count = len(s["original_text"])

        # 计算与前一句的间隔（停顿）
        if i > 0:
            gap = s["start"] - slots[i - 1]["end"]
        else:
            gap = 0.0

        # 推断句子类型
        if char_count <= 3 and duration <= 1.0:
            sentence_type = "短回应"
        elif char_count <= 5 and duration <= 1.5:
            sentence_type = "短句"
        elif "吗" in s["original_text"] or "呢" in s["original_text"]:
            sentence_type = "提问"
        elif char_count >= 10:
            sentence_type = "长叙述"
        else:
            sentence_type = "普通"

        # 推断是否可能是角色切换（停顿较长或句式突变）
        role_switch = False
        if gap > 0.2:
            role_switch = True
        elif i > 0:
            prev_len = len(slots[i - 1]["original_text"])
            if (prev_len <= 3 and char_count >= 8) or (prev_len >= 8 and char_count <= 3):
                role_switch = True

        annotated.append({
            **s,
            "duration": duration,
            "char_count": char_count,
            "gap_before": gap,
            "sentence_type": sentence_type,
            "role_switch": role_switch,
        })

    # 基于角色切换点分配角色标签
    role_labels = ["A", "B", "C", "D", "E"]
    current_role_idx = 0
    for i, item in enumerate(annotated):
        if i > 0 and item["role_switch"]:
            current_role_idx = (current_role_idx + 1) % len(role_labels)
        item["role_label"] = role_labels[current_role_idx]

    return annotated
