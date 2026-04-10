# -*- coding: utf-8 -*-
"""时间轴对齐引擎 — 按原始视频的时间节奏生成新语音。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

from voice_replace.audio_utils import (
    concatenate_audio_files,
    generate_silence_wav,
    mix_bgm_with_speech,
)
from voice_replace.synthesizer import (
    Qwen3TTSEngine,
    EdgeTTSEngine,
    detect_tts_engine,
)
from voice_replace.text_adapter import adapt_new_text
from voice_replace.video_utils import get_duration


def load_segment_timestamps(output_dir: str) -> Optional[List[Dict]]:
    """
    从 Whisper 转写结果中加载 segment 时间戳。

    :param output_dir: 输出目录（包含 step1_transcribe 子目录）
    :return: segments 列表，或 None
    """
    json_path = os.path.join(output_dir, "step1_transcribe", "transcript.json")
    if not os.path.isfile(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("segments", [])


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


def parse_new_lines(new_text_path: str) -> List[str]:
    """
    解析新台词文件，返回非空行列表（去掉 Markdown 标题行）。

    :param new_text_path: 新台词文件路径
    :return: 台词行列表
    """
    with open(new_text_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        result.append(stripped)
    return result


def _analyze_dialogue_structure(slots: List[Dict]) -> List[Dict]:
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
    role_label = "A"
    role_labels = ["A", "B", "C", "D", "E"]
    current_role_idx = 0
    for i, item in enumerate(annotated):
        if i > 0 and item["role_switch"]:
            current_role_idx = (current_role_idx + 1) % len(role_labels)
        item["role_label"] = role_labels[current_role_idx]

    return annotated


def create_editable_transcript(
    transcript_path: str,
    output_dir: str,
) -> str:
    """
    从 Whisper 转写结果生成带时间戳和对话结构分析的可编辑文件。

    生成的文件包含：
    - 原视频的对话结构分析（角色推断、句式类型、时长字数等）
    - 每句台词的详细标注，方便用户参考原视频节奏编写新台词

    :param transcript_path: 转写文本路径
    :param output_dir: 输出目录
    :return: 可编辑文件路径
    """
    edit_path = os.path.join(output_dir, "transcript_for_edit.md")

    segments = load_segment_timestamps(output_dir)

    if segments:
        slots = segments_as_slots(segments)
        annotated = _analyze_dialogue_structure(slots)

        # 统计角色分布
        roles_used = sorted(set(a["role_label"] for a in annotated))
        role_count = len(roles_used)
        dialogue_mode = "多人对话" if role_count > 1 else "单人独白"

        lines = [
            "# 视频台词编辑",
            "# 每行对应原视频的一个语句片段，请逐行替换为新台词。",
            f"# 行数应与原始片段数一致（{len(slots)} 行），以确保时间对齐。",
            "# 以 # 开头的行会被忽略。",
            "",
            "# ============================================================",
            "# 📊 原视频对话结构分析",
            "# ============================================================",
            f"# 对话模式: {dialogue_mode}（推断出 {role_count} 个角色: {', '.join('角色' + r for r in roles_used)}）",
            "# 编写建议:",
            "#   1. 保持与原视频相同的对话节奏（问答式/独白式）",
            "#   2. 每句新台词的字数尽量接近原文字数（影响语音时长）",
            "#   3. 短回应句（1-3字）对应短回应，长叙述句对应长叙述",
            "#   4. 角色切换处保持对话的自然过渡",
            "",
        ]

        for i, a in enumerate(annotated):
            duration = a["duration"]
            char_count = a["char_count"]
            role = a["role_label"]
            stype = a["sentence_type"]
            gap_info = ""
            if a["gap_before"] > 0.1:
                gap_info = f" 停顿{a['gap_before']:.1f}s"

            lines.append(
                f"# --- 第 {i + 1} 句 "
                f"[{a['start']:.2f}s - {a['end']:.2f}s] "
                f"时长{duration:.1f}s {char_count}字 "
                f"角色{role} {stype}{gap_info}"
            )
            lines.append(
                f"# 原文: {a['original_text']}"
            )
            lines.append(a["original_text"])
            lines.append("")

        content = "\n".join(lines)
    else:
        with open(transcript_path, "r", encoding="utf-8") as f:
            original_text = f.read().strip()
        content = f"# 视频台词编辑\n# 每行一句台词\n\n{original_text}\n"

    with open(edit_path, "w", encoding="utf-8") as f:
        f.write(content)

    return edit_path


def synthesize_timeline_aligned(
    new_text_path: str,
    reference_voice_path: str,
    output_dir: str,
    video_duration: float,
    llm_api_key: str = "",
    llm_base_url: str = "",
    llm_model: str = "gpt-4o-mini",
    force_adapt: bool = False,
    bgm_volume: float = 0.15,
    no_bgm: bool = False,
) -> str:
    """
    时间轴对齐合成：按原始视频的时间节奏生成新语音。

    流程：
    1. 读取 Whisper 的 segment 时间戳
    2. 将 segments 作为时间槽
    3. 智能分句适配（如需要，调用大模型将新台词按原视频节奏拆分）
    4. 将新台词逐行与时间槽一一对应
    5. 逐句 TTS 生成
    6. 按原始时间点插入，中间用静音填充
    7. 生成与原视频等长的完整音频
    8. 混合处理后的背景音（去除观众笑声等杂音，保留 BGM）

    :param new_text_path: 新台词文件路径
    :param reference_voice_path: 参考音频路径
    :param output_dir: 输出目录
    :param video_duration: 原视频时长（秒）
    :param llm_api_key: 大模型 API Key（用于智能分句）
    :param llm_base_url: 大模型 API Base URL
    :param llm_model: 大模型名称（默认 gpt-4o-mini）
    :param force_adapt: 强制使用大模型适配
    :param bgm_volume: 背景音音量比例（0.0~1.0，默认 0.15）
    :param no_bgm: 是否禁用背景音混合
    :return: full_speech.wav 文件路径
    """
    tts_dir = os.path.join(output_dir, "step4_tts")
    os.makedirs(tts_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("📄🔊 步骤 3-4/5：时间轴对齐合成")
    print("=" * 60)

    # 1. 加载原始时间戳
    segments = load_segment_timestamps(output_dir)
    if not segments:
        print("[错误] 未找到 Whisper 转写的时间戳数据", file=sys.stderr)
        sys.exit(1)

    # 2. 每个 segment 作为一个时间槽
    slots = segments_as_slots(segments)
    print(f"  原始 segments（时间槽）: {len(slots)} 个")
    for i, s in enumerate(slots):
        print(
            f"    槽 {i}: [{s['start']:.2f}s - {s['end']:.2f}s] "
            f"{s['original_text'][:30]}"
        )

    # 3. 解析新台词
    new_lines = parse_new_lines(new_text_path)
    print(f"  新台词: {len(new_lines)} 句")

    # 3.5 智能分句适配（如果新台词不符合原视频节奏，调用大模型拆分）
    new_lines = adapt_new_text(
        new_lines=new_lines,
        segments=slots,
        output_dir=output_dir,
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=llm_model,
        force_adapt=force_adapt,
    )
    print(f"  适配后台词: {len(new_lines)} 句")

    # 4. 对齐策略
    if len(new_lines) > len(slots):
        print(f"  ⚠️ 新台词 ({len(new_lines)} 句) 多于时间槽 ({len(slots)} 个)")
        print("  策略：多余的台词将紧接在最后一个时间槽之后")
    elif len(new_lines) < len(slots):
        print(f"  ⚠️ 新台词 ({len(new_lines)} 句) 少于时间槽 ({len(slots)} 个)")
        print("  策略：多余的时间槽将保持静音")

    # 5. 初始化 TTS 引擎（直接调用，不再通过 subprocess）
    print("\n  正在加载 TTS 引擎...")
    engine = Qwen3TTSEngine(reference_voice=reference_voice_path)
    engine.set_periodic_reload(30)
    sample_rate = engine.sample_rate

    # 6. 逐句生成 TTS
    sentences_dir = os.path.join(tts_dir, "sentences")
    os.makedirs(sentences_dir, exist_ok=True)

    generated_durations = []
    for idx, line in enumerate(new_lines):
        sent_file = os.path.join(sentences_dir, f"sent_{idx:03d}.wav")
        print(f"  🔊 生成中: [{idx + 1}/{len(new_lines)}] {line[:40]}...")

        start_t = time.time()
        duration = engine.synthesize(text=line, output_path=sent_file)
        gen_time = time.time() - start_t

        print(f"    时长: {duration:.2f}秒（生成耗时 {gen_time:.1f}秒）")

        if os.path.isfile(sent_file) and duration > 0:
            generated_durations.append((sent_file, duration))
        else:
            generated_durations.append((None, 0.0))

    # 7. 按时间轴拼接
    print("\n  🎯 按时间轴对齐拼接...")

    # 计算 TTS 语音片段的平均 RMS 响度，用于笑声音量归一化
    tts_avg_rms = _compute_tts_average_rms(generated_durations, sample_rate)
    if tts_avg_rms > 0:
        print(f"  📊 TTS 语音平均响度: RMS={tts_avg_rms:.0f}")
    else:
        print("  [提示] 无法计算 TTS 响度，笑声将保持原始音量")

    # 预计算原始 segments 之间的间隙（笑声/音效候选区域）
    # 这些区域可能包含笑声、铃声、音效等非语音内容
    vocals_path = _find_clean_vocals(output_dir)
    original_audio_path = _find_original_audio(output_dir)
    original_gaps = _compute_original_gaps(slots)
    if original_gaps:
        print(f"  🔍 检测到 {len(original_gaps)} 个原始间隙")
        for og in original_gaps:
            print(f"    间隙: [{og['start']:.2f}s - {og['end']:.2f}s] "
                  f"时长 {og['duration']:.2f}s")
        if vocals_path:
            print(f"  🎤 已加载原始人声轨道")
        if original_audio_path:
            print(f"  🎵 已加载原始完整音频（用于保留铃声/音效）")
    else:
        print("  [提示] 无间隙，gap 将用静音填充")

    # 预计算哪些句子紧接笑声/音效间隙（需要右对齐）
    pre_laugh_slots = _find_pre_laugh_slots(
        slots, original_gaps, vocals_path, original_audio_path,
    )
    if pre_laugh_slots:
        print(f"  🎭 检测到 {len(pre_laugh_slots)} 个笑声前句子需要右对齐: "
              f"{pre_laugh_slots}")

    audio_pieces = []
    current_pos = 0.0

    for idx in range(max(len(slots), len(new_lines))):
        if idx < len(slots):
            target_start = slots[idx]["start"]
        else:
            target_start = current_pos

        # 填充到目标时间点
        gap = target_start - current_pos
        if gap > 0.01:
            gap_file = os.path.join(tts_dir, f"gap_{idx:03d}.wav")
            # 检查这个 gap 是否落在原始 segments 之间的间隙中
            # 只有原始间隙才可能包含笑声，其他情况用静音
            matched_og = _match_original_gap(
                current_pos, target_start, original_gaps,
            )
            if matched_og:
                # 尝试从原始音频中保留间隙内容（铃声/音效/笑声）
                _extract_gap_from_original(
                    vocals_path=vocals_path,
                    original_audio_path=original_audio_path,
                    original_gap=matched_og,
                    needed_duration=gap,
                    output_path=gap_file,
                    sample_rate=sample_rate,
                    target_rms=tts_avg_rms,
                )
                audio_pieces.append((gap_file, f"间隙填充 {gap:.2f}s"))
            else:
                generate_silence_wav(gap, sample_rate, gap_file)
                audio_pieces.append((gap_file, f"静音 {gap:.2f}s"))
            current_pos = target_start

        # 插入新台词音频
        if idx < len(generated_durations) and generated_durations[idx][0]:
            sent_file, sent_dur = generated_durations[idx]

            # 检查是否需要加速
            if idx < len(slots):
                window_duration = slots[idx]["end"] - slots[idx]["start"]
                if sent_dur > window_duration * 1.2 and window_duration > 0.5:
                    speed_factor = sent_dur / window_duration
                    if speed_factor <= 2.0:
                        print(
                            f"    句 {idx}: 加速 x{speed_factor:.2f} "
                            f"({sent_dur:.2f}s → {window_duration:.2f}s)"
                        )
                        sped_file = os.path.join(tts_dir, f"sent_{idx:03d}_sped.wav")
                        atempo = f"atempo={speed_factor:.4f}"
                        subprocess.run(
                            [
                                "ffmpeg", "-y", "-i", sent_file,
                                "-filter:a", atempo,
                                "-loglevel", "error", sped_file,
                            ],
                            capture_output=True,
                        )
                        if os.path.isfile(sped_file):
                            sent_file = sped_file
                            sent_dur = get_duration(sped_file)

            # 笑声前的句子：右对齐（台词靠后，前面补静音）
            # 这样台词说完后笑声立刻接上，不会有尴尬的空白
            if idx in pre_laugh_slots and idx < len(slots):
                window_duration = slots[idx]["end"] - slots[idx]["start"]
                slack = window_duration - sent_dur
                if slack > 0.05:
                    pre_pad_file = os.path.join(
                        tts_dir, f"pre_laugh_pad_{idx:03d}.wav",
                    )
                    generate_silence_wav(slack, sample_rate, pre_pad_file)
                    audio_pieces.append(
                        (pre_pad_file,
                         f"笑声前静音补齐 {slack:.2f}s（句{idx}右对齐）")
                    )
                    current_pos += slack
                    print(
                        f"    句 {idx}: 右对齐 → "
                        f"前补静音 {slack:.2f}s，台词紧贴笑声"
                    )

            audio_pieces.append(
                (sent_file, f"句 {idx}: {new_lines[idx][:20]}... ({sent_dur:.2f}s)")
            )
            current_pos += sent_dur
            print(
                f"    [{target_start:.2f}s] 句 {idx}: "
                f"{new_lines[idx][:30]}... → {sent_dur:.2f}s"
            )

    # 8. 尾部填充到视频总时长（用静音，尾部不填原声避免带入杂音）
    if current_pos < video_duration:
        tail_gap = video_duration - current_pos
        tail_file = os.path.join(tts_dir, "tail_fill.wav")
        generate_silence_wav(tail_gap, sample_rate, tail_file)
        audio_pieces.append((tail_file, f"尾部静音 {tail_gap:.2f}s"))
        print(f"    尾部填充静音: {tail_gap:.2f}s")

    # 9. 用 FFmpeg concat 拼接所有片段
    full_speech_path = os.path.join(tts_dir, "full_speech.wav")
    filelist_path = os.path.join(tts_dir, "concat_list.txt")
    with open(filelist_path, "w") as f:
        for piece_file, desc in audio_pieces:
            abs_path = os.path.abspath(piece_file)
            f.write(f"file '{abs_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", filelist_path,
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        full_speech_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[错误] 音频拼接失败: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    final_dur = get_duration(full_speech_path)
    print(f"\n  ✅ 时间轴对齐合成完成")
    print(f"    纯语音时长: {final_dur:.2f}s（原视频: {video_duration:.2f}s）")

    # 10. 混合背景音（BGM）
    if not no_bgm:
        bgm_path = _find_bgm_file(output_dir)
        if bgm_path:
            print(f"\n  🎵 混合背景音（音量: {bgm_volume:.0%}）...")
            mixed_path = os.path.join(tts_dir, "full_speech_with_bgm.wav")
            mix_bgm_with_speech(
                speech_path=full_speech_path,
                bgm_path=bgm_path,
                output_path=mixed_path,
                bgm_volume=bgm_volume,
                sample_rate=sample_rate,
            )
            if os.path.isfile(mixed_path) and get_duration(mixed_path) > 0:
                # 用混合后的音频替换原文件
                os.replace(mixed_path, full_speech_path)
                final_dur = get_duration(full_speech_path)
                print(f"    混合后时长: {final_dur:.2f}s")
            else:
                print("    [警告] BGM 混合失败，使用纯语音")
        else:
            print("\n  [提示] 未找到背景音文件，跳过 BGM 混合")
    else:
        print("\n  [提示] 已禁用 BGM 混合")

    print(f"    输出: {full_speech_path}")

    return full_speech_path


def _find_bgm_file(output_dir: str) -> Optional[str]:
    """
    查找处理后的背景音文件。

    优先查找 clean_bgm.wav（已抑制观众笑声），
    其次查找 raw_bgm.wav，
    最后尝试从 reference_info.json 中获取。

    :param output_dir: 输出目录
    :return: 背景音文件路径，或 None
    """
    voice_dir = os.path.join(output_dir, "step2_voice")

    # 优先使用处理后的背景音
    clean_bgm = os.path.join(voice_dir, "demucs_output", "clean_bgm.wav")
    if os.path.isfile(clean_bgm):
        return clean_bgm

    # 其次使用原始背景音
    raw_bgm = os.path.join(voice_dir, "demucs_output", "raw_bgm.wav")
    if os.path.isfile(raw_bgm):
        return raw_bgm

    # 从 reference_info.json 中获取
    ref_info_path = os.path.join(voice_dir, "reference_info.json")
    if os.path.isfile(ref_info_path):
        import json as _json
        with open(ref_info_path, "r", encoding="utf-8") as f:
            ref_info = _json.load(f)
        bgm_file = ref_info.get("bgm_file")
        if bgm_file and os.path.isfile(bgm_file):
            return bgm_file

    # 尝试查找 Demucs 原始输出的 no_vocals.wav
    demucs_dir = os.path.join(voice_dir, "demucs_output")
    if os.path.isdir(demucs_dir):
        for root, dirs, files in os.walk(demucs_dir):
            for f in files:
                if (
                    f.lower().endswith(".wav")
                    and "no_vocals" in f.lower()
                ):
                    return os.path.join(root, f)

    return None


def _find_clean_vocals(output_dir: str) -> Optional[str]:
    """
    查找原始人声轨道文件（clean_vocals.wav）。

    用于在 segments 之间的间隙中保留人物的笑声、感叹声等
    非语言声音，而不是用静音填充。

    :param output_dir: 输出目录
    :return: 人声轨道文件路径，或 None
    """
    voice_dir = os.path.join(output_dir, "step2_voice")

    # 优先使用处理后的纯净人声
    clean_vocals = os.path.join(voice_dir, "demucs_output", "clean_vocals.wav")
    if os.path.isfile(clean_vocals):
        return clean_vocals

    # 尝试查找 Demucs 原始输出的 vocals.wav
    demucs_dir = os.path.join(voice_dir, "demucs_output")
    if os.path.isdir(demucs_dir):
        for root, dirs, files in os.walk(demucs_dir):
            for f in files:
                if (
                    f.lower().endswith(".wav")
                    and "vocals" in f.lower()
                    and "no_vocals" not in f.lower()
                    and "clean" not in f.lower()
                ):
                    return os.path.join(root, f)

    return None


def _find_original_audio(output_dir: str) -> Optional[str]:
    """
    查找原始完整音频文件（包含所有声音：人声+铃声+音效+BGM）。

    用于在 segments 之间的间隙中保留非语音内容（如电话铃声、
    音效等），这些内容在人声分离后的轨道中可能丢失。

    :param output_dir: 输出目录
    :return: 原始音频文件路径，或 None
    """
    # 优先使用 step1_transcribe 中提取的原始音频
    audio_path = os.path.join(output_dir, "step1_transcribe", "audio.wav")
    if os.path.isfile(audio_path):
        return audio_path

    return None


def _compute_original_gaps(
    slots: List[Dict],
    min_gap: float = 0.3,
) -> List[Dict]:
    """
    计算原始 segments 之间的间隙列表。

    只有这些间隙才可能包含笑声等非语言声音。
    短间隙（<0.5s）通常只是自然停顿，不需要特殊处理。

    :param slots: 时间槽列表
    :param min_gap: 最小间隙时长（秒），低于此值视为自然停顿
    :return: 间隙列表，每个元素包含 start, end, duration, before_slot_idx
    """
    gaps = []
    for i in range(1, len(slots)):
        gap_start = slots[i - 1]["end"]
        gap_end = slots[i]["start"]
        gap_duration = gap_end - gap_start
        if gap_duration >= min_gap:
            gaps.append({
                "start": gap_start,
                "end": gap_end,
                "duration": gap_duration,
                "before_slot_idx": i,
            })
    return gaps


def _match_original_gap(
    fill_start: float,
    fill_end: float,
    original_gaps: List[Dict],
    overlap_threshold: float = 0.3,
) -> Optional[Dict]:
    """
    检查当前需要填充的时间段是否与某个原始间隙重叠。

    只有当填充区间与原始间隙有足够重叠时，才认为匹配。
    这样可以避免因 TTS 时长偏差导致截取到原台词内容。

    :param fill_start: 填充区间开始时间
    :param fill_end: 填充区间结束时间
    :param original_gaps: 原始间隙列表
    :param overlap_threshold: 重叠比例阈值（0~1）
    :return: 匹配的原始间隙，或 None
    """
    fill_duration = fill_end - fill_start
    if fill_duration <= 0:
        return None

    for og in original_gaps:
        # 计算重叠区间
        overlap_start = max(fill_start, og["start"])
        overlap_end = min(fill_end, og["end"])
        overlap = max(0, overlap_end - overlap_start)

        # 重叠比例：相对于原始间隙的比例
        if og["duration"] > 0 and overlap / og["duration"] >= overlap_threshold:
            return og

    return None


def _compute_tts_average_rms(
    generated_durations: List[Tuple],
    sample_rate: int = 24000,
) -> float:
    """
    计算所有 TTS 语音片段的平均 RMS 响度。

    用于笑声音量归一化，使笑声的响度与 TTS 语音保持一致。

    :param generated_durations: TTS 生成结果列表 [(文件路径, 时长), ...]
    :param sample_rate: 采样率
    :return: 平均 RMS 值，如果无法计算则返回 0.0
    """
    import numpy as np
    import wave

    rms_values = []
    for sent_file, sent_dur in generated_durations:
        if not sent_file or sent_dur <= 0 or not os.path.isfile(sent_file):
            continue
        try:
            with wave.open(sent_file, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
            if len(audio_data) > 0:
                rms = np.sqrt(np.mean(audio_data ** 2))
                if rms > 0:
                    rms_values.append(rms)
        except Exception:
            continue

    if not rms_values:
        return 0.0
    return float(np.mean(rms_values))


def _extract_gap_from_original(
    vocals_path: Optional[str],
    original_audio_path: Optional[str],
    original_gap: Dict,
    needed_duration: float,
    output_path: str,
    sample_rate: int = 24000,
    energy_threshold: float = 200.0,
    target_rms: float = 0.0,
) -> str:
    """
    从原始音频中截取间隙内容，保留铃声、音效、笑声等非语音内容。

    策略（优先级从高到低）：
    1. 先检查原始完整音频中该间隙是否有内容（铃声、音效等）
    2. 如果原始音频有内容，直接使用（保留铃声/音效）
    3. 如果原始音频无内容，检查人声轨道是否有笑声
    4. 如果都没有内容，生成静音

    :param vocals_path: 原始人声轨道路径（可选）
    :param original_audio_path: 原始完整音频路径（可选，包含铃声/音效）
    :param original_gap: 原始间隙信息（包含 start, end, duration）
    :param needed_duration: 需要填充的时长（秒）
    :param output_path: 输出文件路径
    :param sample_rate: 采样率
    :param energy_threshold: RMS 能量阈值
    :param target_rms: 目标 RMS 响度
    :return: 输出文件路径
    """
    import numpy as np
    import wave

    og_start = original_gap["start"]
    og_end = original_gap["end"]
    og_duration = original_gap["duration"]

    # 策略 1：尝试从原始完整音频中截取（保留铃声/音效）
    if original_audio_path and os.path.isfile(original_audio_path):
        tmp_orig = output_path + ".orig_full.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", original_audio_path,
            "-ss", f"{og_start:.4f}",
            "-t", f"{og_duration:.4f}",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            tmp_orig,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0 and os.path.isfile(tmp_orig):
            try:
                with wave.open(tmp_orig, "rb") as wf:
                    raw = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
                rms = np.sqrt(np.mean(audio_data ** 2)) if len(audio_data) > 0 else 0.0
            except Exception:
                rms = 0.0

            if rms >= energy_threshold:
                print(f"      间隙 [{og_start:.2f}s-{og_end:.2f}s] "
                      f"RMS={rms:.0f} ≥ {energy_threshold:.0f}，"
                      f"保留原始音频（铃声/音效/笑声）")

                # 响度归一化
                if target_rms > 0 and rms > 0:
                    gain = target_rms / rms
                    gain = max(0.3, min(3.0, gain))
                    if abs(gain - 1.0) > 0.05:
                        _normalize_audio_volume(tmp_orig, gain, sample_rate)
                        print(f"      音量归一化: 增益 x{gain:.2f}")

                # 调整时长
                _adjust_gap_duration(
                    tmp_orig, og_duration, needed_duration,
                    output_path, sample_rate,
                )

                # 清理临时文件
                if os.path.isfile(tmp_orig) and tmp_orig != output_path:
                    os.remove(tmp_orig)

                if os.path.isfile(output_path):
                    return output_path

            # 原始音频能量不足，清理临时文件
            if os.path.isfile(tmp_orig):
                os.remove(tmp_orig)

    # 策略 2：回退到人声轨道检查笑声
    if vocals_path and os.path.isfile(vocals_path):
        return _extract_gap_with_energy_check(
            vocals_path=vocals_path,
            original_gap=original_gap,
            needed_duration=needed_duration,
            output_path=output_path,
            sample_rate=sample_rate,
            energy_threshold=energy_threshold,
            target_rms=target_rms,
        )

    # 策略 3：都没有，生成静音
    print(f"      间隙 [{og_start:.2f}s-{og_end:.2f}s] 无可用音源，用静音")
    generate_silence_wav(needed_duration, sample_rate, output_path)
    return output_path


def _adjust_gap_duration(
    input_path: str,
    source_duration: float,
    needed_duration: float,
    output_path: str,
    sample_rate: int = 24000,
) -> None:
    """
    调整间隙音频的时长以匹配需要的 gap 时长。

    :param input_path: 输入音频路径
    :param source_duration: 源音频时长
    :param needed_duration: 需要的时长
    :param output_path: 输出路径
    :param sample_rate: 采样率
    """
    if abs(source_duration - needed_duration) < 0.05:
        # 时长接近，直接使用
        if input_path != output_path:
            os.replace(input_path, output_path)
    elif source_duration > needed_duration:
        # 源更长，居中截取
        offset = (source_duration - needed_duration) / 2
        trim_cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ss", f"{offset:.4f}",
            "-t", f"{needed_duration:.4f}",
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-loglevel", "error",
            output_path,
        ]
        subprocess.run(trim_cmd, capture_output=True, text=True)
    else:
        # 源更短，前后补静音
        pad_total = needed_duration - source_duration
        pad_before = pad_total / 2
        pad_after = pad_total - pad_before

        pad_before_file = output_path + ".pad_before.wav"
        pad_after_file = output_path + ".pad_after.wav"
        generate_silence_wav(pad_before, sample_rate, pad_before_file)
        generate_silence_wav(pad_after, sample_rate, pad_after_file)

        concat_list = output_path + ".concat.txt"
        with open(concat_list, "w") as f:
            f.write(f"file '{os.path.abspath(pad_before_file)}'\n")
            f.write(f"file '{os.path.abspath(input_path)}'\n")
            f.write(f"file '{os.path.abspath(pad_after_file)}'\n")

        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-loglevel", "error",
            output_path,
        ]
        subprocess.run(concat_cmd, capture_output=True, text=True)

        # 清理临时文件
        for tmp in [pad_before_file, pad_after_file, concat_list]:
            if os.path.isfile(tmp):
                os.remove(tmp)


def _extract_gap_with_energy_check(
    vocals_path: str,
    original_gap: Dict,
    needed_duration: float,
    output_path: str,
    sample_rate: int = 24000,
    energy_threshold: float = 200.0,
    target_rms: float = 0.0,
) -> str:
    """
    从原始人声轨道中截取间隙音频，并用能量检测判断是否有笑声。

    策略：
    1. 从原始人声轨道中截取原始间隙对应的时间段
    2. 检测截取音频的 RMS 能量
    3. 如果能量高于阈值（说明有笑声/感叹声），使用该音频
    4. 如果能量低于阈值（说明是静音），生成静音
    5. 如果截取的时长与需要的时长不同，进行裁剪或补静音
    6. 对保留的笑声进行响度归一化，使其与 TTS 语音音量一致

    :param vocals_path: 原始人声轨道路径
    :param original_gap: 原始间隙信息（包含 start, end, duration）
    :param needed_duration: 需要填充的时长（秒）
    :param output_path: 输出文件路径
    :param sample_rate: 采样率
    :param energy_threshold: RMS 能量阈值，高于此值认为有笑声
    :param target_rms: 目标 RMS 响度（TTS 语音的平均响度），用于音量归一化
    :return: 输出文件路径
    """
    import numpy as np
    import wave

    og_start = original_gap["start"]
    og_end = original_gap["end"]
    og_duration = original_gap["duration"]

    # 先截取原始间隙的音频到临时文件
    tmp_path = output_path + ".tmp.wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", vocals_path,
        "-ss", f"{og_start:.4f}",
        "-t", f"{og_duration:.4f}",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-loglevel", "error",
        tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not os.path.isfile(tmp_path):
        generate_silence_wav(needed_duration, sample_rate, output_path)
        return output_path

    # 读取音频数据，检测能量
    try:
        with wave.open(tmp_path, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        rms = np.sqrt(np.mean(audio_data ** 2)) if len(audio_data) > 0 else 0.0
    except Exception:
        rms = 0.0

    if rms < energy_threshold:
        # 能量太低，说明是静音区域，不需要填充原声
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        generate_silence_wav(needed_duration, sample_rate, output_path)
        print(f"      间隙 [{og_start:.2f}s-{og_end:.2f}s] "
              f"RMS={rms:.0f} < {energy_threshold:.0f}，用静音")
        return output_path

    print(f"      间隙 [{og_start:.2f}s-{og_end:.2f}s] "
          f"RMS={rms:.0f} ≥ {energy_threshold:.0f}，保留笑声")

    # 对笑声进行响度归一化，使其与 TTS 语音音量一致
    if target_rms > 0 and rms > 0:
        gain = target_rms / rms
        # 限制增益范围，避免过度放大或缩小（0.3x ~ 3.0x）
        gain = max(0.3, min(3.0, gain))
        if abs(gain - 1.0) > 0.05:
            _normalize_audio_volume(tmp_path, gain, sample_rate)
            print(f"      音量归一化: 增益 x{gain:.2f} "
                  f"(原RMS={rms:.0f} → 目标RMS={target_rms:.0f})")

    # 有笑声，调整时长匹配需要的 gap 时长
    if abs(og_duration - needed_duration) < 0.05:
        # 时长接近，直接使用
        os.replace(tmp_path, output_path)
    elif og_duration > needed_duration:
        # 原始间隙更长，居中截取需要的时长
        offset = (og_duration - needed_duration) / 2
        trim_cmd = [
            "ffmpeg", "-y",
            "-i", tmp_path,
            "-ss", f"{offset:.4f}",
            "-t", f"{needed_duration:.4f}",
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            output_path,
        ]
        subprocess.run(trim_cmd, capture_output=True, text=True)
        os.remove(tmp_path)
    else:
        # 原始间隙更短，用原始间隙 + 前后补静音
        pad_total = needed_duration - og_duration
        pad_before = pad_total / 2
        pad_after = pad_total - pad_before

        pad_before_file = output_path + ".pad_before.wav"
        pad_after_file = output_path + ".pad_after.wav"
        generate_silence_wav(pad_before, sample_rate, pad_before_file)
        generate_silence_wav(pad_after, sample_rate, pad_after_file)

        concat_list = output_path + ".concat.txt"
        with open(concat_list, "w") as f:
            f.write(f"file '{os.path.abspath(pad_before_file)}'\n")
            f.write(f"file '{os.path.abspath(tmp_path)}'\n")
            f.write(f"file '{os.path.abspath(pad_after_file)}'\n")

        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-acodec", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-loglevel", "error",
            output_path,
        ]
        subprocess.run(concat_cmd, capture_output=True, text=True)

        # 清理临时文件
        for tmp in [tmp_path, pad_before_file, pad_after_file, concat_list]:
            if os.path.isfile(tmp):
                os.remove(tmp)

    if not os.path.isfile(output_path):
        generate_silence_wav(needed_duration, sample_rate, output_path)

    return output_path


def _find_pre_laugh_slots(
    slots: List[Dict],
    original_gaps: List[Dict],
    vocals_path: Optional[str],
    original_audio_path: Optional[str] = None,
) -> set:
    """
    找出紧接笑声/音效间隙之前的句子索引。

    这些句子需要右对齐（台词靠后放置），使台词结束后笑声/音效
    立刻接上，避免中间出现尴尬的空白。

    :param slots: 时间槽列表
    :param original_gaps: 原始间隙列表
    :param vocals_path: 人声轨道路径（None 表示无人声轨道）
    :param original_audio_path: 原始完整音频路径（None 表示无原始音频）
    :return: 需要右对齐的句子索引集合
    """
    if not original_gaps:
        return set()
    if not vocals_path and not original_audio_path:
        return set()

    pre_laugh = set()
    for og in original_gaps:
        # 找到这个间隙前面的那个句子
        # before_slot_idx 是间隙后面的句子索引
        # 所以间隙前面的句子索引是 before_slot_idx - 1
        before_idx = og.get("before_slot_idx", 0)
        if before_idx > 0:
            pre_laugh.add(before_idx - 1)
    return pre_laugh


def _normalize_audio_volume(
    wav_path: str,
    gain: float,
    sample_rate: int = 24000,
) -> None:
    """
    对 WAV 文件进行音量增益调整（原地修改）。

    通过乘以增益系数来调整音量，并进行 clipping 防止溢出。

    :param wav_path: WAV 文件路径
    :param gain: 增益系数（>1 放大，<1 缩小）
    :param sample_rate: 采样率
    """
    import numpy as np
    import wave

    try:
        with wave.open(wav_path, "rb") as wf:
            params = wf.getparams()
            raw = wf.readframes(wf.getnframes())

        audio_data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        audio_data = audio_data * gain

        # Clipping 防止溢出 int16 范围
        audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)

        with wave.open(wav_path, "wb") as wf:
            wf.setparams(params)
            wf.writeframes(audio_data.tobytes())
    except Exception:
        pass
