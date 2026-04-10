#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频语音替换工具 - 命令行入口。

完整流程：
1. 从原始视频中提取语音并转写为文字（Whisper）
2. 从原始视频中提取音色参考片段（Demucs + 智能筛选）
3. 大模型智能生成/分句 - 根据主题自动创作新台词，或将自由格式台词按原视频节奏拆分
4. 用克隆的原音色朗读新内容（Qwen3-TTS 时间轴对齐合成）
5. 用 FFmpeg 将新音频替换回原视频

使用方法：
    # 🔥 最推荐：只给一个方向/主题，大模型自动创作新台词
    python -m voice_replace --input 原始视频.mp4 --output_dir 输出目录 \\
        --topic "介绍人工智能技术的发展历程"

    # 方式二：准备新台词文本，大模型自动分句对齐
    python -m voice_replace --input 原始视频.mp4 --output_dir 输出目录 \\
        --new_text 新台词.txt

    # 方式三：手动模式（精确控制每句台词）
    # 第一步：提取原视频文字 + 音色参考（自动完成）
    python -m voice_replace --input 原始视频.mp4 --output_dir 输出目录
    # 第二步：编辑 输出目录/transcript_for_edit.md
    # 第三步：用修改后的文字生成新音频并替换
    python -m voice_replace --input 原始视频.mp4 --output_dir 输出目录 \\
        --new_text 输出目录/transcript_for_edit.md --skip_extract
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from voice_replace.subtitle_remover import remove_subtitles
from voice_replace.timeline import (
    create_editable_transcript,
    synthesize_timeline_aligned,
)
from voice_replace.transcriber import extract_transcript
from voice_replace.video_utils import get_duration, replace_audio_track
from voice_replace.voice_extractor import extract_reference_voice


def check_environment() -> list:
    """
    检查运行环境，返回缺失依赖列表。

    :return: 缺失依赖的错误信息列表
    """
    missing = []

    # 系统工具
    if not shutil.which("ffmpeg"):
        missing.append("FFmpeg 未安装。安装方法: brew install ffmpeg")
    if not shutil.which("ffprobe"):
        missing.append("ffprobe 未安装。安装方法: brew install ffmpeg")

    # Python 依赖
    try:
        import whisper  # noqa: F401
    except ImportError:
        missing.append("openai-whisper 未安装。安装方法: pip install openai-whisper")

    try:
        from qwen_tts import Qwen3TTSModel  # noqa: F401
    except ImportError:
        missing.append("qwen-tts 未安装。安装方法: pip install -U qwen-tts")

    return missing


def ensure_environment() -> None:
    """确保运行环境满足要求，否则退出。"""
    missing = check_environment()
    if missing:
        print("\n[环境检查] ❌ 以下依赖缺失：", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        print("\n请安装缺失依赖后重试。", file=sys.stderr)
        sys.exit(1)
    print("[环境检查] ✅ 所有依赖已就绪")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="视频语音替换工具 - 替换视频中的语言内容，保持原音色不变",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：

  # 🔥 最推荐：只给一个方向/主题，大模型自动创作新台词
  python -m voice_replace --input video.mp4 --output_dir output \\
      --topic "介绍人工智能技术的发展历程"

  # 方式二：准备新台词文本，大模型自动分句对齐
  python -m voice_replace --input video.mp4 --output_dir output \\
      --new_text my_script.txt

  # 指定其他大模型
  python -m voice_replace --input video.mp4 --output_dir output \\
      --topic "聊聊量子计算" --llm_model Kimi-K2.5

  # 手动模式：先提取，再编辑，最后生成
  python -m voice_replace --input video.mp4 --output_dir output
  # （编辑 output/transcript_for_edit.md）
  python -m voice_replace --input video.mp4 --output_dir output \\
      --new_text output/transcript_for_edit.md --skip_extract
        """,
    )

    parser.add_argument(
        "--input", required=True,
        help="输入视频文件路径",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="输出目录（所有中间文件和最终结果都在这里）",
    )
    parser.add_argument(
        "--topic", default="",
        help="新台词的主题/方向描述（最推荐的方式）。"
             "只需给出一个方向，大模型会根据原视频的时间轴和对话结构"
             "自动创作全新的台词。例如: '介绍人工智能技术的发展历程'。"
             "与 --new_text 互斥，优先使用 --topic。",
    )
    parser.add_argument(
        "--new_text", default="",
        help="新的文本文件路径（Markdown 格式）。"
             "不提供时只执行提取步骤（步骤 1-2），"
             "提供时执行完整替换流程（步骤 1-5）。",
    )
    parser.add_argument(
        "--output", default="",
        help="最终输出视频路径（默认: 输出目录/<原文件名>_replaced.mp4）",
    )
    parser.add_argument(
        "--language", default="zh",
        help="视频语言代码（默认: zh）",
    )
    parser.add_argument(
        "--whisper_model", default="base",
        help="Whisper 模型（默认: base，可选: tiny/small/medium/large）",
    )
    parser.add_argument(
        "--speaker", default="Vivian",
        help="Qwen3-TTS 预设音色（当无参考音频时使用，默认: Vivian）",
    )
    parser.add_argument(
        "--speed_factor", type=float, default=1.0,
        help="语速倍率（默认: 1.0，建议 1.15-1.3 加快语速）",
    )
    parser.add_argument(
        "--bgm_volume", type=float, default=0.15,
        help="背景音音量比例（0.0~1.0，默认: 0.15 即 15%%）。"
             "设为 0 等同禁用背景音。",
    )
    parser.add_argument(
        "--no_bgm", action="store_true",
        help="禁用背景音混合（仅保留新语音，不混入原视频的背景音乐）",
    )
    parser.add_argument(
        "--skip_extract", action="store_true",
        help="跳过步骤 1-2（已有提取结果时使用）",
    )

    # 字幕去除参数
    parser.add_argument(
        "--remove_subtitle", action="store_true",
        help="启用字幕去除预处理（去除视频中的硬字幕）",
    )
    parser.add_argument(
        "--subtitle_mode", default="auto",
        choices=["auto", "vse", "smart", "fast"],
        help="字幕去除模式（默认: auto）。"
             "auto: 自动选择（优先 VSE → 智能模式 → 快速模式）；"
             "vse: Video-subtitle-extractor 引擎（效果最好）；"
             "smart: PaddleOCR + inpainting（效果好但慢）；"
             "fast: FFmpeg 模糊（速度快但效果一般）",
    )
    parser.add_argument(
        "--subtitle_region", default="",
        help="手动指定字幕区域，格式: y_start,y_end,x_start,x_end。"
             "例如: 680,720,100,1180。不指定则自动检测。",
    )

    # 大模型智能分句参数
    parser.add_argument(
        "--llm_api_key", default="",
        help="大模型 API Key（已内置 HAI 平台默认 Key，通常无需设置）。"
             "也可通过环境变量 OPENAI_API_KEY 覆盖。",
    )
    parser.add_argument(
        "--llm_base_url", default="",
        help="大模型 API Base URL（已内置 HAI 平台地址，通常无需设置）。"
             "也可通过环境变量 OPENAI_BASE_URL 覆盖。"
             "支持 OpenAI、DeepSeek、通义千问等兼容 API。",
    )
    parser.add_argument(
        "--llm_model", default="",
        help="大模型名称（默认: DeepSeek-V3.1）。"
             "HAI 平台可用: DeepSeek-V3.1, DeepSeek-R1, "
             "Kimi-K2.5, Qwen3-235B-A22B, Qwen3-32B-FP8 等。",
    )
    parser.add_argument(
        "--force_adapt", action="store_true",
        help="强制使用大模型适配台词（即使行数已匹配）",
    )

    return parser.parse_args()


def main() -> int:
    """主入口。"""
    args = parse_args()

    input_video = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output_dir)

    # 验证输入文件
    if not os.path.isfile(input_video):
        print(f"[错误] 输入视频不存在: {input_video}", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)

    # 环境检查
    ensure_environment()

    print("\n" + "=" * 60)
    print("🎬 视频语音替换工具")
    print("=" * 60)
    print(f"  输入视频: {input_video}")
    print(f"  输出目录: {output_dir}")
    if args.remove_subtitle:
        print("  字幕去除: ✅ 已启用")
    if args.topic:
        print(f"  创作主题: {args.topic}")
        print("  模式: 🔥 主题创作（大模型自动生成新台词，步骤 0-5）")
    elif args.new_text:
        print(f"  新文本: {args.new_text}")
        print("  模式: 完整替换流程（步骤 0-5）")
    else:
        print("  模式: 仅提取（步骤 0-2）")

    # ---- 步骤 0（可选）：去除字幕 ----
    effective_video = input_video
    if args.remove_subtitle:
        subtitle_region = None
        if args.subtitle_region:
            parts = args.subtitle_region.split(",")
            if len(parts) == 4:
                subtitle_region = tuple(int(p.strip()) for p in parts)
            else:
                print(
                    "[错误] --subtitle_region 格式错误，"
                    "应为: y_start,y_end,x_start,x_end",
                    file=sys.stderr,
                )
                return 1

        effective_video = remove_subtitles(
            input_video, output_dir,
            mode=args.subtitle_mode,
            subtitle_region=subtitle_region,
        )

    # ---- 步骤 1：提取语音文字 ----
    transcript_path = os.path.join(output_dir, "step1_transcribe", "transcript.txt")
    if args.skip_extract and os.path.isfile(transcript_path):
        print(f"\n[跳过] 步骤 1：已有转写结果 {transcript_path}")
    else:
        transcript_path = extract_transcript(
            effective_video, output_dir,
            language=args.language,
            model=args.whisper_model,
        )

    # ---- 步骤 2：提取音色参考 ----
    ref_voice_path = os.path.join(output_dir, "step2_voice", "reference_voice.wav")
    if args.skip_extract and os.path.isfile(ref_voice_path):
        print(f"\n[跳过] 步骤 2：已有参考音频 {ref_voice_path}")
    else:
        ref_voice_path = extract_reference_voice(effective_video, output_dir)

    # 仅在未提供新文本且未指定主题时，生成可编辑的文本文件
    if not args.new_text and not args.topic:
        edit_path = create_editable_transcript(transcript_path, output_dir)

    # 如果没有提供新文本且没有指定主题，到此为止
    if not args.new_text and not args.topic:
        print("\n" + "=" * 60)
        print("✅ 提取完成！")
        print("=" * 60)
        print(f"\n  原始转写文本: {transcript_path}")
        print(f"  可编辑文本:   {edit_path}")
        print(f"  参考音频:     {ref_voice_path}")
        print(f"\n📝 下一步（三选一）：")
        print(f"\n  � 最推荐：只给一个方向，大模型自动创作新台词")
        print(f"     python -m voice_replace \\")
        print(f"       --input {input_video} \\")
        print(f"       --output_dir {output_dir} \\")
        print(f"       --topic \"你想要的台词方向\" \\")
        print(f"       --skip_extract")
        print(f"\n  🚀 方式二：准备一段新台词文本，大模型自动分句对齐")
        print(f"     python -m voice_replace \\")
        print(f"       --input {input_video} \\")
        print(f"       --output_dir {output_dir} \\")
        print(f"       --new_text 新台词.txt \\")
        print(f"       --skip_extract")
        print(f"\n  ✏️  方式三：逐句编辑 {edit_path}")
        print(f"     python -m voice_replace \\")
        print(f"       --input {input_video} \\")
        print(f"       --output_dir {output_dir} \\")
        print(f"       --new_text {edit_path} \\")
        print(f"       --skip_extract")
        return 0

    # ---- 主题创作模式 vs 新文本模式 ----
    new_text_path = ""
    if args.topic:
        # 主题创作模式：大模型根据主题自动生成新台词
        from voice_replace.text_adapter import generate_text_from_topic
        from voice_replace.dialogue_analysis import segments_as_slots
        from voice_replace.timeline import load_segment_timestamps

        segments = load_segment_timestamps(output_dir)
        if not segments:
            print("[错误] 未找到 Whisper 转写的时间戳数据，请先运行提取步骤", file=sys.stderr)
            return 1

        print("\n" + "=" * 60)
        print("🧠 步骤 3/5：大模型根据主题创作新台词")
        print("=" * 60)
        print(f"  主题: {args.topic}")
        print(f"  原视频片段数: {len(segments)}")

        slots = segments_as_slots(segments)
        generated_lines = generate_text_from_topic(
            topic=args.topic,
            segments=slots,
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            model=args.llm_model,
        )

        if generated_lines is None:
            print("[错误] 大模型创作失败，请检查网络或 API Key", file=sys.stderr)
            return 1

        # 将生成的台词保存为临时文件
        generated_text_path = os.path.join(output_dir, "generated_script.txt")
        with open(generated_text_path, "w", encoding="utf-8") as f:
            f.write(f"# 大模型根据主题自动创作的台词\n")
            f.write(f"# 主题: {args.topic}\n\n")
            for line in generated_lines:
                f.write(f"{line}\n")

        print(f"\n  📝 生成的台词已保存: {generated_text_path}")
        print(f"  台词预览:")
        for i, line in enumerate(generated_lines):
            print(f"    第{i + 1}句: {line}")

        new_text_path = generated_text_path
    else:
        # 新文本模式：验证文件存在
        new_text_path = os.path.abspath(args.new_text)
        if not os.path.isfile(new_text_path):
            print(f"[错误] 新文本文件不存在: {new_text_path}", file=sys.stderr)
            return 1

    # 获取原视频时长（使用去字幕后的视频）
    video_duration = get_duration(effective_video)

    # ---- 步骤 3+4：时间轴对齐合成 ----
    print("\n" + "=" * 60)
    print("📄🔊 步骤 3-4/5：时间轴对齐合成")
    print("=" * 60)

    new_audio_path = synthesize_timeline_aligned(
        new_text_path, ref_voice_path, output_dir, video_duration,
        llm_api_key=args.llm_api_key,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        force_adapt=args.force_adapt,
        bgm_volume=args.bgm_volume,
        no_bgm=args.no_bgm,
    )

    # ---- 步骤 5：替换视频音轨 ----
    print("\n" + "=" * 60)
    print("🎬 步骤 5/5：替换视频音轨")
    print("=" * 60)

    if args.output:
        final_output = os.path.abspath(args.output)
    else:
        input_stem = Path(input_video).stem
        input_ext = Path(input_video).suffix
        final_output = os.path.join(output_dir, f"{input_stem}_replaced{input_ext}")

    replace_audio_track(effective_video, new_audio_path, final_output)

    # ---- 完成 ----
    print("\n" + "=" * 60)
    print("🎉 视频语音替换完成！")
    print("=" * 60)
    print(f"\n  最终视频: {final_output}")
    print(f"\n  中间文件：")
    print(f"    转写文本:   {transcript_path}")
    print(f"    参考音频:   {ref_voice_path}")
    print(f"    新语音:     {new_audio_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
