# -*- coding: utf-8 -*-
"""
字幕去除模块 — 集成 Video-subtitle-extractor (VSE)。

支持三种模式：
1. VSE 模式（默认推荐）：调用 VSE 项目的字幕检测 + inpainting 去除
2. 智能模式：PaddleOCR 检测字幕区域 + OpenCV inpainting 修复
3. 快速模式：FFmpeg 对字幕区域进行模糊/覆盖处理

VSE 安装步骤：
    1. git clone https://github.com/YaoFANGUK/video-subtitle-extractor.git \\
           third_party/video-subtitle-extractor
    2. pip install paddlepaddle paddleocr opencv-python-headless

使用方法：
    from voice_replace.subtitle_remover import remove_subtitles

    # VSE 模式（效果最好）
    output = remove_subtitles("input.mp4", "output_dir", mode="vse")

    # 快速模式（速度快，效果一般）
    output = remove_subtitles("input.mp4", "output_dir", mode="fast")
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# VSE 项目在 third_party 中的路径
_VSE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party",
    "video-subtitle-extractor",
)


def _check_vse_available() -> bool:
    """
    检查 VSE（Video-subtitle-extractor）是否已安装并可用。

    :return: VSE 是否可用
    """
    backend_dir = os.path.join(_VSE_DIR, "backend")
    if not os.path.isdir(backend_dir):
        return False

    # 检查核心文件是否存在
    main_file = os.path.join(backend_dir, "main.py")
    if not os.path.isfile(main_file):
        return False

    # 检查 PaddleOCR 依赖
    try:
        import cv2  # noqa: F401
        from paddleocr import PaddleOCR  # noqa: F401
        return True
    except ImportError:
        return False


def _check_paddle_deps() -> bool:
    """
    检查 PaddleOCR 相关依赖是否可用。

    :return: 依赖是否满足
    """
    try:
        import cv2  # noqa: F401
        from paddleocr import PaddleOCR  # noqa: F401
        return True
    except ImportError:
        return False


def _remove_subtitles_vse(
    video_path: str,
    output_path: str,
) -> str:
    """
    VSE 模式：调用 Video-subtitle-extractor 的核心引擎去除字幕。

    VSE 的处理流程：
    1. 帧差法检测字幕变化区域（过滤固定 logo/水印）
    2. PaddleOCR 精确检测字幕文字位置
    3. 字幕区域聚类，确定字幕行位置
    4. 生成逐帧 mask
    5. 使用 inpainting 算法修复字幕区域

    :param video_path: 输入视频路径
    :param output_path: 输出视频路径
    :return: 输出视频路径
    """
    backend_dir = os.path.join(_VSE_DIR, "backend")

    # 将 VSE 的 backend 目录加入 sys.path
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    # 同时将 VSE 根目录加入（部分 import 需要）
    if _VSE_DIR not in sys.path:
        sys.path.insert(0, _VSE_DIR)

    try:
        # 导入 VSE 核心模块
        from backend.main import SubtitleRemover
    except ImportError as exc:
        print(
            f"  [错误] 无法导入 VSE 核心模块: {exc}\n"
            f"  请确认 VSE 已正确安装在: {_VSE_DIR}",
            file=sys.stderr,
        )
        raise

    print("  正在初始化 VSE 字幕去除引擎...")

    # 创建 VSE 的 SubtitleRemover 实例
    # VSE 的 SubtitleRemover 接受视频路径作为参数
    remover = SubtitleRemover(video_path)

    print("  正在执行字幕检测与去除（VSE 引擎）...")
    print("  （此过程可能需要较长时间，取决于视频长度和分辨率）")

    # 执行字幕去除
    # VSE 的 run() 方法会处理完整流程：
    # 检测字幕区域 → 生成 mask → inpainting 修复 → 输出视频
    remover.run()

    # VSE 默认输出到同目录下，文件名带 _no_sub 后缀
    # 需要找到 VSE 的输出文件并移动到我们的目标路径
    vse_output = _find_vse_output(video_path, remover)

    if vse_output and os.path.isfile(vse_output):
        # 如果 VSE 输出路径和我们的目标路径不同，移动文件
        if os.path.abspath(vse_output) != os.path.abspath(output_path):
            import shutil
            shutil.move(vse_output, output_path)
        print(f"  VSE 处理完成: {output_path}")
    else:
        print(
            "  [警告] VSE 输出文件未找到，尝试查找默认输出位置...",
            file=sys.stderr,
        )
        # 尝试查找 VSE 可能的输出位置
        vse_output = _find_vse_output_fallback(video_path)
        if vse_output and os.path.isfile(vse_output):
            import shutil
            shutil.move(vse_output, output_path)
            print(f"  VSE 处理完成（从备选路径）: {output_path}")
        else:
            print(
                "  [错误] VSE 处理完成但未找到输出文件",
                file=sys.stderr,
            )
            sys.exit(1)

    return output_path


def _find_vse_output(video_path: str, remover: object) -> Optional[str]:
    """
    查找 VSE 的输出文件路径。

    VSE 的输出路径取决于其内部逻辑，通常是：
    - 与输入视频同目录，文件名加 _no_sub 后缀
    - 或者在 remover 对象的属性中

    :param video_path: 输入视频路径
    :param remover: VSE 的 SubtitleRemover 实例
    :return: 输出文件路径，未找到返回 None
    """
    # 尝试从 remover 对象获取输出路径
    if hasattr(remover, "output_path"):
        return remover.output_path

    if hasattr(remover, "video_out_name"):
        return remover.video_out_name

    # 尝试默认命名规则
    return _find_vse_output_fallback(video_path)


def _find_vse_output_fallback(video_path: str) -> Optional[str]:
    """
    根据 VSE 的默认命名规则查找输出文件。

    :param video_path: 输入视频路径
    :return: 输出文件路径，未找到返回 None
    """
    video_dir = os.path.dirname(video_path)
    video_stem = Path(video_path).stem
    video_ext = Path(video_path).suffix

    # VSE 常见的输出命名模式
    candidates = [
        os.path.join(video_dir, f"{video_stem}_no_sub{video_ext}"),
        os.path.join(video_dir, f"{video_stem}_nosub{video_ext}"),
        os.path.join(video_dir, f"{video_stem}_removed{video_ext}"),
        os.path.join(video_dir, f"{video_stem}_clean{video_ext}"),
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return None


def _detect_subtitle_region_auto(
    video_path: str,
    sample_count: int = 10,
    bottom_ratio: float = 0.3,
) -> Optional[tuple]:
    """
    自动检测视频中字幕所在区域。

    通过采样多帧，用 PaddleOCR 检测文字位置，
    统计出字幕最常出现的区域（通常在画面底部）。

    :param video_path: 视频文件路径
    :param sample_count: 采样帧数
    :param bottom_ratio: 只检测画面底部的比例（默认 30%）
    :return: (y_start, y_end, x_start, x_end) 字幕区域坐标，
             检测失败返回 None
    """
    import cv2
    import numpy as np
    from paddleocr import PaddleOCR

    # 初始化 PaddleOCR（仅检测，不识别，加快速度）
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang="ch",
        show_log=False,
        det_db_thresh=0.3,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [警告] 无法打开视频: {video_path}", file=sys.stderr)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if total_frames <= 0:
        cap.release()
        return None

    # 只关注画面底部区域
    y_threshold = int(frame_height * (1 - bottom_ratio))

    # 均匀采样帧
    step = max(1, total_frames // sample_count)
    text_boxes = []

    print(f"  正在采样 {sample_count} 帧检测字幕区域...")
    for i in range(0, min(total_frames, sample_count * step), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        # 只截取底部区域进行检测
        bottom_region = frame[y_threshold:, :]
        results = ocr.ocr(bottom_region, cls=False)

        if results and results[0]:
            for line in results[0]:
                box = line[0]
                # 将坐标转换回完整帧的坐标系
                y_min = int(min(p[1] for p in box)) + y_threshold
                y_max = int(max(p[1] for p in box)) + y_threshold
                x_min = int(min(p[0] for p in box))
                x_max = int(max(p[0] for p in box))
                text_boxes.append((y_min, y_max, x_min, x_max))

    cap.release()

    if not text_boxes:
        print("  [信息] 未检测到字幕区域", file=sys.stderr)
        return None

    # 统计字幕区域的边界（取所有检测框的并集，适当扩展）
    all_y_min = min(box[0] for box in text_boxes)
    all_y_max = max(box[1] for box in text_boxes)
    all_x_min = min(box[2] for box in text_boxes)
    all_x_max = max(box[3] for box in text_boxes)

    # 上下各扩展 10 像素，左右各扩展 20 像素
    padding_y = 10
    padding_x = 20
    y_start = max(0, all_y_min - padding_y)
    y_end = min(frame_height, all_y_max + padding_y)
    x_start = max(0, all_x_min - padding_x)
    x_end = min(frame_width, all_x_max + padding_x)

    print(f"  检测到字幕区域: y=[{y_start}, {y_end}], x=[{x_start}, {x_end}]")
    print(f"  画面尺寸: {frame_width}x{frame_height}")

    return (y_start, y_end, x_start, x_end)


def _remove_subtitles_inpaint(
    video_path: str,
    output_path: str,
    subtitle_region: Optional[tuple] = None,
    sample_count: int = 15,
    inpaint_radius: int = 5,
) -> str:
    """
    智能模式：逐帧检测字幕并用 inpainting 修复。

    对每一帧：
    1. 用 PaddleOCR 检测底部文字区域
    2. 生成 mask
    3. 用 OpenCV inpainting（Navier-Stokes 或 Telea 算法）修复

    :param video_path: 输入视频路径
    :param output_path: 输出视频路径
    :param subtitle_region: 预设字幕区域 (y_start, y_end, x_start, x_end)，
                            None 则自动检测
    :param sample_count: 自动检测时的采样帧数
    :param inpaint_radius: inpainting 修复半径
    :return: 输出视频路径
    """
    import cv2
    import numpy as np
    from paddleocr import PaddleOCR

    # 自动检测字幕区域
    if subtitle_region is None:
        subtitle_region = _detect_subtitle_region_auto(
            video_path, sample_count=sample_count,
        )

    if subtitle_region is None:
        print("  [信息] 未检测到字幕，直接复制原视频")
        _copy_video(video_path, output_path)
        return output_path

    y_start, y_end, x_start, x_end = subtitle_region

    # 初始化 PaddleOCR
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang="ch",
        show_log=False,
        det_db_thresh=0.3,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 临时输出（无音频的视频）
    temp_video = output_path + ".temp_nosound.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (frame_width, frame_height))

    print(f"  正在逐帧去除字幕（共 {total_frames} 帧）...")
    frame_idx = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 只对字幕区域进行 OCR 检测和 inpainting
        sub_region = frame[y_start:y_end, x_start:x_end]
        mask = np.zeros(
            (y_end - y_start, x_end - x_start), dtype=np.uint8,
        )

        results = ocr.ocr(sub_region, cls=False)
        if results and results[0]:
            for line in results[0]:
                box = line[0]
                pts = np.array(box, dtype=np.int32)
                # 扩展检测框
                cv2.fillConvexPoly(mask, pts, 255)
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=2)
                processed_count += 1

        # 如果检测到文字，进行 inpainting
        if mask.any():
            inpainted = cv2.inpaint(
                sub_region, mask, inpaint_radius, cv2.INPAINT_NS,
            )
            frame[y_start:y_end, x_start:x_end] = inpainted

        writer.write(frame)

        # 进度显示
        if frame_idx % 100 == 0:
            progress = frame_idx / total_frames * 100
            print(f"    进度: {progress:.1f}% ({frame_idx}/{total_frames})")

    cap.release()
    writer.release()

    print(f"  共处理 {processed_count} 个字幕文字框")

    # 将原视频的音频合并到处理后的视频
    _merge_audio(video_path, temp_video, output_path)

    # 清理临时文件
    if os.path.isfile(temp_video):
        os.remove(temp_video)

    return output_path


def _remove_subtitles_fast(
    video_path: str,
    output_path: str,
    subtitle_region: Optional[tuple] = None,
    blur_strength: int = 15,
    sample_count: int = 15,
) -> str:
    """
    快速模式：用 FFmpeg 对字幕区域进行模糊处理。

    速度快，但效果不如 inpainting。

    :param video_path: 输入视频路径
    :param output_path: 输出视频路径
    :param subtitle_region: 预设字幕区域 (y_start, y_end, x_start, x_end)
    :param blur_strength: 模糊强度（boxblur radius，建议 5-20）
    :param sample_count: 自动检测时的采样帧数
    :return: 输出视频路径
    """
    # 如果没有预设区域，尝试自动检测
    if subtitle_region is None:
        if _check_paddle_deps():
            subtitle_region = _detect_subtitle_region_auto(
                video_path, sample_count=sample_count,
            )
        else:
            # 没有 PaddleOCR，使用默认的底部 12% 区域
            print("  [信息] PaddleOCR 不可用，使用默认底部区域")

    if subtitle_region is not None:
        y_start, y_end, x_start, x_end = subtitle_region
        height = y_end - y_start
        width = x_end - x_start

        # boxblur radius 不能超过 min(w,h)/2，需要动态限制
        max_radius = min(width, height) // 2
        safe_radius = min(blur_strength, max_radius)
        # 用 power 参数（迭代次数）增强模糊效果
        blur_param = f"{safe_radius}:{safe_radius}:3"

        vf = (
            f"split[a][b];"
            f"[b]crop={width}:{height}:{x_start}:{y_start},"
            f"boxblur={blur_param}[blur];"
            f"[a][blur]overlay={x_start}:{y_start}"
        )
    else:
        # 默认模糊底部 12% 区域
        safe_radius = min(blur_strength, 20)
        blur_param = f"{safe_radius}:{safe_radius}:5"

        vf = (
            "split[a][b];"
            "[b]crop=iw:ih*0.12:0:ih*0.88,"
            f"boxblur={blur_param}[blur];"
            "[a][blur]overlay=0:H*0.88"
        )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:a", "copy",
        output_path,
    ]

    print("  正在用 FFmpeg 模糊字幕区域...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"  [错误] FFmpeg 处理失败:\n{result.stderr}",
            file=sys.stderr,
        )
        sys.exit(1)

    return output_path


def _copy_video(src: str, dst: str) -> None:
    """
    无损复制视频文件。

    :param src: 源文件路径
    :param dst: 目标文件路径
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", src,
        "-c", "copy",
        dst,
    ]
    subprocess.run(cmd, capture_output=True, text=True)


def _merge_audio(
    original_video: str,
    processed_video: str,
    output_path: str,
) -> None:
    """
    将原视频的音频合并到处理后的视频中。

    :param original_video: 原视频（提供音频）
    :param processed_video: 处理后的视频（提供画面）
    :param output_path: 输出路径
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", processed_video,
        "-i", original_video,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # 如果合并失败（可能原视频没有音频），直接复制视频
        print(
            "  [警告] 音频合并失败，输出无音频视频",
            file=sys.stderr,
        )
        _copy_video(processed_video, output_path)


def remove_subtitles(
    video_path: str,
    output_dir: str,
    mode: str = "auto",
    subtitle_region: Optional[tuple] = None,
    sample_count: int = 15,
    inpaint_radius: int = 5,
    blur_strength: int = 15,
) -> str:
    """
    去除视频中的硬字幕。

    :param video_path: 输入视频路径
    :param output_dir: 输出目录
    :param mode: 去除模式
        - "auto": 自动选择（优先 VSE → 智能模式 → 快速模式）
        - "vse": VSE 模式（Video-subtitle-extractor，效果最好）
        - "smart": 智能模式（PaddleOCR + inpainting，效果好但慢）
        - "fast": 快速模式（FFmpeg 模糊，速度快但效果一般）
    :param subtitle_region: 预设字幕区域 (y_start, y_end, x_start, x_end)，
                            None 则自动检测
    :param sample_count: 自动检测字幕区域时的采样帧数
    :param inpaint_radius: inpainting 修复半径（智能模式）
    :param blur_strength: 模糊强度（快速模式）
    :return: 去除字幕后的视频路径
    """
    print("\n" + "=" * 60)
    print("🔤 字幕去除（预处理）")
    print("=" * 60)

    # 确定输出路径
    step_dir = os.path.join(output_dir, "step0_subtitle_removal")
    os.makedirs(step_dir, exist_ok=True)

    input_stem = Path(video_path).stem
    output_path = os.path.join(step_dir, f"{input_stem}_no_sub.mp4")

    # 如果已经处理过，直接返回
    if os.path.isfile(output_path):
        print(f"  已有去字幕结果: {output_path}")
        return output_path

    print(f"  输入视频: {video_path}")
    print(f"  输出路径: {output_path}")

    # 先尝试去除软字幕（无损操作）
    _remove_soft_subtitles(video_path, output_path)

    # 选择模式
    if mode == "auto":
        if _check_vse_available():
            mode = "vse"
        elif _check_paddle_deps():
            mode = "smart"
        else:
            mode = "fast"

    # 执行对应模式
    if mode == "vse":
        if not _check_vse_available():
            print(
                "  [错误] VSE 模式不可用，请确认：\n"
                f"    1. VSE 已 clone 到: {_VSE_DIR}\n"
                "    2. 已安装依赖: pip install paddlepaddle paddleocr "
                "opencv-python-headless\n"
                "  或使用 --subtitle_mode fast 降级为快速模式",
                file=sys.stderr,
            )
            sys.exit(1)
        print("  模式: VSE（Video-subtitle-extractor）")
        print("  引擎: 帧差法字幕检测 + PaddleOCR + inpainting")
        result = _remove_subtitles_vse(video_path, output_path)
    elif mode == "smart":
        if not _check_paddle_deps():
            print(
                "  [错误] 智能模式需要 PaddleOCR，请安装：\n"
                "    pip install paddlepaddle paddleocr",
                file=sys.stderr,
            )
            sys.exit(1)
        print("  模式: 智能模式（PaddleOCR + inpainting）")
        result = _remove_subtitles_inpaint(
            video_path, output_path,
            subtitle_region=subtitle_region,
            sample_count=sample_count,
            inpaint_radius=inpaint_radius,
        )
    else:
        print("  模式: 快速模式（FFmpeg 模糊）")
        if not _check_vse_available():
            print(
                "  提示: clone VSE 并安装依赖可启用更好的去字幕效果\n"
                f"    git clone https://github.com/YaoFANGUK/"
                f"video-subtitle-extractor.git \\\n"
                f"        {_VSE_DIR}",
            )
        result = _remove_subtitles_fast(
            video_path, output_path,
            subtitle_region=subtitle_region,
            blur_strength=blur_strength,
            sample_count=sample_count,
        )

    print(f"  ✅ 字幕去除完成: {result}")
    return result


def _remove_soft_subtitles(video_path: str, output_path: str) -> None:
    """
    去除视频中的软字幕轨道（如果有的话）。

    软字幕是独立的字幕轨道（SRT/ASS 等），可以无损去除。
    此函数仅作为预处理，硬字幕仍需后续步骤处理。

    :param video_path: 输入视频路径
    :param output_path: 输出视频路径（此处仅检测，不实际输出）
    """
    # 检测是否有字幕轨
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "s",
        "-show_entries", "stream=index,codec_name",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(f"  检测到软字幕轨: {result.stdout.strip()}")
        print("  将在处理过程中自动去除软字幕轨")
    else:
        print("  未检测到软字幕轨")


def remove_soft_subtitles_only(video_path: str, output_path: str) -> str:
    """
    仅去除软字幕轨道（无损操作，速度极快）。

    适用于字幕是外挂字幕轨（SRT/ASS）的情况。

    :param video_path: 输入视频路径
    :param output_path: 输出视频路径
    :return: 输出视频路径
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-c", "copy",
        "-sn",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"  [错误] 去除软字幕失败:\n{result.stderr}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  ✅ 软字幕去除完成: {output_path}")
    return output_path


def print_setup_guide() -> None:
    """打印 VSE 安装指南。"""
    print("""
╔══════════════════════════════════════════════════════════╗
║        Video-subtitle-extractor (VSE) 安装指南          ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  步骤 1: Clone VSE 仓库                                 ║
║  ─────────────────────────────────────────────────────── ║
║  git clone https://github.com/YaoFANGUK/                ║
║      video-subtitle-extractor.git \\                     ║
║      third_party/video-subtitle-extractor                ║
║                                                          ║
║  步骤 2: 安装 Python 依赖                               ║
║  ─────────────────────────────────────────────────────── ║
║  pip install paddlepaddle paddleocr \\                   ║
║      opencv-python-headless                              ║
║                                                          ║
║  步骤 3: 使用                                            ║
║  ─────────────────────────────────────────────────────── ║
║  python -m voice_replace --input video.mp4 \\            ║
║      --output_dir output --remove_subtitle               ║
║                                                          ║
║  模式说明：                                              ║
║  • vse   — VSE 引擎（效果最好，需要步骤 1-2）           ║
║  • smart — PaddleOCR + inpainting（需要步骤 2）         ║
║  • fast  — FFmpeg 模糊（无需额外依赖）                  ║
║  • auto  — 自动选择最佳可用模式                         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_setup_guide()

    # 检查 VSE 可用性
    if _check_vse_available():
        print("✅ VSE 已安装并可用")
    else:
        print("❌ VSE 不可用")
        if not os.path.isdir(_VSE_DIR):
            print(f"   原因: VSE 目录不存在 ({_VSE_DIR})")
        elif not _check_paddle_deps():
            print("   原因: PaddleOCR 依赖未安装")
