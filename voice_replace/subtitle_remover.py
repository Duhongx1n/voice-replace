# -*- coding: utf-8 -*-
"""
字幕去除模块 — 集成 Video-subtitle-extractor (VSE)。

支持三种模式：
1. VSE 模式（默认推荐）：调用 VSE 项目的字幕检测 + inpainting 去除
2. 智能模式：OCR 检测字幕区域 + OpenCV inpainting 修复
3. 快速模式：FFmpeg 对字幕区域进行模糊/覆盖处理

OCR 引擎优先级：
    1. RapidOCR（基于 ONNX Runtime，无需 PaddlePaddle，支持 Python 3.13+）
    2. PaddleOCR（本地安装，需要 PaddlePaddle 引擎，仅支持 Python ≤ 3.12）
    3. PaddleOCR 桥接环境（独立 Python 3.12 虚拟环境，通过子进程调用）

安装 OCR 依赖（三选一）：
    # 方式一：RapidOCR（轻量，兼容性好）
    pip install rapidocr-onnxruntime opencv-python-headless

    # 方式二：PaddleOCR（需要 Python ≤ 3.12）
    pip install paddlepaddle paddleocr opencv-python-headless

    # 方式三：PaddleOCR 桥接环境（推荐，解决 Python 版本问题）
    python3.12 -m venv .venv_paddle
    .venv_paddle/bin/pip install paddlepaddle paddleocr opencv-python-headless

使用方法：
    from voice_replace.subtitle_remover import remove_subtitles

    # VSE 模式（效果最好）
    output = remove_subtitles("input.mp4", "output_dir", mode="vse")

    # 快速模式（速度快，效果一般）
    output = remove_subtitles("input.mp4", "output_dir", mode="fast")
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# 项目根目录
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# VSE 项目在 third_party 中的路径
_VSE_DIR = os.path.join(_PROJECT_DIR, "third_party", "video-subtitle-extractor")

# PaddleOCR 独立虚拟环境路径（Python 3.12）
_PADDLE_VENV_DIR = os.path.join(_PROJECT_DIR, ".venv_paddle")
_PADDLE_PYTHON = os.path.join(_PADDLE_VENV_DIR, "bin", "python")
_PADDLE_BRIDGE_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "paddle_ocr_bridge.py",
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

    # 检查 PaddleOCR 及其底层 paddle 引擎依赖
    return _check_paddle_deps()


def _check_paddle_deps() -> bool:
    """
    检查 PaddleOCR 相关依赖是否可用。

    不仅检查 paddleocr 包是否能导入，还检查底层的 paddle
    引擎是否可用（paddleocr 是 Python 包装层，实际运行
    需要 paddlepaddle 核心引擎，该引擎不支持 Python 3.13+）。

    :return: 依赖是否满足
    """
    try:
        import cv2  # noqa: F401
        import paddle  # noqa: F401
        from paddleocr import PaddleOCR  # noqa: F401
        return True
    except (ImportError, Exception):
        return False


def _check_rapidocr_deps() -> bool:
    """
    检查 RapidOCR（ONNX Runtime 版）是否可用。

    RapidOCR 是 PaddleOCR 的 ONNX Runtime 移植版，
    不依赖 PaddlePaddle 引擎，支持 Python 3.13+。

    :return: 依赖是否满足
    """
    try:
        import cv2  # noqa: F401
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401
        return True
    except (ImportError, Exception):
        return False


def _check_paddle_bridge_deps() -> bool:
    """
    检查 PaddleOCR 桥接环境是否可用。

    桥接环境是一个独立的 Python 3.12 虚拟环境，
    安装了 PaddleOCR，通过子进程调用来避免
    Python 版本不兼容的问题。

    :return: 桥接环境是否可用
    """
    if not os.path.isfile(_PADDLE_PYTHON):
        return False
    if not os.path.isfile(_PADDLE_BRIDGE_SCRIPT):
        return False
    # 快速验证：尝试导入 paddleocr
    try:
        result = subprocess.run(
            [_PADDLE_PYTHON, "-c", "import paddleocr; print('ok')"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except (subprocess.TimeoutExpired, Exception):
        return False


def _check_any_ocr_deps() -> bool:
    """
    检查是否有任何可用的 OCR 引擎。

    优先级：PaddleOCR（桥接环境）→ PaddleOCR（本地）→ RapidOCR
    （PaddleOCR 精度更高，优先使用）

    :return: 是否有可用的 OCR 引擎
    """
    return _check_paddle_bridge_deps() or _check_paddle_deps() or _check_rapidocr_deps()


def _create_ocr_engine():
    """
    创建 OCR 引擎实例。

    优先级：PaddleOCR（桥接环境）→ PaddleOCR（本地）→ RapidOCR
    （PaddleOCR 精度更高，优先使用）

    返回一个统一接口的 OCR 包装对象，提供 detect(image) 方法，
    返回检测到的文字框列表 [(box, text, confidence), ...]。

    :return: OCR 引擎包装对象
    :raises RuntimeError: 没有可用的 OCR 引擎
    """
    if _check_paddle_bridge_deps():
        return _PaddleBridgeOCRWrapper()
    elif _check_paddle_deps():
        return _PaddleOCRWrapper()
    elif _check_rapidocr_deps():
        return _RapidOCRWrapper()
    else:
        raise RuntimeError(
            "没有可用的 OCR 引擎。请安装以下任一依赖：\n"
            "  pip install rapidocr-onnxruntime opencv-python-headless\n"
            "  pip install paddlepaddle paddleocr opencv-python-headless\n"
            "  或运行: python3.12 -m venv .venv_paddle && "
            ".venv_paddle/bin/pip install paddlepaddle paddleocr opencv-python-headless"
        )


class _RapidOCRWrapper:
    """RapidOCR 引擎包装器，提供统一的 OCR 接口。"""

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self._ocr = RapidOCR()
        self.name = "RapidOCR"

    def detect(self, image):
        """
        检测图像中的文字区域。

        :param image: OpenCV 格式的图像（numpy array）
        :return: 检测到的文字框列表 [(box, text, confidence), ...]
                 box 格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        result = self._ocr(image)
        if result is None or result[0] is None:
            return []
        # RapidOCR 返回: ([[box, text, confidence], ...], timing)
        return [(item[0], item[1], item[2]) for item in result[0]]


class _PaddleOCRWrapper:
    """PaddleOCR 引擎包装器，提供统一的 OCR 接口。"""

    def __init__(self):
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(
            use_angle_cls=False,
            lang="ch",
            show_log=False,
            det_db_thresh=0.3,
        )
        self.name = "PaddleOCR"

    def detect(self, image):
        """
        检测图像中的文字区域。

        :param image: OpenCV 格式的图像（numpy array）
        :return: 检测到的文字框列表 [(box, text, confidence), ...]
                 box 格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        results = self._ocr.ocr(image, cls=False)
        if not results or not results[0]:
            return []
        # PaddleOCR 返回: [[box, (text, confidence)], ...]
        return [(line[0], line[1][0], line[1][1]) for line in results[0]]


class _PaddleBridgeOCRWrapper:
    """
    PaddleOCR 桥接引擎包装器。

    通过子进程调用独立的 Python 3.12 虚拟环境中的 PaddleOCR，
    解决 Python 3.13+ 无法安装 PaddlePaddle 的问题。
    图像通过临时文件传递，结果通过 JSON 返回。
    """

    def __init__(self):
        self.name = "PaddleOCR（桥接环境 Python 3.12）"
        self._python = _PADDLE_PYTHON
        self._script = _PADDLE_BRIDGE_SCRIPT

    def detect(self, image):
        """
        检测图像中的文字区域（通过桥接脚本）。

        :param image: OpenCV 格式的图像（numpy array）
        :return: 检测到的文字框列表 [(box, text, confidence), ...]
                 box 格式: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        import cv2
        import tempfile

        # 将图像保存为临时文件
        with tempfile.NamedTemporaryFile(
            suffix=".png", delete=False,
        ) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image)

        try:
            result = subprocess.run(
                [self._python, self._script, "detect", tmp_path],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                return []

            data = json.loads(result.stdout)
            detections = data.get("detections", [])
            return [
                (d["box"], d["text"], d["confidence"])
                for d in detections
            ]
        except (json.JSONDecodeError, subprocess.TimeoutExpired, Exception):
            return []
        finally:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

    def detect_video_region(
        self,
        video_path: str,
        sample_count: int = 10,
        bottom_ratio: float = 0.3,
    ) -> Optional[tuple]:
        """
        通过桥接脚本采样视频帧，自动检测字幕区域。

        此方法直接调用桥接脚本的 detect_video_region 命令，
        避免逐帧传递图像的开销。

        :param video_path: 视频文件路径
        :param sample_count: 采样帧数
        :param bottom_ratio: 只检测画面底部的比例
        :return: (y_start, y_end, x_start, x_end) 或 None
        """
        try:
            result = subprocess.run(
                [
                    self._python, self._script,
                    "detect_video_region", video_path,
                    "--samples", str(sample_count),
                    "--bottom_ratio", str(bottom_ratio),
                ],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                return None

            data = json.loads(result.stdout)
            region = data.get("region")
            if region is None:
                return None

            detections = data.get("detections", [])
            frame_size = data.get("frame_size", [0, 0])
            print(f"  检测到字幕区域: y=[{region[0]}, {region[1]}], "
                  f"x=[{region[2]}, {region[3]}]")
            print(f"  画面尺寸: {frame_size[0]}x{frame_size[1]}")
            print(f"  检测到 {len(detections)} 个文字框")
            return tuple(region)
        except (json.JSONDecodeError, subprocess.TimeoutExpired, Exception) as exc:
            print(f"  [警告] 桥接 OCR 检测失败: {exc}", file=sys.stderr)
            return None


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


def _cluster_subtitle_boxes_local(
    text_boxes: list,
    frame_height: int,
    frame_width: int,
    y_tolerance: float = 0.03,
) -> list:
    """
    对检测到的文字框进行 Y 坐标聚类，筛选出真正的字幕区域。

    字幕的特征：
    1. 多帧在相近的 Y 坐标位置重复出现
    2. 通常水平居中
    3. 宽度较大（一般占画面宽度 20% 以上）

    非字幕文字（水印、logo、角标）的特征：
    1. 位置偏角落
    2. 宽度较小
    3. 可能只在少数帧出现

    :param text_boxes: 所有检测框 [(y_min, y_max, x_min, x_max), ...]
    :param frame_height: 画面高度
    :param frame_width: 画面宽度
    :param y_tolerance: Y 坐标聚类容差（占画面高度的比例）
    :return: 筛选后的字幕框列表
    """
    if not text_boxes:
        return []

    # 计算每个框的 Y 中心坐标
    y_centers = [(box[0] + box[1]) / 2.0 for box in text_boxes]
    tolerance_px = frame_height * y_tolerance

    # 简单聚类：按 Y 中心坐标排序后合并相近的框
    indexed = sorted(enumerate(y_centers), key=lambda x: x[1])
    clusters = []
    current_cluster = [indexed[0][0]]
    current_y = indexed[0][1]

    for idx, y_center in indexed[1:]:
        if abs(y_center - current_y) <= tolerance_px:
            current_cluster.append(idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]
            current_y = y_center
    clusters.append(current_cluster)

    # 筛选字幕 cluster：
    # 1. 平均宽度 >= 画面宽度的 20%
    # 2. 水平位置偏中间（中心 x 在画面中间 80% 范围内）
    subtitle_indices = []
    min_width = frame_width * 0.20
    x_margin = frame_width * 0.10

    for cluster in clusters:
        widths = [text_boxes[idx][3] - text_boxes[idx][2] for idx in cluster]
        avg_width = sum(widths) / len(widths)
        x_centers = [
            (text_boxes[idx][2] + text_boxes[idx][3]) / 2.0
            for idx in cluster
        ]
        avg_x_center = sum(x_centers) / len(x_centers)

        is_wide_enough = avg_width >= min_width
        is_centered = x_margin <= avg_x_center <= (frame_width - x_margin)

        if is_wide_enough and is_centered:
            subtitle_indices.extend(cluster)

    # 如果筛选后没有结果，返回所有框（兜底）
    if not subtitle_indices:
        return text_boxes

    return [text_boxes[i] for i in subtitle_indices]


def _detect_subtitle_region_auto(
    video_path: str,
    sample_count: int = 10,
    bottom_ratio: float = 0.3,
) -> Optional[tuple]:
    """
    自动检测视频中字幕所在区域。

    通过采样多帧，用 OCR 引擎检测文字位置，
    统计出字幕最常出现的区域（通常在画面底部）。
    优先使用 RapidOCR，fallback 到 PaddleOCR。

    :param video_path: 视频文件路径
    :param sample_count: 采样帧数
    :param bottom_ratio: 只检测画面底部的比例（默认 30%）
    :return: (y_start, y_end, x_start, x_end) 字幕区域坐标，
             检测失败返回 None
    """
    import cv2

    # 创建统一的 OCR 引擎
    ocr = _create_ocr_engine()
    print(f"  OCR 引擎: {ocr.name}")

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
        detections = ocr.detect(bottom_region)

        for box, text, confidence in detections:
            # 过滤低置信度检测（水印、噪声等）
            if confidence < 0.5:
                continue
            # 过滤空文本
            if not text or not text.strip():
                continue
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

    # 使用聚类算法筛选真正的字幕框，过滤水印/logo 等干扰
    filtered_boxes = _cluster_subtitle_boxes_local(
        text_boxes, frame_height, frame_width,
    )

    if not filtered_boxes:
        print("  [信息] 聚类筛选后无有效字幕区域", file=sys.stderr)
        return None

    print(f"  检测到 {len(text_boxes)} 个文字框，"
          f"筛选后保留 {len(filtered_boxes)} 个字幕框")

    # 统计筛选后的字幕区域边界
    all_y_min = min(box[0] for box in filtered_boxes)
    all_y_max = max(box[1] for box in filtered_boxes)
    all_x_min = min(box[2] for box in filtered_boxes)
    all_x_max = max(box[3] for box in filtered_boxes)

    # 上下各扩展 5 像素，左右各扩展 10 像素（缩小 padding，更精确）
    padding_y = 5
    padding_x = 10
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

    # 创建统一的 OCR 引擎
    ocr = _create_ocr_engine()
    print(f"  OCR 引擎: {ocr.name}")

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

        detections = ocr.detect(sub_region)
        for box, text, confidence in detections:
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
        if _check_paddle_bridge_deps():
            # 优先使用桥接环境的 PaddleOCR（精度最高）
            print("  使用 PaddleOCR 桥接环境（Python 3.12）检测字幕区域...")
            bridge = _PaddleBridgeOCRWrapper()
            subtitle_region = bridge.detect_video_region(
                video_path, sample_count=sample_count,
            )
        elif _check_paddle_deps() or _check_rapidocr_deps():
            # 本地 OCR 引擎可用，直接检测
            subtitle_region = _detect_subtitle_region_auto(
                video_path, sample_count=sample_count,
            )
        else:
            # 没有可用的 OCR 引擎，使用默认的底部 12% 区域
            print("  [信息] OCR 引擎不可用，使用默认底部区域")

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
        elif _check_any_ocr_deps():
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
        if not _check_any_ocr_deps():
            print(
                "  [错误] 智能模式需要 OCR 引擎，请安装以下任一依赖：\n"
                "    pip install rapidocr-onnxruntime  # 推荐\n"
                "    pip install paddlepaddle paddleocr  # 需要 Python ≤ 3.12",
                file=sys.stderr,
            )
            sys.exit(1)
        print("  模式: 智能模式（OCR + inpainting）")
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

    # 检查 OCR 引擎可用性
    if _check_rapidocr_deps():
        print("✅ RapidOCR 已安装并可用（推荐）")
    elif _check_paddle_deps():
        print("✅ PaddleOCR 已安装并可用（本地）")
    elif _check_paddle_bridge_deps():
        print("✅ PaddleOCR 桥接环境可用（Python 3.12 虚拟环境）")
    else:
        print("❌ 无可用的 OCR 引擎")
        print("   安装方法（三选一）:")
        print("   1. pip install rapidocr-onnxruntime")
        print("   2. pip install paddlepaddle paddleocr  # 需要 Python ≤ 3.12")
        print("   3. python3.12 -m venv .venv_paddle && "
              ".venv_paddle/bin/pip install paddlepaddle paddleocr")

    # 检查 VSE 可用性
    if _check_vse_available():
        print("✅ VSE 已安装并可用")
    else:
        print("❌ VSE 不可用")
        if not os.path.isdir(_VSE_DIR):
            print(f"   原因: VSE 目录不存在 ({_VSE_DIR})")
        elif not _check_paddle_deps():
            print("   原因: PaddleOCR 依赖未安装（VSE 需要 PaddlePaddle）")
