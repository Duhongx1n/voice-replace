#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR 桥接脚本 — 运行在独立的 Python 3.12 虚拟环境中。

此脚本通过命令行参数接收任务，将 OCR 检测结果以 JSON 格式
输出到 stdout，供主项目（Python 3.14）通过子进程调用。

适配 PaddleOCR 3.4.0+ 的新 predict() API。

支持的命令：
    detect  — 对单张图片进行 OCR 检测
    detect_video_region — 采样视频帧，自动检测字幕区域

使用方法（由 subtitle_remover.py 自动调用）：
    .venv_paddle/bin/python paddle_ocr_bridge.py detect <image_path>
    .venv_paddle/bin/python paddle_ocr_bridge.py detect_video_region <video_path> [options]
"""

from __future__ import annotations

import json
import os
import sys


def _init_ocr():
    """
    初始化 PaddleOCR 引擎。

    使用 PaddleOCR 3.4.0+ 的新 API 参数。

    :return: PaddleOCR 实例
    """
    # 禁用模型源检查，加速启动
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    from paddleocr import PaddleOCR
    return PaddleOCR(
        use_textline_orientation=False,
        lang="ch",
        text_det_thresh=0.3,
    )


def _parse_predict_result(result) -> list:
    """
    解析 PaddleOCR 3.4.0+ predict() 方法的返回结果。

    新版 API 返回 OCRResult 对象，其 .json 属性为：
    {"res": {"dt_polys": [...], "rec_texts": [...], "rec_scores": [...]}}

    :param result: OCRResult 对象
    :return: 检测结果列表 [{"box": [[x,y],...], "text": str, "confidence": float}, ...]
    """
    try:
        data = result.json
        res = data.get("res", {})
        dt_polys = res.get("dt_polys", [])
        rec_texts = res.get("rec_texts", [])
        rec_scores = res.get("rec_scores", [])

        detections = []
        for i, poly in enumerate(dt_polys):
            text = rec_texts[i] if i < len(rec_texts) else ""
            confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.0
            box = [[int(p[0]), int(p[1])] for p in poly]
            detections.append({
                "box": box,
                "text": text,
                "confidence": confidence,
            })
        return detections
    except (AttributeError, KeyError, IndexError, TypeError):
        return []


def detect_image(image_path: str) -> list:
    """
    对单张图片进行 OCR 检测。

    :param image_path: 图片文件路径
    :return: 检测结果列表 [{"box": [[x,y],...], "text": str, "confidence": float}, ...]
    """
    import cv2

    ocr = _init_ocr()
    image = cv2.imread(image_path)
    if image is None:
        return []

    detections = []
    for result in ocr.predict(image):
        detections.extend(_parse_predict_result(result))
    return detections


def _cluster_subtitle_boxes(
    text_boxes: list,
    all_detections: list,
    frame_height: int,
    frame_width: int,
    y_tolerance: float = 0.03,
    min_occurrence_ratio: float = 0.15,
) -> tuple:
    """
    对检测到的文字框进行 Y 坐标聚类，筛选出真正的字幕区域。

    字幕的特征：
    1. 多帧在相近的 Y 坐标位置重复出现
    2. 通常水平居中
    3. 宽度较大（一般占画面宽度 30% 以上）

    非字幕文字（水印、logo、角标）的特征：
    1. 位置偏角落
    2. 宽度较小
    3. 可能只在少数帧出现

    :param text_boxes: 所有检测框 [(y_min, y_max, x_min, x_max), ...]
    :param all_detections: 所有检测详情
    :param frame_height: 画面高度
    :param frame_width: 画面宽度
    :param y_tolerance: Y 坐标聚类容差（占画面高度的比例）
    :param min_occurrence_ratio: 最小出现比例（占总帧数的比例）
    :return: (filtered_boxes, filtered_detections)
    """
    if not text_boxes:
        return ([], [])

    # 计算每个框的 Y 中心坐标
    y_centers = [(box[0] + box[1]) / 2.0 for box in text_boxes]
    tolerance_px = frame_height * y_tolerance

    # 简单聚类：按 Y 中心坐标排序后合并相近的框
    indexed = sorted(enumerate(y_centers), key=lambda x: x[1])
    clusters = []  # 每个 cluster 是一组 index
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

    # 统计每个 cluster 涉及的不同帧数
    total_frames_set = set()
    for det in all_detections:
        total_frames_set.add(det["frame"])
    total_frame_count = max(len(total_frames_set), 1)

    # 筛选字幕 cluster：
    # 1. 出现帧数占比 >= min_occurrence_ratio
    # 2. 平均宽度 >= 画面宽度的 20%
    # 3. 水平位置偏中间（中心 x 在画面中间 80% 范围内）
    subtitle_indices = []
    min_width = frame_width * 0.20
    x_margin = frame_width * 0.10

    for cluster in clusters:
        # 统计该 cluster 涉及的帧数
        cluster_frames = set()
        for idx in cluster:
            cluster_frames.add(all_detections[idx]["frame"])
        frame_ratio = len(cluster_frames) / total_frame_count

        # 计算平均宽度和水平中心
        widths = [text_boxes[idx][3] - text_boxes[idx][2] for idx in cluster]
        avg_width = sum(widths) / len(widths)
        x_centers = [
            (text_boxes[idx][2] + text_boxes[idx][3]) / 2.0
            for idx in cluster
        ]
        avg_x_center = sum(x_centers) / len(x_centers)

        # 字幕判定条件
        is_wide_enough = avg_width >= min_width
        is_centered = x_margin <= avg_x_center <= (frame_width - x_margin)
        is_frequent = frame_ratio >= min_occurrence_ratio

        if is_wide_enough and is_centered and is_frequent:
            subtitle_indices.extend(cluster)

    # 如果严格筛选后没有结果，放宽条件：只要求居中 + 宽度足够
    if not subtitle_indices:
        for cluster in clusters:
            widths = [
                text_boxes[idx][3] - text_boxes[idx][2]
                for idx in cluster
            ]
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

    # 如果仍然没有结果，返回所有框（兜底）
    if not subtitle_indices:
        return (text_boxes, all_detections)

    filtered_boxes = [text_boxes[i] for i in subtitle_indices]
    filtered_detections = [all_detections[i] for i in subtitle_indices]
    return (filtered_boxes, filtered_detections)


def detect_video_region(
    video_path: str,
    sample_count: int = 10,
    bottom_ratio: float = 0.3,
) -> dict:
    """
    采样视频帧，自动检测字幕所在区域。

    通过均匀采样多帧，用 PaddleOCR 检测画面底部的文字位置，
    再通过 Y 坐标聚类筛选出真正的字幕区域，过滤水印/logo 等干扰。

    :param video_path: 视频文件路径
    :param sample_count: 采样帧数
    :param bottom_ratio: 只检测画面底部的比例（默认 30%）
    :return: {"region": [y_start, y_end, x_start, x_end], "detections": [...]}
             或 {"region": null, "error": "..."}
    """
    import cv2

    ocr = _init_ocr()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"region": None, "error": f"无法打开视频: {video_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if total_frames <= 0:
        cap.release()
        return {"region": None, "error": "视频帧数为 0"}

    # 只关注画面底部区域
    y_threshold = int(frame_height * (1 - bottom_ratio))

    # 均匀采样帧
    step = max(1, total_frames // sample_count)
    text_boxes = []
    all_detections = []

    for i in range(0, min(total_frames, sample_count * step), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        # 只截取底部区域进行检测
        bottom_region = frame[y_threshold:, :]

        for result in ocr.predict(bottom_region):
            items = _parse_predict_result(result)
            for item in items:
                box = item["box"]
                text = item["text"]
                confidence = item["confidence"]

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
                all_detections.append({
                    "frame": i,
                    "text": text,
                    "confidence": confidence,
                    "y_min": y_min,
                    "y_max": y_max,
                    "x_min": x_min,
                    "x_max": x_max,
                })

    cap.release()

    if not text_boxes:
        return {"region": None, "detections": [], "info": "未检测到字幕区域"}

    # 使用聚类算法筛选真正的字幕框，过滤水印/logo 等干扰
    filtered_boxes, filtered_detections = _cluster_subtitle_boxes(
        text_boxes, all_detections, frame_height, frame_width,
    )

    if not filtered_boxes:
        return {"region": None, "detections": [], "info": "未检测到字幕区域"}

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

    return {
        "region": [y_start, y_end, x_start, x_end],
        "detections": filtered_detections,
        "frame_size": [frame_width, frame_height],
        "total_detected": len(text_boxes),
        "filtered_count": len(filtered_boxes),
    }


def detect_frame_region(
    image_data_path: str,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
) -> list:
    """
    对帧的指定区域进行 OCR 检测（用于智能模式逐帧处理）。

    :param image_data_path: 帧图片文件路径
    :param y_start: 区域 y 起始坐标
    :param y_end: 区域 y 结束坐标
    :param x_start: 区域 x 起始坐标
    :param x_end: 区域 x 结束坐标
    :return: 检测结果列表
    """
    import cv2

    ocr = _init_ocr()
    frame = cv2.imread(image_data_path)
    if frame is None:
        return []

    sub_region = frame[y_start:y_end, x_start:x_end]
    detections = []
    for result in ocr.predict(sub_region):
        detections.extend(_parse_predict_result(result))
    return detections


def main():
    """
    主入口：解析命令行参数并执行对应的 OCR 任务。

    命令格式：
        python paddle_ocr_bridge.py detect <image_path>
        python paddle_ocr_bridge.py detect_video_region <video_path> [--samples N] [--bottom_ratio F]
        python paddle_ocr_bridge.py detect_frame_region <image_path> <y_start> <y_end> <x_start> <x_end>
    """
    if len(sys.argv) < 3:
        print(json.dumps({
            "error": "用法: python paddle_ocr_bridge.py <command> <args...>",
        }))
        sys.exit(1)

    command = sys.argv[1]

    try:
        if command == "detect":
            image_path = sys.argv[2]
            result = detect_image(image_path)
            print(json.dumps({"detections": result}))

        elif command == "detect_video_region":
            video_path = sys.argv[2]
            sample_count = 10
            bottom_ratio = 0.3

            # 解析可选参数
            i = 3
            while i < len(sys.argv):
                if sys.argv[i] == "--samples" and i + 1 < len(sys.argv):
                    sample_count = int(sys.argv[i + 1])
                    i += 2
                elif sys.argv[i] == "--bottom_ratio" and i + 1 < len(sys.argv):
                    bottom_ratio = float(sys.argv[i + 1])
                    i += 2
                else:
                    i += 1

            result = detect_video_region(video_path, sample_count, bottom_ratio)
            print(json.dumps(result))

        elif command == "detect_frame_region":
            if len(sys.argv) < 7:
                print(json.dumps({
                    "error": "用法: detect_frame_region <image> <y_start> <y_end> <x_start> <x_end>",
                }))
                sys.exit(1)
            image_path = sys.argv[2]
            y_start = int(sys.argv[3])
            y_end = int(sys.argv[4])
            x_start = int(sys.argv[5])
            x_end = int(sys.argv[6])
            result = detect_frame_region(image_path, y_start, y_end, x_start, x_end)
            print(json.dumps({"detections": result}))

        else:
            print(json.dumps({"error": f"未知命令: {command}"}))
            sys.exit(1)

    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
