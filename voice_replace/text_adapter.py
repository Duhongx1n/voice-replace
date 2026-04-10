# -*- coding: utf-8 -*-
"""
智能分句适配模块 — 接入大模型，将自由格式的新台词按原视频节奏拆分。

当用户给出的新台词不符合原视频的分句模板时，本模块会调用大模型（OpenAI 兼容 API），
将新台词智能拆分为与原视频 segment 数量一致、时长比例匹配的逐句台词。

支持的大模型 API：
- HAI 内部大模型平台（DeepSeek-V3.1 / Kimi-K2.5 / Qwen3 等）
- OpenAI（GPT-4o / GPT-4o-mini）
- DeepSeek
- 通义千问（DashScope）
- 任何兼容 OpenAI Chat Completions 格式的 API（如 Ollama、vLLM 等）
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


# ============================================================
# 默认配置 — HAI 内部大模型平台
# ============================================================

# HAI 平台 API Key（可通过 --llm_api_key 或环境变量 OPENAI_API_KEY 覆盖）
DEFAULT_API_KEY = "sk-342d3b26-22e0-4005-9d6e-a25e27b11ba5"

# HAI 平台 API Base URL（公网地址）
DEFAULT_BASE_URL = "https://api.haihub.cn/v1/"

# 默认使用 DeepSeek-V3.1（支持思考模式，128K 上下文，支持 function calling）
DEFAULT_MODEL = "DeepSeek-V3.1"


def check_openai() -> bool:
    """检查 openai 库是否可用。"""
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


def _build_system_prompt() -> str:
    """
    构建系统提示词（分句适配模式）。

    :return: 系统提示词字符串
    """
    return (
        "你是一个专业的视频台词适配助手。你的任务是将用户提供的新台词内容，"
        "按照原视频的分句节奏和对话结构进行智能拆分。\n\n"
        "核心规则：\n"
        "1. 输出的句子数量必须与原视频的片段数量完全一致\n"
        "2. 每句新台词的字数应尽量与对应原始片段的字数比例接近（因为字数影响语音时长）\n"
        "3. 每句必须是语义完整的、可以独立朗读的句子或短语\n"
        "4. 不要在句子中间断开一个词语或成语\n"
        "5. 保持新台词的整体语义和逻辑连贯性\n"
        "6. 如果新台词内容比原视频短，可以适当扩写或添加过渡语\n"
        "7. 如果新台词内容比原视频长，需要精简压缩，但不能丢失核心信息\n"
        "8. ⭐ 必须保持原视频的对话结构：\n"
        "   - 原视频中的短回应句（1-3字）对应的新台词也应该是短回应\n"
        "   - 原视频中的提问句对应的新台词也应该是提问\n"
        "   - 原视频中的长叙述句对应的新台词也应该是长叙述\n"
        "   - 如果原视频是多人问答式对话，新台词也应保持问答式结构\n"
        "9. 输出格式：严格按 JSON 数组格式输出，每个元素是一句台词字符串\n"
        "10. 不要输出任何解释、注释或额外文字，只输出 JSON 数组"
    )


def _build_generation_system_prompt() -> str:
    """
    构建系统提示词（主题创作模式）。

    当用户只给出一个方向/主题时，大模型需要根据原视频的对话结构
    自动创作全新的台词内容。

    :return: 系统提示词字符串
    """
    return (
        "你是一个专业的视频台词创作助手。你的任务是根据用户给出的主题方向，"
        "参考原视频的对话结构和节奏，创作全新的台词内容。\n\n"
        "核心规则：\n"
        "1. 输出的句子数量必须与原视频的片段数量完全一致\n"
        "2. 每句新台词的字数应尽量与对应原始片段的字数比例接近（因为字数影响语音时长）\n"
        "3. 每句必须是语义完整的、可以独立朗读的句子或短语\n"
        "4. 不要在句子中间断开一个词语或成语\n"
        "5. 新台词的整体语义和逻辑必须连贯，围绕用户给出的主题展开\n"
        "6. ⭐ 必须保持原视频的对话结构：\n"
        "   - 原视频中的短回应句（1-3字）对应的新台词也应该是短回应\n"
        "   - 原视频中的提问句对应的新台词也应该是提问\n"
        "   - 原视频中的长叙述句对应的新台词也应该是长叙述\n"
        "   - 如果原视频是多人问答式对话，新台词也应保持问答式结构\n"
        "7. 新台词应该自然流畅，像真人说话一样，不要太书面化\n"
        "8. 输出格式：严格按 JSON 数组格式输出，每个元素是一句台词字符串\n"
        "9. 不要输出任何解释、注释或额外文字，只输出 JSON 数组"
    )


def _build_user_prompt(
    original_segments: List[Dict],
    new_text: str,
) -> str:
    """
    构建用户提示词，包含原视频的对话结构分析。

    :param original_segments: 原视频的 Whisper segments 列表
    :param new_text: 用户提供的新台词（完整文本）
    :return: 用户提示词字符串
    """
    # 分析对话结构
    from voice_replace.dialogue_analysis import analyze_dialogue_structure, segments_as_slots
    slots = segments_as_slots(original_segments)
    annotated = analyze_dialogue_structure(slots)

    # 构建原始分句信息（带对话结构标注）
    seg_info_lines = []
    total_chars = 0
    for i, a in enumerate(annotated):
        text = a.get("original_text", "").strip()
        duration = a["duration"]
        char_count = a["char_count"]
        role = a["role_label"]
        stype = a["sentence_type"]
        total_chars += char_count
        seg_info_lines.append(
            f"  第{i + 1}句: 时长{duration:.1f}秒, {char_count}字, "
            f"角色{role}, 类型[{stype}], "
            f"原文: \"{text}\""
        )

    seg_info = "\n".join(seg_info_lines)
    num_segments = len(original_segments)

    # 统计对话模式
    roles_used = sorted(set(a["role_label"] for a in annotated))
    role_count = len(roles_used)
    dialogue_mode = "多人对话" if role_count > 1 else "单人独白"

    prompt = (
        f"## 原视频对话结构分析\n\n"
        f"对话模式: {dialogue_mode}（推断出 {role_count} 个角色）\n\n"
        f"## 原视频分句信息（共 {num_segments} 个片段，总计 {total_chars} 字）\n\n"
        f"{seg_info}\n\n"
        f"## 新台词内容\n\n"
        f"{new_text}\n\n"
        f"## 任务\n\n"
        f"请将上面的「新台词内容」拆分为恰好 {num_segments} 句，"
        f"使每句的字数尽量与对应原始片段的字数比例接近。\n\n"
        f"重要要求：\n"
        f"1. 保持原视频的对话结构：原文是短回应的地方，新台词也应该是短回应\n"
        f"2. 原文是提问的地方，新台词也应该是提问\n"
        f"3. 角色切换处保持自然过渡\n\n"
        f"直接输出 JSON 数组，格式如：\n"
        f'[\"\u7b2c\u4e00\u53e5\u53f0\u8bcd\", \"\u7b2c\u4e8c\u53e5\u53f0\u8bcd\", ...]\n\n'
        f"注意：数组长度必须恰好为 {num_segments}。"
    )

    return prompt


def _build_topic_generation_prompt(
    original_segments: List[Dict],
    topic: str,
) -> str:
    """
    构建主题创作模式的用户提示词。

    :param original_segments: 原视频的 Whisper segments 列表
    :param topic: 用户给出的主题/方向描述
    :return: 用户提示词字符串
    """
    # 分析对话结构
    from voice_replace.dialogue_analysis import analyze_dialogue_structure, segments_as_slots
    slots = segments_as_slots(original_segments)
    annotated = analyze_dialogue_structure(slots)

    # 构建原始分句信息（带对话结构标注）
    seg_info_lines = []
    total_chars = 0
    for i, a in enumerate(annotated):
        text = a.get("original_text", "").strip()
        duration = a["duration"]
        char_count = a["char_count"]
        role = a["role_label"]
        stype = a["sentence_type"]
        total_chars += char_count
        seg_info_lines.append(
            f"  第{i + 1}句: 时长{duration:.1f}秒, {char_count}字, "
            f"角色{role}, 类型[{stype}], "
            f"原文: \"{text}\""
        )

    seg_info = "\n".join(seg_info_lines)
    num_segments = len(original_segments)

    # 统计对话模式
    roles_used = sorted(set(a["role_label"] for a in annotated))
    role_count = len(roles_used)
    dialogue_mode = "多人对话" if role_count > 1 else "单人独白"

    prompt = (
        f"## 创作主题/方向\n\n"
        f"{topic}\n\n"
        f"## 原视频对话结构分析\n\n"
        f"对话模式: {dialogue_mode}（推断出 {role_count} 个角色）\n\n"
        f"## 原视频分句信息（共 {num_segments} 个片段，总计 {total_chars} 字）\n\n"
        f"{seg_info}\n\n"
        f"## 任务\n\n"
        f"请围绕上面的「创作主题/方向」，参考原视频的对话结构，"
        f"创作恰好 {num_segments} 句全新的台词。\n\n"
        f"重要要求：\n"
        f"1. 每句新台词的字数尽量与对应原始片段的字数比例接近\n"
        f"2. 保持原视频的对话结构：原文是短回应的地方，新台词也应该是短回应\n"
        f"3. 原文是提问的地方，新台词也应该是提问\n"
        f"4. 角色切换处保持自然过渡\n"
        f"5. 台词内容必须围绕给定的主题展开，不要照搬原文\n\n"
        f"直接输出 JSON 数组，格式如：\n"
        f'["\u7b2c\u4e00\u53e5\u53f0\u8bcd", "\u7b2c\u4e8c\u53e5\u53f0\u8bcd", ...]\n\n'
        f"注意：数组长度必须恰好为 {num_segments}。"
    )

    return prompt


def _parse_llm_response(response_text: str, expected_count: int) -> Optional[List[str]]:
    """
    解析大模型返回的 JSON 数组。

    :param response_text: 大模型的原始回复文本
    :param expected_count: 期望的句子数量
    :return: 句子列表，解析失败返回 None
    """
    text = response_text.strip()

    # 尝试直接解析
    try:
        result = json.loads(text)
        if isinstance(result, list) and all(isinstance(s, str) for s in result):
            return result
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 代码块
    json_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    for match in matches:
        try:
            result = json.loads(match.strip())
            if isinstance(result, list) and all(isinstance(s, str) for s in result):
                return result
        except json.JSONDecodeError:
            continue

    # 尝试提取方括号内容
    bracket_pattern = r'\[.*\]'
    matches = re.findall(bracket_pattern, text, re.DOTALL)
    for match in matches:
        try:
            result = json.loads(match)
            if isinstance(result, list) and all(isinstance(s, str) for s in result):
                return result
        except json.JSONDecodeError:
            continue

    print(f"[警告] 无法解析大模型返回的 JSON，原始回复:\n{text[:500]}", file=sys.stderr)
    return None


def _needs_adaptation(
    new_lines: List[str],
    segments: List[Dict],
    tolerance: float = 0.3,
) -> Tuple[bool, str]:
    """
    判断新台词是否需要大模型适配。

    判断标准：
    1. 行数与 segment 数不一致 → 需要适配
    2. 行数一致但字数比例偏差过大 → 需要适配
    3. 行数一致且字数比例接近 → 不需要适配

    :param new_lines: 新台词行列表
    :param segments: 原始 segments 列表
    :param tolerance: 字数比例偏差容忍度（默认 30%）
    :return: (是否需要适配, 原因说明)
    """
    num_lines = len(new_lines)
    num_segments = len(segments)

    # 行数不一致
    if num_lines != num_segments:
        return True, (
            f"新台词行数({num_lines})与原视频片段数({num_segments})不一致"
        )

    # 行数一致，检查字数比例
    original_chars = []
    new_chars = []
    for seg, line in zip(segments, new_lines):
        orig_text = seg.get("original_text", seg.get("text", "")).strip()
        original_chars.append(len(orig_text))
        new_chars.append(len(line))

    total_orig = sum(original_chars)
    total_new = sum(new_chars)

    if total_orig == 0:
        return False, "原始文本为空，跳过适配"

    # 检查每句的字数比例偏差
    large_deviation_count = 0
    for i in range(num_segments):
        if original_chars[i] == 0:
            continue
        orig_ratio = original_chars[i] / total_orig
        new_ratio = new_chars[i] / total_new if total_new > 0 else 0
        deviation = abs(orig_ratio - new_ratio)
        if deviation > tolerance:
            large_deviation_count += 1

    if large_deviation_count > num_segments * 0.5:
        return True, (
            f"超过半数句子({large_deviation_count}/{num_segments})"
            f"的字数比例偏差超过{tolerance:.0%}"
        )

    return False, "新台词已符合原视频节奏，无需适配"


def adapt_text_with_llm(
    new_text: str,
    segments: List[Dict],
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    max_retries: int = 2,
) -> Optional[List[str]]:
    """
    调用大模型将新台词按原视频节奏拆分。

    优先级（从高到低）：
    1. 函数参数（--llm_api_key / --llm_base_url / --llm_model）
    2. 环境变量（OPENAI_API_KEY / OPENAI_BASE_URL）
    3. 内置默认值（HAI 平台 DeepSeek-V3.1）

    :param new_text: 用户提供的新台词（完整文本或多行文本）
    :param segments: 原视频的 segment 列表（含 start, end, original_text/text）
    :param api_key: API Key（也可通过环境变量 OPENAI_API_KEY 设置）
    :param base_url: API Base URL（也可通过环境变量 OPENAI_BASE_URL 设置）
    :param model: 模型名称（默认使用 HAI 平台 DeepSeek-V3.1）
    :param max_retries: 最大重试次数
    :return: 适配后的句子列表，失败返回 None
    """
    if not check_openai():
        print(
            "[错误] openai 库未安装。请运行: pip install openai",
            file=sys.stderr,
        )
        return None

    import openai

    # 确定 API Key、Base URL 和模型（优先级：参数 > 环境变量 > 内置默认值）
    effective_api_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY", "")
        or DEFAULT_API_KEY
    )
    effective_base_url = (
        base_url
        or os.environ.get("OPENAI_BASE_URL", "")
        or DEFAULT_BASE_URL
    )
    effective_model = model or DEFAULT_MODEL

    if not effective_api_key:
        print(
            "[错误] 未提供 LLM API Key。请通过 --llm_api_key 参数或 "
            "OPENAI_API_KEY 环境变量设置。",
            file=sys.stderr,
        )
        return None

    # 创建客户端
    client_kwargs = {"api_key": effective_api_key}
    if effective_base_url:
        client_kwargs["base_url"] = effective_base_url

    client = openai.OpenAI(**client_kwargs)

    num_segments = len(segments)
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(segments, new_text)

    print(f"  [大模型分句] 调用 {effective_model}")
    print(f"    API: {effective_base_url or '(默认)'}")
    print(f"    目标: {num_segments} 句")

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=effective_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
            )

            reply = response.choices[0].message.content
            result = _parse_llm_response(reply, num_segments)

            if result is None:
                print(f"  [大模型分句] 第 {attempt + 1} 次: JSON 解析失败，重试...")
                continue

            if len(result) != num_segments:
                print(
                    f"  [大模型分句] 第 {attempt + 1} 次: "
                    f"返回 {len(result)} 句，期望 {num_segments} 句，重试..."
                )
                # 补充提示重试
                user_prompt += (
                    f"\n\n⚠️ 上次你返回了 {len(result)} 句，但必须恰好是 "
                    f"{num_segments} 句。请重新拆分。"
                )
                continue

            # 验证每句非空
            empty_count = sum(1 for s in result if not s.strip())
            if empty_count > 0:
                print(
                    f"  [大模型分句] 第 {attempt + 1} 次: "
                    f"有 {empty_count} 句为空，重试..."
                )
                continue

            print(f"  [大模型分句] ✅ 成功拆分为 {len(result)} 句")
            return result

        except Exception as e:
            print(
                f"  [大模型分句] 第 {attempt + 1} 次调用失败: {e}",
                file=sys.stderr,
            )

    print("[警告] 大模型分句失败，将使用原始文本", file=sys.stderr)
    return None


def generate_text_from_topic(
    topic: str,
    segments: List[Dict],
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    max_retries: int = 2,
) -> Optional[List[str]]:
    """
    根据主题/方向，调用大模型自动创作与原视频时间轴匹配的新台词。

    用户只需给出一个方向（如"介绍人工智能技术"），大模型会参考
    原视频每句话的时长、角色、句式结构，自动创作出完全匹配时间轴
    的全新台词。

    :param topic: 用户给出的主题/方向描述
    :param segments: 原视频的 segment 列表（含 start, end, original_text/text）
    :param api_key: API Key
    :param base_url: API Base URL
    :param model: 模型名称
    :param max_retries: 最大重试次数
    :return: 生成的台词列表，失败返回 None
    """
    if not check_openai():
        print(
            "[错误] openai 库未安装。请运行: pip install openai",
            file=sys.stderr,
        )
        return None

    import openai

    # 确定 API Key、Base URL 和模型（优先级：参数 > 环境变量 > 内置默认值）
    effective_api_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY", "")
        or DEFAULT_API_KEY
    )
    effective_base_url = (
        base_url
        or os.environ.get("OPENAI_BASE_URL", "")
        or DEFAULT_BASE_URL
    )
    effective_model = model or DEFAULT_MODEL

    if not effective_api_key:
        print(
            "[错误] 未提供 LLM API Key。请通过 --llm_api_key 参数或 "
            "OPENAI_API_KEY 环境变量设置。",
            file=sys.stderr,
        )
        return None

    # 创建客户端
    client_kwargs = {"api_key": effective_api_key}
    if effective_base_url:
        client_kwargs["base_url"] = effective_base_url

    client = openai.OpenAI(**client_kwargs)

    num_segments = len(segments)
    system_prompt = _build_generation_system_prompt()
    user_prompt = _build_topic_generation_prompt(segments, topic)

    print(f"  [大模型创作] 调用 {effective_model}")
    print(f"    API: {effective_base_url or '(默认)'}")
    print(f"    主题: {topic}")
    print(f"    目标: {num_segments} 句")

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=effective_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=4096,
            )

            reply = response.choices[0].message.content
            result = _parse_llm_response(reply, num_segments)

            if result is None:
                print(f"  [大模型创作] 第 {attempt + 1} 次: JSON 解析失败，重试...")
                continue

            if len(result) != num_segments:
                print(
                    f"  [大模型创作] 第 {attempt + 1} 次: "
                    f"返回 {len(result)} 句，期望 {num_segments} 句，重试..."
                )
                user_prompt += (
                    f"\n\n⚠️ 上次你返回了 {len(result)} 句，但必须恰好是 "
                    f"{num_segments} 句。请重新创作。"
                )
                continue

            # 验证每句非空
            empty_count = sum(1 for s in result if not s.strip())
            if empty_count > 0:
                print(
                    f"  [大模型创作] 第 {attempt + 1} 次: "
                    f"有 {empty_count} 句为空，重试..."
                )
                continue

            print(f"  [大模型创作] ✅ 成功创作 {len(result)} 句新台词")
            return result

        except Exception as e:
            print(
                f"  [大模型创作] 第 {attempt + 1} 次调用失败: {e}",
                file=sys.stderr,
            )

    print("[警告] 大模型创作失败", file=sys.stderr)
    return None


def adapt_new_text(
    new_lines: List[str],
    segments: List[Dict],
    output_dir: str,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    force_adapt: bool = False,
) -> List[str]:
    """
    智能适配新台词：判断是否需要适配，需要则调用大模型。

    这是本模块的主入口函数。已内置 HAI 平台的 API Key 和地址，
    无需额外配置即可使用智能分句功能。

    :param new_lines: 用户提供的新台词行列表
    :param segments: 原视频的 segment 列表
    :param output_dir: 输出目录（用于保存适配结果）
    :param api_key: LLM API Key（可选，默认使用内置 HAI 平台 Key）
    :param base_url: LLM API Base URL（可选，默认使用 HAI 平台地址）
    :param model: LLM 模型名称（可选，默认使用 DeepSeek-V3.1）
    :param force_adapt: 强制使用大模型适配（即使行数已匹配）
    :return: 适配后的台词行列表（如果不需要适配或适配失败，返回原始列表）
    """
    if not segments:
        print("  [智能分句] 无原始 segments 信息，跳过适配")
        return new_lines

    # 判断是否需要适配
    needs_adapt, reason = _needs_adaptation(new_lines, segments)

    if force_adapt:
        needs_adapt = True
        reason = "用户强制启用大模型适配"

    if not needs_adapt:
        print(f"  [智能分句] {reason}")
        return new_lines

    print(f"  [智能分句] 需要适配: {reason}")

    # 检查是否有可用的 LLM（参数 > 环境变量 > 内置默认值）
    effective_api_key = (
        api_key
        or os.environ.get("OPENAI_API_KEY", "")
        or DEFAULT_API_KEY
    )
    if not effective_api_key:
        print(
            "  [智能分句] ⚠️ 未配置大模型 API Key，无法进行智能分句。\n"
            "    请通过以下方式之一配置：\n"
            "    1. 命令行参数: --llm_api_key YOUR_KEY\n"
            "    2. 环境变量: export OPENAI_API_KEY=YOUR_KEY\n"
            "    将使用原始文本继续（可能导致时间轴对齐效果不佳）",
        )
        return new_lines

    # 合并新台词为完整文本
    full_new_text = "\n".join(new_lines)

    # 调用大模型
    adapted = adapt_text_with_llm(
        new_text=full_new_text,
        segments=segments,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    if adapted is None:
        print("  [智能分句] 大模型适配失败，使用原始文本")
        return new_lines

    # 保存适配结果
    adapted_path = os.path.join(output_dir, "adapted_transcript.md")
    with open(adapted_path, "w", encoding="utf-8") as f:
        f.write("# 大模型适配后的台词\n")
        f.write(f"# 原始片段数: {len(segments)}, 适配后句数: {len(adapted)}\n\n")
        for i, (seg, line) in enumerate(zip(segments, adapted)):
            orig_text = seg.get("original_text", seg.get("text", "")).strip()
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            f.write(
                f"# --- 第 {i + 1} 句 [{start:.2f}s - {end:.2f}s] "
                f"原文({len(orig_text)}字): {orig_text}\n"
            )
            f.write(f"{line}\n\n")

    print(f"  [智能分句] 适配结果已保存: {adapted_path}")

    # 打印对比
    print(f"\n  [智能分句] 适配对比（原始 → 新台词）：")
    for i, (seg, line) in enumerate(zip(segments, adapted)):
        orig_text = seg.get("original_text", seg.get("text", "")).strip()
        print(
            f"    第{i + 1}句: [{len(orig_text)}字] {orig_text[:25]}..."
            f" → [{len(line)}字] {line[:25]}..."
        )

    return adapted
