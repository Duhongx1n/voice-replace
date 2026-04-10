# -*- coding: utf-8 -*-
"""
TTS 语音合成模块 — 封装 Qwen3-TTS 和 edge-tts 引擎。

支持三种模式：
1. 声音克隆（Base 模型）：提供参考音频克隆音色
2. 预设音色（CustomVoice 模型）：9 种内置音色 + instruct 控制
3. 语音设计（VoiceDesign 模型）：自然语言描述设计全新音色
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from typing import Optional, Tuple

from voice_replace.audio_utils import trim_leading_noise
from voice_replace.video_utils import get_duration


# ============================================================
# 预设音色列表
# ============================================================

QWEN3_SPEAKERS = {
    "Vivian": "明亮的年轻女声",
    "Serena": "温暖、温柔的年轻女声",
    "Uncle_Fu": "成熟的男性声音，醇厚音色",
    "Dylan": "年轻的北京男声",
    "Eric": "活泼的成都男声",
    "Ryan": "富有节奏感的英文男声",
    "Aiden": "阳光的美国男声",
    "Ono_Anna": "俏皮的日本女声",
    "Sohee": "温暖的韩国女声",
}

EDGE_VOICES = {
    "zh-CN-XiaoxiaoNeural": "晓晓（女，温暖自然）",
    "zh-CN-XiaoyiNeural": "晓伊（女，亲切活泼）",
    "zh-CN-YunjianNeural": "云健（男，沉稳大气）",
    "zh-CN-YunxiNeural": "云希（男，年轻阳光）",
    "zh-CN-YunyangNeural": "云扬（男，新闻播报）",
}


# ============================================================
# TTS 引擎检测
# ============================================================

def check_qwen3_tts() -> bool:
    """检查 Qwen3-TTS 是否可用。"""
    try:
        from qwen_tts import Qwen3TTSModel  # noqa: F401
        return True
    except ImportError:
        return False


def check_edge_tts() -> bool:
    """检查 edge-tts 是否可用。"""
    try:
        import edge_tts  # noqa: F401
        return True
    except ImportError:
        return False


def detect_tts_engine(preferred: str = "auto") -> str:
    """
    检测可用的 TTS 引擎。

    :param preferred: "auto"（自动选择）、"qwen3"（强制 Qwen3）、"edge"（强制 edge-tts）
    :return: "qwen3" 或 "edge"
    """
    has_qwen3 = check_qwen3_tts()
    has_edge = check_edge_tts()

    if preferred == "qwen3":
        if has_qwen3:
            return "qwen3"
        print("[错误] 指定使用 Qwen3-TTS 但未安装。请运行: pip install -U qwen-tts", file=sys.stderr)
        sys.exit(1)
    elif preferred == "edge":
        if has_edge:
            return "edge"
        print("[错误] 指定使用 edge-tts 但未安装。请运行: pip install edge-tts", file=sys.stderr)
        sys.exit(1)
    else:
        if has_qwen3:
            return "qwen3"
        if has_edge:
            return "edge"
        print("[错误] 未检测到任何 TTS 引擎！请至少安装其一：", file=sys.stderr)
        print("  方式一（推荐，本地高质量）: pip install -U qwen-tts", file=sys.stderr)
        print("  方式二（在线，需联网）:     pip install edge-tts", file=sys.stderr)
        sys.exit(1)


# ============================================================
# 文本预处理
# ============================================================

def normalize_text_for_tts(text: str) -> str:
    """
    对输入文本做预处理，减少 TTS 杂音/幻觉。

    处理策略：
    1. 将纯数字转为中文读法
    2. 将带单位的数字保留自然读法
    3. 将百分号转为"百分之"
    4. 清除多余符号

    :param text: 原始文本
    :return: 预处理后的文本
    """
    _DIGITS = "零一二三四五六七八九"
    _UNITS_SMALL = ["", "十", "百", "千"]
    _UNITS_BIG = ["", "万", "亿"]

    def _int_to_chinese(n: int) -> str:
        """整数转中文（支持到亿级别）。"""
        if n == 0:
            return "零"
        if n < 0:
            return "负" + _int_to_chinese(-n)

        result = ""
        s = str(n)

        groups = []
        while s:
            groups.append(s[-4:] if len(s) >= 4 else s)
            s = s[:-4]
        groups.reverse()

        for gi, group in enumerate(groups):
            group_val = int(group)
            if group_val == 0:
                continue

            big_unit = (
                _UNITS_BIG[len(groups) - 1 - gi]
                if (len(groups) - 1 - gi) < len(_UNITS_BIG)
                else ""
            )

            if gi > 0 and int(group) < 1000 and len(group) == len(str(int(group))):
                if result and not result.endswith("零"):
                    result += "零"

            g_str = ""
            for di, ch in enumerate(group):
                d = int(ch)
                pos = len(group) - 1 - di
                if d == 0:
                    if g_str and not g_str.endswith("零") and pos > 0:
                        g_str += "零"
                else:
                    if d == 1 and pos == 1 and not g_str:
                        g_str += _UNITS_SMALL[pos]
                    else:
                        g_str += _DIGITS[d] + _UNITS_SMALL[pos]

            g_str = g_str.rstrip("零")
            result += g_str + big_unit

        return result

    def _number_to_chinese(match_str: str) -> str:
        """将数字字符串转为中文读法。"""
        if "." in match_str:
            parts = match_str.split(".", 1)
            integer_part = _int_to_chinese(int(parts[0])) if parts[0] else "零"
            decimal_part = "".join(_DIGITS[int(d)] for d in parts[1])
            return integer_part + "点" + decimal_part
        return _int_to_chinese(int(match_str))

    result = text

    # 1. 百分比
    result = re.sub(
        r'(\d+(?:\.\d+)?)\s*%',
        lambda m: "百分之" + _number_to_chinese(m.group(1)),
        result,
    )

    # 2. 带中文单位的数字
    result = re.sub(
        r'(\d+(?:\.\d+)?)\s*(万|亿|千|百|美元|元|块钱|块|分钟|秒|小时|天|年|月|步|条|个|次|倍|级)',
        lambda m: _number_to_chinese(m.group(1)) + m.group(2),
        result,
    )

    # 3. 带英文单位的数字
    result = re.sub(
        r'(\d+(?:\.\d+)?)\s*(KB|MB|GB|TB|token|Token|tokens|Tokens)',
        lambda m: _number_to_chinese(m.group(1)) + " " + m.group(2),
        result,
        flags=re.IGNORECASE,
    )

    # 4. 剩余纯数字
    result = re.sub(
        r'(?<![a-zA-Z])(\d+(?:\.\d+)?)(?![a-zA-Z%])',
        lambda m: _number_to_chinese(m.group(1)),
        result,
    )

    # 5. 清理
    result = re.sub(r'——', '，', result)
    result = re.sub(r'—', '，', result)
    result = re.sub(r'\s+', ' ', result).strip()

    return result


# ============================================================
# Qwen3-TTS 引擎
# ============================================================

class Qwen3TTSEngine:
    """
    Qwen3-TTS 语音合成引擎。

    支持三种模式：
    1. 声音克隆（Base 模型）：提供参考音频克隆音色
    2. 预设音色（CustomVoice 模型）：内置音色 + instruct 控制
    3. 语音设计（VoiceDesign 模型）：自然语言描述设计音色
    """

    ICL_MAX_DURATION = 15.0

    BASE_MODEL_CANDIDATES = [
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ]

    CUSTOM_MODEL_CANDIDATES = [
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ]

    VOICE_DESIGN_MODEL_CANDIDATES = [
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    ]

    DEFAULT_INSTRUCT = "用标准的新闻播音腔朗读，字正腔圆，语速稍快，节奏稳定，语调平稳，适合专业技术讲解"
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_TOP_K = 30
    DEFAULT_TOP_P = 0.8
    DEFAULT_SEED = 42

    MAX_SECONDS_PER_CHAR = 0.45
    MAX_RETRIES = 3
    CODEC_TOKENS_PER_SECOND = 12
    RUNAWAY_THRESHOLD_RATIO = 0.9
    PREEMPTIVE_RELOAD_RATIO = 1.5

    def __init__(
        self,
        speaker: str = "Vivian",
        reference_voice: str = "",
        ref_text: str = "",
        voice_design: str = "",
        instruct: str = "",
    ):
        """
        初始化 Qwen3-TTS 引擎。

        :param speaker: 预设音色名称
        :param reference_voice: 参考音频路径（声音克隆模式）
        :param ref_text: 参考音频文字内容
        :param voice_design: 语音设计指令
        :param instruct: CustomVoice 模式的语气控制指令
        """
        import torch
        from qwen_tts import Qwen3TTSModel

        self.speaker = speaker
        self.reference_voice = reference_voice
        self.ref_text = ref_text
        self.voice_design = voice_design
        self.instruct = instruct
        self.model = None
        self.model_name = ""
        self.is_clone_mode = False
        self.is_voice_design_mode = bool(voice_design)
        self.x_vector_only = False
        self.sample_rate = 24000
        self._voice_clone_prompt = None
        self._periodic_reload_interval = 30
        self._sentences_since_reload = 0

        # 确定设备和精度
        if torch.cuda.is_available():
            device_map = "cuda:0"
            dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_map = "mps"
            dtype = torch.bfloat16
            attn_impl = "sdpa"
        else:
            device_map = "cpu"
            dtype = torch.float32
            attn_impl = "eager"

        print(f"[Qwen3-TTS] 计算设备: {device_map}, 精度: {dtype}")

        # 判断模式
        has_ref = reference_voice and os.path.isfile(reference_voice)
        if has_ref:
            print(f"[Qwen3-TTS] 检测到参考音频: {reference_voice}")
            print("[Qwen3-TTS] 将使用声音克隆模式（Base 模型）")
            model_candidates = self.BASE_MODEL_CANDIDATES
        elif self.is_voice_design_mode:
            print("[Qwen3-TTS] 将使用语音设计模式（VoiceDesign 模型）")
            model_candidates = self.VOICE_DESIGN_MODEL_CANDIDATES
        else:
            if reference_voice:
                print(f"[警告] 参考音频不存在: {reference_voice}，降级为预设音色模式")
            print("[Qwen3-TTS] 将使用预设音色模式（CustomVoice 模型）")
            if not self.instruct:
                self.instruct = self.DEFAULT_INSTRUCT
            model_candidates = self.CUSTOM_MODEL_CANDIDATES

        # 加载模型
        for model_id in model_candidates:
            try:
                print(f"[Qwen3-TTS] 尝试加载模型: {model_id}")
                self.model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=device_map,
                    dtype=dtype,
                    attn_implementation=attn_impl,
                )
                self.model_name = model_id
                print(f"[Qwen3-TTS] 模型加载成功: {model_id}")
                break
            except Exception as e:
                print(f"[Qwen3-TTS] 加载失败: {model_id} — {e}")
                continue

        if self.model is None:
            raise RuntimeError("无法加载任何 Qwen3-TTS 模型")

        is_base_model = "Base" in self.model_name

        # 设置工作模式
        if has_ref and is_base_model:
            self.is_clone_mode = True
            self._setup_clone_mode(reference_voice)
        elif self.is_voice_design_mode and "VoiceDesign" in self.model_name:
            self.is_clone_mode = False
            print("[Qwen3-TTS] 模式: 语音设计 voice_design")
        elif "CustomVoice" in self.model_name:
            self.is_clone_mode = False
            self._setup_custom_voice_mode()
        else:
            self.is_clone_mode = False
            self.is_voice_design_mode = False

    def _setup_clone_mode(self, reference_voice: str) -> None:
        """设置声音克隆模式。"""
        ref_duration = get_duration(reference_voice)
        print(f"[Qwen3-TTS] 参考音频时长: {ref_duration:.1f}秒")

        effective_ref_audio = reference_voice

        if ref_duration > self.ICL_MAX_DURATION:
            self.x_vector_only = True
            print(f"[Qwen3-TTS] 参考音频超过 ICL 上限，使用 x_vector_only 模式")
            effective_ref_audio = self._trim_ref_audio(
                reference_voice, self.ICL_MAX_DURATION,
            )
        else:
            if not self.ref_text:
                self.ref_text = self._transcribe_ref_audio(reference_voice)
            if self.ref_text:
                self.x_vector_only = False
                print(f"[Qwen3-TTS] 模式: 声音克隆 ICL")
            else:
                self.x_vector_only = True
                print("[Qwen3-TTS] 模式: 声音克隆 x_vector（无参考文本）")

        # 预创建 voice_clone_prompt
        try:
            print("[Qwen3-TTS] 正在分析参考音频特征...")
            self._voice_clone_prompt = self.model.create_voice_clone_prompt(
                ref_audio=effective_ref_audio,
                ref_text=self.ref_text if not self.x_vector_only else None,
                x_vector_only_mode=self.x_vector_only,
            )
            print("[Qwen3-TTS] 参考音频特征提取成功")
        except Exception as e:
            print(f"[警告] 参考音频特征提取失败: {e}")
            if not self.x_vector_only:
                print("[Qwen3-TTS] 降级为 x_vector_only 模式重试...")
                self.x_vector_only = True
                try:
                    self._voice_clone_prompt = self.model.create_voice_clone_prompt(
                        ref_audio=effective_ref_audio,
                        x_vector_only_mode=True,
                    )
                except Exception as e2:
                    print(f"[警告] x_vector_only 也失败: {e2}")

    def _setup_custom_voice_mode(self) -> None:
        """设置预设音色模式。"""
        speakers = self.model.get_supported_speakers()
        if speakers:
            if self.speaker not in speakers:
                lower_speaker = self.speaker.lower()
                matched = [s for s in speakers if s.lower() == lower_speaker]
                if matched:
                    self.speaker = matched[0]
                else:
                    print(f"[警告] 音色 '{self.speaker}' 不可用，使用 '{speakers[0]}'")
                    self.speaker = speakers[0]
        print(f"[Qwen3-TTS] 模式: 预设音色（{self.speaker}）")

    @staticmethod
    def _trim_ref_audio(audio_path: str, max_duration: float) -> str:
        """截取参考音频的前 N 秒。"""
        trimmed_path = os.path.join(
            os.path.dirname(audio_path),
            f"_ref_trimmed_{max_duration:.0f}s.wav",
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-t", str(max_duration),
            "-acodec", "pcm_s16le",
            "-ar", "24000",
            "-ac", "1",
            trimmed_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.isfile(trimmed_path):
            return trimmed_path
        return audio_path

    @staticmethod
    def _transcribe_ref_audio(audio_path: str) -> str:
        """使用 Whisper 自动转录参考音频文字。"""
        try:
            import whisper
            print("[Whisper] 正在转录参考音频...")
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, language="zh")
            text = result.get("text", "").strip()
            if text:
                print(f"[Whisper] 转录结果: {text[:100]}...")
            return text
        except ImportError:
            print("[Whisper] whisper 未安装，无法自动转录")
            return ""
        except Exception as e:
            print(f"[Whisper] 转录失败: {e}")
            return ""

    def _calc_max_new_tokens(self, text: str) -> int:
        """根据文本长度计算合理的 max_new_tokens 上限。"""
        content_chars = len(re.sub(
            r'[，。！？、；：\u201c\u201d\u2018\u2019（）…\s\.\,\!\?\;\:\'\"\-]',
            '', text,
        ))
        max_duration_sec = max(5.0, content_chars * self.MAX_SECONDS_PER_CHAR * 2.0)
        max_tokens = int(max_duration_sec * self.CODEC_TOKENS_PER_SECOND)
        return max(120, min(2400, max_tokens))

    def _is_runaway_generation(self, audio_data, sr: int, text: str) -> bool:
        """检测本次生成是否跑飞了。"""
        max_tokens = self._calc_max_new_tokens(text)
        max_duration = max_tokens / self.CODEC_TOKENS_PER_SECOND
        actual_duration = len(audio_data) / sr
        threshold = max_duration * self.RUNAWAY_THRESHOLD_RATIO
        return actual_duration >= threshold

    def _needs_preemptive_reload(self, audio_data, sr: int, text: str) -> bool:
        """检测是否需要预防性重载。"""
        content_chars = len(re.sub(
            r'[，。！？、；：\u201c\u201d\u2018\u2019（）…\s\.\,\!\?\;\:\'\"\-]',
            '', text,
        ))
        expected_duration = max(2.0, content_chars * self.MAX_SECONDS_PER_CHAR)
        actual_duration = len(audio_data) / sr
        threshold = expected_duration * self.PREEMPTIVE_RELOAD_RATIO
        return actual_duration > threshold

    def _reload_model(self) -> None:
        """释放当前模型并重新加载，清理膨胀的 KV cache 内存。"""
        import gc
        import torch
        from qwen_tts import Qwen3TTSModel

        model_name = self.model_name
        print(f"    [重载] 释放模型 {model_name}...")

        saved_voice_clone_prompt = self._voice_clone_prompt

        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if torch.cuda.is_available():
            device_map = "cuda:0"
            dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_map = "mps"
            dtype = torch.bfloat16
            attn_impl = "sdpa"
        else:
            device_map = "cpu"
            dtype = torch.float32
            attn_impl = "eager"

        print(f"    [重载] 重新加载模型 {model_name}...")
        reload_start = time.time()
        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        reload_time = time.time() - reload_start
        print(f"    [重载] 模型加载完成（耗时 {reload_time:.1f}s）")

        if saved_voice_clone_prompt is not None:
            self._voice_clone_prompt = saved_voice_clone_prompt

    def set_periodic_reload(self, interval: int) -> None:
        """设置定期重载间隔。"""
        self._periodic_reload_interval = max(0, interval)

    def _maybe_periodic_reload(self) -> bool:
        """检查是否需要定期重载模型。"""
        if self._periodic_reload_interval <= 0:
            return False

        self._sentences_since_reload += 1

        if self._sentences_since_reload >= self._periodic_reload_interval:
            print(f"\n    [定期重载] 已连续生成 {self._sentences_since_reload} 句，触发重载...")
            self._reload_model()
            self._sentences_since_reload = 0
            return True

        return False

    def _generate_once(self, text: str):
        """调用模型生成一次音频。"""
        import torch

        torch.manual_seed(self.DEFAULT_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.DEFAULT_SEED)

        max_tokens = self._calc_max_new_tokens(text)

        if self.is_clone_mode:
            if self._voice_clone_prompt is not None:
                return self.model.generate_voice_clone(
                    text=text,
                    language="Chinese",
                    voice_clone_prompt=self._voice_clone_prompt,
                    temperature=self.DEFAULT_TEMPERATURE,
                    top_k=self.DEFAULT_TOP_K,
                    top_p=self.DEFAULT_TOP_P,
                    max_new_tokens=max_tokens,
                )
            return self.model.generate_voice_clone(
                text=text,
                language="Chinese",
                ref_audio=self.reference_voice,
                ref_text=self.ref_text if not self.x_vector_only else None,
                x_vector_only_mode=self.x_vector_only,
                temperature=self.DEFAULT_TEMPERATURE,
                top_k=self.DEFAULT_TOP_K,
                top_p=self.DEFAULT_TOP_P,
                max_new_tokens=max_tokens,
            )
        elif self.is_voice_design_mode:
            return self.model.generate_voice_design(
                text=text,
                instruct=self.voice_design,
                language="Chinese",
                temperature=self.DEFAULT_TEMPERATURE,
                top_k=self.DEFAULT_TOP_K,
                top_p=self.DEFAULT_TOP_P,
                max_new_tokens=max_tokens,
            )
        elif "CustomVoice" in self.model_name:
            kwargs = dict(
                text=text,
                language="Chinese",
                speaker=self.speaker,
                temperature=self.DEFAULT_TEMPERATURE,
                top_k=self.DEFAULT_TOP_K,
                top_p=self.DEFAULT_TOP_P,
                max_new_tokens=max_tokens,
            )
            if self.instruct:
                kwargs["instruct"] = self.instruct
            return self.model.generate_custom_voice(**kwargs)
        else:
            raise RuntimeError(f"模型 {self.model_name} 不支持当前模式")

    @staticmethod
    def _audio_quality_ok(
        audio_data,
        sr: int,
        text: str,
        max_duration: float,
    ) -> Tuple[bool, str]:
        """检测音频质量。"""
        import numpy as np

        duration = len(audio_data) / sr

        if duration > max_duration:
            return False, f"时长 {duration:.2f}s 超过上限 {max_duration:.1f}s"

        content_chars = len(re.sub(r'[，。！？、；：\u201c\u201d\u2018\u2019（）…\s]', '', text))
        min_duration = max(0.3, content_chars * 0.1)
        if duration < min_duration:
            return False, f"时长 {duration:.2f}s 太短"

        frame_size = int(sr * 0.025)
        hop = int(sr * 0.010)
        frames = []
        for i in range(0, len(audio_data) - frame_size, hop):
            frame = audio_data[i:i + frame_size].astype(np.float64)
            rms = np.sqrt(np.mean(frame ** 2))
            frames.append(rms)

        if not frames:
            return False, "音频帧数为零"

        frames = np.array(frames)
        silence_ratio = np.mean(frames < 0.005)
        if silence_ratio > 0.7:
            return False, f"静音比例过高 {silence_ratio:.0%}"

        overall_rms = np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))
        if overall_rms < 0.001:
            return False, f"整体能量过低"

        if len(frames) > 10:
            energy_diff = np.abs(np.diff(frames))
            median_diff = np.median(energy_diff)
            if median_diff > 0:
                spike_ratio = np.mean(energy_diff > median_diff * 15)
                if spike_ratio > 0.15:
                    return False, f"能量突刺比例过高（疑似杂音）"

        return True, "OK"

    def synthesize(self, text: str, output_path: str) -> float:
        """
        合成单个句子的音频（含重试和质量检测）。

        :param text: 要合成的文本
        :param output_path: 输出 WAV 文件路径
        :return: 生成音频的时长（秒）
        """
        import soundfile as sf

        original_text = text
        text = normalize_text_for_tts(text)
        if text != original_text:
            print(f"    [预处理] {original_text[:50]} → {text[:50]}")

        text_max_duration = max(2.5, len(text) * self.MAX_SECONDS_PER_CHAR)

        best_audio = None
        best_sr = self.sample_rate
        best_duration = 0.0
        best_reason = ""
        reloaded_this_sentence = False

        for attempt in range(self.MAX_RETRIES):
            try:
                wavs, sr = self._generate_once(text)
                self.sample_rate = sr

                if not wavs or len(wavs) == 0:
                    best_reason = "模型未生成音频"
                    continue

                audio_data = wavs[0]

                # 跑飞检测
                if self._is_runaway_generation(audio_data, sr, text):
                    if not reloaded_this_sentence:
                        reloaded_this_sentence = True
                        self._reload_model()
                        self._sentences_since_reload = 0
                    best_reason = "EOS失败"
                    continue

                # 质量检测
                is_ok, reason = self._audio_quality_ok(
                    audio_data, sr, text, text_max_duration,
                )

                if is_ok:
                    audio_data, trimmed_ms = trim_leading_noise(audio_data, sr, text)
                    if trimmed_ms > 0:
                        print(f"    [头部清洗] 裁掉前 {trimmed_ms}ms 杂音")
                    sf.write(output_path, audio_data, sr)
                    duration = len(audio_data) / sr

                    # 预防性重载
                    if (
                        not reloaded_this_sentence
                        and self._needs_preemptive_reload(audio_data, sr, text)
                    ):
                        self._reload_model()
                        self._sentences_since_reload = 0
                        reloaded_this_sentence = True

                    # 定期重载
                    if not reloaded_this_sentence:
                        self._maybe_periodic_reload()

                    return duration
                else:
                    print(f"    [质量检测] 第 {attempt + 1} 次不合格: {reason}")
                    duration = len(audio_data) / sr
                    if best_audio is None or duration > best_duration:
                        best_audio = audio_data
                        best_sr = sr
                        best_duration = duration
                        best_reason = reason

            except Exception as e:
                print(f"    [重试] 第 {attempt + 1} 次生成异常: {e}")
                best_reason = str(e)

        # 兜底
        if best_audio is not None:
            print(f"    [兜底] 使用最佳结果（原因: {best_reason}）")
            best_audio, trimmed_ms = trim_leading_noise(best_audio, best_sr, text)
            if trimmed_ms > 0:
                print(f"    [头部清洗] 裁掉前 {trimmed_ms}ms 杂音")
            truncated_samples = int(text_max_duration * best_sr)
            if len(best_audio) > truncated_samples:
                best_audio = best_audio[:truncated_samples]
            sf.write(output_path, best_audio, best_sr)
            return len(best_audio) / best_sr

        print(f"[警告] TTS 合成完全失败: {text[:30]}...")
        return 0.0


# ============================================================
# edge-tts 引擎
# ============================================================

class EdgeTTSEngine:
    """edge-tts 语音合成引擎（微软在线 TTS）。"""

    def __init__(self, voice: str = "zh-CN-XiaoxiaoNeural"):
        self.voice = voice
        self.sample_rate = 24000
        print(f"[edge-tts] 使用语音: {voice}")

    def synthesize(self, text: str, output_path: str) -> float:
        """
        合成单个句子的音频。

        :param text: 要合成的文本
        :param output_path: 输出 WAV 文件路径
        :return: 生成音频的时长（秒）
        """
        import edge_tts

        try:
            mp3_path = output_path.rsplit(".", 1)[0] + ".mp3"

            communicate = edge_tts.Communicate(text=text, voice=self.voice)
            communicate.save_sync(mp3_path)

            cmd = [
                "ffmpeg", "-y",
                "-i", mp3_path,
                "-acodec", "pcm_s16le",
                "-ar", str(self.sample_rate),
                "-ac", "1",
                output_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[错误] MP3→WAV 转换失败: {result.stderr}", file=sys.stderr)
                return 0.0

            if os.path.exists(mp3_path):
                os.remove(mp3_path)

            return get_duration(output_path)

        except Exception as e:
            print(f"[错误] edge-tts 合成失败: {e}", file=sys.stderr)
            return 0.0
