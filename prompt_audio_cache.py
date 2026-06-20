"""
固定提示音缓存。

唤醒确认、没听清兜底这类短提示不需要每次都走在线流式 TTS。
第一次生成 wav 后缓存到本地，后续直接播放，可以降低响应延迟并提升听感稳定性。
"""

from __future__ import annotations

import hashlib
import logging
import struct
from pathlib import Path

from audio_output import AudioOutput
from settings import PROJECT_ROOT


logger = logging.getLogger(__name__)


class PromptTTSGenerator:
    """固定提示音专用的 DashScope wav 生成器。"""

    def __init__(self, api_key: str, model: str = "cosyvoice-v1", voice: str = "longwan") -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice

    def synthesize_to_wav(self, text: str, output_path: str | Path) -> bool:
        """生成短提示语 wav 缓存；主回答不走这里，仍使用 streaming_tts.py 真流式播放。"""
        if not text.strip():
            logger.error("提示音 TTS 输入文本为空")
            return False
        if not self.api_key:
            logger.error("未配置 DASHSCOPE_API_KEY，无法生成提示音")
            return False

        try:
            import dashscope
            from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer
        except ImportError:
            logger.error("未安装 dashscope，请先安装依赖")
            return False

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            dashscope.api_key = self.api_key
            synthesizer = SpeechSynthesizer(
                model=self.model,
                voice=self.voice,
                format=AudioFormat.WAV_22050HZ_MONO_16BIT,
            )
            audio_data = synthesizer.call(text)
            if not audio_data:
                logger.error("提示音 TTS 未返回有效音频数据")
                return False

            if hasattr(audio_data, "content"):
                audio_data = audio_data.content
            output_file.write_bytes(self._fix_wav_header(audio_data))

            logger.info("提示音缓存已生成: %s", output_file)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("提示音 TTS 合成失败: %s", exc)
            return False

    @staticmethod
    def _fix_wav_header(audio_data: bytes) -> bytes:
        """
        修正 DashScope WAV 返回值里的长度占位字段。

        部分 SDK 返回的 RIFF/data 长度是 0x7fffffff，占位值会让 Windows 播放器误判文件长度。
        """
        if not isinstance(audio_data, (bytes, bytearray)):
            return audio_data
        if len(audio_data) < 44:
            return bytes(audio_data)

        fixed = bytearray(audio_data)
        if fixed[0:4] != b"RIFF" or fixed[8:12] != b"WAVE":
            return bytes(fixed)

        struct.pack_into("<I", fixed, 4, len(fixed) - 8)

        offset = 12
        while offset + 8 <= len(fixed):
            chunk_id = fixed[offset : offset + 4]
            chunk_size = struct.unpack_from("<I", fixed, offset + 4)[0]
            data_start = offset + 8
            if chunk_id == b"data":
                struct.pack_into("<I", fixed, offset + 4, max(0, len(fixed) - data_start))
                break
            if chunk_size <= 0 or data_start + chunk_size > len(fixed):
                break
            offset = data_start + chunk_size + (chunk_size % 2)

        return bytes(fixed)


class PromptAudioCache:
    """固定短提示语的 wav 缓存与播放。"""

    def __init__(self, tts_generator: PromptTTSGenerator, audio_output: AudioOutput) -> None:
        self.tts_generator = tts_generator
        self.audio_output = audio_output
        self.cache_dir = PROJECT_ROOT / "data" / "runtime" / "prompt_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def play(self, text: str) -> bool:
        prompt = text.strip()
        if not prompt:
            return False

        wav_path = self.ensure(prompt)
        if not wav_path:
            return False
        logger.info("播放本地提示音缓存: %s", wav_path)
        return self.audio_output.play_wav(wav_path)

    def ensure(self, text: str) -> Path | None:
        """确保提示语已生成缓存，返回 wav 路径。"""
        prompt = text.strip()
        if not prompt:
            return None

        wav_path = self._cache_path(prompt)
        if wav_path.exists():
            return wav_path

        logger.info("首次生成提示音缓存: %s", prompt)
        if not self.tts_generator.synthesize_to_wav(prompt, wav_path):
            logger.warning("提示音缓存生成失败，将交给上层 fallback: %s", prompt)
            return None
        return wav_path

    def load_pcm(self, text: str, expected_sample_rate: int = 22050) -> bytes | None:
        """
        确保提示语缓存存在后，解析出 raw PCM（去 WAV 头，仅 data chunk）。

        应答词缓存用 WAV_22050HZ_MONO_16BIT 生成，与流式 TTS 的
        PCM_22050HZ_MONO_16BIT 同采样率/声道/位宽，可直接灌进同一条播放队列，
        绕开“cosyvoice 服务端攥住过短首片”导致的应答词延迟。
        """
        prompt = text.strip()
        if not prompt:
            return None
        wav_path = self.ensure(prompt)
        if not wav_path:
            return None
        try:
            pcm, sample_rate, channels, sample_width = self._extract_pcm(wav_path.read_bytes())
        except Exception as exc:  # noqa: BLE001
            logger.warning("应答词 PCM 解析失败 %s: %s", wav_path, exc)
            return None
        if not pcm:
            logger.warning("应答词 PCM 为空: %s", wav_path)
            return None
        if (sample_rate, channels, sample_width) != (expected_sample_rate, 1, 2):
            logger.warning(
                "应答词 PCM 格式不匹配（%sHz/%sch/%sB，期望 %sHz/1ch/2B），跳过本地预灌",
                sample_rate,
                channels,
                sample_width,
                expected_sample_rate,
            )
            return None
        return pcm

    @staticmethod
    def _extract_pcm(wav_bytes: bytes) -> tuple[bytes, int, int, int]:
        """从 WAV 字节解析出 (raw_pcm, sample_rate, channels, sample_width_bytes)。"""
        if len(wav_bytes) < 44 or wav_bytes[0:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
            raise ValueError("非法 WAV 头")
        sample_rate = channels = sample_width = 0
        pcm = b""
        offset = 12
        while offset + 8 <= len(wav_bytes):
            chunk_id = wav_bytes[offset : offset + 4]
            chunk_size = struct.unpack_from("<I", wav_bytes, offset + 4)[0]
            data_start = offset + 8
            if chunk_id == b"fmt ":
                channels = struct.unpack_from("<H", wav_bytes, data_start + 2)[0]
                sample_rate = struct.unpack_from("<I", wav_bytes, data_start + 4)[0]
                bits_per_sample = struct.unpack_from("<H", wav_bytes, data_start + 14)[0]
                sample_width = bits_per_sample // 8
            elif chunk_id == b"data":
                # DashScope 偶尔把 data 长度写成占位值，按文件实际剩余长度兜底。
                actual = len(wav_bytes) - data_start
                size = chunk_size if 0 < chunk_size <= actual else actual
                pcm = wav_bytes[data_start : data_start + size]
                break
            if chunk_size <= 0 or data_start + chunk_size > len(wav_bytes):
                break
            offset = data_start + chunk_size + (chunk_size % 2)
        return pcm, sample_rate, channels, sample_width

    def _cache_path(self, text: str) -> Path:
        cache_key = "|".join([self.tts_generator.model, self.tts_generator.voice, text])
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"prompt_{digest}.wav"
