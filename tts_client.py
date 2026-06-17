"""
阿里云 TTS 客户端。
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path


logger = logging.getLogger(__name__)


class DashScopeTTSClient:
    """阿里云 CosyVoice 文本转语音。"""

    def __init__(self, api_key: str, model: str = "cosyvoice-v1", voice: str = "longwan") -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice

    def synthesize_to_wav(self, text: str, output_path: str | Path) -> bool:
        if not text.strip():
            logger.error("TTS 输入文本为空")
            return False
        if not self.api_key:
            logger.error("未配置 DASHSCOPE_API_KEY，无法执行 TTS")
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
                logger.error("TTS 未返回有效音频数据")
                return False

            # DashScope 在非流式模式下会直接返回 wav 字节流。
            if hasattr(audio_data, "content"):
                audio_data = audio_data.content
            audio_data = self._fix_wav_header(audio_data)
            output_file.write_bytes(audio_data)

            logger.info("TTS 音频已生成: %s", output_file)
            return True
        except Exception as exc:
            logger.error("TTS 合成失败: %s", exc)
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
