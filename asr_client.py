"""
阿里云 DashScope 原生 ASR 客户端。
"""

from __future__ import annotations

import logging
import threading
import time
import wave
from pathlib import Path


logger = logging.getLogger(__name__)


class DashScopeASRClient:
    """基于 DashScope 原生流式 ASR SDK 的 wav 识别。"""

    def __init__(
        self,
        api_key: str,
        model: str = "gummy-chat-v1",
        frame_ms: int = 100,
        wait_timeout: float = 8.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.frame_ms = frame_ms
        self.wait_timeout = wait_timeout

    def transcribe_wav(self, wav_path: str | Path) -> str | None:
        audio_path = Path(wav_path)
        if not audio_path.exists():
            logger.error("ASR 输入音频不存在: %s", audio_path)
            return None
        if not self.api_key:
            logger.error("未配置 DASHSCOPE_API_KEY，无法执行 ASR")
            return None

        try:
            import dashscope
            from dashscope.audio.asr import TranslationRecognizerCallback, TranslationRecognizerChat
        except ImportError:
            logger.error("未安装 dashscope，无法执行 ASR")
            return None

        try:
            pcm_data, sample_rate = self._read_standard_wav(audio_path)
        except ValueError as exc:
            logger.error("%s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("读取 wav 失败: %s", exc)
            return None

        dashscope.api_key = self.api_key

        class ASRCallback(TranslationRecognizerCallback):
            def __init__(self) -> None:
                self.last_text = ""
                self.final_text = ""
                self.error_message = ""
                self.completed = threading.Event()

            def on_event(self, request_id, transcription_result, translation_result, usage):
                if not transcription_result:
                    return

                text = (getattr(transcription_result, "text", "") or "").strip()
                if not text:
                    return

                self.last_text = text
                if getattr(transcription_result, "is_sentence_end", False):
                    self.final_text = text
                    logger.info("ASR 识别到完整句子: %s", text)
                else:
                    logger.debug("ASR 中间结果: %s", text)

            def on_complete(self):
                self.completed.set()

            def on_error(self, result):
                self.error_message = str(result)
                self.completed.set()

            def on_close(self):
                self.completed.set()

        callback = ASRCallback()
        recognizer = TranslationRecognizerChat(
            model=self.model,
            format="pcm",
            sample_rate=sample_rate,
            transcription_enabled=True,
            translation_enabled=False,
            callback=callback,
        )

        try:
            recognizer.start()
            frame_bytes = max(1, int(sample_rate * 2 * self.frame_ms / 1000))
            for start in range(0, len(pcm_data), frame_bytes):
                frame = pcm_data[start : start + frame_bytes]
                if not recognizer.send_audio_frame(frame):
                    break
                # 按真实音频节奏推流，避免一次性灌入导致服务端端点判断不稳定。
                time.sleep(self.frame_ms / 1000)

            recognizer.stop()
            callback.completed.wait(self.wait_timeout)

            if callback.error_message:
                logger.error("ASR 调用失败: %s", callback.error_message)
                return None

            text = (callback.final_text or callback.last_text).strip()
            if not text:
                logger.warning("ASR 未返回可读文本")
                return None

            logger.info("ASR 识别成功: %s", text)
            return text
        except Exception as exc:
            logger.error("ASR 调用失败: %s", exc)
            return None

    @staticmethod
    def _read_standard_wav(audio_path: Path) -> tuple[bytes, int]:
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            compression = wav_file.getcomptype()

            if compression != "NONE":
                raise ValueError(f"ASR 仅支持 PCM wav，当前压缩类型: {compression}")
            if channels != 1:
                raise ValueError(f"ASR 仅支持单声道 wav，当前声道数: {channels}")
            if sample_width != 2:
                raise ValueError(f"ASR 仅支持 16-bit wav，当前采样宽度: {sample_width} bytes")
            if sample_rate != 16000:
                raise ValueError(f"ASR 当前要求 16kHz wav，当前采样率: {sample_rate}")

            return wav_file.readframes(wav_file.getnframes()), sample_rate
