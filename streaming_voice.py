"""
实时语音闭环编排模块。
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Iterator

from audio_output import AudioOutput
from llm_client import DeepSeekClient
from settings import PROJECT_ROOT, VoiceCliSettings
from tts_client import DashScopeTTSClient


logger = logging.getLogger(__name__)


class RealtimeASR:
    """用麦克风 PCM 流直接驱动 DashScope ASR。"""

    def __init__(self, api_key: str, model: str, sample_rate: int = 16000, chunk_frames: int = 3200) -> None:
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_frames = chunk_frames

    def listen_once(self, max_duration: int = 6) -> str | None:
        if not self.api_key:
            logger.error("未配置 DASHSCOPE_API_KEY，无法执行实时 ASR")
            return None

        try:
            import dashscope
            import pyaudio
            from dashscope.audio.asr import TranslationRecognizerCallback, TranslationRecognizerChat
        except ImportError as exc:
            logger.error("缺少实时麦克风依赖: %s", exc)
            logger.error("请先安装 pyaudio，例如 conda install -c conda-forge pyaudio")
            return None

        dashscope.api_key = self.api_key

        class Callback(TranslationRecognizerCallback):
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
                    logger.info("实时 ASR 完整句子: %s", text)
                else:
                    logger.debug("实时 ASR 中间结果: %s", text)

            def on_complete(self):
                self.completed.set()

            def on_error(self, result):
                self.error_message = str(result)
                self.completed.set()

            def on_close(self):
                self.completed.set()

        callback = Callback()
        recognizer = TranslationRecognizerChat(
            model=self.model,
            format="pcm",
            sample_rate=self.sample_rate,
            transcription_enabled=True,
            translation_enabled=False,
            callback=callback,
        )

        mic = pyaudio.PyAudio()
        stream = None
        try:
            stream = mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_frames,
            )

            recognizer.start()
            logger.info("实时 ASR 已启动，请开始说话，最长 %s 秒", max_duration)
            start_time = time.time()
            while time.time() - start_time < max_duration and not callback.completed.is_set():
                data = stream.read(self.chunk_frames, exception_on_overflow=False)
                if not recognizer.send_audio_frame(data):
                    logger.info("ASR 服务端提示句子结束")
                    break

            recognizer.stop()
            callback.completed.wait(8.0)

            if callback.error_message:
                logger.error("实时 ASR 失败: %s", callback.error_message)
                return None

            text = (callback.final_text or callback.last_text).strip()
            if not text:
                logger.warning("实时 ASR 未返回可读文本")
                return None
            return text
        except Exception as exc:  # noqa: BLE001
            logger.error("实时 ASR 过程失败: %s", exc)
            return None
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            mic.terminate()


class StreamingTTSPlayer:
    """把流式文本按句合成小 wav 并立即播放。"""

    def __init__(self, tts_client: DashScopeTTSClient, audio_output: AudioOutput) -> None:
        self.tts_client = tts_client
        self.audio_output = audio_output
        self.runtime_dir = PROJECT_ROOT / "data" / "runtime" / "streaming_tts"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def speak_stream(self, text_stream: Iterator[str]) -> str:
        full_text = ""
        buffer = ""
        index = 0

        for piece in text_stream:
            print(piece, end="", flush=True)
            full_text += piece
            buffer += piece

            while True:
                sentence, buffer = self._pop_sentence(buffer)
                if not sentence:
                    break
                index += 1
                self._synthesize_and_play(sentence, index)

        if buffer.strip():
            index += 1
            self._synthesize_and_play(buffer.strip(), index)
        print()
        return full_text.strip()

    def _synthesize_and_play(self, text: str, index: int) -> None:
        output_path = self.runtime_dir / f"tts_chunk_{index:03d}.wav"
        logger.info("流式 TTS 合成第 %s 段: %s", index, text)
        if self.tts_client.synthesize_to_wav(text, output_path):
            self.audio_output.play_wav(output_path)
        else:
            logger.error("流式 TTS 第 %s 段合成失败: %s", index, text)

    @staticmethod
    def _pop_sentence(buffer: str) -> tuple[str | None, str]:
        # 过短的逗号停顿不切分，避免 TTS 请求碎片化。
        match = re.search(r"[。！？.!?]", buffer)
        if match:
            end = match.end()
            return buffer[:end].strip(), buffer[end:]

        soft_match = re.search(r"[，,；;：:]", buffer)
        if soft_match and len(buffer[: soft_match.end()].strip()) >= 16:
            end = soft_match.end()
            return buffer[:end].strip(), buffer[end:]

        return None, buffer


class RealtimeVoiceLoop:
    """单轮实时语音闭环。"""

    def __init__(self, settings: VoiceCliSettings) -> None:
        self.asr = RealtimeASR(settings.dashscope_api_key, settings.asr_model)
        self.llm = DeepSeekClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
        )
        self.tts_player = StreamingTTSPlayer(
            DashScopeTTSClient(
                api_key=settings.dashscope_api_key,
                model=settings.tts_model,
                voice=settings.tts_voice,
            ),
            AudioOutput(),
        )

    def run_once(self, max_duration: int) -> int:
        user_text = self.asr.listen_once(max_duration=max_duration)
        if not user_text:
            logger.error("实时 ASR 未获得文本")
            return 3

        logger.info("用户输入文本: %s", user_text)
        logger.info("开始 DeepSeek 流式回复")
        reply = self.tts_player.speak_stream(self.llm.stream_reply(user_text))
        if not reply:
            logger.error("LLM 未生成有效回复")
            return 4

        logger.info("实时语音闭环完成")
        return 0
