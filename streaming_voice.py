"""
实时语音闭环编排模块。
"""

from __future__ import annotations

import logging
import threading
import time
import audioop
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

from audio_session import AudioSessionLock
from audio_output import AudioOutput
from llm_client import ChatMessage, DeepSeekClient
from prompt_audio_cache import PromptAudioCache, PromptTTSGenerator
from settings import VoiceCliSettings
from streaming_tts import DashScopeStreamingTTSPlayer


logger = logging.getLogger(__name__)


ASREndpointSource = Literal[
    "server_sentence_end",
    "local_silence",
    "initial_silence_timeout",
    "max_duration",
    "empty",
    "error",
]


@dataclass
class ASRListenResult:
    """一次实时 ASR 监听结果，包含竞赛时延统计需要的端点时间。"""

    text: str | None = None
    first_voice_time: float | None = None
    last_voice_time: float | None = None
    sentence_end_time: float | None = None
    endpoint_source: ASREndpointSource = "empty"

    @property
    def utterance_end_time(self) -> float | None:
        """
        官方时延从“用户语句结束”开始计时。

        云端主动判句尾时优先使用云端时间；本地静音兜底时用最后一次有效语音时间，
        避免把等待尾静音的时间从比赛指标里“藏掉”。
        """
        if self.endpoint_source == "server_sentence_end" and self.sentence_end_time is not None:
            return self.sentence_end_time
        return self.last_voice_time


class RealtimeASR:
    """用麦克风 PCM 流直接驱动 DashScope ASR。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        sample_rate: int = 16000,
        chunk_frames: int = 3200,
        audio_session: AudioSessionLock | None = None,
        vad_rms_threshold: int = 500,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_frames = chunk_frames
        self.audio_session = audio_session
        self.vad_rms_threshold = vad_rms_threshold

    def listen_once(
        self,
        max_duration: int = 6,
        initial_silence_timeout: float | None = None,
        end_silence_timeout: float = 1.0,
    ) -> ASRListenResult:
        """
        监听一次语音输入。

        max_duration 是硬上限；initial_silence_timeout 控制“最多等多久开口”；
        end_silence_timeout 控制识别到句尾后等待多久再收口，避免用户刚停顿就被截断。
        """
        if not self.api_key:
            logger.error("未配置 DASHSCOPE_API_KEY，无法执行实时 ASR")
            return ASRListenResult(endpoint_source="error")

        recording_context = self.audio_session.recording() if self.audio_session else suppress(True)
        with recording_context as can_record:
            if not can_record:
                logger.warning("当前音频设备正在播放，跳过本次录音")
                return ASRListenResult(endpoint_source="error")
            return self._listen_once_unlocked(
                max_duration=max_duration,
                initial_silence_timeout=initial_silence_timeout,
                end_silence_timeout=end_silence_timeout,
            )

    def _listen_once_unlocked(
        self,
        max_duration: int = 6,
        initial_silence_timeout: float | None = None,
        end_silence_timeout: float = 1.0,
    ) -> ASRListenResult:
        try:
            import dashscope
            import pyaudio
            from dashscope.audio.asr import TranslationRecognizerCallback, TranslationRecognizerChat
        except ImportError as exc:
            logger.error("缺少实时麦克风依赖: %s", exc)
            logger.error("请先安装 pyaudio，例如 conda install -c conda-forge pyaudio")
            return ASRListenResult(endpoint_source="error")

        dashscope.api_key = self.api_key

        class Callback(TranslationRecognizerCallback):
            def __init__(self) -> None:
                self.last_text = ""
                self.final_text = ""
                self.error_message = ""
                self.completed = threading.Event()
                self.first_text_time: float | None = None
                self.last_text_time: float | None = None
                self.sentence_end_time: float | None = None

            def on_event(self, request_id, transcription_result, translation_result, usage):
                if not transcription_result:
                    return
                text = (getattr(transcription_result, "text", "") or "").strip()
                if not text:
                    return
                now = time.time()
                if self.first_text_time is None:
                    self.first_text_time = now
                self.last_text_time = now
                self.last_text = text
                if getattr(transcription_result, "is_sentence_end", False):
                    self.final_text = text
                    self.sentence_end_time = now
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
        recognizer_started = False
        try:
            stream = mic.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_frames,
            )

            recognizer.start()
            recognizer_started = True
            logger.info("实时 ASR 已启动，请开始说话，最长 %s 秒", max_duration)
            start_time = time.time()
            first_voice_time: float | None = None
            last_voice_time: float | None = None
            endpoint_source: ASREndpointSource = "max_duration"
            voice_streak = 0
            while time.time() - start_time < max_duration and not callback.completed.is_set():
                data = stream.read(self.chunk_frames, exception_on_overflow=False)
                now = time.time()
                if self._is_voice_frame(data):
                    voice_streak += 1
                    if first_voice_time is None and voice_streak >= 2:
                        first_voice_time = now
                        logger.info("本地 VAD 检测到语音活动")
                    if first_voice_time is not None:
                        last_voice_time = now
                else:
                    voice_streak = 0

                if not recognizer.send_audio_frame(data):
                    logger.info("ASR 服务端提示句子结束")
                    endpoint_source = "server_sentence_end"
                    if callback.sentence_end_time is None:
                        callback.sentence_end_time = time.time()
                    break
                if callback.sentence_end_time is not None:
                    logger.info("ASR 服务端提示句子结束")
                    endpoint_source = "server_sentence_end"
                    break
                if callback.first_text_time is not None and first_voice_time is None:
                    # 旧项目 VAD 会综合识别结果 reason；这里把云端已吐字也视为“用户已开口”，
                    # 避免轻声或设备增益偏低时本地 RMS 没过阈值，被开头静音规则提前截断。
                    first_voice_time = callback.first_text_time
                    last_voice_time = callback.last_text_time or now
                    logger.info("ASR 已返回文本，按有效语音继续监听")
                elif callback.last_text_time is not None and first_voice_time is not None:
                    last_voice_time = max(last_voice_time or callback.last_text_time, callback.last_text_time)
                if (
                    initial_silence_timeout is not None
                    and first_voice_time is None
                    and now - start_time >= initial_silence_timeout
                ):
                    logger.info("开头静音超过 %.1f 秒，结束本次监听", initial_silence_timeout)
                    endpoint_source = "initial_silence_timeout"
                    break
                if (
                    first_voice_time is not None
                    and last_voice_time is not None
                    and now - last_voice_time >= end_silence_timeout
                ):
                    logger.info("本地静音超过 %.1f 秒，判定用户输入完成", end_silence_timeout)
                    endpoint_source = "local_silence"
                    break

            # 先停止本地麦克风采集，再通知云端 ASR 收尾，避免后续 TTS 播放被继续采集。
            self._close_stream(stream)
            stream = None
            recognizer.stop()
            recognizer_started = False
            callback.completed.wait(8.0)

            if callback.error_message:
                logger.error("实时 ASR 失败: %s", callback.error_message)
                return ASRListenResult(
                    first_voice_time=first_voice_time,
                    last_voice_time=last_voice_time,
                    sentence_end_time=callback.sentence_end_time,
                    endpoint_source="error",
                )

            text = (callback.final_text or callback.last_text).strip()
            if not text:
                logger.warning("实时 ASR 未返回可读文本")
                return ASRListenResult(
                    first_voice_time=first_voice_time,
                    last_voice_time=last_voice_time,
                    sentence_end_time=callback.sentence_end_time,
                    endpoint_source="empty",
                )
            return ASRListenResult(
                text=text,
                first_voice_time=first_voice_time,
                last_voice_time=last_voice_time,
                sentence_end_time=callback.sentence_end_time,
                endpoint_source=endpoint_source,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("实时 ASR 过程失败: %s", exc)
            return ASRListenResult(endpoint_source="error")
        finally:
            if stream:
                self._close_stream(stream)
            if recognizer_started:
                with suppress(Exception):
                    recognizer.stop()
                callback.completed.wait(2.0)
            with suppress(Exception):
                mic.terminate()

    @staticmethod
    def _close_stream(stream) -> None:
        """兼容 Windows 句柄释放：关闭前先停流，单项失败不影响后续清理。"""
        with suppress(Exception):
            if stream.is_active():
                stream.stop_stream()
        with suppress(Exception):
            stream.close()

    def _is_voice_frame(self, data: bytes) -> bool:
        """用本地 PCM 能量做轻量 VAD，避免 ASR 延迟吐字导致误判开头静音。"""
        if not data:
            return False
        try:
            return audioop.rms(data, 2) >= self.vad_rms_threshold
        except Exception:  # noqa: BLE001
            return False


class RealtimeVoiceLoop:
    """持续唤醒流程复用的实时语音编排。"""

    def __init__(self, settings: VoiceCliSettings) -> None:
        self.audio_session = AudioSessionLock()
        self.asr = RealtimeASR(
            settings.dashscope_api_key,
            settings.asr_model,
            audio_session=self.audio_session,
        )
        self.llm = DeepSeekClient(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
        )
        self.prompt_tts_generator = PromptTTSGenerator(
            api_key=settings.dashscope_api_key,
            model=settings.tts_model,
            voice=settings.tts_voice,
        )
        self.audio_output = AudioOutput()
        self.streaming_tts_player = DashScopeStreamingTTSPlayer(
            api_key=settings.dashscope_api_key,
            model=settings.tts_model,
            voice=settings.tts_voice,
        )
        self.prompt_audio_cache = PromptAudioCache(
            self.prompt_tts_generator,
            self.audio_output,
        )

    def answer_text(
        self,
        user_text: str,
        history: list[ChatMessage] | None = None,
        utterance_end_time: float | None = None,
    ) -> str:
        """复用同一条流式 LLM + 分句 TTS 链路，供实时入口和持续监听入口调用。"""
        logger.info("用户输入文本: %s", user_text)
        logger.info("开始 DeepSeek 流式回复")
        with self.audio_session.speaking() as can_speak:
            if not can_speak:
                logger.error("音频设备正忙，无法播放回复")
                return ""
            reply = self.streaming_tts_player.speak_stream(
                self.llm.stream_reply(user_text, history=history),
                utterance_end_time=utterance_end_time,
            )
        if not reply:
            logger.error("LLM 未生成有效回复")
            return ""
        return reply

    def speak_text(self, text: str) -> bool:
        """播放固定提示语，不经过 LLM，优先使用本地 wav 缓存。"""
        if not text.strip():
            return False
        with self.audio_session.speaking() as can_speak:
            if not can_speak:
                logger.error("音频设备正忙，无法播放提示语")
                return False
            if self.prompt_audio_cache.play(text):
                return True
            logger.warning("提示音缓存播放失败，降级为在线流式 TTS")
            spoken = self.streaming_tts_player.speak_stream(iter([text.strip()]))
        return bool(spoken)

    def warmup_prompt_audio(self, *texts: str) -> None:
        """预生成常用提示音，避免唤醒后第一次才等待在线 TTS。"""
        for text in texts:
            if text.strip():
                self.prompt_audio_cache.ensure(text)

    def interrupt_tts(self) -> None:
        """外部中断当前 TTS，用于 Ctrl+C 或后续唤醒打断。"""
        self.streaming_tts_player.interrupt()
        self.audio_session.force_idle()
