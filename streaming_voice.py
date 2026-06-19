"""
实时语音闭环编排模块。

RealtimeASR 用麦克风 PCM 流驱动 DashScope ASR（叠加本地 RMS VAD 兜底端点判断）；
RealtimeVoiceLoop 把 ASR、DeepSeek LLM、流式 TTS、提示音缓存和音频互斥锁组装成
一条可复用链路，供 wake_loop 调用。
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

try:
    import audioop  # Python 3.13 已按 PEP 594 移除，板端若用新版本需装 audioop-lts。
except ImportError:  # pragma: no cover - 取决于运行环境
    audioop = None

from audio_session import AudioSessionLock
from audio_output import AudioOutput
from llm_client import ChatMessage, DeepSeekClient
from prompt_audio_cache import PromptAudioCache, PromptTTSGenerator
from settings import VoiceCliSettings
from streaming_tts import DashScopeStreamingTTSPlayer


logger = logging.getLogger(__name__)


_SPECULATIVE_END = object()


class SpeculativeLLMStream:
    """
    本地 VAD 判定停声后提前启动的 LLM 文本流。

    它只做“取文本、不播声音”：最终 ASR 文本确认后，若和投机文本足够一致，上层再把
    已经缓存/仍在生成的文本片段交给 TTS；若不一致则丢弃，避免错答先播出来。
    """

    def __init__(
        self,
        llm: DeepSeekClient,
        user_text: str,
        history: list[ChatMessage] | None = None,
    ) -> None:
        self.llm = llm
        self.user_text = user_text.strip()
        self.history = list(history or [])
        self.started_at: float = time.time()
        self.first_token_at: float | None = None
        self._queue: queue.Queue[str | BaseException | object] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        try:
            for piece in self.llm.stream_reply(self.user_text, history=self.history):
                if self._stop_event.is_set():
                    break
                if piece and self.first_token_at is None:
                    self.first_token_at = time.time()
                    logger.info("投机 LLM 首 token 到达: %.3fs", self.first_token_at - self.started_at)
                if piece:
                    self._queue.put(piece)
        except Exception as exc:  # noqa: BLE001
            self._queue.put(exc)
        finally:
            self._queue.put(_SPECULATIVE_END)

    def iter_pieces(self) -> Iterator[str]:
        """把后台已缓存/正在生成的片段转成普通迭代器，供 TTS 复用。"""
        while True:
            item = self._queue.get()
            if item is _SPECULATIVE_END:
                return
            if isinstance(item, BaseException):
                raise RuntimeError(f"投机 LLM 预取失败: {item}") from item
            yield str(item)

    def discard(self) -> None:
        """终态 ASR 文本不一致时尽快停止消费投机流。"""
        self._stop_event.set()

    def is_compatible_with(self, final_text: str, threshold: float = 0.86) -> bool:
        """保守判断投机文本能否复用，宁可少命中，也不能错答先播。"""
        try:
            from difflib import SequenceMatcher
        except ImportError:  # pragma: no cover - 标准库理论上不会缺
            return self.user_text == final_text.strip()

        left = self._normalize_question(self.user_text)
        right = self._normalize_question(final_text)
        if not left or not right:
            return False
        if left == right:
            return True
        shorter, longer = sorted((left, right), key=len)
        if len(shorter) >= 4 and shorter in longer and len(longer) - len(shorter) <= 4:
            extra = longer.replace(shorter, "", 1)
            # 只允许“怎么 -> 怎么样”“要吗/呢/啊”这类语气尾巴差异复用。
            # 如果最终文本多出“公园/老人/室内”等实词约束，必须丢弃预取，避免抢时间但答错方向。
            return bool(extra) and self._is_safe_endpoint_extra(extra)
        if shorter in longer:
            return False
        return SequenceMatcher(None, left, right).ratio() >= threshold

    @staticmethod
    def _normalize_question(text: str) -> str:
        text = text.lower()
        return re.sub(r"[\s，。！？,.!?;；:：、\"'“”‘’（）()]", "", text)

    @staticmethod
    def _is_safe_endpoint_extra(extra: str) -> bool:
        """判断 ASR 最终文本比投机文本多出的尾巴是否只是语气/问句补全。"""
        safe_chars = set("样吗呢么嘛的了啊呀吧")
        return all(char in safe_chars for char in extra)


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
    vad_last_voice_time: float | None = None
    sentence_end_time: float | None = None
    endpoint_source: ASREndpointSource = "empty"
    # 投机相关：本地 VAD 停声那一刻的最佳识别文本，以及最终是否走了投机路径。
    # 上层据此把 LLM 首 token 等待与 ASR 云端收尾并行，并在终态文本不一致时校正。
    speculative_text: str | None = None
    is_endpoint_speculative: bool = False

    @property
    def utterance_end_time(self) -> float | None:
        """
        官方时延从“用户语句结束”开始计时，评委用秒表起点就是“嘴巴停下”。

        必须优先用纯本地 VAD 记录的最后一次有效语音时间（vad_last_voice_time），
        它只由麦克风 PCM 能量决定，不受云端吐字时间影响；而 last_voice_time 会被
        云端 partial 结果的到达时间往后拉，导致“用户说完 -> 云端判句尾”的等待被从
        指标里扣掉，日志因此明显低于人耳实测。仅当本地 VAD 全程未触发（如缺 audioop）
        时才退回 last_voice_time / 云端句尾时间。
        """
        return self.vad_last_voice_time or self.last_voice_time or self.sentence_end_time

    @property
    def endpoint_confirm_delay(self) -> float | None:
        """用户停声（本地 VAD）到云端句尾确认的耗时，解释体感时延与日志时延差异。"""
        base = self.vad_last_voice_time or self.last_voice_time
        if base is None or self.sentence_end_time is None:
            return None
        return max(0.0, self.sentence_end_time - base)


class RealtimeASR:
    """用麦克风 PCM 流直接驱动 DashScope ASR。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        sample_rate: int = 16000,
        chunk_frames: int = 1600,
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
        end_silence_timeout: float = 0.6,
        on_endpoint: Callable[[str], None] | None = None,
    ) -> ASRListenResult:
        """
        监听一次语音输入。

        max_duration 是硬上限；initial_silence_timeout 控制“最多等多久开口”；
        end_silence_timeout 控制识别到句尾后等待多久再收口，避免用户刚停顿就被截断。
        on_endpoint 在本地 VAD 判定用户停声那一刻被调用一次（带当前最佳识别文本），
        供上层投机式预发 LLM，把首 token 等待和 ASR 收尾并行。
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
                on_endpoint=on_endpoint,
            )

    def _listen_once_unlocked(
        self,
        max_duration: int = 6,
        initial_silence_timeout: float | None = None,
        end_silence_timeout: float = 1.0,
        on_endpoint: Callable[[str], None] | None = None,
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
            # 纯本地 VAD 的最后有效语音时间，只由麦克风 PCM 能量决定，不被云端吐字时间污染，
            # 作为官方时延 t 的诚实起点（评委秒表起点 = 用户嘴巴停下那一刻）。
            vad_last_voice_time: float | None = None
            endpoint_source: ASREndpointSource = "max_duration"
            voice_streak = 0
            endpoint_notified = False
            while time.time() - start_time < max_duration and not callback.completed.is_set():
                data = stream.read(self.chunk_frames, exception_on_overflow=False)
                now = time.time()
                if self._is_voice_frame(data):
                    voice_streak += 1
                    if first_voice_time is None and voice_streak >= 2:
                        first_voice_time = now
                        # ⑥ 首帧确认耗时：从开始监听到本地 VAD 认定用户开口。
                        logger.info("本地 VAD 检测到语音活动（开口确认耗时 %.3fs）", now - start_time)
                    if first_voice_time is not None:
                        last_voice_time = now
                        vad_last_voice_time = now
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
                # 本地端点必须用纯 VAD 的最后语音时间；last_voice_time 会被云端 partial
                # 到达时间刷新，继续用它会把 1s 静音窗口拖到云端识别之后，无法提前预取 LLM。
                endpoint_base_time = vad_last_voice_time or last_voice_time
                if (
                    first_voice_time is not None
                    and endpoint_base_time is not None
                    and now - endpoint_base_time >= end_silence_timeout
                ):
                    logger.info("本地静音超过 %.1f 秒，判定用户输入完成", end_silence_timeout)
                    best_text = (callback.final_text or callback.last_text).strip()
                    if on_endpoint and best_text and not endpoint_notified:
                        endpoint_notified = True
                        # 本地端点先到时可以投机启动 LLM；最终 ASR 文本返回后由上层决定是否采纳。
                        on_endpoint(best_text)
                    endpoint_source = "local_silence"
                    break

            # 先停止本地麦克风采集，再通知云端 ASR 收尾，避免后续 TTS 播放被继续采集。
            self._close_stream(stream)
            stream = None
            recognizer.stop()
            recognizer_started = False
            # ② ASR 收尾耗时：stop() 后等云端回最终结果，这段夹在官方时延 t 里，是隐藏大头。
            asr_finalize_start = time.time()
            completed_in_time = callback.completed.wait(8.0)
            logger.debug(
                "ASR 收尾耗时: %.3fs（端点=%s，云端%s在 8s 内完成）",
                time.time() - asr_finalize_start,
                endpoint_source,
                "已" if completed_in_time else "未",
            )
            if vad_last_voice_time is not None and callback.sentence_end_time is not None:
                # 用户停声（本地 VAD）到云端判句尾的等待：旧 t 起点被云端吐字时间往后拉，把这段排除在外，
                # 但评委秒表从“嘴巴停下”就开始算，所以它正是日志 t 与人耳实测（手机计时）的主要差值。
                logger.debug(
                    "用户停声->云端句尾确认延迟: %.3fs（旧日志 t 不含这段，但计入人耳实测）",
                    max(0.0, callback.sentence_end_time - vad_last_voice_time),
                )

            if callback.error_message:
                logger.error("实时 ASR 失败: %s", callback.error_message)
                return ASRListenResult(
                    first_voice_time=first_voice_time,
                    last_voice_time=last_voice_time,
                    vad_last_voice_time=vad_last_voice_time,
                    sentence_end_time=callback.sentence_end_time,
                    endpoint_source="error",
                )

            text = (callback.final_text or callback.last_text).strip()
            if not text:
                logger.warning("实时 ASR 未返回可读文本")
                return ASRListenResult(
                    first_voice_time=first_voice_time,
                    last_voice_time=last_voice_time,
                    vad_last_voice_time=vad_last_voice_time,
                    sentence_end_time=callback.sentence_end_time,
                    endpoint_source="empty",
                )
            return ASRListenResult(
                text=text,
                first_voice_time=first_voice_time,
                last_voice_time=last_voice_time,
                vad_last_voice_time=vad_last_voice_time,
                sentence_end_time=callback.sentence_end_time,
                endpoint_source=endpoint_source,
                speculative_text=text if endpoint_notified else None,
                is_endpoint_speculative=endpoint_notified,
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
        if audioop is None:
            # 缺少 audioop（如 Python 3.13 未装 audioop-lts）时退化为始终有效帧，
            # 本地 VAD 只是端点判断的兜底，云端 ASR 仍能正常工作。
            return True
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
        asr_endpoint_confirm_delay: float | None = None,
        prefetched_stream: SpeculativeLLMStream | None = None,
    ) -> str:
        """复用同一条流式 LLM + 分句 TTS 链路，供实时入口和持续监听入口调用。"""
        logger.info("用户输入文本: %s", user_text)
        if prefetched_stream:
            logger.debug("复用投机 LLM 预取流，投机文本: %s", prefetched_stream.user_text)
            text_stream = prefetched_stream.iter_pieces()
            llm_start_time = prefetched_stream.started_at
            llm_first_token_time = prefetched_stream.first_token_at
        else:
            logger.info("开始 DeepSeek 流式回复")
            text_stream = self.llm.stream_reply(user_text, history=history)
            llm_start_time = None
            llm_first_token_time = None
        with self.audio_session.speaking() as can_speak:
            if not can_speak:
                logger.error("音频设备正忙，无法播放回复")
                return ""
            reply = self.streaming_tts_player.speak_stream(
                text_stream,
                utterance_end_time=utterance_end_time,
                asr_endpoint_confirm_delay=asr_endpoint_confirm_delay,
                llm_start_time=llm_start_time,
                llm_first_token_time=llm_first_token_time,
            )
        if not reply:
            logger.error("LLM 未生成有效回复")
            return ""
        return reply

    def start_speculative_reply(
        self,
        user_text: str,
        history: list[ChatMessage] | None = None,
    ) -> SpeculativeLLMStream | None:
        """本地停声时提前启动 LLM；只预取文本，不占用播放设备。"""
        user_text = user_text.strip()
        if not user_text:
            return None
        if not self.llm.api_key:
            logger.warning("未配置 DeepSeek API Key，跳过投机 LLM 预取")
            return None
        logger.debug("本地端点触发投机 LLM 预取: %s", user_text)
        return SpeculativeLLMStream(self.llm, user_text, history=history)

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
