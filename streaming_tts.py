"""
DashScope 真流式 TTS 播放。

区别于旧的“分段合成 wav 再播放”，这里使用 SpeechSynthesizer 的 callback
接收 PCM 音频块，并通过连续输出流播放，减少段与段之间的空隙。
"""

from __future__ import annotations

import logging
import platform
import queue
import re
import shutil
import subprocess
import threading
import time
import ctypes
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field

from audio_output import get_sound_card_index, set_sound_mixer_command


logger = logging.getLogger(__name__)


@dataclass
class StreamingTTSMetrics:
    llm_start_time: float = 0.0
    utterance_end_time: float | None = None
    llm_first_token_time: float | None = None
    tts_first_audio_time: float | None = None
    playback_first_write_time: float | None = None
    text_done_time: float | None = None
    playback_done_time: float | None = None

    def log_summary(self) -> None:
        # 比赛要求显示“用户语句结束 -> TTS 播报第一个字”的时延。
        # 这里用首次写入播放设备近似“第一个字开始播报”，比 TTS 首包更接近用户听感。
        if self.utterance_end_time and self.playback_first_write_time:
            latency = max(0.0, self.playback_first_write_time - self.utterance_end_time)
            logger.info("系统响应时延 t=%.3fs，等级=%s", latency, self.competition_level(latency))
        if self.llm_first_token_time:
            logger.info("LLM 首 token 延迟: %.2fs", self.llm_first_token_time - self.llm_start_time)
        if self.tts_first_audio_time:
            base = self.text_done_time or self.llm_start_time
            logger.info("TTS 首音频包延迟: %.2fs", self.tts_first_audio_time - base)
        if self.playback_first_write_time:
            logger.info("首次写入音频输出延迟: %.2fs", self.playback_first_write_time - self.llm_start_time)
        if self.playback_done_time:
            logger.info("流式 TTS 总耗时: %.2fs", self.playback_done_time - self.llm_start_time)

    @staticmethod
    def competition_level(latency: float) -> str:
        if latency <= 2.0:
            return "优秀"
        if latency <= 4.0:
            return "及格"
        return "不及格"


@dataclass
class StreamingTTSConfig:
    sample_rate: int = 22050
    channels: int = 1
    sample_width_bytes: int = 2
    first_flush_chars: int = 10
    min_flush_chars: int = 16
    long_flush_chars: int = 40
    playback_buffer_ms: int = 120
    queue_timeout_seconds: float = 10.0
    linux_card_name: str = "lahainayupikiot"
    metrics: StreamingTTSMetrics = field(default_factory=StreamingTTSMetrics)


class PCMStreamPlayer:
    """跨平台连续 PCM 播放后端。"""

    def __init__(self, config: StreamingTTSConfig) -> None:
        self.config = config
        self._system = platform.system().lower()
        self._pyaudio = None
        self._stream = None
        self._aplay_process = None
        self._waveout = None

    def open(self) -> bool:
        if self._system == "windows":
            return self._open_windows()
        if self._system == "linux":
            return self._open_linux()
        logger.error("当前系统暂不支持 PCM 连续播放: %s", self._system)
        return False

    def write(self, pcm_data: bytes) -> bool:
        if not pcm_data:
            return True
        try:
            if self._system == "windows" and self._waveout:
                return self._waveout.write(pcm_data)
            if self._system == "windows" and self._stream:
                self._stream.write(pcm_data)
                return True
            if self._system == "linux" and self._aplay_process and self._aplay_process.stdin:
                if self._aplay_process.poll() is not None:
                    logger.error("aplay 进程已退出")
                    return False
                self._aplay_process.stdin.write(pcm_data)
                self._aplay_process.stdin.flush()
                return True
        except Exception as exc:  # noqa: BLE001
            logger.error("PCM 连续播放写入失败: %s", exc)
            return False
        return False

    def close(self, drain: bool = True) -> None:
        if self._system == "windows":
            with suppress(Exception):
                if self._waveout:
                    self._waveout.close(drain=drain)
            with suppress(Exception):
                if self._stream:
                    self._stream.stop_stream()
            with suppress(Exception):
                if self._stream:
                    self._stream.close()
            with suppress(Exception):
                if self._pyaudio:
                    self._pyaudio.terminate()
            self._stream = None
            self._pyaudio = None
            self._waveout = None
            return

        if self._system == "linux":
            proc = self._aplay_process
            self._aplay_process = None
            if not proc:
                return
            with suppress(Exception):
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

    def _open_windows(self) -> bool:
        self._waveout = WindowsWaveOutPCMPlayer(
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            sample_width_bytes=self.config.sample_width_bytes,
        )
        if self._waveout.open():
            return True
        self._waveout = None
        logger.warning("Windows waveOut 打开失败，尝试 PyAudio fallback")
        try:
            import pyaudio

            self._pyaudio = pyaudio.PyAudio()
            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                output=True,
                frames_per_buffer=1024,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("打开 Windows PyAudio 输出失败: %s", exc)
            self.close(drain=False)
            return False

    def _open_linux(self) -> bool:
        if not shutil.which("aplay"):
            logger.error("未找到 aplay，无法进行 Linux PCM 连续播放")
            return False

        card = get_sound_card_index(self.config.linux_card_name)
        if card:
            # 与旧 wav 播放路径保持一致，先配置板端声卡通路再写入 raw PCM。
            set_sound_mixer_command(card)
        device_args = ["-D", f"hw:{card},0"] if card else []
        cmd = [
            "aplay",
            "-t",
            "raw",
            "-r",
            str(self.config.sample_rate),
            "-f",
            "S16_LE",
            "-c",
            str(self.config.channels),
            *device_args,
        ]
        try:
            self._aplay_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("启动 aplay 连续播放失败: %s", exc)
            self.close()
            return False


class WindowsWaveOutPCMPlayer:
    """Windows 原生 waveOut PCM 播放后端，避免 PyAudio stream.write 兼容性问题。"""

    CALLBACK_NULL = 0
    WAVE_MAPPER = 0xFFFFFFFF
    WHDR_DONE = 0x00000001

    class WAVEFORMATEX(ctypes.Structure):
        _fields_ = [
            ("wFormatTag", ctypes.c_ushort),
            ("nChannels", ctypes.c_ushort),
            ("nSamplesPerSec", ctypes.c_uint),
            ("nAvgBytesPerSec", ctypes.c_uint),
            ("nBlockAlign", ctypes.c_ushort),
            ("wBitsPerSample", ctypes.c_ushort),
            ("cbSize", ctypes.c_ushort),
        ]

    class WAVEHDR(ctypes.Structure):
        _fields_ = [
            ("lpData", ctypes.c_void_p),
            ("dwBufferLength", ctypes.c_uint),
            ("dwBytesRecorded", ctypes.c_uint),
            ("dwUser", ctypes.c_void_p),
            ("dwFlags", ctypes.c_uint),
            ("dwLoops", ctypes.c_uint),
            ("lpNext", ctypes.c_void_p),
            ("reserved", ctypes.c_void_p),
        ]

    def __init__(self, sample_rate: int, channels: int, sample_width_bytes: int) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width_bytes = sample_width_bytes
        self._handle = ctypes.c_void_p()
        self._winmm = ctypes.WinDLL("winmm")
        self._pending_headers: list[tuple[ctypes.Array, WindowsWaveOutPCMPlayer.WAVEHDR]] = []

    def open(self) -> bool:
        bits_per_sample = self.sample_width_bytes * 8
        fmt = self.WAVEFORMATEX(
            1,
            self.channels,
            self.sample_rate,
            self.sample_rate * self.channels * self.sample_width_bytes,
            self.channels * self.sample_width_bytes,
            bits_per_sample,
            0,
        )
        result = self._winmm.waveOutOpen(
            ctypes.byref(self._handle),
            ctypes.c_uint(self.WAVE_MAPPER),
            ctypes.byref(fmt),
            0,
            0,
            self.CALLBACK_NULL,
        )
        if result != 0:
            logger.error("waveOutOpen 失败，错误码: %s", result)
            return False
        return True

    def write(self, pcm_data: bytes) -> bool:
        if not self._handle:
            return False
        self._cleanup_done_headers()
        while len(self._pending_headers) >= 8:
            self._cleanup_done_headers()
            time.sleep(0.01)

        buffer = ctypes.create_string_buffer(pcm_data)
        header = self.WAVEHDR(
            ctypes.cast(buffer, ctypes.c_void_p),
            len(pcm_data),
            0,
            None,
            0,
            0,
            None,
            None,
        )
        result = self._winmm.waveOutPrepareHeader(self._handle, ctypes.byref(header), ctypes.sizeof(header))
        if result != 0:
            logger.error("waveOutPrepareHeader 失败，错误码: %s", result)
            return False

        result = self._winmm.waveOutWrite(self._handle, ctypes.byref(header), ctypes.sizeof(header))
        if result != 0:
            logger.error("waveOutWrite 失败，错误码: %s", result)
            with suppress(Exception):
                self._winmm.waveOutUnprepareHeader(self._handle, ctypes.byref(header), ctypes.sizeof(header))
            return False

        # 保留 buffer/header 生命周期，让 waveOut 自己排队连续播放。
        self._pending_headers.append((buffer, header))
        return True

    def close(self, drain: bool = True) -> None:
        if not self._handle:
            return
        if drain:
            self._wait_all_done()
        else:
            with suppress(Exception):
                self._winmm.waveOutReset(self._handle)
        self._cleanup_done_headers()
        with suppress(Exception):
            self._winmm.waveOutClose(self._handle)
        self._handle = ctypes.c_void_p()
        self._pending_headers.clear()

    def _cleanup_done_headers(self) -> None:
        active: list[tuple[ctypes.Array, WindowsWaveOutPCMPlayer.WAVEHDR]] = []
        for buffer, header in self._pending_headers:
            if header.dwFlags & self.WHDR_DONE:
                with suppress(Exception):
                    self._winmm.waveOutUnprepareHeader(self._handle, ctypes.byref(header), ctypes.sizeof(header))
            else:
                active.append((buffer, header))
        self._pending_headers = active

    def _wait_all_done(self) -> None:
        while any(not (header.dwFlags & self.WHDR_DONE) for _, header in self._pending_headers):
            self._cleanup_done_headers()
            time.sleep(0.01)


class DashScopeStreamingTTSPlayer:
    """LLM 文本流到 DashScope TTS 音频流的桥接器。"""

    def __init__(self, api_key: str, model: str, voice: str, config: StreamingTTSConfig | None = None) -> None:
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.config = config or StreamingTTSConfig()
        self.audio_queue: queue.Queue[tuple[str, bytes | str | None]] = queue.Queue()
        self.interrupt_event = threading.Event()
        self._playback_thread: threading.Thread | None = None
        self._player: PCMStreamPlayer | None = None

    def speak_stream(self, text_stream: Iterator[str], utterance_end_time: float | None = None) -> str:
        if not self.api_key:
            logger.error("未配置 DASHSCOPE_API_KEY，无法执行流式 TTS")
            return ""

        self.interrupt_event.clear()
        self._clear_audio_queue()
        self.config.metrics = StreamingTTSMetrics(
            llm_start_time=time.time(),
            utterance_end_time=utterance_end_time,
        )

        if not self._start_playback_thread():
            return ""

        full_text = ""
        buffer = ""
        has_flushed_text = False
        synthesizer = None
        try:
            import dashscope
            from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer

            dashscope.api_key = self.api_key

            player = self

            class Callback(ResultCallback):
                def on_data(self, data: bytes) -> None:
                    if player.interrupt_event.is_set():
                        return
                    if player.config.metrics.tts_first_audio_time is None:
                        player.config.metrics.tts_first_audio_time = time.time()
                    player.audio_queue.put(("audio", bytes(data)))

                def on_error(self, message) -> None:
                    player.audio_queue.put(("error", str(message)))

                def on_complete(self) -> None:
                    player.audio_queue.put(("end", None))

            synthesizer = SpeechSynthesizer(
                model=self.model,
                voice=self.voice,
                format=AudioFormat.PCM_22050HZ_MONO_16BIT,
                callback=Callback(),
            )

            for piece in text_stream:
                if self.interrupt_event.is_set():
                    break
                if piece and self.config.metrics.llm_first_token_time is None:
                    self.config.metrics.llm_first_token_time = time.time()
                print(piece, end="", flush=True)
                full_text += piece
                buffer += piece
                while True:
                    chunk, buffer = self._pop_flush_chunk(buffer, has_flushed_text)
                    if not chunk:
                        break
                    has_flushed_text = True
                    logger.info("流式 TTS 发送文本片段: %s", chunk)
                    synthesizer.streaming_call(chunk)
                    with suppress(Exception):
                        synthesizer.streaming_flush()

            if buffer.strip() and not self.interrupt_event.is_set():
                logger.info("流式 TTS 发送尾段: %s", buffer.strip())
                synthesizer.streaming_call(buffer.strip())

            self.config.metrics.text_done_time = time.time()
            if not self.interrupt_event.is_set():
                synthesizer.streaming_complete()
            print()
        except Exception as exc:  # noqa: BLE001
            logger.error("流式 TTS 失败: %s", exc)
            self.audio_queue.put(("error", str(exc)))
        finally:
            if self.interrupt_event.is_set() and synthesizer is not None:
                with suppress(Exception):
                    synthesizer.close()
            self._wait_playback_done()
            self.config.metrics.playback_done_time = time.time()
            self.config.metrics.log_summary()

        return full_text.strip()

    def speak_text(self, text: str) -> bool:
        return bool(self.speak_stream(iter([text.strip()])))

    def interrupt(self) -> None:
        self.interrupt_event.set()
        self._clear_audio_queue()
        self.audio_queue.put(("interrupt", None))
        if self._player:
            self._player.close(drain=False)

    def _start_playback_thread(self) -> bool:
        self._player = PCMStreamPlayer(self.config)
        if not self._player.open():
            return False
        self._playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self._playback_thread.start()
        return True

    def _playback_worker(self) -> None:
        assert self._player is not None
        pending_audio = bytearray()
        bytes_per_second = self.config.sample_rate * self.config.channels * self.config.sample_width_bytes
        min_buffer_bytes = max(1, int(bytes_per_second * self.config.playback_buffer_ms / 1000))
        try:
            while not self.interrupt_event.is_set():
                try:
                    item_type, data = self.audio_queue.get(timeout=self.config.queue_timeout_seconds)
                except queue.Empty:
                    logger.warning("流式 TTS 音频队列等待超时")
                    break
                if item_type == "audio" and isinstance(data, bytes):
                    pending_audio.extend(data)
                    # DashScope callback 可能给很碎的小 PCM 块，先攒到很短的播放缓冲再写入，
                    # 兼顾首字延迟和 Windows waveOut 小块播放的稳定性。
                    if len(pending_audio) < min_buffer_bytes:
                        continue
                    if not self._write_pending_audio(pending_audio):
                        break
                elif item_type == "end":
                    if pending_audio:
                        self._write_pending_audio(pending_audio)
                    break
                elif item_type == "interrupt":
                    break
                elif item_type == "error":
                    logger.error("流式 TTS 播放收到错误: %s", data)
                    break
        finally:
            self._player.close()

    def _write_pending_audio(self, pending_audio: bytearray) -> bool:
        """合并小 PCM 块后再写入，减少 Windows 逐块播放的碎裂感。"""
        assert self._player is not None
        if not pending_audio:
            return True
        data = bytes(pending_audio)
        pending_audio.clear()
        if self.config.metrics.playback_first_write_time is None:
            self.config.metrics.playback_first_write_time = time.time()
        return self._player.write(data)

    def _wait_playback_done(self) -> None:
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=30)
            if self._playback_thread.is_alive():
                logger.warning("流式 TTS 播放线程结束超时，执行中断")
                self.interrupt()
                self._playback_thread.join(timeout=3)

    def _clear_audio_queue(self) -> None:
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _pop_flush_chunk(self, buffer: str, has_flushed_text: bool) -> tuple[str | None, str]:
        chunk, rest = self._pop_sentence(buffer, has_flushed_text)
        if chunk:
            return chunk, rest
        limit = self.config.long_flush_chars
        if not has_flushed_text:
            # 首段更激进，优先尽快送入 TTS；后续段落更保守，减少段间碎裂。
            limit = min(limit, self.config.first_flush_chars + 8)
        if len(buffer.strip()) >= limit:
            return buffer.strip(), ""
        return None, buffer

    def _pop_sentence(self, buffer: str, has_flushed_text: bool) -> tuple[str | None, str]:
        match = re.search(r"[。！？.!?]", buffer)
        if match and len(buffer[: match.end()].strip()) >= self.config.first_flush_chars:
            end = match.end()
            return buffer[:end].strip(), buffer[end:]

        min_chars = self.config.min_flush_chars if has_flushed_text else self.config.first_flush_chars
        hard_pause = re.search(r"[；;]", buffer)
        if hard_pause and len(buffer[: hard_pause.end()].strip()) >= min_chars:
            end = hard_pause.end()
            return buffer[:end].strip(), buffer[end:]

        soft_pause = re.search(r"[，,：:]", buffer)
        if soft_pause and len(buffer[: soft_pause.end()].strip()) >= min_chars:
            end = soft_pause.end()
            return buffer[:end].strip(), buffer[end:]

        return None, buffer
