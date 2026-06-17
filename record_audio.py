"""
固定时长麦克风录音模块。
"""

from __future__ import annotations

import ctypes
import logging
import platform
import subprocess
import time
import wave
from pathlib import Path


logger = logging.getLogger(__name__)


class AudioRecorder:
    """把麦克风录成标准 WAV。"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width

    def record_to_wav(self, duration_seconds: int, output_path: str | Path) -> Path | None:
        if duration_seconds <= 0:
            logger.error("录音时长必须大于 0")
            return None

        output_file = Path(output_path).expanduser().resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)

        system_name = platform.system().lower()
        if system_name == "windows":
            ok = self._record_windows(duration_seconds, output_file)
        else:
            ok = self._record_alsa(duration_seconds, output_file)

        return output_file if ok else None

    def _record_alsa(self, duration_seconds: int, output_file: Path) -> bool:
        """Linux / 板端优先复用 arecord。"""
        arecord_cmd = [
            "arecord",
            "-t",
            "wav",
            "-r",
            str(self.sample_rate),
            "-f",
            "S16_LE",
            "-c",
            str(self.channels),
            "-d",
            str(duration_seconds),
            str(output_file),
        ]

        recording_started = False
        try:
            logger.info("开始录音: %s 秒", duration_seconds)
            subprocess.run(arecord_cmd, check=True, capture_output=True, text=True)
            logger.info("录音完成: %s", output_file)
            return True
        except subprocess.CalledProcessError as exc:
            logger.error("录音失败: %s", exc.stderr or exc)
            return False

    def _record_windows(self, duration_seconds: int, output_file: Path) -> bool:
        """
        Windows 下使用 WinMM 录音接口，避免额外依赖。

        这里直接生成 16k/mono/16bit PCM WAV，方便后续 ASR 复用。
        """

        winmm = ctypes.windll.winmm

        WIM_DATA = 0x3C0
        CALLBACK_FUNCTION = 0x00030000
        WAVE_MAPPER = 0xFFFFFFFF
        MMSYSERR_NOERROR = 0
        BUFFER_MS = 250

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

        callback_type = ctypes.WINFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )

        # 显式声明 WinMM 签名，避免 64 位 Windows 句柄被 ctypes 默认截成 32 位整数。
        winmm.waveInOpen.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
            ctypes.POINTER(WAVEFORMATEX),
            callback_type,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        winmm.waveInOpen.restype = ctypes.c_uint
        winmm.waveInPrepareHeader.argtypes = [ctypes.c_void_p, ctypes.POINTER(WAVEHDR), ctypes.c_uint]
        winmm.waveInPrepareHeader.restype = ctypes.c_uint
        winmm.waveInAddBuffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(WAVEHDR), ctypes.c_uint]
        winmm.waveInAddBuffer.restype = ctypes.c_uint
        winmm.waveInStart.argtypes = [ctypes.c_void_p]
        winmm.waveInStart.restype = ctypes.c_uint
        winmm.waveInStop.argtypes = [ctypes.c_void_p]
        winmm.waveInStop.restype = ctypes.c_uint
        winmm.waveInReset.argtypes = [ctypes.c_void_p]
        winmm.waveInReset.restype = ctypes.c_uint
        winmm.waveInUnprepareHeader.argtypes = [ctypes.c_void_p, ctypes.POINTER(WAVEHDR), ctypes.c_uint]
        winmm.waveInUnprepareHeader.restype = ctypes.c_uint
        winmm.waveInClose.argtypes = [ctypes.c_void_p]
        winmm.waveInClose.restype = ctypes.c_uint

        class RecorderState:
            def __init__(self) -> None:
                self.chunks: list[bytes] = []
                self.active = True
                self.headers: list[WAVEHDR] = []
                self.buffers: list[ctypes.Array[ctypes.c_char]] = []
                self.wave_in = ctypes.c_void_p()
                self.callback = None

        state = RecorderState()
        bytes_per_frame = self.channels * self.sample_width
        buffer_frames = max(1, int(self.sample_rate * BUFFER_MS / 1000))
        buffer_bytes = buffer_frames * bytes_per_frame
        buffer_count = max(4, int(duration_seconds * 1000 / BUFFER_MS) + 2)

        fmt = WAVEFORMATEX(
            wFormatTag=1,
            nChannels=self.channels,
            nSamplesPerSec=self.sample_rate,
            nAvgBytesPerSec=self.sample_rate * bytes_per_frame,
            nBlockAlign=bytes_per_frame,
            wBitsPerSample=self.sample_width * 8,
            cbSize=0,
        )

        def _wave_in_proc(hwi, msg, instance, header_ptr, _dw2):
            if msg != WIM_DATA or not state.active:
                return
            try:
                header_ref = ctypes.cast(header_ptr, ctypes.POINTER(WAVEHDR))
                header = header_ref.contents
                if header.dwBytesRecorded:
                    state.chunks.append(ctypes.string_at(header.lpData, header.dwBytesRecorded))
                if state.active:
                    winmm.waveInAddBuffer(hwi, header_ref, ctypes.sizeof(WAVEHDR))
            except Exception as exc:  # noqa: BLE001
                logger.error("录音回调异常: %s", exc)

        state.callback = callback_type(_wave_in_proc)

        result = winmm.waveInOpen(
            ctypes.byref(state.wave_in),
            WAVE_MAPPER,
            ctypes.byref(fmt),
            state.callback,
            0,
            CALLBACK_FUNCTION,
        )
        if result != MMSYSERR_NOERROR:
            logger.error("打开麦克风失败，错误码: %s", result)
            return False

        try:
            for _ in range(buffer_count):
                buffer = ctypes.create_string_buffer(buffer_bytes)
                header = WAVEHDR()
                header.lpData = ctypes.cast(buffer, ctypes.c_void_p).value
                header.dwBufferLength = buffer_bytes
                header.dwBytesRecorded = 0
                header.dwUser = None
                header.dwFlags = 0
                header.dwLoops = 0
                header.lpNext = None
                header.reserved = None

                state.buffers.append(buffer)
                state.headers.append(header)

                result = winmm.waveInPrepareHeader(state.wave_in, ctypes.byref(header), ctypes.sizeof(WAVEHDR))
                if result != MMSYSERR_NOERROR:
                    logger.error("准备录音缓冲区失败，错误码: %s", result)
                    return False
                result = winmm.waveInAddBuffer(state.wave_in, ctypes.byref(header), ctypes.sizeof(WAVEHDR))
                if result != MMSYSERR_NOERROR:
                    logger.error("加入录音缓冲区失败，错误码: %s", result)
                    return False

            result = winmm.waveInStart(state.wave_in)
            if result != MMSYSERR_NOERROR:
                logger.error("启动录音失败，错误码: %s", result)
                return False

            logger.info("开始录音: %s 秒", duration_seconds)
            recording_started = True
            time.sleep(duration_seconds)
        except Exception as exc:  # noqa: BLE001
            logger.error("Windows 录音失败: %s", exc)
            return False
        finally:
            state.active = False
            winmm.waveInStop(state.wave_in)
            winmm.waveInReset(state.wave_in)
            for header in state.headers:
                winmm.waveInUnprepareHeader(state.wave_in, ctypes.byref(header), ctypes.sizeof(WAVEHDR))
            winmm.waveInClose(state.wave_in)

            if state.chunks:
                with wave.open(str(output_file), "wb") as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(self.sample_width)
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(b"".join(state.chunks))
                logger.info("录音完成: %s", output_file)
            else:
                logger.error("未采集到任何音频数据")

        return recording_started and bool(state.chunks) and output_file.exists()
