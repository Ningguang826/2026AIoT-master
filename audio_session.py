"""
最小音频会话锁。

当前 CLI 只需要防止“录音”和“TTS 播放”同时占用设备，不引入旧项目完整
优先级系统。后续如果要支持播放中唤醒打断，再扩展这里。
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from enum import Enum
from typing import Iterator


logger = logging.getLogger(__name__)


class AudioSessionState(Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    SPEAKING = "SPEAKING"


class AudioSessionLock:
    """录音/播放互斥的轻量状态锁。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state = AudioSessionState.IDLE

    @property
    def state(self) -> AudioSessionState:
        with self._lock:
            return self._state

    @contextmanager
    def recording(self) -> Iterator[bool]:
        entered = self._enter(AudioSessionState.RECORDING)
        try:
            yield entered
        finally:
            if entered:
                self._leave(AudioSessionState.RECORDING)

    @contextmanager
    def speaking(self) -> Iterator[bool]:
        entered = self._enter(AudioSessionState.SPEAKING)
        try:
            yield entered
        finally:
            if entered:
                self._leave(AudioSessionState.SPEAKING)

    def _enter(self, target: AudioSessionState) -> bool:
        with self._lock:
            if self._state != AudioSessionState.IDLE:
                logger.warning("音频设备忙，当前状态: %s，无法进入: %s", self._state.value, target.value)
                return False
            self._state = target
            return True

    def _leave(self, target: AudioSessionState) -> None:
        with self._lock:
            if self._state == target:
                self._state = AudioSessionState.IDLE

    def force_idle(self) -> None:
        """异常恢复时强制回到空闲状态。"""
        with self._lock:
            self._state = AudioSessionState.IDLE
