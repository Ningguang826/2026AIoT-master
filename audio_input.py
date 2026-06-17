"""
音频输入模块。
"""

from __future__ import annotations

from pathlib import Path


class AudioInput:
    """当前阶段先保留最小输入能力。"""

    def resolve_wav(self, wav_path: str) -> Path:
        audio_path = Path(wav_path).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")
        return audio_path

