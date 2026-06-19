"""
配置加载。

从 .env 读取 DashScope / DeepSeek 的 API key 与模型名，集中提供给各模块。
只加载环境变量，不把任何 key 写入代码或日志。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent


def _load_env() -> Path | None:
    """
    依次尝试加载常见的 .env 位置。

    优先读取当前项目目录，兼容从工作区根目录、项目目录或测试脚本启动。
    这里只加载环境变量，不把任何 key 写入代码或日志。
    """
    candidates = [
        PROJECT_ROOT / ".env",
        Path.cwd() / ".env",
        PROJECT_ROOT.parent / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path)
            return env_path
    return None


@dataclass
class VoiceCliSettings:
    dashscope_api_key: str = ""
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    asr_model: str = "gummy-chat-v1"
    tts_model: str = "cosyvoice-v1"
    tts_voice: str = "longwan"

    @classmethod
    def load(cls) -> "VoiceCliSettings":
        _load_env()
        return cls(
            dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY", "").strip(),
            deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY", "").strip(),
            deepseek_base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
            deepseek_model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip(),
            asr_model=os.environ.get("ASR_MODEL", "gummy-chat-v1").strip(),
            tts_model=os.environ.get("TTS_MODEL", "cosyvoice-v1").strip(),
            tts_voice=os.environ.get("TTS_VOICE", "longwan").strip(),
        )
