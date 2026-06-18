"""
持续唤醒语音助手命令行入口。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _configure_windows_asyncio_policy() -> None:
    """
    Windows 默认 Proactor 事件循环偶尔会在第三方 SDK 退出时打印无效句柄告警。

    当前 CLI 不依赖 Proactor 专属能力，切到 Selector 策略可以减少
    `Cancelling an overlapped future failed` 这类退出噪声。
    """
    if sys.platform != "win32":
        return
    try:
        import asyncio

        if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).debug("设置 Windows asyncio 策略失败: %s", exc)


_configure_windows_asyncio_policy()

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from settings import VoiceCliSettings
from wake_loop import WakeLoop, WakeLoopConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="持续唤醒语音助手命令行工具")
    parser.add_argument("--wake-loop", action="store_true", help="启动短窗 ASR 触发词持续监听")
    parser.add_argument("--wake-listen-seconds", type=int, default=10, help="持续监听短窗秒数")
    parser.add_argument("--wake-question-seconds", type=int, default=18, help="唤醒后正式录音硬上限秒数")
    parser.add_argument("--wake-initial-silence-seconds", type=float, default=8.0, help="唤醒后最多等待用户开口秒数")
    parser.add_argument("--wake-end-silence-seconds", type=float, default=1.0, help="检测到本地静音多久判定输入完成")
    parser.add_argument("--wake-question-retries", type=int, default=3, help="激活后连续无有效问题时的最多重试次数")
    parser.add_argument("--wake-rounds", type=int, default=3, help="完成回答的问题轮数，0 表示一直运行")
    parser.add_argument("--exit-after-idle-return", action="store_true", help="测试用：唤醒后回到待机前退出")
    parser.add_argument(
        "--wake-words",
        default="小智,小志,小知,小子,你好,hello,hi,开始对话",
        help="逗号分隔的触发词列表",
    )
    parser.add_argument("--wake-reply", default="你好，我在。", help="唤醒后播放的确认提示语")
    parser.add_argument(
        "--unclear-reply",
        default="不好意思呀，我刚才没听清，可以再说一遍吗？",
        help="唤醒后未听清问题时播放的兜底提示语",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = VoiceCliSettings.load()

    if not args.wake_loop:
        logger.info("当前主线只保留持续唤醒入口，已自动按 --wake-loop 启动")

    trigger_words = tuple(word.strip() for word in args.wake_words.split(",") if word.strip())
    wake_loop = WakeLoop(
        settings,
        WakeLoopConfig(
            trigger_words=trigger_words or ("小智", "小志", "小知", "小子", "你好", "hello", "hi", "开始对话"),
            wake_reply=args.wake_reply,
            unclear_reply=args.unclear_reply,
            listen_seconds=args.wake_listen_seconds,
            question_seconds=args.wake_question_seconds,
            question_initial_silence_seconds=args.wake_initial_silence_seconds,
            question_end_silence_seconds=args.wake_end_silence_seconds,
            question_retry_limit=args.wake_question_retries,
            max_rounds=args.wake_rounds,
            exit_after_idle_return=args.exit_after_idle_return,
        ),
    )
    return wake_loop.run()


if __name__ == "__main__":
    sys.exit(main())
