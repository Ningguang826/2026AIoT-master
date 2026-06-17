"""
最小语音闭环命令行入口。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from audio_input import AudioInput
from audio_output import AudioOutput
from asr_client import DashScopeASRClient
from llm_client import DeepSeekClient
from record_audio import AudioRecorder
from settings import PROJECT_ROOT, VoiceCliSettings
from streaming_voice import RealtimeVoiceLoop
from tts_client import DashScopeTTSClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="最小语音闭环命令行工具")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-wav", help="输入 wav 文件路径")
    group.add_argument("--input-text", help="手动输入文本，跳过 ASR")
    group.add_argument("--record-seconds", type=int, help="先录制固定时长 wav，再进入 ASR")
    group.add_argument("--realtime-seconds", type=int, help="实时麦克风输入并流式播报的最长时长")
    parser.add_argument(
        "--record-output",
        default=str(PROJECT_ROOT / "data" / "runtime" / "input_recording.wav"),
        help="录音输出 wav 路径",
    )
    parser.add_argument(
        "--output-wav",
        default=str(PROJECT_ROOT / "data" / "runtime" / "voice_cli_reply.wav"),
        help="TTS 输出音频路径",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = VoiceCliSettings.load()

    if args.realtime_seconds:
        realtime_loop = RealtimeVoiceLoop(settings)
        return realtime_loop.run_once(args.realtime_seconds)

    audio_input = AudioInput()
    audio_recorder = AudioRecorder()
    audio_output = AudioOutput()
    asr_client = DashScopeASRClient(
        api_key=settings.dashscope_api_key,
        model=settings.asr_model,
    )
    llm_client = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
    )
    tts_client = DashScopeTTSClient(
        api_key=settings.dashscope_api_key,
        model=settings.tts_model,
        voice=settings.tts_voice,
    )

    user_text: str | None = None

    if args.input_text:
        user_text = args.input_text.strip()
        logger.info("使用手动文本输入模式")
    else:
        input_wav = args.input_wav
        if args.record_seconds:
            logger.info("使用麦克风预录模式")
            record_start = time.time()
            recorded_path = audio_recorder.record_to_wav(args.record_seconds, args.record_output)
            logger.info("录音耗时: %.2fs", time.time() - record_start)
            if not recorded_path:
                logger.error("录音失败，请检查麦克风权限或设备占用")
                return 5
            input_wav = str(recorded_path)

        try:
            wav_path = audio_input.resolve_wav(input_wav)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 2

        logger.info("使用 wav 输入模式: %s", wav_path)
        asr_start = time.time()
        user_text = asr_client.transcribe_wav(wav_path)
        logger.info("ASR 耗时: %.2fs", time.time() - asr_start)
        if not user_text:
            logger.error("ASR 失败，可改用 --input-text 继续联调")
            return 3

    logger.info("用户输入文本: %s", user_text)

    llm_start = time.time()
    reply_text = llm_client.reply(user_text)
    logger.info("LLM 耗时: %.2fs", time.time() - llm_start)
    logger.info("LLM 回复: %s", reply_text)

    tts_start = time.time()
    output_path = Path(args.output_wav).expanduser().resolve()
    tts_ok = tts_client.synthesize_to_wav(reply_text, output_path)
    logger.info("TTS 耗时: %.2fs", time.time() - tts_start)
    if not tts_ok:
        logger.error("TTS 失败，文本回复如下: %s", reply_text)
        return 4

    played = audio_output.play_wav(output_path)
    if not played:
        logger.warning("音频未实际播放，请手动检查文件: %s", output_path)

    logger.info("最小语音闭环执行结束")
    return 0


if __name__ == "__main__":
    sys.exit(main())
