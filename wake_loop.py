"""
持续唤醒监听最小状态机。

第一版使用“短窗 ASR + 触发词”模拟唤醒，先验证持续待机、触发对话、
播报完成后继续听追问，连续多次无有效输入后再回待机，不接入复杂离线 wake word 引擎。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from llm_client import ChatMessage
from settings import VoiceCliSettings
from streaming_voice import ASRListenResult, RealtimeVoiceLoop


logger = logging.getLogger(__name__)


class WakeState(Enum):
    """持续监听的最小状态集合。"""

    IDLE_LISTENING = "IdleListening"
    ACTIVATED_RECORDING = "ActivatedRecording"
    THINKING = "Thinking"
    SPEAKING = "Speaking"
    ERROR_RECOVERY = "ErrorRecovery"


@dataclass
class WakeLoopConfig:
    """持续监听参数，默认值优先服务演示稳定性。"""

    trigger_words: tuple[str, ...] = ("小智", "小志", "小知", "小子", "你好", "hello", "hi", "开始对话")
    wake_reply: str = "你好，我在。"
    unclear_reply: str = "不好意思呀，我刚才没听清，可以再说一遍吗？"
    listen_seconds: int = 10
    question_seconds: int = 18
    question_initial_silence_seconds: float = 8.0
    question_end_silence_seconds: float = 1.0
    question_retry_limit: int = 3
    max_rounds: int = 3
    exit_after_idle_return: bool = False
    state: WakeState = field(default=WakeState.IDLE_LISTENING, init=False)


QuestionListenStatus = Literal["valid", "empty", "unclear", "incomplete"]


@dataclass
class QuestionListenResult:
    """一次用户问题监听的分类结果。"""

    status: QuestionListenStatus
    text: str | None = None
    utterance_end_time: float | None = None


class WakeLoop:
    """单线程持续监听循环。"""

    def __init__(self, settings: VoiceCliSettings, config: WakeLoopConfig | None = None) -> None:
        self.config = config or WakeLoopConfig()
        self.voice_loop = RealtimeVoiceLoop(settings)

    def run(self) -> int:
        completed_rounds = 0
        logger.info("持续监听已启动，触发词: %s", " / ".join(self.config.trigger_words))
        if self.config.max_rounds > 0:
            logger.info("第一版会在完成 %s 轮对话后退出，便于验证资源释放", self.config.max_rounds)
        else:
            logger.info("当前为持续运行模式，按 Ctrl+C 退出")
        self.voice_loop.warmup_prompt_audio(self.config.wake_reply, self.config.unclear_reply)

        while self.config.max_rounds <= 0 or completed_rounds < self.config.max_rounds:
            try:
                self.config.state = WakeState.IDLE_LISTENING
                logger.info("待机监听中，请说触发词")
                trigger_result = self.voice_loop.asr.listen_once(max_duration=self.config.listen_seconds)
                trigger_text = trigger_result.text
                if not trigger_text:
                    logger.info("待机短窗未识别到有效文本，继续监听")
                    continue

                logger.info("待机识别文本: %s", trigger_text)
                if not self._has_trigger(trigger_text):
                    logger.info("未命中触发词，继续待机")
                    continue

                self.config.state = WakeState.ACTIVATED_RECORDING
                logger.info("已唤醒，先播放确认提示")
                self.voice_loop.speak_text(self.config.wake_reply)
                completed_rounds = self._run_active_dialog(completed_rounds)
                if self.config.exit_after_idle_return:
                    logger.info("已按测试参数在回到待机前退出")
                    return 0
            except KeyboardInterrupt:
                logger.info("收到中断信号，持续监听退出")
                self.voice_loop.interrupt_tts()
                return 130
            except Exception as exc:  # noqa: BLE001
                self.config.state = WakeState.ERROR_RECOVERY
                logger.error("持续监听异常，已回到恢复状态: %s", exc)

        logger.info("持续监听验证完成，共完成 %s 轮", completed_rounds)
        return 0

    def _run_active_dialog(self, completed_rounds: int) -> int:
        """
        唤醒后的连续对话循环。

        回答完成后不立刻回到唤醒监听，而是继续等待追问；只有连续多次没有有效问题
        才退回待机，避免“有没有室内的推荐”这类追问被当作唤醒词判断。
        """
        miss_count = 0
        dialog_history: list[ChatMessage] = []
        while self.config.max_rounds <= 0 or completed_rounds < self.config.max_rounds:
            logger.info("请说出问题或继续追问")
            result = self._listen_user_question_once(miss_count + 1)
            if result.status == "empty":
                miss_count += 1
                logger.info(
                    "本次未检测到有效语音，连续空输入 %s/%s",
                    miss_count,
                    self.config.question_retry_limit,
                )
                if miss_count >= self.config.question_retry_limit:
                    logger.info("连续 %s 次无有效输入，回到待机监听", self.config.question_retry_limit)
                    return completed_rounds
                continue

            if result.status == "incomplete":
                miss_count += 1
                logger.info(
                    "识别到疑似未说完内容，继续等待补充 %s/%s: %s",
                    miss_count,
                    self.config.question_retry_limit,
                    result.text,
                )
                if miss_count >= self.config.question_retry_limit:
                    logger.info("连续 %s 次未获得完整问题，回到待机监听", self.config.question_retry_limit)
                    return completed_rounds
                continue

            if result.status == "unclear":
                miss_count += 1
                logger.info(
                    "识别到疑似无效输入，连续无效输入 %s/%s: %s",
                    miss_count,
                    self.config.question_retry_limit,
                    result.text,
                )
                self.voice_loop.speak_text(self.config.unclear_reply)
                if miss_count >= self.config.question_retry_limit:
                    logger.info("连续 %s 次未获得清晰问题，回到待机监听", self.config.question_retry_limit)
                    return completed_rounds
                continue

            miss_count = 0
            user_text = result.text or ""
            self.config.state = WakeState.THINKING
            logger.info("已识别问题，开始生成回复")

            # answer_text 内部同步完成 TTS 播放；播放期间不再启动监听，保证录音和播放互斥。
            self.config.state = WakeState.SPEAKING
            logger.info("本轮携带短期上下文消息数: %s", len(dialog_history))
            reply = self.voice_loop.answer_text(
                user_text,
                history=dialog_history,
                utterance_end_time=result.utterance_end_time,
            )
            if not reply:
                self.config.state = WakeState.ERROR_RECOVERY
                logger.warning("本轮回复失败，清理后回到待机")
                return completed_rounds

            dialog_history.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply},
                ]
            )
            # 借鉴旧项目 messages[-10:] 的做法，只保留最近短期上下文，避免拖慢首 token。
            dialog_history = dialog_history[-10:]
            completed_rounds += 1
            logger.info("第 %s 轮对话完成，保持激活状态继续听追问", completed_rounds)

        return completed_rounds

    def _listen_user_question_once(self, attempt: int) -> QuestionListenResult:
        logger.info(
            "开始第 %s/%s 次用户问题监听，开头静音上限 %.1f 秒",
            attempt,
            self.config.question_retry_limit,
            self.config.question_initial_silence_seconds,
        )
        result = self.voice_loop.asr.listen_once(
            max_duration=self.config.question_seconds,
            initial_silence_timeout=self.config.question_initial_silence_seconds,
            end_silence_timeout=self.config.question_end_silence_seconds,
        )
        user_text = result.text
        if not user_text:
            # 空麦克风输入只安静重试，不播放“没听清”，避免无人在说话时反复打扰。
            return QuestionListenResult(status="empty")
        if self._is_incomplete_starter(user_text):
            return self._listen_continuation(user_text, result, attempt)
        if self._is_unclear_question(user_text):
            return QuestionListenResult(status="unclear", text=user_text)
        return QuestionListenResult(status="valid", text=user_text, utterance_end_time=result.utterance_end_time)

    def _listen_continuation(
        self,
        prefix_text: str,
        prefix_result: ASRListenResult,
        attempt: int,
    ) -> QuestionListenResult:
        """
        处理被 1 秒端点切开的半句话。

        竞赛模式不能用很长句末静音硬等用户思考，所以遇到明显未说完的短句时，
        立即再听一个补充窗口并拼接，既保留停顿容错，也不让完整问题固定多等 2 秒。
        """
        logger.info("疑似未说完，进入补充监听: %s", prefix_text)
        continuation = self.voice_loop.asr.listen_once(
            max_duration=min(10, self.config.question_seconds),
            initial_silence_timeout=max(3.0, min(5.0, self.config.question_initial_silence_seconds)),
            end_silence_timeout=self.config.question_end_silence_seconds,
        )
        if not continuation.text:
            # 补听仍为空时才把它当作一次未完成输入；如果补听成功，会合并后正常回答。
            return QuestionListenResult(
                status="incomplete",
                text=prefix_text,
                utterance_end_time=prefix_result.utterance_end_time,
            )

        merged_text = self._merge_question_text(prefix_text, continuation.text)
        logger.info("补充监听合并问题: %s", merged_text)
        if self._is_incomplete_starter(merged_text):
            return QuestionListenResult(
                status="incomplete",
                text=merged_text,
                utterance_end_time=continuation.utterance_end_time or prefix_result.utterance_end_time,
            )
        if self._is_unclear_question(merged_text):
            return QuestionListenResult(status="unclear", text=merged_text)
        return QuestionListenResult(
            status="valid",
            text=merged_text,
            utterance_end_time=continuation.utterance_end_time or prefix_result.utterance_end_time,
        )

    def _has_trigger(self, text: str) -> bool:
        normalized = self._normalize_trigger_text(text)
        trigger_words = [self._normalize_trigger_text(word) for word in self.config.trigger_words]
        if any(word and word in normalized for word in trigger_words):
            return True

        # ASR 偶尔会把“小智”识别成“小子/小只/小纸”等，先用短词近似匹配兜底。
        return any(self._is_close_trigger(normalized, word) for word in trigger_words if len(word) >= 2)

    def _is_unclear_question(self, text: str) -> bool:
        """
        粗略过滤 ASR 把噪声、语气词或触发词残留识别成文本的情况。

        这里不追求语义理解，只过滤明显不像问题/指令的短文本，避免把“嗯/啊/呃”
        继续送给 LLM；正常追问如“有没有室内的推荐”会直接通过。
        """
        normalized = self._normalize_trigger_text(text)
        if not normalized:
            return True
        if self._has_trigger(text) and len(normalized) <= 4:
            return True
        filler_words = {
            "嗯",
            "啊",
            "呃",
            "额",
            "哦",
            "喂",
            "嗯嗯",
            "啊啊",
            "呃呃",
            "额额",
            "那个",
            "这个",
        }
        if normalized in filler_words:
            return True
        if len(normalized) <= 1:
            return True
        if normalized in {"有哪", "哪有"}:
            return True
        return False

    def _is_incomplete_starter(self, text: str) -> bool:
        """
        过滤“我想问一下”这类还没进入正题的开场白。

        DashScope 服务端有时会把这类短开场白直接判成句子结束；如果立刻送给 LLM，
        就会回答“您请说”，并在 `--wake-rounds 1` 测试中提前退出。
        """
        normalized = self._normalize_trigger_text(text)
        incomplete_starters = {
            "我想问一下",
            "我想问问",
            "想问一下",
            "请问一下",
            "我问一下",
            "问一下",
            "那个我想问一下",
            "这个我想问一下",
            "还有没有一些",
            "还有没有",
            "那有没有一些",
            "有没有一些",
            "那也没有一些",
            "那有没有",
            "那室外的呢",
            "那室内的呢",
            "有哪",
        }
        return normalized in incomplete_starters

    @staticmethod
    def _merge_question_text(prefix: str, continuation: str) -> str:
        """拼接续听文本时去掉前段句号，避免“我想问一下。成都...”割裂语义。"""
        prefix = prefix.strip().rstrip("。！？!?，,；;：:、 ")
        continuation = continuation.strip()
        if not prefix:
            return continuation
        if not continuation:
            return prefix
        return f"{prefix}，{continuation}"

    @staticmethod
    def _normalize_trigger_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[\s，。！？,.!?;；:：、\"'“”‘’（）()]", "", text)
        return text

    @staticmethod
    def _is_close_trigger(text: str, trigger: str) -> bool:
        if trigger in {"hello", "hi"}:
            return trigger in text
        if not text or not trigger:
            return False

        for start in range(0, max(1, len(text) - len(trigger) + 1)):
            candidate = text[start : start + len(trigger)]
            if len(candidate) != len(trigger):
                continue
            same_count = sum(1 for left, right in zip(candidate, trigger) if left == right)
            if same_count >= len(trigger) - 1:
                return True
        return False
