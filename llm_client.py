"""
DeepSeek 文本问答客户端。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TypedDict

from openai import OpenAI


logger = logging.getLogger(__name__)


class ChatMessage(TypedDict):
    role: str
    content: str


class DeepSeekClient:
    """最小 DeepSeek 对话客户端，支持唤醒会话内的短期历史。"""

    SYSTEM_PROMPT = (
        "你是一个适合嵌入式竞赛演示的中文语音助手，请用口语化中文简洁回答，优先一到两句话。"
        "回答开头先给一个 6 到 10 个字的短语或短句，方便语音尽快开始播报，再继续补充关键信息。"
        "用户可能连续追问，你必须继承上一轮的地点、对象和约束；如果上下文不足，先简短澄清，不要自行切换城市或主题。"
    )

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        # 持久化客户端，避免每轮对话重复创建 HTTP 客户端带来额外连接开销。
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

    def reply(self, user_text: str, history: list[ChatMessage] | None = None) -> str:
        if not user_text.strip():
            return "我没有听清楚你的问题。"
        if not self.api_key:
            logger.error("未配置 DEEPSEEK_API_KEY，返回兜底回复")
            return "当前网络问答服务暂时不可用，请稍后再试。"

        try:
            assert self.client is not None
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(user_text, history),
                temperature=0.7,
            )
            content = response.choices[0].message.content or ""
            content = content.strip()
            if not content:
                return "我暂时没有生成有效回复。"
            return content
        except Exception as exc:
            logger.error("DeepSeek 调用失败: %s", exc)
            return "当前网络问答服务暂时不可用，请稍后再试。"

    def stream_reply(self, user_text: str, history: list[ChatMessage] | None = None) -> Iterator[str]:
        """流式生成 DeepSeek 回复文本片段。"""
        if not user_text.strip():
            yield "我没有听清楚你的问题。"
            return
        if not self.api_key:
            logger.error("未配置 DEEPSEEK_API_KEY，返回兜底回复")
            yield "当前网络问答服务暂时不可用，请稍后再试。"
            return

        try:
            assert self.client is not None
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self._build_messages(user_text, history),
                temperature=0.7,
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content or ""
                if piece:
                    yield piece
        except Exception as exc:
            logger.error("DeepSeek 流式调用失败: %s", exc)
            yield "当前网络问答服务暂时不可用，请稍后再试。"

    def _build_messages(self, user_text: str, history: list[ChatMessage] | None = None) -> list[ChatMessage]:
        """按旧项目 messages[-10:] 思路构造短期上下文窗口。"""
        recent_history = (history or [])[-10:]
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            *recent_history,
            {"role": "user", "content": user_text.strip()},
        ]
