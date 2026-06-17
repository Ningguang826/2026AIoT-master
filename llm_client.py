"""
DeepSeek 文本问答客户端。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from openai import OpenAI


logger = logging.getLogger(__name__)


class DeepSeekClient:
    """最小单轮 DeepSeek 对话客户端。"""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model

    def reply(self, user_text: str) -> str:
        if not user_text.strip():
            return "我没有听清楚你的问题。"
        if not self.api_key:
            logger.error("未配置 DEEPSEEK_API_KEY，返回兜底回复")
            return "当前网络问答服务暂时不可用，请稍后再试。"

        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个适合嵌入式竞赛演示的中文语音助手，请简洁、自然地回答。",
                    },
                    {"role": "user", "content": user_text},
                ],
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

    def stream_reply(self, user_text: str) -> Iterator[str]:
        """流式生成 DeepSeek 回复文本片段。"""
        if not user_text.strip():
            yield "我没有听清楚你的问题。"
            return
        if not self.api_key:
            logger.error("未配置 DEEPSEEK_API_KEY，返回兜底回复")
            yield "当前网络问答服务暂时不可用，请稍后再试。"
            return

        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            stream = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个适合嵌入式竞赛演示的中文语音助手，请简洁、自然地回答。",
                    },
                    {"role": "user", "content": user_text},
                ],
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
