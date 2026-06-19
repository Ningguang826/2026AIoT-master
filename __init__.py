"""
持续唤醒端云语音助手 CLI。

闭环：短窗唤醒监听 -> 实时麦克风 ASR -> DeepSeek 流式回复
-> DashScope 真流式 TTS -> 扬声器播放 -> 继续追问。
"""

