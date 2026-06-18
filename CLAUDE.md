# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## 项目定位

这是当前主线项目：持续唤醒端云语音助手 CLI。当前不再保留早期批处理 wav 调试入口，主线只围绕：

`短窗唤醒监听 -> 实时麦克风 ASR -> DeepSeek 流式回复 -> DashScope 真流式 TTS -> 扬声器播放 -> 继续追问`

纯 CLI、简单状态机优先，不引入旧项目 PyQt GUI、MQTT 或复杂线程池。

## 运行

```bash
python main_cli.py --wake-loop --wake-rounds 3
python main_cli.py --wake-loop --wake-rounds 0
```

说明：

- `--wake-rounds 3`：完成 3 轮有效回答后退出，适合主机侧验证。
- `--wake-rounds 0`：常驻运行，按 `Ctrl+C` 退出，适合演示。
- `--exit-after-idle-return`：测试用，唤醒后连续空输入回待机前自动退出。
- 不带 `--wake-loop` 时，程序也会按持续唤醒主线启动。

## 当前模块职责

- `main_cli.py`：持续唤醒 CLI 参数与入口。
- `wake_loop.py`：持续监听状态机，负责待机监听、触发词判断、唤醒后连续追问、空输入重试与回待机。
- `streaming_voice.py`：实时麦克风 ASR、本地 RMS VAD、DeepSeek 流式回复与 TTS 播放编排。
- `streaming_tts.py`：DashScope callback 真流式 TTS，Windows 用 waveOut，Linux / SC171 用 `aplay` stdin 连续写入 PCM。
- `audio_session.py`：录音 / 播放互斥锁，防止 TTS 被麦克风重新识别。
- `llm_client.py`：DeepSeek 客户端，支持流式输出与唤醒会话短期上下文。
- `prompt_audio_cache.py`：唤醒确认、没听清兜底等固定提示音 wav 缓存；内部包含短提示语专用 DashScope wav 生成器。
- `audio_output.py`：固定提示音 wav 播放，以及 Linux / SC171 声卡查找和 mixer 配置复用。
- `settings.py`：`.env` 与模型配置加载。

## 已删除的早期开发入口

以下早期链路已经从主线移除：

- `--input-text`
- `--input-wav`
- `--record-seconds`
- `--realtime-seconds`
- `audio_input.py`
- `record_audio.py`
- `asr_client.py`
- `streaming_voice.py::StreamingTTSPlayer`
- `tts_client.py`

这些代码曾用于打通最小闭环和单轮实时链路。当前主机侧测试已证明持续唤醒链路可跑通，因此不再保留，避免后续维护混乱。

## 当前主机侧测试命令

```bash
python -m py_compile main_cli.py wake_loop.py streaming_voice.py streaming_tts.py audio_session.py prompt_audio_cache.py llm_client.py audio_output.py settings.py
python main_cli.py --wake-loop --wake-rounds 1
python main_cli.py --wake-loop --wake-rounds 3
python main_cli.py --wake-loop --wake-rounds 1 --wake-end-silence-seconds 1.0
python main_cli.py --wake-loop --wake-rounds 1 --exit-after-idle-return
python main_cli.py --wake-loop --wake-rounds 0
```

主机侧通过标准：

- `--wake-rounds 1` 可完成 `待机 -> 唤醒 -> 提问 -> 回答 -> 退出`。
- `--wake-rounds 3` 可连续追问 3 轮，并继承上一轮地点、对象和约束。
- 空输入连续 3 次后回到待机，不反复播放“没听清”。
- “我想问一下”这类开场白不会被当作完整问题回答。
- 用户停顿 1 到 2 秒时不明显截断。
- TTS 播放期间不启动录音，播放后恢复监听。

## 开发约束

- 修改新功能前先检查 `2025AIoT-master/` 是否有可复用经验，但不要搬旧 GUI / 复杂线程池。
- 代码注释与文档使用中文。
- API key、`.env`、数据库凭据不写入代码。
- 修改主线代码后同步更新：
  - `.codex/任务进度面板.md`
  - `.codex/开发计划.md`
  - 必要时同步 `.codex/当前实现逻辑.md`
