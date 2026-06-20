# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

持续唤醒端云语音助手 CLI。主线只围绕一条闭环：

`短窗唤醒监听 -> 实时麦克风 ASR -> DeepSeek 流式回复 -> DashScope 真流式 TTS -> 扬声器播放 -> 继续追问`

纯 CLI、单线程状态机优先。不引入旧项目 PyQt GUI、MQTT 或复杂线程池。目标平台是 Windows（开发主机）和 Linux / 广和通 SC171 开发板（部署端）。

## 环境与依赖

> 本机运行环境固定在 conda 虚拟环境 **`D:\anaconda\envs\AIoT`**（依赖都装在这里）。跑任何 `python` / `py_compile` 都应使用该环境，例如 `D:\anaconda\envs\AIoT\python.exe ...` 或先 `conda activate AIoT`。

运行依赖外部 SDK，需先安装：

```bash
pip install dashscope openai python-dotenv
# pyaudio 在 Windows / Linux 上用 pip 常装不上，建议：
conda install -c conda-forge pyaudio
```

必须在项目根目录（或工作区根、上级目录）放置 `.env`，`settings.py` 会按 `项目目录 -> 当前工作目录 -> 上级目录` 顺序查找：

- `DASHSCOPE_API_KEY`：ASR（gummy-chat-v1）和 TTS（cosyvoice-v1）共用。
- `DEEPSEEK_API_KEY`：DeepSeek 对话。
- 可选覆盖项：`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`、`ASR_MODEL`、`TTS_MODEL`、`TTS_VOICE`。

缺少 key 时各模块会记日志并返回兜底结果，不会崩溃，但闭环跑不通。

## 运行

```bash
python main_cli.py --wake-loop --wake-rounds 3   # 完成 3 轮后退出，适合主机验证
python main_cli.py --wake-loop --wake-rounds 0   # 常驻运行，Ctrl+C 退出，适合演示
```

说明：

- 不带 `--wake-loop` 也会自动按持续唤醒主线启动（只保留这一条入口）。
- `--exit-after-idle-return`：测试用，唤醒后连续空输入回待机前自动退出。
- 其余参数（`--wake-listen-seconds`、`--wake-question-seconds`、`--wake-end-silence-seconds`、`--wake-question-retries` 等）见 `main_cli.py::parse_args`，默认值优先保证演示稳定。

## 测试

无单元测试框架，验证靠静态编译 + 真实麦克风手动跑闭环。

静态检查（改完代码先跑这个）：

```bash
python -m py_compile main_cli.py wake_loop.py streaming_voice.py streaming_tts.py audio_session.py prompt_audio_cache.py llm_client.py audio_output.py settings.py
```

完整主机侧手动测试用例（含每步要念的话术）封装在 `run_host_tests.ps1`，会把输出写进 `test_after_cleanup.log`：

```powershell
.\run_host_tests.ps1
```

主机侧通过标准：

- `--wake-rounds 1` 完成 `待机 -> 唤醒 -> 提问 -> 回答 -> 退出`。
- `--wake-rounds 3` 连续追问 3 轮并继承上一轮地点、对象、约束。
- 空输入连续 3 次后回待机，不反复播放“没听清”。
- “我想问一下”这类开场白不被当作完整问题回答。
- 用户停顿 1 到 2 秒不被明显截断。
- TTS 播放期间不启动录音，播放后恢复监听。

## 架构大图

调用链按层组装，每层只关心一件事：

- `main_cli.py` 解析参数 -> 构造 `WakeLoopConfig` 与 `WakeLoop`。
- `wake_loop.py::WakeLoop` 是**唯一的单线程状态机**（`IdleListening / ActivatedRecording / Thinking / Speaking / ErrorRecovery`），持有一个 `RealtimeVoiceLoop`。它负责触发词判断、唤醒后连续追问、半句话续听拼接、空/无效输入重试与回待机。所有“这句话算不算有效问题”的启发式都在这里（`_is_unclear_question` / `_is_incomplete_starter` / `_has_trigger` 含 ASR 误识别近似匹配）。
- `streaming_voice.py::RealtimeVoiceLoop` 是编排门面，把 ASR、LLM、TTS、提示音缓存、音频锁组装在一起，对外暴露 `asr.listen_once / speak_text / answer_text / warmup_prompt_audio / interrupt_tts`。
- `streaming_voice.py::RealtimeASR` 用麦克风 PCM 直接驱动 DashScope ASR，叠加本地 RMS VAD 兜底端点判断，返回带端点时间的 `ASRListenResult`。
- `llm_client.py::DeepSeekClient` 走 OpenAI 兼容接口流式输出，`stream_reply` 产出文本片段迭代器。`SYSTEM_PROMPT` 要求“开头先给 6-10 字短语”以便尽快开播。
- `streaming_tts.py::DashScopeStreamingTTSPlayer` 把 LLM 文本流桥接成 TTS 音频流再连续播放。

跨模块的关键设计（改动前务必理解，单看一个文件看不出来）：

1. **录音/播放互斥**：`audio_session.py::AudioSessionLock` 是单一互斥锁（IDLE/RECORDING/SPEAKING）。`RealtimeASR.listen_once` 进 `recording()`、`answer_text/speak_text` 进 `speaking()`。这保证 TTS 播放期间麦克风不录音，否则自家播报会被当成用户输入重新识别。任何新增的录音或播放路径都必须经过这把锁。

2. **真流式而非分段合成**：LLM 文本流在 `_pop_flush_chunk` / `_pop_sentence` 按标点和长度切片，首段更激进（尽快出声），后续段更保守（减少段间碎裂），逐片喂给 DashScope `streaming_call`。播放线程 `_playback_worker` 把回调返回的碎 PCM 块攒到 `playback_buffer_ms` 再写设备。

3. **跨平台 PCM 播放后端**：`PCMStreamPlayer` 按平台切换——Windows 用 `WindowsWaveOutPCMPlayer`（ctypes 直调 winmm，绕开 PyAudio 写入兼容问题，失败再 fallback 到 PyAudio）；Linux / SC171 用 `aplay` 子进程 stdin 连续写 raw PCM，并在开流前调 `audio_output.py` 查声卡索引、配 amixer 通路。固定提示音 wav 播放走另一路：Windows `winsound`、Linux `aplay`。

4. **竞赛时延指标**：`StreamingTTSMetrics` 统计“用户语句结束 -> 首字播报”时延并分级（≤2s 优秀 / ≤4s 及格）。计时起点是 `ASRListenResult.utterance_end_time`（云端判句尾用云端时间，本地静音兜底用最后有效语音时间），所以端点时间会从 ASR 一路透传到 TTS。

5. **短期上下文窗口**：唤醒会话内多轮历史在 `WakeLoop._run_active_dialog` 累积，并裁到最近 `[-10:]`（`DeepSeekClient._build_messages` 也再裁一次），避免拖慢首 token。回待机会清空历史。

6. **提示音缓存**：唤醒确认、“没听清”兜底等固定短语不走在线流式 TTS，由 `prompt_audio_cache.py` 首次生成 wav 缓存到 `data/runtime/prompt_cache/`（按 模型|音色|文本 的 sha1 命名），后续直接播放。`PromptTTSGenerator._fix_wav_header` 会修正 DashScope WAV 的长度占位字段。启动时 `warmup_prompt_audio` 预生成，避免唤醒后首次等待。

## 工作区上下文

本项目位于嵌入式竞赛工作区下，与 `../2025AIoT-master/`（旧 PyQt5 智能学习助手，作为复用素材）并列。

- 开发新功能前，先检查 `2025AIoT-master/` 是否有可复用经验（接口、命令、音频处理、错误处理），但**不要搬旧 GUI / 复杂线程池**。当前多处实现（`messages[-10:]` 上下文、Linux 声卡 amixer 配置）即借鉴自旧项目。
- 工作区根 `../AGENTS.md`、`../CLAUDE.md` 描述整体规则；本项目 `.codex/` 下有 `AGENTS.md`、`任务进度面板.md`、`开发计划.md`、`当前实现逻辑.md`。

## 开发约束

- 代码注释与文档使用中文，技术术语和标识符保持原文。
- 修改主线代码后同步更新 `.codex/任务进度面板.md`、`.codex/开发计划.md`，必要时同步 `.codex/当前实现逻辑.md`。

## Claude Code 工具调用安全规则

绝不在正文中手写、拼接或复述 `(tool_use) name=... id=toolu... input={...}` 这类工具调用文本。工具调用必须作为 Claude Code 结构化工具调用块发出；说明文字和工具调用要分离，不要把多个 Read/Bash 调用挤在同一段正文里。

如果需要连续读取多个文件，优先使用一个只读 Bash/PowerShell 命令完成，或发出多个独立的真实 Read 工具调用块。若发现工具调用已被打印成普通文本，应立即改用真实工具调用重新执行，不要等待用户提醒。
