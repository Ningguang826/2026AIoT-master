# 端侧 LLM 接入调研结论

> 用途：板端联调前的依据。记录官方端侧 LLM 的真实接口、模型选型、prompt 模板和对现有架构的影响。
> 来源：`官方资料/大模型实战/本地端侧/`（fmodel + 参考代码）、`AIoT开发套件V3.pdf`（已适配模型仓库）。

## 1. 路线决策

- 不微调，走「官方已量化端侧模型 + 系统提示词做角色塑造」。
- 主机阶段仍用云端 DeepSeek 验证链路；端侧 LLM 适配是板端联调阶段的事，主机无法测（需 `fiboaisdk` + license）。

## 2. 可用的端侧 LLM（手头已有 fmodel）

| 模型 | 框架 / 算力 | 文件 | 输出特性 |
|------|-----------|------|---------|
| Qwen3-0.6B | QNN / **DSP(NPU)** | `qwen3-0.6b_..._qnn_2.28_dsp_....fmodel` | 标准对话模型，输出较干净 |
| deepseek-r1-distill-qwen-1.5b | MNN / **CPU** | `deepseek-r1-qwen-1.5b_..._mnn_3.0.5_cpu_....fmodel` | R1 推理蒸馏，爱重复/啰嗦/带标记，需大量后处理 |

**优先选 Qwen3-0.6B（NPU）**，理由：

- 0.6B + NPU 时延更可能达标（核心目标是压 `t`）；deepseek-r1-1.5b 是 CPU 跑，更慢。
- 语音助手要「快 + 短 + 口语」，不需要 R1 的思考链；R1 的重复/思考特性反而不利。
- deepseek-r1-1.5b 留作「回答质量不够」时的备选。

## 3. 端侧推理接口（fiboaisdk.api_nlp_py）

与现有云端 `DeepSeekClient` 形态完全不同，关键差异见第 5 节。

```python
from fiboaisdk.api_aisdk_py import api_nlp_py as nlp_api
from fiboaisdk.api_aisdk_py import license_py as license_api

# 1) license 初始化（key1/2/3.pem + license.bin，路径 /home/fibo/qcom_6490_license/）
license_api.Init(key1, key2, key3, license_data)   # 返回 0 为成功

# 2) 加载模型
api = nlp_api.NLPAPI()
api.Init(model_path, "")           # 第二参数为超参，示例留空

# 3) 同步生成（无 messages、无流式）
result = nlp_api.ResultNlpText()
api.GenerateSync(context, result)  # context 是手工拼好的整段 prompt 字符串
text = result.text                 # 一次性拿到完整文本

# 4) 释放
api.Release()
```

## 4. prompt 模板（两个模型不同，都要手工拼字符串）

Qwen3-0.6B（推荐）：

```
<|im_start|>system
{系统提示词}<|im_end|>
<|im_start|>user
{用户输入}<|im_end|>
<|im_start|>assistant
```

deepseek-r1-qwen-1.5b：

```
<|begin_of_text|>{系统提示词}{用户输入}<｜Assistant｜>
```

注意：deepseek-r1 的官方参考代码带一个 100+ 行的 `extract_response_fixed_format`，专门去模板残留、去重复句、去重复关键词、截断到 800 字、按句号收口——说明其原始输出很脏，移植时这套清洗必须一起带。

## 5. 对现有架构的影响（板端适配要改的点）

现状：`llm_client.py::DeepSeekClient.stream_reply` 是「OpenAI messages 数组 + 流式生成器」。端侧三处根本不同：

1. **无 messages**：system / 历史 / 用户输入要自己拼成一整个字符串；多轮历史按同样模板循环拼接。0.6B 上下文窗口小，`messages[-10:]` 需缩到 `[-2:]`～`[-4:]` 实测。
2. **非流式**：`GenerateSync` 一次性返回。现有「边收 token 边分句播报」用不上，要改成「等整段生成完 → 再分句 → 送 TTS」。**这会增加首字延迟**（等模型生成完才出声），是端侧时延的主要新增项，需实测。
3. **输出要清洗**：尤其 deepseek-r1，需移植官方的去重复/去残留/收口逻辑。

落地形态：新增独立端侧适配器（如 `FiboLocalLLMClient`），封装模板选择、`GenerateSync`、输出清洗；为减少上层改动，可把「一次性返回」包成单块生成器，使 `streaming_voice.answer_text` 与 TTS 衔接尽量少改。

## 6. 角色塑造要点（小模型规律）

- 0.6B / 1.5B 量化后指令遵循远弱于云端，长而堆形容词的人设易记不全、多轮跑偏。
- 有效写法：**短、具体、给行为约束、给 1-2 个 few-shot 示例、硬限输出长度**。
- 现有云端 `SYSTEM_PROMPT`（要求开头给 6-10 字短语那段）不能照搬，需在端侧重写并实测调。
- 多轮一致性是硬伤，建议每轮在 context 里重申角色，历史窗口尽量短。

## 7. 待确认 / 待办

- 板端实测 Qwen3-0.6B 的 `GenerateSync` 单次耗时，确认非流式是否拖垮 `t`。
- 确认 `api.Init` 第二参（超参字符串）支持哪些项（如最大生成长度、温度），可否用于限长降时延。
- license 文件获取与板端放置路径（参考 ASR 案例 `/home/fibo/qcom_6490_license/`）。
- 端侧是否有流式 / 增量生成接口（当前示例只有 `GenerateSync`）；若有可显著改善首字延迟。
