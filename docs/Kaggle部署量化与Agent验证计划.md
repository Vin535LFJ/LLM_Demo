# Kaggle 部署、量化推理与 GUI-Agent 验证计划

## 1. 项目定位

本计划面向“用 Kaggle 双 T4 GPU 学习大模型部署、量化推理与 Agent 工程化”的近期目标制定。项目不追求一次性搭建生产系统，而是通过可复现实验掌握以下能力：

1. 在本地与 Kaggle 环境中完成模型下载、缓存、加载、推理与日志记录。
2. 部署并验证 Open-AutoGLM(9B) 与 emotion2vec（语音情绪识别）两类模型。
3. 使用 llama.cpp 与 vLLM 进行量化推理对比，建立速度、显存、精度三维评测表。
4. 验证远端 Agent 到本地 GUI 的调用链路，形成基础界面自动化交互 Demo。
5. 沉淀双 T4 环境下的部署、量化、推理优化标准流程。

## 2. 阶段计划总览

| 阶段 | 周期 | 核心任务 | 关键产物 |
| --- | --- | --- | --- |
| 阶段 0 | 0.5 周 | 梳理仓库、准备环境、确定模型与数据缓存策略 | 环境检查清单、目录规范、依赖版本表 |
| 阶段 1 | 1 周 | 本地测试环境搭建与最小推理验证 | 本地部署记录、最小推理脚本、错误排查记录 |
| 阶段 2 | 1 周 | Kaggle 双 T4 环境准备与模型缓存 | Kaggle Notebook、Dataset 缓存说明、GPU 基线数据 |
| 阶段 3 | 1–2 周 | Open-AutoGLM(9B) 部署与运行验证 | 文本推理 Demo、显存/延迟基线、问题清单 |
| 阶段 4 | 1 周 | emotion2vec 语音情绪识别部署与验证 | 音频样例、识别结果表、推理耗时记录 |
| 阶段 5 | 2 周 | llama.cpp 与 vLLM 量化对比测试 | 量化实验矩阵、CSV 指标、对比图表 |
| 阶段 6 | 1–2 周 | GUI-Agent 远端到本地调用链路验证 | 架构图、接口协议、自动化交互 Demo |
| 阶段 7 | 1 周 | 总结标准化调优流程与技术选型建议 | 部署测试文档、两类场景技术选型参考 |

## 3. 环境与目录规划

### 3.1 推荐目录

```text
LLM_Demo/
├── docs/
│   ├── 学习计划.md
│   ├── 白皮书.md
│   └── Kaggle部署量化与Agent验证计划.md
├── notebooks/
│   ├── 01_kaggle_env_check.ipynb
│   ├── 02_open_autoglm_baseline.ipynb
│   ├── 03_emotion2vec_baseline.ipynb
│   ├── 04_llamacpp_quant_benchmark.ipynb
│   ├── 05_vllm_quant_benchmark.ipynb
│   └── 06_gui_agent_demo.ipynb
├── scripts/
│   ├── check_env.py
│   ├── run_open_autoglm.py
│   ├── run_emotion2vec.py
│   ├── benchmark_llamacpp.py
│   ├── benchmark_vllm.py
│   └── gui_agent_bridge.py
├── data/
│   ├── audio_samples/
│   └── prompts/
└── outputs/
    ├── logs/
    ├── benchmarks/
    └── figures/
```

### 3.2 环境基线

| 环境 | 用途 | 重点检查项 |
| --- | --- | --- |
| 本地 CPU/GPU 环境 | 依赖安装、脚本调试、小模型链路验证、GUI 自动化调试 | Python、CUDA、PyTorch、浏览器/桌面自动化依赖 |
| Kaggle 单 T4 | 单卡部署、显存基线、轻量量化验证 | `nvidia-smi`、磁盘空间、离线 Dataset 挂载 |
| Kaggle 双 T4 | vLLM 张量并行、批量推理、吞吐测试 | 双卡可见性、`tensor_parallel_size=2`、显存峰值 |

## 4. 模型部署验证计划

### 4.1 Open-AutoGLM(9B)

**目标**：完成 9B 级语言模型在本地/Kaggle 的加载、推理和基线评测，确认双 T4 下可运行的精度与量化方案。

**执行步骤**：

1. 确认模型来源、许可证、权重格式、Tokenizer 与 `trust_remote_code` 需求。
2. 在本地使用小样例 Prompt 验证 Transformers 推理链路。
3. 在 Kaggle 上将模型权重保存为 Dataset，避免每次 Notebook 重新下载。
4. 记录 FP16/BF16、8bit、4bit 方案的加载结果；若 FP16 OOM，记录 OOM 日志作为基线。
5. 固定 Prompt 集，记录 TTFT、总延迟、tokens/s、峰值显存和输出质量。

**验收标准**：

- 至少完成 5 条固定 Prompt 的稳定推理。
- 输出一张包含显存、延迟、吞吐、主观质量评分的基线表。
- 明确写出双 T4 上推荐的加载方式和不推荐方式。

### 4.2 emotion2vec（语音情绪识别）

**目标**：完成语音情绪识别模型的部署验证，形成音频输入到情绪标签输出的最小闭环。

**执行步骤**：

1. 准备 10–20 条短音频样例，覆盖中性、开心、愤怒、悲伤等常见情绪。
2. 统一音频格式，例如 16 kHz、单声道、WAV。
3. 编写最小推理脚本，输出情绪标签、置信度、单条音频耗时。
4. 记录 CPU 与 GPU 推理耗时差异。
5. 汇总错误案例，分析噪声、语速、音量、采样率对识别结果的影响。

**验收标准**：

- 能批量处理指定目录下的音频文件。
- 输出标准 CSV：`file,duration_sec,pred_label,confidence,latency_ms,device`。
- 形成“语音情绪识别场景”的技术选型建议。

## 5. 量化推理对比实验

### 5.1 核心维度

| 维度 | 指标 | 采集方式 |
| --- | --- | --- |
| 推理速度 | TTFT、总延迟、tokens/s、请求吞吐 | Python 计时、vLLM metrics、日志解析 |
| 资源占用 | GPU 峰值显存、CPU 内存、模型磁盘大小 | `nvidia-smi`、`psutil`、文件大小统计 |
| 效果精度 | PPL、固定 Prompt 人工评分、任务正确率 | WikiText/自定义题集、人工打分表 |

### 5.2 llama.cpp 实验

**重点**：验证 GGUF 多比特量化在低资源、本地或边缘部署场景下的可用性。

| 实验项 | 建议配置 | 输出 |
| --- | --- | --- |
| 权重转换 | HF/Safetensors → GGUF | 转换日志、模型大小 |
| 量化格式 | Q4_K_M、Q5_K_M、Q8_0 | 显存/内存、速度、质量表 |
| 运行设备 | CPU、GPU offload、全 GPU（如可行） | tokens/s 对比 |
| Prompt 集 | 问答、摘要、工具调用规划 | 输出样例与人工评分 |

### 5.3 vLLM 实验

**重点**：验证双 T4 上的高吞吐推理、连续批处理、张量并行与 AWQ/GPTQ 等量化模型加载。

| 实验项 | 建议配置 | 输出 |
| --- | --- | --- |
| 基线 | Transformers vs vLLM | 单请求延迟、tokens/s |
| 并发 | batch size 1/4/8/16 | 吞吐-延迟曲线 |
| 双卡 | `tensor_parallel_size=2` | 双卡显存分布、吞吐提升 |
| 量化 | AWQ/GPTQ/可用 INT8 或 4bit | 精度损失与速度收益 |

### 5.4 统一实验矩阵

| 框架 | 精度/格式 | 单卡可行性 | 双卡可行性 | 速度 | 显存 | 质量 | 推荐场景 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformers | FP16/BF16 | 待测 | 待测 | 待测 | 待测 | 基线 | 调试、功能验证 |
| Transformers + bitsandbytes | INT8/NF4 | 待测 | 待测 | 待测 | 待测 | 待测 | Kaggle 快速验证 |
| llama.cpp | GGUF Q4/Q5/Q8 | 待测 | 不作为重点 | 待测 | 待测 | 待测 | 本地/边缘低资源部署 |
| vLLM | FP16/AWQ/GPTQ | 待测 | 待测 | 待测 | 待测 | 待测 | 高吞吐服务化推理 |

## 6. GUI-Agent 方案验证

### 6.1 目标架构

```text
远端 Agent / Notebook / API Client
        │
        │ HTTP/WebSocket/RPC
        ▼
本地 Agent Bridge 服务
        │
        ├── GUI 状态采集：截图、窗口标题、控件树
        ├── 动作执行：点击、输入、快捷键、滚动
        └── 日志记录：动作、坐标、结果、异常截图
        ▼
本地 GUI 应用 / 浏览器 / 桌面程序
```

### 6.2 验证范围

1. 远端发送任务：例如“打开页面、输入内容、点击按钮、读取结果”。
2. 本地 Bridge 接收任务并执行 GUI 动作。
3. 每一步保存截图、动作日志和返回状态。
4. 异常时返回错误类型、截图和可复现步骤。

### 6.3 最小 Demo

- Demo A：远端指令控制本地浏览器打开指定网页并完成一次搜索。
- Demo B：远端指令控制一个本地测试 GUI，完成输入框填写、按钮点击和结果读取。
- Demo C：将模型输出的结构化动作计划转为 GUI 自动化动作序列。

## 7. 标准化调优流程

每次部署或量化实验都按以下模板记录：

1. **环境记录**：GPU、驱动、CUDA、Python、核心依赖版本。
2. **模型记录**：模型名称、参数量、权重格式、上下文长度、许可证。
3. **加载策略**：精度、量化方式、device map、并行策略。
4. **输入集**：Prompt 或音频样例版本，固定随机种子。
5. **性能指标**：TTFT、总延迟、tokens/s、峰值显存、CPU 内存、磁盘大小。
6. **效果指标**：PPL/任务正确率/人工评分/错误案例。
7. **结论**：推荐配置、不推荐配置、适用场景、下一步优化。

## 8. 最终交付物

### 8.1 部署测试文档

- 本地环境搭建说明。
- Kaggle 双 T4 Notebook 操作说明。
- Open-AutoGLM(9B) 部署验证记录。
- emotion2vec 部署验证记录。
- llama.cpp 与 vLLM 量化测试报告。
- GUI-Agent 调用链路验证报告。

### 8.2 两类场景技术选型参考

| 场景 | 主要目标 | 初步推荐方向 | 需要用实验确认的问题 |
| --- | --- | --- | --- |
| 高吞吐文本生成/Agent 后端 | 并发、吞吐、GPU 利用率 | vLLM + AWQ/GPTQ 或双 T4 张量并行 | T4 上量化算子兼容性、吞吐提升幅度、质量损失 |
| 本地/边缘/低资源推理 | 易部署、低内存、可离线运行 | llama.cpp + GGUF Q4/Q5 | 中文与工具调用能力损失、CPU/GPU offload 速度 |
| 语音情绪识别 | 稳定识别、低延迟、批处理 | emotion2vec + 轻量 API 封装 | 噪声鲁棒性、音频预处理影响、CPU 是否足够 |
| GUI 自动化 Agent | 可控、安全、可审计 | 远端 Agent + 本地 Bridge + 截图/动作日志 | 动作成功率、异常恢复、人机确认边界 |

## 9. 风险与应对

| 风险 | 表现 | 应对 |
| --- | --- | --- |
| Kaggle 网络或磁盘限制 | 模型下载失败、Session 重启后丢失 | 提前制作 Dataset 缓存，保存依赖 wheel 与模型权重 |
| T4 显存不足 | FP16 OOM、长上下文失败 | 优先 4bit/8bit，缩短上下文，使用双卡或 vLLM 量化模型 |
| 量化模型效果下降 | 回答不稳定、任务正确率下降 | 固定评测集，保留 FP16/高比特基线，比较 AWQ/GPTQ/GGUF |
| GUI 自动化不稳定 | 坐标漂移、窗口焦点丢失 | 使用控件树优先，坐标点击兜底，动作后截图校验 |
| 依赖版本冲突 | vLLM、CUDA、PyTorch 不兼容 | 为每个实验记录版本，必要时拆分 Notebook/环境 |

## 10. 近期里程碑

- **M1：环境闭环**：本地与 Kaggle 均能运行环境检查脚本并输出 GPU/依赖信息。
- **M2：模型闭环**：Open-AutoGLM(9B) 与 emotion2vec 均完成最小输入到输出验证。
- **M3：量化闭环**：至少完成 llama.cpp 三种 GGUF 量化与 vLLM 两种配置对比。
- **M4：Agent 闭环**：远端指令能够驱动本地 GUI 完成 3 步以上自动化任务。
- **M5：文档闭环**：形成部署测试文档、实验表格、场景技术选型参考和复盘结论。

## 11. 面向移动端 AI 应用岗位的项目升级计划

### 11.1 结论：当前两个模型能否支撑求职

**仅部署 Open-AutoGLM(9B) 与 emotion2vec 还不足以直接支撑“移动端 AI 应用 APP 研发”岗位。** 这两个模型可以证明你理解大模型与语音模型的部署、推理和评测流程，但岗位核心仍然是移动端工程、跨端产品化、端云协同、流式体验、弱网容错、端侧推理与性能稳定性。因此，本项目必须从“模型部署实验”升级为“移动端 AI 原生应用工程化 Demo”。

更准确的定位如下：

| 能力项 | Open-AutoGLM(9B) | emotion2vec | 对岗位的支撑程度 | 需要补强 |
| --- | --- | --- | --- | --- |
| AI 对话、总结、写作辅助 | 可作为云端/远端 LLM 服务后端 | 不适用 | 中 | 移动端流式 UI、会话上下文、Prompt 模板、模型路由 |
| 语音输入/语音情绪理解 | 不适用 | 可作为语音情绪识别能力 | 中 | 端侧音频采集、权限、降噪、播报、ASR/TTS 链路 |
| 量化与推理优化 | 可做 Kaggle/vLLM/llama.cpp 对比 | 可做 CPU/GPU 延迟对比 | 中 | 移动端 GGUF/ONNX/Core ML/MNN/NCNN 实测 |
| 端侧模型部署 | 9B 模型通常不适合直接端侧运行 | 小模型更适合端侧探索 | 低到中 | 增加 0.5B–3B LLM、Embedding、ASR/OCR 等端侧模型 |
| 移动端 APP 工程能力 | 不覆盖 | 不覆盖 | 低 | 必须新增 iOS/Android/Flutter/React Native/KMP 工程 Demo |
| 弱网、缓存、重试、灰度、埋点 | 不覆盖 | 不覆盖 | 低 | 必须新增端上工程能力与可观测性模块 |

### 11.2 项目求职定位

为了匹配岗位职责，本项目建议包装为：

> **“面向移动端 AI 原生应用的端云协同大模型 Demo：包含云端 LLM 服务、语音情绪识别、端侧量化推理探索、流式对话、多轮上下文、弱网缓存与 GUI/APP 自动化验证。”**

这样比单纯写“部署 Open-AutoGLM(9B) 与 emotion2vec”更贴近岗位关键词：

- AI 对话、翻译、总结、写作辅助。
- 语音输入、播报、情绪识别。
- 流式输出、上下文管理、会话历史、Prompt 模板。
- 模型路由、请求重试、缓存、弱网/无网处理。
- 端侧推理、量化模型、离线能力、隐私保护。
- 首 Token 延迟、内存、功耗、Crash/ANR、兼容性。
- iOS/Android 跨端产品从 0 到 1。

### 11.3 必须新增的移动端工程模块

如果目标是应聘该类岗位，本项目后续必须新增至少一个移动端客户端。建议优先选择你最熟悉的一条路线：

| 路线 | 推荐程度 | 适合人群 | 交付物 |
| --- | --- | --- | --- |
| Android Kotlin 原生 | 高 | 目标 Android 岗或泛移动端岗 | Android App、SSE/WebSocket 流式对话、Room 会话存储、端侧 GGUF/ONNX Demo |
| iOS Swift 原生 | 高 | 目标 iOS 岗 | iOS App、流式对话、Core ML/llama.cpp 接入、Keychain/本地缓存 |
| Flutter | 高 | 目标跨端岗 | 一套 iOS/Android UI、统一状态管理、流式对话、插件桥接端侧推理 |
| React Native | 中 | 已熟悉前端生态 | 跨端 App、Native Module 接 llama.cpp/ONNX Runtime Mobile |
| Kotlin Multiplatform | 中 | 有 Kotlin 基础 | 共享会话、网络、缓存逻辑，多端 UI 分别实现 |

最低可交付版本应包含：

1. **AI 对话页**：支持流式输出、停止生成、重新生成、复制、收藏。
2. **会话系统**：本地保存多轮会话、上下文裁剪、Token 预算估算。
3. **Prompt 模板**：总结、翻译、写作辅助、情绪安抚等模板。
4. **语音能力页**：录音、上传 emotion2vec 服务、展示情绪标签与置信度。
5. **弱网与失败处理**：超时、重试、断点提示、缓存兜底、错误码展示。
6. **设置页**：模型选择、云端/端侧模式切换、温度、最大 Token、隐私选项。
7. **性能面板**：TTFT、总耗时、tokens/s、失败率、请求 ID、模型版本。

### 11.4 端云协同架构升级

```text
移动端 App（iOS/Android/Flutter）
  ├── Chat UI：流式渲染、Markdown、停止生成、重试
  ├── Session：会话历史、上下文裁剪、Token 预算
  ├── Voice：录音、权限、音频格式转换、播报
  ├── Offline：缓存、弱网队列、端侧小模型推理
  ├── Observability：埋点、Crash/ANR、性能日志
  └── Model Manager：模型下载、校验、版本、灰度
          │
          │ HTTPS / SSE / WebSocket
          ▼
AI Gateway / Backend
  ├── OpenAI-compatible API
  ├── Prompt 模板与安全策略
  ├── 模型路由：Open-AutoGLM / 其他 LLM / 降级模型
  ├── 请求重试、限流、缓存
  └── 指标采集：TTFT、TPOT、tokens/s、错误率
          │
          ├── vLLM：高吞吐云端 LLM 推理
          ├── llama.cpp：GGUF 本地/边缘推理验证
          └── emotion2vec：语音情绪识别服务
```

### 11.5 简历项目应呈现的亮点

简历中不要只写“部署了某某模型”。建议写成面向业务与工程闭环的项目成果：

- 设计并实现一个移动端 AI 助手 Demo，支持流式对话、多轮会话、Prompt 模板、语音情绪识别与模型路由。
- 使用 vLLM 部署 9B 级语言模型服务，记录 TTFT、tokens/s、显存峰值与并发吞吐，并与 llama.cpp GGUF 量化方案进行对比。
- 在 Kaggle 双 T4 环境完成 FP16、INT8、4bit、GGUF、AWQ/GPTQ 等方案验证，输出速度、资源占用、效果精度三维评测报告。
- 在移动端侧验证量化模型加载、模型文件下载与校验、本地缓存、弱网重试和隐私开关等能力。
- 建立端云协同链路：移动端 App → AI Gateway → vLLM/llama.cpp/emotion2vec，并接入请求日志和性能指标面板。

### 11.6 面试可讲述的问题清单

完成升级后，应能围绕以下问题展开回答：

1. **流式体验**：为什么要用 SSE 或 WebSocket？如何处理停止生成、断线重连、重复 Token 与 UI 回压？
2. **上下文管理**：多轮会话如何裁剪？如何估算 Token？历史消息何时摘要化？
3. **模型路由**：什么时候走云端 9B 模型？什么时候走端侧小模型？失败后如何降级？
4. **量化取舍**：GGUF Q4/Q5、AWQ、GPTQ、bitsandbytes 4bit 的速度、显存、质量差异是什么？
5. **端侧部署**：模型文件如何下载、校验、解压、版本管理和灰度？如何控制包体积？
6. **性能优化**：如何优化首 Token 延迟、内存占用、功耗、卡顿、Crash/ANR？
7. **弱网处理**：请求超时、重试、幂等、缓存、离线提示如何设计？
8. **语音链路**：录音权限、采样率、音频格式、情绪识别延迟和错误案例如何处理？
9. **安全合规**：如何做隐私保护、敏感内容过滤、日志脱敏和用户数据清理？

### 11.7 新增里程碑：从模型实验到可求职项目

| 里程碑 | 目标 | 验收标准 |
| --- | --- | --- |
| M6：移动端壳工程 | 建立 Android/iOS/Flutter 任一客户端 | 能打开 App，完成基础导航、设置页、日志面板 |
| M7：流式对话 | 接入云端 LLM API | 支持流式输出、停止、重试、多轮历史 |
| M8：语音情绪页 | 接入 emotion2vec | 能录音/选择音频并展示情绪识别结果 |
| M9：端侧推理 Demo | 接入 llama.cpp/ONNX/Core ML/MNN 任一框架 | 能加载小模型或 GGUF 量化模型并离线返回结果 |
| M10：弱网与缓存 | 完成移动端工程能力补强 | 支持超时重试、会话本地存储、无网提示 |
| M11：性能与稳定性报告 | 对齐岗位性能要求 | 输出启动耗时、内存、TTFT、Crash/ANR、耗电优化记录 |
| M12：求职材料 | 项目包装与面试准备 | README、架构图、演示视频、实验报告、简历描述全部完成 |

### 11.8 推荐最终仓库成果形态

```text
LLM_Demo/
├── mobile_app/                 # Android/iOS/Flutter 客户端
├── server/                     # AI Gateway、SSE/WebSocket、模型路由
├── inference/                  # vLLM、llama.cpp、emotion2vec 推理脚本
├── notebooks/                  # Kaggle 双 T4 实验
├── docs/
│   ├── 移动端AI应用架构设计.md
│   ├── Kaggle部署量化与Agent验证计划.md
│   ├── 端侧模型部署与量化对比报告.md
│   ├── 弱网缓存与流式体验设计.md
│   └── 面试问题与项目复盘.md
└── outputs/
    ├── benchmarks/
    ├── figures/
    └── demo_video/
```

### 11.9 最终判断

Open-AutoGLM(9B) 与 emotion2vec **适合作为项目中的“云端大模型能力”和“语音情绪能力”两个模块**，但它们不是该岗位的全部重点。要真正支撑移动端 AI 岗位求职，项目必须补齐：

1. 一个可运行的移动端 App。
2. 流式对话、多轮上下文、Prompt 模板和会话历史。
3. 弱网、缓存、重试、模型路由和降级。
4. 端侧量化推理 Demo，例如 llama.cpp/GGUF、ONNX Runtime Mobile、Core ML、MNN 或 NCNN。
5. 性能稳定性报告，包括 TTFT、内存、卡顿、Crash/ANR、包体积、功耗。

完成这些后，本项目就可以从“模型部署学习项目”升级为“移动端 AI 应用端云协同项目”，能够更有力地支撑你应聘该岗位或类似岗位。

## 12. Kaggle Notebook 分阶段实施总计划

### 12.1 为什么需要重新完整规划

是的，接下来应该把项目完整规划成“目标明确、方案闭环、分阶段交付、每阶段可验证”的工程项目，而不是零散地跑几个模型。原因有三点：

1. **求职需要讲清楚完整链路**：岗位关注的不只是模型能否跑起来，还关注移动端产品化、服务接入、流式体验、上下文管理、弱网容错、端侧推理、性能稳定性与协作交付。
2. **Kaggle 实验需要可复现**：Kaggle Session 会重启，网络、磁盘、GPU 资源都有约束，必须将模型缓存、依赖安装、实验数据、日志与输出表格标准化。
3. **项目成果需要可展示**：最终需要形成 Notebook、脚本、报告、架构图、指标表、Demo 视频和简历描述，而不是只保留运行截图。

因此，建议将本项目拆成三条主线同步推进：

| 主线 | 目标 | 最终证明的能力 |
| --- | --- | --- |
| Kaggle 模型部署与量化主线 | 在双 T4 上完成 Open-AutoGLM(9B)、vLLM、llama.cpp、量化对比 | 大模型部署、推理优化、评测分析 |
| 语音与多模态能力主线 | 完成 emotion2vec 语音情绪识别验证 | AI 语音能力接入、音频预处理、模型服务化 |
| 移动端 AI 应用工程主线 | 规划并逐步实现 App/API/端侧推理/弱网缓存 | 匹配移动端 AI 应用岗位的工程闭环 |

### 12.2 总体项目目标

本项目最终目标是完成一个能用于求职展示的 **移动端 AI 应用端云协同实验项目**：

> 在 Kaggle 双 T4 环境中完成云端大模型与语音模型部署、量化推理和性能评测；同时设计面向移动端 App 的 AI Gateway、流式对话、上下文管理、模型路由、弱网缓存与端侧量化推理方案，最终形成可复现实验 Notebook、技术报告和移动端 AI 应用架构方案。

拆解为 5 个可验收目标：

1. **模型能跑**：Open-AutoGLM(9B) 与 emotion2vec 均能在固定环境中完成最小推理。
2. **指标可量化**：每种部署/量化方案都输出 TTFT、总延迟、tokens/s、显存、内存、模型大小和质量评分。
3. **方案可对比**：Transformers、vLLM、llama.cpp、不同量化格式之间有统一实验矩阵。
4. **移动端可落地**：明确 App、AI Gateway、模型服务、端侧推理、缓存和弱网处理的接口与实现路径。
5. **成果可求职**：输出 README、架构图、Notebook、报告、简历描述、面试问题复盘。

### 12.3 总体技术方案

```text
Kaggle / Cloud 实验层
  ├── Notebook 01：环境检查、依赖、GPU、磁盘、模型缓存
  ├── Notebook 02：Open-AutoGLM(9B) Transformers 基线推理
  ├── Notebook 03：Open-AutoGLM(9B) vLLM 推理与双 T4 测试
  ├── Notebook 04：llama.cpp / GGUF 量化对比
  ├── Notebook 05：emotion2vec 语音情绪识别
  └── Notebook 06：统一 Benchmark 汇总与报告生成
          │
          ▼
AI 服务层
  ├── OpenAI-compatible API / FastAPI
  ├── SSE / WebSocket 流式输出
  ├── Prompt 模板、上下文裁剪、会话历史
  ├── 模型路由、降级、重试、缓存
  └── 指标日志：TTFT、tokens/s、错误率、显存
          │
          ▼
移动端应用层（后续实现）
  ├── 对话页、语音页、历史会话页、设置页
  ├── 弱网提示、本地缓存、重试、离线模式
  ├── 端侧小模型 / GGUF / ONNX / Core ML / MNN Demo
  └── 性能稳定性：启动、内存、卡顿、Crash/ANR、耗电
```

### 12.4 Kaggle Notebook 拆分计划

| Notebook | 名称 | 目标 | 主要内容 | 产出 |
| --- | --- | --- | --- | --- |
| 01 | `01_kaggle_env_check.ipynb` | 建立 Kaggle 基线环境 | GPU/CPU/磁盘检查、依赖版本、目录创建、Dataset 挂载测试 | `outputs/logs/env_report.md` |
| 02 | `02_model_cache_prepare.ipynb` | 解决模型重复下载问题 | 下载/挂载模型、校验文件、保存 Kaggle Dataset、记录许可证与版本 | `outputs/logs/model_cache_report.md` |
| 03 | `03_open_autoglm_transformers_baseline.ipynb` | 跑通 9B 模型基线 | Transformers 加载、Prompt 集、FP16/8bit/4bit 尝试、OOM 记录 | `outputs/benchmarks/open_autoglm_baseline.csv` |
| 04 | `04_open_autoglm_vllm_benchmark.ipynb` | 测试 vLLM 与双 T4 | vLLM API、并发、`tensor_parallel_size=2`、吞吐/延迟曲线 | `outputs/benchmarks/vllm_benchmark.csv` |
| 05 | `05_llamacpp_gguf_quant.ipynb` | 测试 GGUF 量化 | 模型转换、Q4/Q5/Q8、CPU/GPU offload、质量样例 | `outputs/benchmarks/llamacpp_gguf.csv` |
| 06 | `06_emotion2vec_ser.ipynb` | 跑通语音情绪识别 | 音频预处理、批量推理、CPU/GPU 对比、错误案例 | `outputs/benchmarks/emotion2vec.csv` |
| 07 | `07_api_gateway_streaming_demo.ipynb` | 验证服务接入方式 | FastAPI/SSE/OpenAI-compatible API、重试、缓存、日志 | `outputs/logs/api_gateway_demo.md` |
| 08 | `08_benchmark_report.ipynb` | 汇总实验报告 | 读取所有 CSV，生成图表、结论、选型建议 | `outputs/reports/final_benchmark_report.md` |
| 09 | `09_mobile_app_design.ipynb` | 输出移动端落地方案 | App 页面、接口协议、状态管理、弱网与端侧推理方案 | `outputs/reports/mobile_ai_app_design.md` |

### 12.5 分阶段实施计划

#### 阶段 A：项目基建与实验规范

**目标**：先把 Kaggle 实验环境、目录、输出格式和记录模板统一下来。

**任务**：

1. 创建标准目录：`notebooks/`、`scripts/`、`data/`、`outputs/logs/`、`outputs/benchmarks/`、`outputs/reports/`。
2. 编写环境检查 Notebook：记录 GPU 型号、显存、CUDA、Python、PyTorch、Transformers、vLLM、llama.cpp 版本。
3. 设计统一 Benchmark CSV 字段。
4. 准备固定 Prompt 集和音频样例清单。

**验收标准**：

- 能在 Kaggle 上一键运行环境检查。
- 每次实验都能保存日志和 CSV。
- 所有 Notebook 使用相同输出目录与命名规则。

#### 阶段 B：模型缓存与最小推理

**目标**：解决 Kaggle 重启后模型重复下载的问题，并跑通两个核心模型。

**任务**：

1. 将 Open-AutoGLM(9B) 权重缓存为 Kaggle Dataset 或挂载路径。
2. 将 emotion2vec 权重和音频样例缓存为 Dataset。
3. 分别完成一个最小推理样例。
4. 记录模型加载时间、磁盘大小、显存峰值和失败日志。

**验收标准**：

- Open-AutoGLM(9B) 能对固定 Prompt 返回文本。
- emotion2vec 能对固定音频返回情绪标签。
- 所有模型路径、版本和运行命令写入报告。

#### 阶段 C：Open-AutoGLM(9B) 基线与量化

**目标**：明确 9B 模型在 Kaggle T4 上哪些方式能跑，哪些会 OOM，哪些值得后续优化。

**任务**：

1. Transformers FP16/BF16 基线加载测试。
2. bitsandbytes INT8、NF4、FP4、double quant 测试。
3. 固定 10–20 条 Prompt，测量 TTFT、总延迟、tokens/s、峰值显存。
4. 记录输出质量：正确性、完整性、幻觉、格式遵循能力。

**验收标准**：

- 得到一张 Open-AutoGLM(9B) 部署方式对比表。
- 明确 Kaggle 单 T4/双 T4 推荐配置。
- 形成“云端 LLM 服务基线结论”。

#### 阶段 D：vLLM 高吞吐推理测试

**目标**：验证岗位中“APP 与大模型服务接入与优化”的服务端推理基础。

**任务**：

1. 使用 vLLM 加载模型或可兼容替代模型。
2. 测试单请求、批量请求、并发请求。
3. 测试双 T4 张量并行：`tensor_parallel_size=2`。
4. 输出吞吐-延迟曲线，分析 TTFT 与 batch size 的关系。

**验收标准**：

- 输出 vLLM 与 Transformers 的延迟/吞吐对比。
- 给出适合移动端 AI App 后端的推荐服务配置。
- 记录不兼容项和替代方案。

#### 阶段 E：llama.cpp / GGUF 端侧量化启发实验

**目标**：为岗位加分项“端侧模型部署与量化模型运行”积累证据。

**任务**：

1. 选择合适的小模型或可转换模型进行 GGUF 转换。
2. 对比 Q4_K_M、Q5_K_M、Q8_0 等格式。
3. 测试 CPU、GPU offload、不同线程数或 batch 配置。
4. 总结哪些模型适合移动端/边缘端，哪些只适合云端。

**验收标准**：

- 输出 GGUF 量化对比表。
- 形成端侧推理选型建议：llama.cpp、MLC-LLM、ONNX Runtime Mobile、Core ML、MNN、NCNN 的后续验证优先级。

#### 阶段 F：emotion2vec 语音情绪识别

**目标**：形成“语音输入/播报/情绪理解”方向的 AI 能力模块。

**任务**：

1. 准备标准音频样例。
2. 统一采样率、声道和格式。
3. 批量推理并输出情绪标签、置信度和延迟。
4. 分析噪声、音量、语速、采样率对结果的影响。

**验收标准**：

- 输出 emotion2vec 结果 CSV。
- 形成移动端语音能力接入建议，包括录音权限、音频格式、上传策略、隐私提示。

#### 阶段 G：AI Gateway 与移动端接口设计

**目标**：把 Kaggle 模型实验转成移动端 App 可接入的服务方案。

**任务**：

1. 设计 OpenAI-compatible `/chat/completions` 接口。
2. 支持 SSE 或 WebSocket 流式输出。
3. 增加 Prompt 模板、上下文裁剪、会话 ID、请求 ID。
4. 设计模型路由策略：云端 9B、轻量模型、端侧模型、失败降级。
5. 设计缓存、重试、超时和错误码。

**验收标准**：

- 输出接口协议文档。
- Notebook 或脚本能模拟移动端请求并收到流式结果。
- 生成移动端需要处理的状态机说明。

#### 阶段 H：移动端 App 方案与后续实现

**目标**：将本项目从 Kaggle 实验延伸到真正匹配岗位的移动端工程项目。

**任务**：

1. 确定移动端技术栈：Android Kotlin、iOS Swift 或 Flutter。
2. 设计页面：对话页、语音页、历史页、设置页、性能面板。
3. 设计本地存储：会话、Prompt 模板、模型配置、缓存队列。
4. 设计端侧推理：llama.cpp/GGUF 或 ONNX/Core ML/MNN 小模型 Demo。
5. 设计性能指标：启动耗时、首 Token、内存、耗电、Crash/ANR、包体积。

**验收标准**：

- 输出移动端架构设计文档。
- 明确第一版 App 的 MVP 范围。
- 能将 Kaggle 侧实验数据转化为 App 技术选型依据。

### 12.6 每个 Notebook 的统一输出规范

每个 Notebook 都必须在开头记录：

```text
实验名称：
实验日期：
Kaggle GPU：
Python 版本：
CUDA / PyTorch / Transformers / vLLM 版本：
模型名称与版本：
模型路径：
量化方式：
输入数据版本：
输出目录：
```

每个 Benchmark CSV 建议统一字段：

```text
experiment_id,model,framework,precision,quant_method,device,gpu_count,
prompt_id,input_tokens,output_tokens,ttft_ms,total_latency_ms,tokens_per_sec,
peak_gpu_memory_mb,cpu_memory_mb,model_size_gb,quality_score,error_type,notes
```

每个阶段至少输出 4 类材料：

1. Notebook 源文件。
2. CSV 指标文件。
3. Markdown 结论报告。
4. 关键日志或截图。

### 12.7 优先级建议

如果时间有限，建议按以下优先级执行：

| 优先级 | 内容 | 原因 |
| --- | --- | --- |
| P0 | 环境检查、模型缓存、Open-AutoGLM(9B) 最小推理、emotion2vec 最小推理 | 先证明核心模型可运行 |
| P1 | vLLM Benchmark、llama.cpp/GGUF Benchmark、统一报告 | 形成量化与推理优化亮点 |
| P2 | AI Gateway、SSE 流式接口、Prompt/上下文管理 | 对齐移动端 AI App 服务接入要求 |
| P3 | 移动端 App MVP、弱网缓存、端侧推理 Demo | 对齐目标岗位核心职责与加分项 |
| P4 | GUI-Agent、本地 GUI 自动化、演示视频 | 作为扩展亮点，不应优先于移动端主线 |

### 12.8 最终交付检查清单

| 类别 | 交付物 | 是否必须 |
| --- | --- | --- |
| 项目说明 | README、项目目标、架构图、运行顺序 | 必须 |
| Kaggle 实验 | 8–9 个分阶段 Notebook | 必须 |
| 模型报告 | Open-AutoGLM(9B)、emotion2vec 部署报告 | 必须 |
| 量化报告 | vLLM、llama.cpp、GGUF、AWQ/GPTQ/INT8/4bit 对比 | 必须 |
| 服务方案 | AI Gateway、SSE/WebSocket、模型路由、缓存重试 | 必须 |
| 移动端方案 | App MVP 设计、接口协议、弱网状态机、端侧推理方案 | 必须 |
| 求职材料 | 简历项目描述、面试问题、Demo 视频、截图 | 必须 |
| GUI-Agent | 远端 Agent 与本地 GUI 自动化验证 | 可选增强 |

### 12.9 下一步执行顺序

接下来建议不要立刻写复杂代码，而是按以下顺序推进：

1. **先建 `01_kaggle_env_check.ipynb`**：确认 Kaggle 双 T4、依赖、磁盘和输出目录。
2. **再建 `02_model_cache_prepare.ipynb`**：解决模型权重和数据集缓存问题。
3. **再跑两个最小闭环**：Open-AutoGLM(9B) 文本生成、emotion2vec 音频情绪识别。
4. **再做量化对比**：vLLM 与 llama.cpp 分别做，不要混在一个 Notebook。
5. **最后汇总报告**：用统一 CSV 生成表格和图，沉淀岗位导向的技术选型结论。

这样做的好处是：每一步都有独立成果；中途失败也能保留日志；最终可以自然组织成 README、报告、简历和面试故事线。
