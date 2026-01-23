# Fun-ASR-GGUF

将 [Fun-ASR-Nano](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) 模型转换为可以在本地高效运行的格式，实现**准确、快速的离线语音识别**。主要依赖了 [llama.cpp](https://github.com/ggml-org/llama.cpp) 对 LLM Decoder 的加速推理。

### 核心特性

- ✅ **纯本地运行** - 无需网络，数据不外传
- ✅ **速度快** - 混合推理架构，支持 GPU 加速
- ✅ **准确率高** - Encoder 保持 FP32 精度
- ✅ **内存占用小** - CTC Decoder 和 LLM Decoder 使用 INT8 量化
- ✅ **上下文增强** - 可提供上下文信息，进一步提升识别准确率
- ✅ **支持热词** - 通过 CTC 预识别提取热词，提高专业领域识别准确率
- ✅ **时间戳精确** - 字符级时间戳对齐
- ✅ **较长音频支持** - 单次可识别长达 60 秒的音频

---

## 快速开始

### 1. 安装依赖

导出模型需要：

```bash
pip install torch torchaudio transformers onnxruntime modelscope onnxscript sentencepiece
```

推理需要：

```bash
pip install onnx onnxruntime numpy pydub gguf watchdog rich pypinyin
```

>  `pydub` 用于音频格式转换，需要系统安装 [ffmpeg](https://ffmpeg.org/download.html)


从 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 下载预编译二进制文件：

- Windows: 下载 `llama-bXXXX-bin-win-vulkan-x64.zip`

解压后将以 `dll` 文件放入 `fun_asr_gguf/` 文件夹：

> MacOS 和 Linux 也有对应的预编译文件，但我没有做测试

### 2. 下载模型（可选，如已有导出模型可跳过）

下载原始模型

```bash
pip install modelscope
modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512 --local_dir ./Fun-ASR-Nano-2512
```

导出模型

```bash
# 导出 Encoder (FP32) + CTC Decoder (INT8)
python 01-Export-Encoder-Adaptor-CTC.py

# 导出 LLM Decoder (INT8)
python 02-Export-Decoder-GGUF.py
```

### 3. 运行识别

```python
from fun_asr_gguf import create_asr_engine

engine = create_asr_engine(
    encoder_onnx_path="model/Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx",
    ctc_onnx_path="model/Fun-ASR-Nano-CTC.int8.onnx",
    decoder_gguf_path="model/Fun-ASR-Nano-Decoder.q8_0.gguf",
    tokens_path="model/tokens.txt",
    hotwords_path="hot.txt", # 可选：热词文件路径，支持运行期间实时修改
    similar_threshold=0.6,   # 可选：热词模糊匹配阈值，默认 0.6
    max_hotwords=10,         # 可选：最多提供给 LLM 的热词数量，默认 10
)
engine.initialize()

result = engine.transcribe("audio.mp3", language="中文")
print(result.text)
```

就这么简单！

> 单段音频长度在60秒内可准确识别，过长会有问题

---

## 工作原理

```
音频输入
    ↓
┌─────────────────────────────────────────────┐
│  Encoder (ONNX, FP32)        → 音频特征      │  保持最高精度
│  CTC Decoder (ONNX, INT8)    → 粗识别结果    │  快速预识别
└─────────────────────────────────────────────┘
    ↓              ↓              ↓
  音频特征      时间戳         热词候选
    ↓              ↓              ↓
┌─────────────────────────────────────────────┐
│  构建 Prompt (Prefix + 音频 + Suffix)        │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  LLM Decoder (GGUF, INT8, llama.cpp)        │  支持 Vulkan GPU
│  ↓                                          │
│  生成最终识别文本                             │
└─────────────────────────────────────────────┘
    ↓
  时间戳对齐 → 输出结果
```

### 为什么这样做？

1. **CTC 预识别** 提供两样东西：
   - 粗识别结果（用于筛选热词，提供给LLM）
   - 时间戳信息（为LLM的文本输出赋予时间戳）

2. **混合架构** 各取所长：
   - ONNX Runtime：运行 Encoder 和 CTC，稳定可靠
   - llama.cpp：运行 LLM，支持 GPU 加速(Vulkan, Metal)，速度超快

3. **精度策略**：
   - **Encoder 用 FP32** - 保证识别准确率（INT8 影响大）
   - **CTC 和 LLM 用 INT8** - 节省内存，速度提升，准确率影响小

### 内存占用

 - **Encoder**（约 200M 参数）
   - FP32：~800MB 内存
   - INT8：~200MB 内存（但不推荐，准确率下降明显）
 - **CTC Decoder**：内存占用可忽略（几十 MB）
 - **LLM Decoder**（约 600M 参数）
   - INT8：~1.2GB 内存或显存（显卡内会用fp16进行计算，但比原生fp16要快）
   - FP16：~1.2GB 内存或显存

 **总内存占用**（推荐 FP32 Encoder + INT8 Decoder）：约 **2GB**（CPU 内存或 GPU 显存）

---

## 使用示例

### 基础用法

```python
from fun_asr_gguf import create_asr_engine

engine = create_asr_engine(
    encoder_onnx_path="model/Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx",
    ctc_onnx_path="model/Fun-ASR-Nano-CTC.int8.onnx",
    decoder_gguf_path="model/Fun-ASR-Nano-Decoder.q8_0.gguf",
    tokens_path="model/tokens.txt",
    hotwords_path="hot.txt",  # 可选：热词文件路径
    similar_threshold=0.6,    # 可选：热词匹配阈值
    max_hotwords=10,          # 可选：最多召回热词数
)
engine.initialize()

# 转录音频
result = engine.transcribe("audio.mp3")
print(result.text)           # 识别文本
print(result.segments)       # 带时间戳的分段
print(result.timings)        # 各阶段耗时
```

### 指定语言和上下文

```python
result = engine.transcribe(
    "audio.mp3",
    language="中文",        # None=自动检测, "中文", "英文", "日文" 等
    context="这是技术会议讨论深度学习"  # 上下文信息可以提高准确率
)
```


### 热词配置

创建 `hot.txt` 文件（每行一个热词）：

```text
督工
静静
深度学习
神经网络
```

**特性：**
1. **实时更新**：识别程序运行期间，你可以随时修改 `hot.txt` 并保存，程序会通过 `watchdog` 自动更新内存中的热词库，无需重启。
2. **模糊召回**：热词可以有几千条、上万条，程序会根据 CTC 的粗识别结果进行音素级别的模糊匹配，找出相似度在 `similar_threshold` 以上的热词，并取前 `max_hotwords` 条（默认10条）作为上下文提供给 LLM 纠错。

---

## 性能参考

以下是在小新Pro16GT（U9-258H + RTX5050）笔记本上的效果，60秒的睡前消息音频，转录用时2.59秒。

需要注意的是，**LLM Decoder 所需时间取决于吐出文字的数量，不适合用 RTF 描述**，睡前消息音频的文字密度非常高，短短60秒就有350个字，但这段音频的速度可以作为下限参考，即 RTF 最慢也不会慢过 0.04

在文字密度更低的音频上，识别速度还能更快。

```
======================================================================
处理音频: input.mp3
======================================================================

[1] 加载音频...
    音频长度: 60.00s

[2] 音频编码...
    耗时: 934.21ms

[3] CTC 解码...
    CTC: 大家好二零二六年一月十一日星期日欢迎收看一千零四起事间消息请静静介绍话题去年十月十九
日九百六十七期节目说到韦内瑞拉问题我们回顾一下你当时的评论无论是从集节的兵力来看还这种动机来 
看特朗普政府并不打算对韦伦瑞拉政权发动全面的进攻最多是发动象征性的轰炸进行政投击在诺贝尔和平 
鸟发给了韦内瑞拉反对派之后美国军队进攻的概率进一步降低现在美国突袭韦内瑞拉抓走了总统马杜罗杜 
工你怎么看待两个月之前的判断当初的判断不变美国对于韦内瑞拉的突袭性质依然是政治投击不能算是地 
面战争入侵的美国军队总数是以两百站在韦伦瑞拉领土上的时间不超过一个小时算是地面战争或者全面进 
攻实在有点勉强当然美国动用总力量并不小一五十架先进飞机加上经年累月不止的情报网络这放在东亚或 
者欧洲也不是一支很小的力量用到美国的西半球主场压倒韦伦瑞拉的军队那是必然的
    热词: ['睡前消息', '督工']
    耗时: 133.76ms

[4] 准备 Prompt...
--------------- Prefix Prompt ---------------
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
请结合上下文信息，更加准确地完成语音转写任务。


**上下文信息：**这是1004期睡前消息节目，主持人叫督工，助理叫静静


热词列表：[睡前消息, 督工]
语音转写：
----------------------------------------
    Prefix: 72 tokens
    Suffix: 5 tokens

[5] LLM 解码...
======================================================================
大家好，2026年1月11日星期日，欢迎收看1004期《睡前消息》。请静静介绍话题。去年10月19日967期节目说到委内瑞拉问题，我们回顾一下你当时的评论。无论是从集结的兵力来看，还是从动机来看，特朗普政府并不打算对委内瑞拉政权发动全面的进攻，最多是发动象征性的轰炸进行政治投机。在诺贝尔和平奖发给了委内瑞拉反对派之后，美国军队进攻的概率进一步降低。现在美国突袭委内瑞拉，抓走了总统马杜罗。杜工，你怎么看待两个月之前的判断？当初的判断不变，美国对于委内瑞拉的突袭性质依然是政治投机，不能算是地面战争。入侵的美国军队总数是以200占在委内瑞拉领土上的时间不超过一个小时，算是地面战争或者全面进攻，实在有强。当然，美国动用总力量并不小，150架先进飞机，加上经年累月部署的情报网络，这放在东亚或者欧洲也不是一只很小的力量，用到美国的西半球主场压倒委内瑞拉的军队那是必然的。
======================================================================

[6] 时间戳对齐
    对齐耗时: 118.78ms
    结果预览: 大(1.23s) 家(1.35s) 好(1.47s) ，(1.62s) 2(1.77s) 0(1.89s) 2(2.01s) 6(2.13s) 年(2.25s) 1(2.43s) ...

[统计]
  音频长度:  60.00s
  Decoder输入:   6994 tokens/s (总: 203, prefix:72, audio:126, suffix:5)
  Decoder输出:    213 tokens/s (总: 256)

[转录耗时]
  - 音频编码：   934ms
  - CTC解码：    134ms
  - LLM读取：     29ms
  - LLM生成：   1202ms
  - 时间戳对齐:  119ms
  - 总耗时：    2.59s
```

同一段音频，纯 CPU 推理速度：

```
[统计]
  音频长度:  60.00s
  Decoder输入:    318 tokens/s (总: 203, prefix:72, audio:126, suffix:5)
  Decoder输出:     51 tokens/s (总: 255)

[转录耗时]
  - 音频编码：   922ms
  - CTC解码：    145ms
  - LLM读取：    637ms
  - LLM生成：   5016ms
  - 时间戳对齐:  149ms
  - 总耗时：    7.04s
```

---

## 常见问题

### Q: Encoder 为什么不用 INT8？

A: 测试发现 Encoder 用 INT8 会有可观察到的准确率下降，建议保持 FP32，延迟上也就差个100毫秒左右。

### Q: 如何选择量化级别？

A:
- **Encoder**：尽量 FP32，保证精度
- **CTC Decoder 和 LLM Decoder**：推荐 Q8_0（INT8），速度快且准确率影响小

### Q: 支持哪些语言？

A: Fun-ASR-Nano-2512 支持中文、英文、日文。Fun-ASR-MLT-Nano-2512 还支持更多语言（粤语、韩文、越南语等）。

### Q: 如何提高识别准确率？

A:
1. 使用 `context` 参数提供上下文信息
2. 配置 `hot.txt` 添加领域热词
3. 指定正确的 `language` 参数

---

## 文件说明

### 核心文件

- `01-Export-Encoder-Adaptor-CTC.py` - 导出 Encoder 和 CTC Decoder
- `02-Export-Decoder-GGUF.py` - 导出 LLM Decoder
- `03-Inference.py` - 完整的使用示例
- `fun_asr_gguf/` - 核心推理引擎

### 导出的模型

```
model/
├── Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx   # 音频编码器 (FP32)
├── Fun-ASR-Nano-CTC.int8.onnx               # CTC 解码器 (INT8)
├── Fun-ASR-Nano-Decoder.q8_0.gguf           # LLM 解码器 (INT8)
└── tokens.txt                               # CTC Token 映射
```

---

## 技术细节

### 架构设计

- **Encoder**：ONNX 格式，FP32，提取音频特征
- **CTC Decoder**：ONNX 格式，INT8 量化，用于时间戳和热词候选
- **LLM Decoder**：GGUF 格式，INT8 量化，llama.cpp 推理


---

## 致谢

- [Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) - 原始模型
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - GGUF 推理引擎
