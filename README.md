# FunASR-Nano GGUF 转换与推理指南

本项目提供了一套完整的脚本，用于将 FunASR-Nano 模型转换为 GGUF 格式，并实现纯本地推理（支持 ONNX Encoder + GGUF Decoder）。

## 1. 准备工作

请在项目根目录下，下载模型文件，以及克隆以下三个必要的仓库代码：

```powershell
# 1. 下载 FunASR-Nano 模型 (需安装 modelscope)
pip install modelscope
modelscope download --model FunAudioLLM/Fun-ASR-Nano-2512 --local_dir ./Fun-ASR-Nano-2512

# 2. 克隆 FunASR 源代码 (用于模型导出)
git clone https://github.com/FunAudioLLM/Fun-ASR.git

# 3. 克隆 llama.cpp 源代码 (用于 GGUF 转换)
git clone https://github.com/ggml-org/llama.cpp.git

# 4. 克隆 llama-cpp-python 源代码 (用于供 AI 查询 API，修改脚本)
git clone https://github.com/abetlen/llama-cpp-python.git
```


## 2. 环境依赖

确保已安装以下 Python 库：

```bash
pip install torch torchaudio transformers numpy onnx pydub onnxruntime funasr
# 如果需要 GPU 支持
# pip install onnxruntime-gpu
```

以及 `llama-cpp-python` (用于推理)：
```bash
pip install llama-cpp-python
```

## 3. 执行步骤

### 第一步：导出 ONNX 模型 (Audio Encoder)

运行脚本 `01_Export_ONNX.py`：

```powershell
python 01_Export_ONNX.py
```

*   **功能**：加载 FunASR 模型，提取音频编码器（Encoder）和 Embedding层，导出为 ONNX 格式。
*   **输出**：`model-gguf/` 目录下的 `.onnx` 文件。

### 第二步：导出 GGUF 模型 (LLM Decoder)

运行脚本 `02_Export_GGUF.py`：

```powershell
python 02_Export_GGUF.py
```

*   **功能**：
    1.  提取 LLM 部分的权重，保存为 Hugging Face 标准格式（Safetensors）。
    2.  调用 `llama.cpp` 的转换工具，将 HF 模型转换为 GGUF 格式。
*   **输出**：`model-gguf/qwen3-0.6b-asr.gguf`

### 第三步：运行推理

运行脚本 `03_Inference.py`：

```powershell
python 03_Inference.py
```

*   **功能**：
    1.  使用 ONNX Runtime 运行 Audio Encoder，从音频提取 Embedding。
    2.  使用 `llama-cpp-python` 加载 GGUF 模型。
    3.  通过 "Pure Embedding" 模式，将音频特征直接注入 LLM 生成文本。
*   **配置**：可以在脚本中修改 `test_audio = './input.mp3'` 来测试不同的音频文件。

推理效果，可以见 `04_Inference.ipynb` 中的输出。

## 目录结构说明

执行完上述步骤后，`model-gguf` 文件夹将包含完整的推理所需文件：

*   `FunASR_Nano_Encoder.onnx` : 音频编码器
*   `FunASR_Nano_Decoder_Embed.onnx` : 文本 Embedder
*   `qwen3-0.6b-asr.gguf` : LLM 主模型
*   `Qwen3-0.6B/` : 分词器文件 (Tokenizer)


## 速度说明

如果启用了 vulkan 编译 llama-cpp-python，跑 CPU 的版本的脚本的话，需要显示用 `$env:VK_ICD_FILENAMES="none"`  环境变量禁止 Vulkan。

在 GPU 上跑，我的机器实测一次最多注入8个 embedding，限制到了注入 embedding 的速度，似乎是 ubatch 有限，但这是当前版本的 llama-cpp-python 的限制，实际上 llama.cpp 跑到 512 的 ubatch 都能跑，

FP16 模型 RTX 5050 Vulkan 解码速度 `03_Inference_gpu.py`：
```
[*] Generating Text:
----------------------------------------
，星期日，欢迎收看一千零四起事件消息，请静静介绍话题。去年十月十九日，九百六十七期节目说到委内瑞拉
问题，我们回顾一下你当时的评。
----------------------------------------
[*] Generation Speed: 134.18 tokens/s
```

FP16 模型 CPU 解码速度 `03_Inference_cpu.py`：
```
=== 推理切片 [0:160000] ===
Final Input Shape: (155, 1024)
2026-01-20 08:37:04,333 - INFO - 注入 Total Embeddings Shape: (155, 1024)
2026-01-20 08:37:04,363 - INFO - 正在注入融合 Embedding...
init: embeddings required but some input tokens were not marked as outputs -> overriding
2026-01-20 08:37:05,293 - INFO - 开始生成文本 (最大 1024 tokens)...

，星期日，欢迎收看一千零四期誓言消息，请静静介绍话题。去年十月十九日，九百六十七期节目说到委内瑞拉
问题，我们回顾一下你当时的评。


2026-01-20 08:37:06,647 - INFO - 解码速度: 33.27 tokens/s (45 tokens in 1.35s)
```


INT8 模型 CPU 解码速度 `03_Inference_cpu_int8.py`：
```
=== 推理切片 [0:160000] ===
Final Input Shape: (155, 1024)
2026-01-20 08:44:50,934 - INFO - 注入 Total Embeddings Shape: (155, 1024)
2026-01-20 08:44:50,954 - INFO - 正在注入融合 Embedding...
init: embeddings required but some input tokens were not marked as outputs -> overriding
2026-01-20 08:44:51,937 - INFO - 开始生成文本 (最大 1024 tokens)...

，星期日，欢迎收看一千零四期誓言消息，请静静介绍话题。去年十月十九日，九百六十七期节目说到委内瑞拉
问题，我们回顾一下你当时的评。


2026-01-20 08:44:52,792 - INFO - 解码速度: 52.64 tokens/s (45 tokens in 0.85s)
```