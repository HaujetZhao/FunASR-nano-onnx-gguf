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
