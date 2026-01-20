# pypi 版本的 llama-cpp-python 默认没有 vulkan
# 要想支持 vulkan 需要从源码编译 llama-cpp-python
# 电脑上要先装上 vulkan SDK
# 通过设置环境变量 CMAKE_ARGS = "-DGGML_VULKAN=ON"  启用 vulkan
# 再编译 llama-cpp-python


import ctypes
import numpy as np
import logging
import time
import onnxruntime
import torch
import torchaudio
import sys
import os
from pydub import AudioSegment
from transformers import AutoTokenizer
import pickle

from llama_cpp import (
    Llama,
    llama_batch_init,
    llama_batch_free,
    llama_decode,
    llama_get_logits,
    llama_kv_self_clear,
)

# =========================================================================
# 配置部分
# =========================================================================

# 日志设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference_gpu.log", encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 模型路径
model_dir = r'./model-gguf'
tokenizer_path = f'{model_dir}/Qwen3-0.6B'

# ONNX 模型
onnx_encoder = f'{model_dir}/FunASR_Nano_Encoder.onnx'

# GGUF 模型 (用于解码)
gguf_model_path = f'{model_dir}/qwen3-0.6b-asr.gguf'

# 输入音频
test_audio = r'./input.mp3'

# Prompt 配置 (Prefix + Audio + Suffix)
PREFIX_PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
SUFFIX_PROMPT = "<|im_end|>\n<|im_start|>assistant\n"

# 音频处理参数
SAMPLE_RATE = 16000
USE_NORMALIZER = True
MAX_INPUT_AUDIO_LENGTH = 320000
SLIDING_WINDOW = 0

# 模型参数
MAX_SEQ_LEN = 1024
STOP_TOKEN = [151643, 151645]
MAX_THREADS = 0  # 0 = Auto

# GPU 特定参数
CHUNK_SIZE = 8  # GPU 推理的黄金数值，避免 NaN

# =========================================================================
# 辅助函数
# =========================================================================

def load_gguf_embeddings(model_path):
    """Load the raw token embedding table from the GGUF file."""
    gguf_py_path = os.path.abspath("./llama-cpp-python/vendor/llama.cpp/gguf-py")
    if gguf_py_path not in sys.path:
        sys.path.append(gguf_py_path)

    try:
        from gguf import GGUFReader
    except ImportError:
        logger.error("Could not import GGUFReader. Please ensure ./llama-cpp-python/vendor/llama.cpp/gguf-py exists.")
        sys.exit(1)

    logger.info(f"Reading GGUF tensors from {model_path}...")
    reader = GGUFReader(model_path, 'r')

    try:
        tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
    except StopIteration:
        logger.error("Could not find 'token_embd.weight' in GGUF file.")
        sys.exit(1)

    weights = np.array(tensor.data).reshape(-1, 1024)
    logger.info(f"Loaded Embeddings: Shape={weights.shape}, Type={weights.dtype}")

    if weights.dtype != np.float32:
        logger.info("Converting embeddings from float16/other to float32...")
        weights = weights.astype(np.float32)

    return weights

def normalizer(_audio, target_value=8192.0):
    """音频归一化处理"""
    _audio = _audio.astype(np.float32)
    rms = np.sqrt(np.mean((_audio * _audio), dtype=np.float32), dtype=np.float32)
    _audio *= (target_value / (rms + 1e-7))
    np.clip(_audio, -32768.0, 32767.0, out=_audio)
    return _audio.astype(np.int16)

def decode_with_pure_embeddings_gpu(llm, audio_embeddings, max_new_tokens=200):
    """
    纯 Embedding 解码函数 (GPU 版本)
    完全复制自 06 的优化版本
    """

    # 1. 准备数据
    embeds = audio_embeddings.squeeze()
    if len(embeds.shape) == 1:
        embeds = embeds.reshape(1, -1)

    n_tokens, n_dim = embeds.shape
    logger.info(f"注入 Total Embeddings Shape: {embeds.shape}")

    # 确保数据类型正确
    embeds = embeds.astype(np.float32)

    # 2. 初始化 Batch
    batch_embd = llama_batch_init(2048, n_dim, 1)
    batch_embd.token = ctypes.cast(None, ctypes.POINTER(ctypes.c_int32))  # 标记为 embedding

    batch_text = llama_batch_init(1, 0, 1)

    ctx = llm.ctx
    llama_kv_self_clear(ctx)
    llm.n_tokens = 0

    try:
        # ---------------------------------------------------------------------
        # 阶段 A: 分块注入融合 Embedding (GPU 优化)
        # ---------------------------------------------------------------------
        print(f"\n[*] Start Injection (Chunk Size: {CHUNK_SIZE})...")
        inject_start = time.time()

        for i in range(0, n_tokens, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, n_tokens)
            current_len = end - i

            # 准备数据切片
            chunk_data = embeds[i:end]
            if not chunk_data.flags['C_CONTIGUOUS']:
                chunk_data = np.ascontiguousarray(chunk_data)

            # 设置 Batch
            batch_embd.n_tokens = current_len
            for k in range(current_len):
                batch_embd.pos[k] = i + k
                batch_embd.n_seq_id[k] = 1
                batch_embd.seq_id[k][0] = 0
                # 仅在整个序列的最后一个 token 开启 logits
                is_last = (i + k == n_tokens - 1)
                batch_embd.logits[k] = 1 if is_last else 0

            # 内存拷贝
            ctypes.memmove(batch_embd.embd, chunk_data.ctypes.data, chunk_data.nbytes)

            # 解码
            if llama_decode(ctx, batch_embd) != 0:
                print(f"[!] Error during injection at index {i}")
                return ""

            llm.n_tokens += current_len

            # 简易进度条
            if i % 32 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()

        inject_time = time.time() - inject_start
        print(f"\n[OK] Injection Done. Time: {inject_time:.4f}s (Avg: {n_tokens/inject_time:.1f} t/s)")

        # ---------------------------------------------------------------------
        # 阶段 B: 文本生成 (Greedy Search) - 完全复制自 06
        # ---------------------------------------------------------------------
        print("\n[*] Generating Text:")
        print("-" * 40)

        vocab_size = llm.n_vocab()
        gen_start = time.time()
        gen_count = 0
        eos_token = llm.token_eos()

        full_text = ""

        for _ in range(max_new_tokens):
            # 获取 Logits
            logits = np.ctypeslib.as_array(llama_get_logits(ctx), shape=(vocab_size,))
            token_id = int(np.argmax(logits))

            if token_id == eos_token or token_id in STOP_TOKEN:
                break

            # 打印字符
            try:
                txt = llm.detokenize([token_id]).decode('utf-8', errors='ignore')
                print(txt, end="", flush=True)
                full_text += txt
                gen_count += 1
            except:
                pass

            # 下一步
            batch_text.n_tokens = 1
            batch_text.token[0] = token_id
            batch_text.pos[0] = llm.n_tokens
            batch_text.n_seq_id[0] = 1
            batch_text.seq_id[0][0] = 0
            batch_text.logits[0] = 1

            if llama_decode(ctx, batch_text) != 0:
                break
            llm.n_tokens += 1

        print("\n" + "-" * 40)
        gen_time = time.time() - gen_start
        print(f"[*] Generation Speed: {gen_count/gen_time:.2f} tokens/s")

    finally:
        # 释放资源
        llama_batch_free(batch_embd)
        llama_batch_free(batch_text)

    return full_text

# =========================================================================
# 主程序
# =========================================================================

def main():
    print('\nStarting GPU Inference Engine...')

    # 1. 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 2. 加载 GGUF 模型 (GPU 优化配置)
    print(f'\nLoading GGUF model: {gguf_model_path}')

    llm = Llama(
        model_path=gguf_model_path,
        n_ctx=MAX_SEQ_LEN + 2048,
        n_batch=2048,
        n_ubatch=64,  # 改为 64，与 06 一致
        n_gpu_layers=-1,
        main_gpu=0,
        split_mode=0,
        embedding=True,
        verbose=False
    )
    print('✅ GGUF model loaded successfully (Vulkan Backend Active)')

    # 3. 从 GGUF 文件中加载 Raw Embeddings
    print('\nLoading Raw Embeddings from GGUF...')
    all_embeddings = load_gguf_embeddings(gguf_model_path)

    # 4. 准备 Prefix 和 Suffix 的 Embeddings
    prefix_tokens = tokenizer.encode(PREFIX_PROMPT, add_special_tokens=True)
    suffix_tokens = tokenizer.encode(SUFFIX_PROMPT, add_special_tokens=True)

    prefix_emb = all_embeddings[prefix_tokens]
    suffix_emb = all_embeddings[suffix_tokens]

    logger.info(f"Prefix Shape: {prefix_emb.shape}, Suffix Shape: {suffix_emb.shape}")

    # 5. 初始化 ONNX Audio Encoder
    print('\nLoading ONNX Audio Encoder...')
    session_opts = onnxruntime.SessionOptions()
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.intra_op_num_threads = MAX_THREADS
    ort_session_A = onnxruntime.InferenceSession(onnx_encoder, sess_options=session_opts, providers=['CPUExecutionProvider'])

    in_name_A = [x.name for x in ort_session_A.get_inputs()]
    out_name_A = [x.name for x in ort_session_A.get_outputs()]
    shape_value_in_A = ort_session_A._inputs_meta[0].shape[-1]

    # Query Embed for Audio Encoder input
    query_embed_input = np.ones((1, 10, 1024), dtype=np.float32)

    # 6. 处理音频并推理
    for audio_file in [test_audio]:
        print("-" * 80)
        print(f"Test Input Audio: {audio_file}")

        # 加载和归一化音频
        audio = np.array(AudioSegment.from_file(audio_file).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
        if USE_NORMALIZER:
            audio = normalizer(audio, 8192.0)

        audio_len = len(audio)
        audio = audio.reshape(1, 1, -1)

        # 定义输入长度
        if isinstance(shape_value_in_A, str):
             INPUT_AUDIO_LENGTH = min(MAX_INPUT_AUDIO_LENGTH, audio_len)
        else:
             INPUT_AUDIO_LENGTH = shape_value_in_A

        stride_step = INPUT_AUDIO_LENGTH if SLIDING_WINDOW <= 0 else SLIDING_WINDOW

        # Padding
        if audio_len < INPUT_AUDIO_LENGTH:
             pad_len = INPUT_AUDIO_LENGTH - audio_len
             pad_samples = np.zeros((1, 1, pad_len), dtype=audio.dtype)
             audio = np.concatenate((audio, pad_samples), axis=-1)

        aligned_len = audio.shape[-1]

        asr_result = ""
        slice_start = 0
        slice_end = INPUT_AUDIO_LENGTH

        while slice_end <= aligned_len:
            # 6.1 运行 ONNX Audio Encoder
            input_feed_A = {}
            input_feed_A[in_name_A[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(audio[..., slice_start: slice_end], 'cpu', 0)
            input_feed_A[in_name_A[1]] = onnxruntime.OrtValue.ortvalue_from_numpy(query_embed_input, 'cpu', 0)

            all_outputs_A = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)

            # audio_features: (1, Seq_Len, 1024)
            audio_features = all_outputs_A[0].numpy()

            # 6.2 拼接: [Prefix, Audio, Suffix]
            audio_features_sq = audio_features.squeeze(0)

            final_embedding = np.concatenate([prefix_emb, audio_features_sq, suffix_emb], axis=0)

            print(f"\n=== 推理切片 [{slice_start}:{slice_end}] ===")
            print(f"Final Input Shape: {final_embedding.shape}")

            try:
                # 6.3 调用 LLM 解码 (GPU 版本)
                result_text = decode_with_pure_embeddings_gpu(
                    llm,
                    final_embedding,
                    max_new_tokens=MAX_SEQ_LEN
                )
                asr_result += result_text

            except Exception as e:
                logger.error(f"解码发生错误: {e}")
                import traceback
                traceback.print_exc()

            slice_start += stride_step
            slice_end = slice_start + INPUT_AUDIO_LENGTH

if __name__ == "__main__":
    main()
