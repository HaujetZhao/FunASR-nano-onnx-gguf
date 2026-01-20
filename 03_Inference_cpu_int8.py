import os
os.environ['VK_ICD_FILENAMES'] = 'none'   # 禁用 Vulkan，以确保只使用 CPU

import ctypes
import numpy as np
import logging
import time
import onnxruntime
import torch
import torchaudio
import sys
import os
import pickle
from pydub import AudioSegment
from transformers import AutoTokenizer
from llama_cpp import (
    Llama,
    llama_batch_init,
    llama_batch_free,
    llama_decode,
    llama_get_logits,
    llama_kv_self_clear,  # 新版 API：清理缓存
)

# =========================================================================
# 配置部分
# =========================================================================

# 日志设置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference_int8.log", encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 模型路径
model_dir = r'./model-gguf'
tokenizer_path = f'{model_dir}/Qwen3-0.6B'

# ONNX 模型
onnx_encoder = f'{model_dir}/FunASR_Nano_Encoder.onnx'       # Audio Encoder

# GGUF 模型 (用于解码) - 指向 Int8 模型
gguf_model_path = f'{model_dir}/qwen3-0.6b-asr-q8_0.gguf'

# 输入音频
test_audio = r'./input.mp3'

# Prompt 配置 (Prefix + Audio + Suffix)
# 注意：Audio Encoder 已经移除了硬编码的 Prompt，现在需要我们手动拼接。
PREFIX_PROMPT = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
SUFFIX_PROMPT = "<|im_end|>\n<|im_start|>assistant\n"

# 音频处理参数 (需与模型训练时一致)
SAMPLE_RATE = 16000
USE_NORMALIZER = True
MAX_INPUT_AUDIO_LENGTH = 320000 
SLIDING_WINDOW = 0 # 0 表示根据音频长度自动分段 (这里简化为一次性处理或整段)

# 模型参数
MAX_SEQ_LEN = 1024
STOP_TOKEN = [151643, 151645] # Qwen 的特殊停止 Token
MAX_THREADS = 0 # 0 = Auto

# =========================================================================
# 辅助函数
# =========================================================================

def load_gguf_embeddings(model_path):
    """Load the raw token embedding table from the GGUF file."""
    # 尝试加载 gguf 库
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
        
    logger.info(f"Tensor Info: type={tensor.tensor_type}, shape={tensor.shape}")

    # GGML_TYPE_Q8_0 = 8
    if tensor.tensor_type == 8:
        logger.info("Detected Q8_0 embeddings. Dequantizing to F32...")
        
        # Q8_0 Block Structure:
        # d: float16 (2 bytes)
        # qs: int8[32] (32 bytes)
        # Block size = 34 bytes
        # Number of values per block = 32
        
        block_size_bytes = 34
        num_values_per_block = 32
        
        raw_data = tensor.data
        n_blocks = len(raw_data) // block_size_bytes
        
        # 使用 numpy 进行向量化解压
        
        # 1. 将 buffer 转换为 uint8 数组以便切片
        data_u8 = np.frombuffer(raw_data, dtype=np.uint8)
        
        # 2. 计算 n_blocks (使用总字节数，而不是 len(raw_data))
        # raw_data 可能是 numpy.memmap，其 shape 是逻辑上的 (rows, stride)，导致 len() 只返回行数
        n_blocks = data_u8.size // block_size_bytes
        logger.info(f"Total Blocks: {n_blocks} (from {data_u8.size} bytes)")
        
        # 3. Reshape 成 (n_blocks, 34)
        blocks = data_u8.reshape(n_blocks, block_size_bytes)
        
        # 3. 提取 delta (前2字节) -> float16
        # 注意：这里假设是 Little Endian
        deltas = blocks[:, :2].view(np.float16).flatten() # shape: (n_blocks,)
        
        # 4. 提取 quants (后32字节) -> int8
        quants = blocks[:, 2:].view(np.int8) # shape: (n_blocks, 32)
        
        # 5. 计算值: value = delta * quant
        # 扩展 delta 维度以便广播: (n_blocks, 1) * (n_blocks, 32)
        decoded = deltas[:, np.newaxis] * quants
        
        # 6. Flatten 并 reshape 为 embedding 形状
        # shape: (n_vocab, hidden_size)
        weights = decoded.flatten().astype(np.float32).reshape(-1, 1024)
        
    else:
        # 假设是 F32 或 F16
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

def decode_with_pure_embeddings(llm_obj, audio_embeddings, max_new_tokens=200):
    """
    纯 Embedding 解码函数 (Pure Embedding Decoding)
    """
    
    # 1. 准备数据
    embeds = audio_embeddings.squeeze()
    if len(embeds.shape) == 1:
        embeds = embeds.reshape(1, -1)
    
    n_tokens, n_dim = embeds.shape
    logger.info(f"注入 Total Embeddings Shape: {embeds.shape}")

    # 2. 初始化 Batch
    batch_embd = llama_batch_init(n_tokens, n_dim, 1)        
    
    # batch_text: 用于存放生成的 Token IDs
    batch_text = llama_batch_init(1, 0, 1)

    ctx = llm_obj.ctx
    
    # 3. 清理上下文缓存 (KV Cache)
    llama_kv_self_clear(llm_obj.ctx) 
    
    try:
        # ---------------------------------------------------------------------
        # 阶段 A: 注入融合 Embedding
        # ---------------------------------------------------------------------
        logger.info("正在注入融合 Embedding...")
        
        batch_embd.n_tokens = n_tokens
        llm_obj.n_tokens = 0 # 重置 LLM 内部计数器
        
        # 关键：batch.token 设置为 NULL，告知底层使用 embedding
        batch_embd.token = ctypes.cast(None, ctypes.POINTER(ctypes.c_int32))

        for i in range(n_tokens):
            batch_embd.pos[i] = i
            batch_embd.n_seq_id[i] = 1
            batch_embd.seq_id[i][0] = 0
            
            # 只在最后一个 Token 开启 Logits 计算
            batch_embd.logits[i] = 1 if i == n_tokens - 1 else 0

        # 使用 ctypes.memmove 高效拷贝 Numpy 数据到 C 指针
        if not embeds.flags['C_CONTIGUOUS']:
            embeds = np.ascontiguousarray(embeds)
        
        ctypes.memmove(batch_embd.embd, embeds.ctypes.data, embeds.nbytes)
        
        # 执行解码
        if llama_decode(ctx, batch_embd) != 0:
             raise RuntimeError("Audio embedding decoding failed")
        
        llm_obj.n_tokens += n_tokens

        # ---------------------------------------------------------------------
        # 阶段 B: 文本生成 (Greedy Search)
        # ---------------------------------------------------------------------
        generated_text = ""
        logger.info(f"开始生成文本 (最大 {max_new_tokens} tokens)...\n")
        
        eos_token = llm_obj.token_eos()
        vocab_size = llm_obj.n_vocab()
        
        batch_text.n_tokens = 1
        
        gen_start_time = time.time()
        tokens_generated = 0
        
        for step in range(max_new_tokens):
            # 1. 获取 Logits
            logits_ptr = llama_get_logits(ctx)
            logits_arr = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
            
            # 2. 贪婪采样 (Argmax)
            token_id = int(np.argmax(logits_arr))
            
            # 3. 检查停止条件
            if token_id == eos_token or token_id in STOP_TOKEN:
                break
                
            # 4. 解码 token 为文本
            try:
                text_piece = llm_obj.detokenize([token_id]).decode('utf-8', errors='ignore')
                print(text_piece, end="", flush=True)
                generated_text += text_piece
                tokens_generated += 1
            except Exception:
                pass
                
            # 5. 把生成的 Token 喂回去 (Autoregressive)
            batch_text.token[0] = token_id
            batch_text.pos[0] = llm_obj.n_tokens
            batch_text.n_seq_id[0] = 1
            batch_text.seq_id[0][0] = 0
            batch_text.logits[0] = 1
            
            if llama_decode(ctx, batch_text) != 0:
                break
            
            llm_obj.n_tokens += 1
            
        print('\n\n')
        gen_duration = time.time() - gen_start_time
        tps = tokens_generated / gen_duration if gen_duration > 0 else 0

        logger.info(f"解码速度: {tps:.2f} tokens/s ({tokens_generated} tokens in {gen_duration:.2f}s)\n\n")
        
    finally:
        # 释放资源
        llama_batch_free(batch_embd)
        llama_batch_free(batch_text)

    return generated_text

# =========================================================================
# 主程序
# =========================================================================

def main():
    print('\nStarting INT8 Refactored Inference Engine...')
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 2. 加载 GGUF 模型 (Int8)
    print(f'\nLoading GGUF model: {gguf_model_path}')
    if not os.path.exists(gguf_model_path):
        print(f"Error: Model not found at {gguf_model_path}. Please run 02_Export_GGUF_int8.py first.")
        return

    llm = Llama(
        model_path=gguf_model_path,
        n_ctx=MAX_SEQ_LEN + 1024, # 适当增加 Context 窗口
        n_threads=MAX_THREADS,
        embedding=True, # 必须开启
        verbose=False
    )
    print('GGUF model loaded successfully!')

    # 3. 从 GGUF 文件中加载 Raw Embeddings
    print('\nLoading Raw Embeddings from GGUF...')
    all_embeddings = load_gguf_embeddings(gguf_model_path)
    
    # 4. 准备 Prefix 和 Suffix 的 Embeddings
    prefix_tokens = tokenizer.encode(PREFIX_PROMPT, add_special_tokens=False) 
    suffix_tokens = tokenizer.encode(SUFFIX_PROMPT, add_special_tokens=False)
    
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
    
    # Query Embed for Audio Encoder input (Original logic kept this input)
    # 假设 hidden_size = 1024
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
            
            audio_features = all_outputs_A[0].numpy()
            
            # 6.2 拼接: [Prefix, Audio, Suffix]
            audio_features_sq = audio_features.squeeze(0) 
            
            final_embedding = np.concatenate([prefix_emb, audio_features_sq, suffix_emb], axis=0)
            
            # Optionally save pickle (code exists in 03, added here for consistency if needed)
            # os.makedirs("pickles_int8", exist_ok=True)
            # ...

            print(f"\n=== 推理切片 [{slice_start}:{slice_end}] ===")
            print(f"Final Input Shape: {final_embedding.shape}")
            
            try:
                # 6.3 调用 LLM 解码
                result_text = decode_with_pure_embeddings(
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
