import os
os.environ['VK_ICD_FILENAMES'] = 'none'   # 禁用 Vulkan，以确保只使用 CPU

import pickle
import numpy as np
import logging
import ctypes
import time
import os
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# GGUF 模型路径
GGUF_MODEL_PATH = r'./model-gguf/qwen3-0.6b-asr.gguf'

# Pickle 文件路径 (默认)
PICKLE_PATH = r'./pickles/embedding_slice_0_320000.pkl' # 默认文件名，会尝试自动寻找最近的或使用指定文件

# 参数
MAX_SEQ_LEN = 1024
STOP_TOKEN = [151643, 151645]
MAX_THREADS = 0

# =========================================================================
# 解码函数 (复用自 03_Inference.py)
# =========================================================================

def decode_with_pure_embeddings(llm_obj, audio_embeddings, max_new_tokens=200):
    """
    纯 Embedding 解码函数
    """
    
    # 1. 准备数据
    embeds = audio_embeddings.squeeze()
    if len(embeds.shape) == 1:
        embeds = embeds.reshape(1, -1)
    
    n_tokens, n_dim = embeds.shape
    logger.info(f"注入 Embedding Shape: {embeds.shape}")

    # 2. 初始化 Batch
    batch_embd = llama_batch_init(n_tokens, n_dim, 1)        
    batch_text = llama_batch_init(1, 0, 1)

    ctx = llm_obj.ctx
    
    # 3. 清理上下文缓存
    llama_kv_self_clear(llm_obj.ctx) 
    
    try:
        # A. 注入 Embedding
        logger.info("正在注入 Embedding...")
        
        batch_embd.n_tokens = n_tokens
        llm_obj.n_tokens = 0 
        
        batch_embd.token = ctypes.cast(None, ctypes.POINTER(ctypes.c_int32))

        for i in range(n_tokens):
            batch_embd.pos[i] = i
            batch_embd.n_seq_id[i] = 1
            batch_embd.seq_id[i][0] = 0
            batch_embd.logits[i] = 1 if i == n_tokens - 1 else 0

        if not embeds.flags['C_CONTIGUOUS']:
            embeds = np.ascontiguousarray(embeds)
        
        ctypes.memmove(batch_embd.embd, embeds.ctypes.data, embeds.nbytes)
        
        if llama_decode(ctx, batch_embd) != 0:
             raise RuntimeError("Audio embedding decoding failed")
        
        llm_obj.n_tokens += n_tokens

        # B. 文本生成
        generated_text = ""
        logger.info(f"开始生成文本...\n")
        
        eos_token = llm_obj.token_eos()
        vocab_size = llm_obj.n_vocab()
        
        batch_text.n_tokens = 1
        
        gen_start_time = time.time()
        tokens_generated = 0
        
        for step in range(max_new_tokens):
            logits_ptr = llama_get_logits(ctx)
            logits_arr = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,))
            token_id = int(np.argmax(logits_arr))
            
            if token_id == eos_token or token_id in STOP_TOKEN:
                break
                
            try:
                text_piece = llm_obj.detokenize([token_id]).decode('utf-8', errors='ignore')
                print(text_piece, end="", flush=True)
                generated_text += text_piece
                tokens_generated += 1
            except Exception:
                pass
                
            batch_text.token[0] = token_id
            batch_text.pos[0] = llm_obj.n_tokens
            batch_text.n_seq_id[0] = 1
            batch_text.seq_id[0][0] = 0
            batch_text.logits[0] = 1
            
            if llama_decode(ctx, batch_text) != 0:
                break
            
            llm_obj.n_tokens += 1
            
        print('\n')
        gen_duration = time.time() - gen_start_time
        tps = tokens_generated / gen_duration if gen_duration > 0 else 0
        logger.info(f"解码速度: {tps:.2f} tokens/s ({tokens_generated} tokens in {gen_duration:.2f}s)\n")
        
    finally:
        llama_batch_free(batch_embd)
        llama_batch_free(batch_text)

    return generated_text

# =========================================================================
# 主程序
# =========================================================================

def main():
    target_pickle = PICKLE_PATH
    
    # 如果指定文件不存在，尝试找目录下最新的 pkl
    if not os.path.exists(target_pickle):
        logger.warning(f"指定文件 {target_pickle} 不存在，尝试搜索 pickles 目录...")
        if os.path.exists("pickles"):
            files = [os.path.join("pickles", f) for f in os.listdir("pickles") if f.endswith(".pkl")]
            if files:
                target_pickle = max(files, key=os.path.getctime)
                logger.info(f"找到最新 Pickle 文件: {target_pickle}")
            else:
                logger.error("pickles 目录下没有找到 .pkl 文件")
                return
        else:
            logger.error("pickles 目录不存在")
            return
            
    print(f'\nLoading Pickle: {target_pickle}')
    with open(target_pickle, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded embeddings, shape: {embeddings.shape}")

    print(f'\nLoading GGUF model: {GGUF_MODEL_PATH}')
    llm = Llama(
        model_path=GGUF_MODEL_PATH,
        n_ctx=MAX_SEQ_LEN + 1024,
        n_threads=MAX_THREADS,
        embedding=True,
        verbose=False
    )
    print('GGUF model loaded successfully!')

    print("\n--- Start Inference from Pickle ---")
    decode_with_pure_embeddings(llm, embeddings, max_new_tokens=MAX_SEQ_LEN)
    print("\n--- End Inference ---")

if __name__ == "__main__":
    main()
