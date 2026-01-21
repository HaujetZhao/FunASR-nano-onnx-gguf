import onnxruntime as ort
import numpy as np
import librosa
import base64
import time
import os

# =========================================================================
# 配置
# =========================================================================
MODEL_DIR = "./model-gguf"
ENCODER_ONNX = f"{MODEL_DIR}/Fun-ASR-Nano-Encoder.int8.onnx"
CTC_ONNX = f"{MODEL_DIR}/Fun-ASR-Nano-CTC.int8.onnx"
TOKENS_PATH = f"{MODEL_DIR}/tokens.txt"
AUDIO_PATH = "input2.mp3" # 请确保当前目录有此文件，或者修改为现有文件

# =========================================================================
# 工具函数
# =========================================================================

def load_tokens(filename):
    id2token = dict()
    with open(filename, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if len(parts) == 1:
                t, i = " ", parts[0]
            else:
                t, i = parts
            id2token[int(i)] = t
    return id2token

def decode_ctc(logits, id2token, blank_id):
    # logits shape: [1, T, vocab_size]
    indices = np.argmax(logits[0], axis=-1)
    
    # 时间戳换算参数
    # STFT hop=10ms, LFR stride=6, 所以每帧 60ms
    frame_shift_ms = 60
    offset_ms = -30
    
    results = []
    last_idx = -1
    for i, idx in enumerate(indices):
        # CTC 核心逻辑：跳过空白符和连续重复帧
        if idx == blank_id or idx == last_idx:
            last_idx = idx
            continue
        
        last_idx = idx
        token_b64 = id2token.get(idx, "")
        if not token_b64: continue
        
        try:
            token_text = base64.b64decode(token_b64).decode("utf-8")
        except:
            token_text = f"[ERR:{idx}]"
            
        # 计算当前帧对应的绝对时间 (秒)
        timestamp = max((i * frame_shift_ms + offset_ms) / 1000.0, 0.0)
        
        # 识别内置时间戳 Token (虽然本模型通常不输出，但逻辑保留)
        is_ts_token = False
        if token_text.startswith("<|") and token_text.endswith("|>"):
            try:
                # 如果模型输出了内置时间戳，则解析它
                timestamp = float(token_text[2:-2])
                is_ts_token = True
            except ValueError:
                pass
        
        results.append({
            "text": token_text,
            "start": timestamp,
            "is_ts_token": is_ts_token
        })
                
    full_text = "".join([r["text"] for r in results if not r["is_ts_token"]])
    
    # 构造带详细时间信息的列表字符串
    # 格式: [0.12s] 你 (0.30s) 是 ...
    detailed_lines = []
    for r in results:
        if r["is_ts_token"]:
            detailed_lines.append(f" #⚓{r['start']:.2f}s# ")
        else:
            detailed_lines.append(f"[{r['start']:.2f}s]{r['text']} ")
            
    return full_text, "".join(detailed_lines), results

# =========================================================================
# 主程序
# =========================================================================

def main():
    print("\n" + "="*50)
    print("SenseVoice Nano CTC Standalone Inference Test")
    print("="*50)

    # 1. 加载资源
    print(f"\n[1/5] Loading tokens from {TOKENS_PATH}...")
    id2token = load_tokens(TOKENS_PATH)
    blank_id = max(id2token.keys())
    print(f"      Total tokens: {len(id2token)}, Blank ID: {blank_id}")

    print(f"\n[2/5] Loading ONNX Sessions...")
    sess_opts = ort.SessionOptions()
    # sess_opts.intra_op_num_threads = 4
    
    encoder_sess = ort.InferenceSession(ENCODER_ONNX, sess_options=sess_opts, providers=['CPUExecutionProvider'])
    ctc_sess = ort.InferenceSession(CTC_ONNX, sess_options=sess_opts, providers=['CPUExecutionProvider'])
    print(f"      Encoder: {os.path.basename(ENCODER_ONNX)}")
    print(f"      CTC Head: {os.path.basename(CTC_ONNX)}")

    # 2. 加载音频
    if not os.path.exists(AUDIO_PATH):
        print(f"\n[!] Error: Audio file {AUDIO_PATH} not found. Please provide an audio file.")
        return

    print(f"\n[3/5] Processing Audio: {AUDIO_PATH}...")
    audio, _ = librosa.load(AUDIO_PATH, sr=16000)
    # 转换为 int16 (导出模型 forward 第一步是 .float()，通常输入建议保持一致)
    # 在 01-Export 脚本中，输入是 torch.int16
    audio_int16 = (audio * 32768).astype(np.int16)
    audio_input = audio_int16[np.newaxis, np.newaxis, :] # [1, 1, samples]

    # 3. 推理第一阶段：Encoder
    print(f"\n[4/5] Running Encoder Branch...")
    start_t = time.perf_counter()
    
    # 模拟 LLM 系统 prompt token (这里随便给 10 个)
    dummy_query = np.ones((1, 10, 1024), dtype=np.float32)
    
    # 输出：concat_embed, ids_len, enc_output
    enc_outputs = encoder_sess.run(None, {
        "audio": audio_input,
        "query_embed": dummy_query
    })
    enc_output = enc_outputs[2]
    enc_t = time.perf_counter() - start_t
    print(f"      Encoder Finished in {enc_t:.3f}s, Output shape: {enc_output.shape}")

    # 4. 推理第二阶段：CTC Head
    print(f"\n[5/5] Running CTC Head...")
    start_t = time.perf_counter()
    logits = ctc_sess.run(None, {"enc_output": enc_output})[0]
    ctc_t = time.perf_counter() - start_t
    print(f"      CTC Head Finished in {ctc_t:.3f}s, Logits shape: {logits.shape}")

    # 5. 解码
    print(f"\nDecoding CTC Logits with Character-level Timestamps...")
    full_text, detailed_text, char_timestamps = decode_ctc(logits, id2token, blank_id)

    print("\n" + "-"*30)
    print("CTC TRANSCRIPTION RESULT")
    print("-"*30)
    print(f"Text:\n{full_text}")
    print("\nDetailed Timestamps:")
    print(detailed_text)
    print("-"*30)
    print(f"Total Inference Time: {enc_t + ctc_t:.3f}s")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
