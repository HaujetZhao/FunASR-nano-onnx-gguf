import os
import sys
import warnings
import logging
import torch
import numpy as np
import base64
from pathlib import Path
from export_config import MODEL_DIR, EXPORT_DIR
import fun_asr_gguf.export.model_definition as model_def

# Suppress warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

OUTPUT_DIR = str(EXPORT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_dir = str(MODEL_DIR)
weight_path = os.path.join(model_dir, "model.pt")

# Standardized names
ONNX_ENCODER_FP32 = f'{OUTPUT_DIR}/Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx'
ONNX_CTC_FP32 = f'{OUTPUT_DIR}/Fun-ASR-Nano-CTC.fp32.onnx'
TOKENS_PATH = f'{OUTPUT_DIR}/tokens.txt'

SAMPLE_RATE = 16000
NFFT_STFT = 400
WINDOW_LENGTH = 400
HOP_LENGTH = 160
N_MELS = 80
OPSET = 18

def generate_sensevoice_vocab(tiktoken_path):
    print(f"Generating vocabulary from {tiktoken_path}...")
    tokens = []
    with open(tiktoken_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): tokens.append(line.split()[0])
    
    special_labels = [
        "<|endoftext|>", "<|startoftranscript|>",
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", 
        "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", 
        "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", 
        "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", 
        "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", 
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", 
        "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", 
        "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh",
        "ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM", "Laughter", "/Laughter", "Applause", "/Applause",
        "HAPPY", "SAD", "ANGRY", "NEUTRAL",
        "translate", "transcribe", "startoflm", "startofprev", "nospeech", "notimestamps"
    ]
    for label in special_labels:
        if not label.startswith("<|"): label = f"<|{label}|>"
        tokens.append(base64.b64encode(label.encode()).decode())
    for i in range(1, 51): tokens.append(base64.b64encode(f"<|SPECIAL_TOKEN_{i}|>".encode()).decode())
    for i in range(1500): tokens.append(base64.b64encode(f"<|{i * 0.02:.2f}|>".encode()).decode())
    tokens.append(base64.b64encode("<blk>".encode()).decode())
    return tokens

def main():
    print("\n[Step 01] Exporting ONNX FP32 Models...")
    
    tiktoken_path = os.path.join(model_dir, "multilingual.tiktoken")
    if os.path.exists(tiktoken_path):
        tokens = generate_sensevoice_vocab(tiktoken_path)
        with open(TOKENS_PATH, "w", encoding="utf-8") as f:
            for i, t in enumerate(tokens): f.write(f"{t} {i}\n")
    else:
        print("Warning: tiktoken file not found, vocab generation skipped.")
        tokens = ["dummy"] * 60515 
    
    hybrid = model_def.HybridSenseVoice(vocab_size=len(tokens))
    hybrid.load_weights(weight_path)
    hybrid.eval()
    
    with torch.no_grad():
        print(f"\n[1/2] Exporting Clean Encoder-Adaptor...")
        enc_wrapper = model_def.CleanEncoderExportWrapper(hybrid).eval()
        
        # 模拟 1 秒音频对应的 LFR 特征: (1, 16.67, 560) -> 见 19 号脚本
        # 1s 音频 = 100 mel frames. 100 // 6 = 16.67
        dummy_lfr = torch.randn(1, 17, 560)
        dummy_mask = torch.ones(1, 17)
        
        # Define dynamic symbols for Encoder
        lfr_frames = torch.export.Dim("lfr_frames", min=1, max=16384)
        
        torch.onnx.export(
            enc_wrapper, (dummy_lfr, dummy_mask), ONNX_ENCODER_FP32,
            input_names=['lfr_feat', 'mask'], 
            output_names=['enc_output', 'adaptor_output'],
            dynamic_shapes={
                'lfr_feat': {1: lfr_frames}, 
                'mask': {1: lfr_frames}
            },
            # Note: outputs will automatically be inferred as dynamic based on the inputs
            opset_version=OPSET,
            dynamo=True
        )

        print(f"\n[2/2] Exporting Clean CTC Head...")
        ctc_wrapper = model_def.CTCHeadExportWrapper(hybrid).eval()
        dummy_enc = torch.randn(1, 100, 512)
        
        # Define dynamic symbols for CTC
        enc_len = torch.export.Dim("enc_len", min=1, max=16384)
        
        torch.onnx.export(
            ctc_wrapper, (dummy_enc,), ONNX_CTC_FP32,
            input_names=['enc_output'], 
            output_names=['topk_log_probs', 'topk_indices'],
            dynamic_shapes={
                'enc_output': {1: enc_len}
            },
            opset_version=OPSET,
            dynamo=True
        )
        
    print(f"\n✅ Export complete:\n  - {ONNX_ENCODER_FP32}\n  - {ONNX_CTC_FP32}")

if __name__ == "__main__":
    main()
