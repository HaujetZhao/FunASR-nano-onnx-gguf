import gc
import time
import torch
import torchaudio
import numpy as np
import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import base64
import sys
from pathlib import Path

# 添加 rknn 到搜索路径以便导入模型定义
sys.path.insert(0, str(Path(__file__).cwd() / "rknn"))

import torch_model
import adaptor
from STFT_Process import STFT_Process

# =========================================================================
# 配置部分
# =========================================================================

# 输出目录
OUTPUT_DIR = r'./model-gguf'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 源模型路径
model_dir = r'./Fun-ASR-Nano-2512'
weight_path = os.path.join(model_dir, "model.pt")

# 输出 ONNX 路径
onnx_encoder_fp32 = f'{OUTPUT_DIR}/Fun-ASR-Nano-Encoder.fp32.onnx'
onnx_encoder_int8 = f'{OUTPUT_DIR}/Fun-ASR-Nano-Encoder.int8.onnx'
onnx_ctc_fp32 = f'{OUTPUT_DIR}/Fun-ASR-Nano-CTC.fp32.onnx'
onnx_ctc_int8 = f'{OUTPUT_DIR}/Fun-ASR-Nano-CTC.int8.onnx'
tokens_path = f'{OUTPUT_DIR}/tokens.txt'

# 参数配置
SAMPLE_RATE = 16000
WINDOW_TYPE = 'hamming'
N_MELS = 80
NFFT_STFT = 400
WINDOW_LENGTH = 400
HOP_LENGTH = 160
PRE_EMPHASIZE = 0.97
LFR_M = 7
LFR_N = 6
MAX_INPUT_AUDIO_LENGTH = SAMPLE_RATE * 30 # 30s
OPSET = 18 # 匹配建议的 opset 以避免转换失败

# =========================================================================
# 词表生成逻辑
# =========================================================================

def generate_sensevoice_vocab(tiktoken_path):
    print(f"Generating vocabulary from {tiktoken_path}...")
    tokens = []
    with open(tiktoken_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tokens.append(line.split()[0])
    
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
        
    for i in range(1, 51):
        tokens.append(base64.b64encode(f"<|SPECIAL_TOKEN_{i}|>".encode()).decode())
        
    for i in range(1500):
        tokens.append(base64.b64encode(f"<|{i * 0.02:.2f}|>".encode()).decode())
        
    tokens.append(base64.b64encode("<blk>".encode()).decode())
    return tokens

# =========================================================================
# 模型定义
# =========================================================================

class HybridSenseVoice(torch.nn.Module):
    def __init__(self, encoder_dim=512, llm_dim=1024, vocab_size=60515):
        super().__init__()
        self.audio_encoder = torch_model.SenseVoiceEncoderSmall()
        self.audio_adaptor = adaptor.Transformer(encoder_dim=encoder_dim, llm_dim=llm_dim, n_layer=2)
        self.ctc_decoder = adaptor.Transformer(encoder_dim=encoder_dim, llm_dim=encoder_dim, n_layer=5)
        self.ctc_proj = torch_model.CTC(odim=vocab_size, encoder_output_size=encoder_dim)
        
    def load_weights(self, path):
        state_dict = torch.load(path, map_location="cpu")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("audio_encoder."): new_state_dict[k] = v
            elif k.startswith("audio_adaptor."): new_state_dict[k] = v
            elif k.startswith("ctc_decoder."): new_state_dict[k] = v
            elif k.startswith("ctc.ctc_lo."): new_state_dict[k.replace("ctc.ctc_lo", "ctc_proj.ctc_lo")] = v
        self.load_state_dict(new_state_dict, strict=False)
        print("Weights loaded successfully.")

class EncoderExportWrapper(torch.nn.Module):
    def __init__(self, hybrid_model, stft_model, pre_emphasis=0.97, lfr_m=7, lfr_n=6):
        super().__init__()
        self.hybrid_model = hybrid_model
        self.stft_model = stft_model
        self.pre_emphasis_val = float(pre_emphasis)
        self.pre_emphasis = torch.tensor(pre_emphasis, dtype=torch.float32).view(1, 1, -1)
        self.fbank = (torchaudio.functional.melscale_fbanks(NFFT_STFT // 2 + 1, 20, SAMPLE_RATE // 2, N_MELS, SAMPLE_RATE, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.lfr_m_factor = (lfr_m - 1) // 2

    def forward(self, audio, query_embed):
        audio = audio.float()
        audio = audio - torch.mean(audio)
        if self.pre_emphasis_val > 0:
            audio = torch.cat([audio[..., :1], audio[..., 1:] - self.pre_emphasis * audio[..., :-1]], dim=-1)
        real, imag = self.stft_model(audio, 'constant')
        mel = (torch.matmul(self.fbank, real * real + imag * imag).transpose(1, 2) + 1e-7).log()
        
        # 1. 稳定的 LFR 逻辑
        T = mel.shape[1]
        T_lfr = (T + self.lfr_n - 1) // self.lfr_n
        
        # 充分填充以确保切片长度一致
        # 需要确保最后一个切片还能取到索引 (T_lfr-1)*n + (m-1)
        pad_len = (T_lfr * self.lfr_n + self.lfr_m) - T
        left_pad = mel[:, [0]].repeat(1, self.lfr_m_factor, 1)
        right_pad = mel[:, [-1]].repeat(1, pad_len, 1)
        padded = torch.cat([left_pad, mel, right_pad], dim=1)
        
        # 使用切片拼接
        lfr_list = []
        for i in range(self.lfr_m):
            feat = padded[:, i : i + T_lfr * self.lfr_n : self.lfr_n]
            lfr_list.append(feat[:, :T_lfr, :]) # 强制截断到 T_lfr 以防万一
        
        x = torch.cat(lfr_list, dim=-1)
        
        # 2. 模型执行
        enc_output = self.hybrid_model.audio_encoder(x)
        llm_embeds = self.hybrid_model.audio_adaptor(enc_output)
        
        # 合并 LLM 嵌入
        concat_embed = torch.cat([query_embed, llm_embeds], dim=1)
        
        # 确保 ids_len 是正确的 1D Tensor [num_tokens]
        ids_len = torch.tensor([concat_embed.shape[1]], dtype=torch.int64)
        
        return concat_embed, ids_len, enc_output

class CTCHeadExportWrapper(torch.nn.Module):
    def __init__(self, hybrid_model):
        super().__init__()
        self.hybrid_model = hybrid_model
    def forward(self, enc_output):
        h = self.hybrid_model.ctc_decoder(enc_output)
        logits = self.hybrid_model.ctc_proj.ctc_lo(h)
        return logits

def main():
    print("\n[Hybrid Export] Initializing...")
    tiktoken_path = os.path.join(model_dir, "multilingual.tiktoken")
    tokens = generate_sensevoice_vocab(tiktoken_path)
    with open(tokens_path, "w", encoding="utf-8") as f:
        for i, t in enumerate(tokens): f.write(f"{t} {i}\n")
    hybrid = HybridSenseVoice(vocab_size=len(tokens))
    hybrid.load_weights(weight_path)
    hybrid.eval()
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()

    with torch.no_grad():
        print(f"\n[1/4] Exporting Encoder Branch...")
        enc_wrapper = EncoderExportWrapper(hybrid, custom_stft)
        audio = torch.ones((1, 1, SAMPLE_RATE * 5), dtype=torch.int16)
        query = torch.ones((1, 10, 1024), dtype=torch.float32)
        torch.onnx.export(
            enc_wrapper, (audio, query), onnx_encoder_fp32,
            input_names=['audio', 'query_embed'], output_names=['concat_embed', 'ids_len', 'enc_output'],
            dynamic_axes={'audio': {2: 'audio_len'}, 'query_embed': {1: 'num_token'}, 'concat_embed': {1: 'num_token'}, 'enc_output': {1: 'enc_len'}},
            opset_version=OPSET
        )
        print(f"\n[2/4] Exporting CTC Head...")
        ctc_wrapper = CTCHeadExportWrapper(hybrid)
        dummy_enc = torch.randn(1, 100, 512)
        torch.onnx.export(
            ctc_wrapper, (dummy_enc,), onnx_ctc_fp32,
            input_names=['enc_output'], output_names=['logits'],
            dynamic_axes={'enc_output': {1: 'enc_len'}, 'logits': {1: 'enc_len'}},
            opset_version=OPSET
        )
        print("\n[3/4] Quantizing Encoder...")
        quantize_dynamic(onnx_encoder_fp32, onnx_encoder_int8, op_types_to_quantize=["MatMul"], per_channel=True, reduce_range=False, weight_type=QuantType.QUInt8)
        print(f"\n[4/4] Quantizing CTC Head...")
        quantize_dynamic(onnx_ctc_fp32, onnx_ctc_int8, op_types_to_quantize=["MatMul"], per_channel=True, reduce_range=False, weight_type=QuantType.QUInt8)
    print("\n[Success] Hybrid models ready.")

if __name__ == "__main__":
    main()
