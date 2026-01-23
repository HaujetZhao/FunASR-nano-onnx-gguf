import os
import sys
import warnings
import logging

# =========================================================================
# Environment Setup (Must be before importing torch)
# =========================================================================

# Force CPU usage to avoid CUDA version mismatch warnings (RTX 5050 vs Old PyTorch)
# and to ensure the ONNX graph is device-agnostic.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Suppress specific warnings to clean up output
warnings.filterwarnings("ignore", category=DeprecationWarning) # Ignore Legacy Exporter warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*Constant folding.*") # Ignore slice warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*dynamic_axes.*") 
logging.getLogger("onnxruntime").setLevel(logging.ERROR) # Quiet ONNX Runtime
logging.getLogger("root").setLevel(logging.ERROR)        # Quiet Quantizer

import gc
import time
import torch
import torchaudio
import numpy as np
import onnxruntime
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import base64
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

# Import local model dependencies (standalone - no rknn folder needed)
import fun_asr_gguf.export_torch_model as torch_model

# =========================================================================
# Configuration
# =========================================================================

OUTPUT_DIR = r'./model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_dir = r'./Fun-ASR-Nano-2512'
weight_path = os.path.join(model_dir, "model.pt")

# Output filenames
onnx_encoder_fp32 = f'{OUTPUT_DIR}/Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx'
onnx_encoder_int8 = f'{OUTPUT_DIR}/Fun-ASR-Nano-Encoder-Adaptor.int8.onnx'
onnx_ctc_fp32 = f'{OUTPUT_DIR}/Fun-ASR-Nano-CTC.fp32.onnx'
onnx_ctc_int8 = f'{OUTPUT_DIR}/Fun-ASR-Nano-CTC.int8.onnx'
tokens_path = f'{OUTPUT_DIR}/tokens.txt'

# Parameters
SAMPLE_RATE = 16000
WINDOW_TYPE = 'hamming'
N_MELS = 80
NFFT_STFT = 400
WINDOW_LENGTH = 400
HOP_LENGTH = 160
PRE_EMPHASIZE = 0.97
LFR_M = 7
LFR_N = 6 
DOWNSAMPLE_RATE = 1 # Matched with model.pt (k=1)
OPSET = 18

# =========================================================================
# Custom Adaptor Classes (Replicating Official Logic + RKNN Compatibility)
# =========================================================================

class MultiHeadedAttention(nn.Module):
    """Copied from rknn/adaptor.py"""
    def __init__(self, n_head, n_feat, dropout_rate):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)
            min_value = -float("inf")
            scores = scores.masked_fill(mask, min_value)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, query, key, value, mask):
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.d_k ** (-0.5)
        return self.forward_attention(v, scores, mask)

class EncoderLayer(nn.Module):
    """Copied from rknn/adaptor.py"""
    def __init__(self, size, self_attn, feed_forward, dropout_rate, normalize_before=True, concat_after=False, stochastic_depth_rate=0.0):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        self.norm2 = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x, mask=None, cache=None):
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if cache is None:
            x_q = x
        else:
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + 1.0 * self.concat_linear(x_concat)
        else:
            x = residual + 1.0 * self.dropout(self.self_attn(x_q, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + 1.0 * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)
        return x, mask

class CorrectTransformerAdaptor(nn.Module):
    """
    Corrected Adaptor with Padding logic (even if k=1).
    """
    def __init__(self, downsample_rate=8, encoder_dim=512, llm_dim=1024, ffn_dim=2048, n_layer=2, **kwargs):
        super().__init__()
        self.k = downsample_rate
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, ffn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_dim, self.llm_dim)
        
        self.blocks = None
        if n_layer > 0:
            self.blocks = nn.ModuleList([
                EncoderLayer(
                    llm_dim,
                    MultiHeadedAttention(kwargs.get("attention_heads", 8), llm_dim, kwargs.get("attention_dropout_rate", 0.0)),
                    torch_model.PositionwiseFeedForward(llm_dim, llm_dim // 4, kwargs.get("dropout_rate", 0.0)),
                    kwargs.get("dropout_rate", 0.0)
                )
                for i in range(n_layer)
            ])

    def forward(self, x, ilens=None):
        batch_size, seq_len, dim = x.size()
        
        chunk_num = (seq_len - 1) // self.k + 1
        pad_num = chunk_num * self.k - seq_len
        x = F.pad(x, (0, 0, 0, pad_num, 0, 0), value=0.0)
        
        x = x.contiguous()
        x = x.view(batch_size, chunk_num, dim * self.k)
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        olens = None
        masks = None
        if ilens is not None:
            olens = (ilens - 1) // self.k + 1
            pass 
        
        if self.blocks is not None:
            for block in self.blocks:
                x, masks = block(x, masks)
                
        return x, olens

# =========================================================================
# Model Definition
# =========================================================================

class HybridSenseVoice(torch.nn.Module):
    def __init__(self, encoder_dim=512, llm_dim=1024, vocab_size=60515):
        super().__init__()
        self.audio_encoder = torch_model.SenseVoiceEncoderSmall()
        # Use our CORRECT adaptor with downsample_rate=1 (matched to weights)
        self.audio_adaptor = CorrectTransformerAdaptor(downsample_rate=DOWNSAMPLE_RATE, encoder_dim=encoder_dim, llm_dim=llm_dim, n_layer=2)
        # CTC decoder (k=1)
        self.ctc_decoder = CorrectTransformerAdaptor(downsample_rate=1, encoder_dim=encoder_dim, llm_dim=encoder_dim, n_layer=5)
        self.ctc_proj = torch_model.CTC(odim=vocab_size, encoder_output_size=encoder_dim)
        
    def load_weights(self, path):
        state_dict = torch.load(path, map_location="cpu")
        new_state_dict = {}
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        for k, v in state_dict.items():
            if k.startswith("audio_encoder."): new_state_dict[k] = v
            elif k.startswith("audio_adaptor."): new_state_dict[k] = v
            elif k.startswith("ctc_decoder."): new_state_dict[k] = v
            elif k.startswith("ctc.ctc_lo."): new_state_dict[k.replace("ctc.ctc_lo", "ctc_proj.ctc_lo")] = v
        
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

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

    def forward(self, audio):
        # Audio Processing
        audio = audio.float()
        audio = audio - torch.mean(audio)
        if self.pre_emphasis_val > 0:
            audio = torch.cat([audio[..., :1], audio[..., 1:] - self.pre_emphasis * audio[..., :-1]], dim=-1)
        real, imag = self.stft_model(audio, 'constant')
        mel = (torch.matmul(self.fbank, real * real + imag * imag).transpose(1, 2) + 1e-7).log()
        
        # LFR Processing
        T = mel.shape[1]
        T_lfr = (T + self.lfr_n - 1) // self.lfr_n
        pad_len = (T_lfr * self.lfr_n + self.lfr_m) - T
        left_pad = mel[:, [0]].repeat(1, self.lfr_m_factor, 1)
        right_pad = mel[:, [-1]].repeat(1, pad_len, 1)
        padded = torch.cat([left_pad, mel, right_pad], dim=1)
        
        lfr_list = []
        for i in range(self.lfr_m):
            feat = padded[:, i : i + T_lfr * self.lfr_n : self.lfr_n]
            lfr_list.append(feat[:, :T_lfr, :])
        
        x = torch.cat(lfr_list, dim=-1)
        
        # 1. Encoder Execution (6x)
        enc_output = self.hybrid_model.audio_encoder(x)
        
        # 2. Adaptor Execution (1x)
        adaptor_output, _ = self.hybrid_model.audio_adaptor(enc_output, None)
        
        # 3. Official Slicing (Simulated 8x Downsampling)
        olens_1 = 1 + (T_lfr - 3 + 2) // 2
        olens_2 = 1 + (olens_1 - 3 + 2) // 2
        target_len = (olens_2 - 1) // 2 + 1
        
        # Slice the adaptor output
        adaptor_output = adaptor_output[:, :target_len, :]
        
        return enc_output, adaptor_output

class CTCHeadExportWrapper(torch.nn.Module):
    def __init__(self, hybrid_model):
        super().__init__()
        # Hold only used submodules to avoid redundant weight export
        self.ctc_decoder = hybrid_model.ctc_decoder
        self.ctc_proj = hybrid_model.ctc_proj
        
    def forward(self, enc_output):
        h, _ = self.ctc_decoder(enc_output, None)
        logits = self.ctc_proj.ctc_lo(h)
        return logits



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
# Utils: Merge .data file back to .onnx
# =========================================================================

def merge_onnx_file(model_path):
    """
    Loads an ONNX model (which loads external data automatically), 
    and saves it back to the same path. ONNX 1.16+ defaults to saving 
    as a single file if < 2GB.
    """
    print(f"   [Merge] Merging external data for {os.path.basename(model_path)}...")
    data_file = model_path + ".data"
    
    try:
        # onnx.load automatically reads the .data file if present
        model = onnx.load(model_path)
        
        # onnx.save writes a single file by default if it fits
        onnx.save(model, model_path)
        print(f"   [Merge] Successfully embedded weights into {os.path.basename(model_path)}")
        
        # Cleanup
        if os.path.exists(data_file):
            try:
                os.remove(data_file)
                print(f"   [Cleanup] Deleted external data file: {os.path.basename(data_file)}")
            except:
                pass
                
    except Exception as e:
        print(f"   [Warning] Merge failed: {e}")


def main():
    print("\n[Hybrid Export - DYNAMO ENABLED] Initializing Encoder-Adaptor System...")
    # ... (Initialization code omitted - same as before) ...
    tiktoken_path = os.path.join(model_dir, "multilingual.tiktoken")
    if os.path.exists(tiktoken_path):
        tokens = generate_sensevoice_vocab(tiktoken_path)
        with open(tokens_path, "w", encoding="utf-8") as f:
            for i, t in enumerate(tokens): f.write(f"{t} {i}\n")
    else:
        print("Warning: tiktoken file not found, vocab generation skipped.")
        tokens = ["dummy"] * 60515 
    
    hybrid = HybridSenseVoice(vocab_size=len(tokens))
    hybrid.load_weights(weight_path)
    hybrid.eval()
    
    custom_stft = torch_model.STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()

    with torch.no_grad():
        print(f"\n[1/4] Exporting Dual-Output Encoder (Dynamo=True)...")
        enc_wrapper = EncoderExportWrapper(hybrid, custom_stft, lfr_m=LFR_M, lfr_n=LFR_N).eval()
        
        audio = torch.randn(1, 1, SAMPLE_RATE * 1, dtype=torch.float32) 
        
        torch.onnx.export(
            enc_wrapper, (audio,), onnx_encoder_fp32,
            input_names=['audio'], 
            output_names=['enc_output', 'adaptor_output'],
            dynamic_axes={
                'audio': {2: 'audio_len'}, 
                'enc_output': {1: 'enc_len'}, 
                'adaptor_output': {1: 'adaptor_len'}
            },
            opset_version=OPSET,
            dynamo=True
        )
        print(f"Saved to: {onnx_encoder_fp32}")
        merge_onnx_file(onnx_encoder_fp32)

        print(f"\n[2/4] Exporting CTC Head (Dynamo=True)...")
        # Cleanup potential previous .data file
        data_file = onnx_ctc_fp32 + ".data"
        if os.path.exists(data_file):
            try: os.remove(data_file)
            except: pass

        ctc_wrapper = CTCHeadExportWrapper(hybrid).eval() # Using unified wrapper logic
        dummy_enc = torch.randn(1, 100, 512)
        
        torch.onnx.export(
            ctc_wrapper, (dummy_enc,), onnx_ctc_fp32,
            input_names=['enc_output'], output_names=['logits'],
            dynamic_axes={'enc_output': {1: 'enc_len'}, 'logits': {1: 'enc_len'}},
            opset_version=OPSET,
            dynamo=True
        )
        print(f"Saved to: {onnx_ctc_fp32}")
        merge_onnx_file(onnx_ctc_fp32)

        print("\n[3/4] Quantizing Encoder...")
        quantize_dynamic(onnx_encoder_fp32, onnx_encoder_int8, op_types_to_quantize=["MatMul"], per_channel=True, reduce_range=False, weight_type=QuantType.QUInt8)
        print(f"Saved to: {onnx_encoder_int8}")

        print(f"\n[4/4] Quantizing CTC Head...")
        quantize_dynamic(onnx_ctc_fp32, onnx_ctc_int8, op_types_to_quantize=["MatMul"], per_channel=True, reduce_range=False, weight_type=QuantType.QUInt8)
        print(f"Saved to: {onnx_ctc_int8}")
        
    print("\n[Success] Export complete.")

if __name__ == "__main__":
    main()