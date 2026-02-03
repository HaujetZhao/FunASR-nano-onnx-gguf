import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import base64
import numpy as np

# ============================================================================
# Basic Building Blocks
# ============================================================================

class SinusoidalPositionEncoder(nn.Module):
    def __init__(self, d_model=80, dropout_rate=0.1):
        super().__init__()

    def encode(self, positions: torch.Tensor, depth: int, dtype: torch.dtype):
        positions = positions.type(dtype)
        device = positions.device
        log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype, device=device)) / (depth / 2 - 1)
        inv_timescales = torch.exp(torch.arange(depth / 2, device=device).type(dtype) * (-log_timescale_increment))
        inv_timescales = inv_timescales.unsqueeze(0)
        scaled_time = positions.unsqueeze(-1) * inv_timescales.unsqueeze(1)
        encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2)
        return encoding.type(dtype)

    def forward(self, x, mask=None):
        batch_size, timesteps, input_dim = x.size()
        positions = torch.arange(1, timesteps + 1, device=x.device)[None, :]
        position_encoding = self.encode(positions, input_dim, x.dtype).to(x.device)
        x = x + position_encoding
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim, hidden_units, dropout_rate, activation=None):
        super().__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        if activation is None: activation = torch.nn.ReLU()
        self.activation = activation

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class LayerNorm(nn.LayerNorm):
    def forward(self, input):
        output = F.layer_norm(input.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None, self.bias.float() if self.bias is not None else None, self.eps)
        return output.type_as(input)

# ============================================================================
# SANM Components
# ============================================================================

class MultiHeadedAttentionSANM(nn.Module):
    def __init__(self, n_head, in_feat, n_feat, dropout_rate, kernel_size, sanm_shfit=0):
        super().__init__()
        self.d_k, self.h = n_feat // n_head, n_head
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fsmn_block = nn.Conv1d(n_feat, n_feat, kernel_size, stride=1, padding=0, groups=n_feat, bias=False)
        self.pad_fn = nn.ConstantPad1d(((kernel_size - 1) // 2, kernel_size - 1 - (kernel_size - 1) // 2), 0.0)

    def forward_fsmn(self, inputs, mask):
        if mask is not None: inputs = inputs * mask.unsqueeze(-1)
        x = self.pad_fn(inputs.transpose(1, 2))
        x = self.fsmn_block(x).transpose(1, 2)
        x = self.dropout(x + inputs)
        if mask is not None: x = x * mask.unsqueeze(-1)
        return x

    def forward_attention(self, v_h, scores, mask):
        if mask is not None:
            # Score Masking: (B, 1, 1, T_key)
            m = mask.unsqueeze(1).unsqueeze(2).eq(0)
            scores = scores.masked_fill(m, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(m, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(self.dropout(attn), v_h).permute(0, 2, 1, 3).flatten(2)
        return self.linear_out(x)

    def forward(self, x, mask):
        q_k_v = self.linear_q_k_v(x)
        q, k, v = torch.split(q_k_v, self.h * self.d_k, dim=-1)
        q_h = q.unflatten(-1, (self.h, self.d_k)).transpose(1, 2)
        k_h = k.unflatten(-1, (self.h, self.d_k)).transpose(1, 2)
        v_h = v.unflatten(-1, (self.h, self.d_k)).transpose(1, 2)
        
        fsmn_memory = self.forward_fsmn(v, mask)
        scores = torch.matmul(q_h * (self.d_k ** -0.5), k_h.transpose(-2, -1))
        att_outs = self.forward_attention(v_h, scores, mask)
        
        out = att_outs + fsmn_memory
        if mask is not None: out = out * mask.unsqueeze(-1)
        return out

class EncoderLayerSANM(nn.Module):
    def __init__(self, in_size, size, self_attn, feed_forward, dropout_rate, normalize_before=True):
        super().__init__()
        self.self_attn, self.feed_forward = self_attn, feed_forward
        self.norm1, self.norm2 = LayerNorm(in_size), LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.in_size, self.size, self.normalize_before = in_size, size, normalize_before

    def forward(self, x, mask):
        residual = x
        if self.normalize_before: x = self.norm1(x)
        x = self.self_attn(x, mask)
        if self.in_size != self.size: return self.dropout(x), mask
        x = residual + self.dropout(x)
        if not self.normalize_before: x = self.norm1(x)
        
        residual = x
        if self.normalize_before: x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before: x = self.norm2(x)
        if mask is not None: x = x * mask.unsqueeze(-1)
        return x, mask

# ============================================================================
# DML Optimized Adaptor Components
# ============================================================================

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate):
        super().__init__()
        self.d_k, self.h = n_feat // n_head, n_head
        self.linear_q, self.linear_k, self.linear_v, self.linear_out = nn.Linear(n_feat, n_feat), nn.Linear(n_feat, n_feat), nn.Linear(n_feat, n_feat), nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        q = self.linear_q(query).unflatten(-1, (self.h, self.d_k)).transpose(1, 2)
        k = self.linear_k(key).unflatten(-1, (self.h, self.d_k)).transpose(1, 2)
        v = self.linear_v(value).unflatten(-1, (self.h, self.d_k)).transpose(1, 2)
        scores = torch.matmul(q * (self.d_k ** -0.5), k.transpose(-2, -1))
        
        if mask is not None:
            m = mask.unsqueeze(1).unsqueeze(2).eq(0)
            scores = scores.masked_fill(m, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(m, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
            
        x = torch.matmul(self.dropout(attn), v).transpose(1, 2).flatten(2)
        out = self.linear_out(x)
        if mask is not None: out = out * mask.unsqueeze(-1)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout_rate, normalize_before=True):
        super().__init__()
        self.self_attn, self.feed_forward = self_attn, feed_forward
        self.norm1, self.norm2 = nn.LayerNorm(size, eps=1e-12), nn.LayerNorm(size, eps=1e-12)
        self.dropout, self.normalize_before = nn.Dropout(dropout_rate), normalize_before

    def forward(self, x, mask=None):
        residual = x
        if self.normalize_before: x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, mask))
        if not self.normalize_before: x = self.norm1(x)
        residual = x
        if self.normalize_before: x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before: x = self.norm2(x)
        if mask is not None: x = x * mask.unsqueeze(-1)
        return x, mask

# ============================================================================
# Main Models
# ============================================================================

class SenseVoiceEncoderSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size, self.output_size, self.attention_heads, self.linear_units, self.num_blocks, self.tp_blocks, self.dropout_rate, self.attention_dropout_rate, self.kernel_size = 560, 512, 4, 2048, 50, 20, 0.1, 0.1, 11
        self.embed = SinusoidalPositionEncoder()
        self.encoders0 = nn.ModuleList([EncoderLayerSANM(560, 512, MultiHeadedAttentionSANM(4, 560, 512, 0.1, 11), PositionwiseFeedForward(512, 2048, 0.1), 0.1)])
        self.encoders = nn.ModuleList([EncoderLayerSANM(512, 512, MultiHeadedAttentionSANM(4, 512, 512, 0.1, 11), PositionwiseFeedForward(512, 2048, 0.1), 0.1) for _ in range(self.num_blocks - 1)])
        self.tp_encoders = nn.ModuleList([EncoderLayerSANM(512, 512, MultiHeadedAttentionSANM(4, 512, 512, 0.1, 11), PositionwiseFeedForward(512, 2048, 0.1), 0.1) for _ in range(self.tp_blocks)])
        self.after_norm, self.tp_norm = LayerNorm(512), LayerNorm(512)

    def forward(self, x, mask):
        x = self.embed(x * (512**0.5), mask)
        for layer in self.encoders0: x, _ = layer(x, mask)
        for layer in self.encoders:  x, _ = layer(x, mask)
        x = self.after_norm(x)
        if mask is not None: x = x * mask.unsqueeze(-1)
        for layer in self.tp_encoders: x, _ = layer(x, mask)
        x = self.tp_norm(x)
        if mask is not None: x = x * mask.unsqueeze(-1)
        return x

class CorrectTransformerAdaptor(nn.Module):
    def __init__(self, downsample_rate=1, encoder_dim=512, llm_dim=1024, ffn_dim=2048, n_layer=2, **kwargs):
        super().__init__()
        self.k = downsample_rate
        self.linear1, self.relu, self.linear2 = nn.Linear(encoder_dim * self.k, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, llm_dim)
        self.blocks = nn.ModuleList([EncoderLayer(llm_dim, MultiHeadedAttention(kwargs.get("attention_heads", 8), llm_dim, 0.0), PositionwiseFeedForward(llm_dim, llm_dim // 4, 0.0), 0.0) for _ in range(n_layer)]) if n_layer > 0 else None

    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.size()
        chunk_num = (seq_len - 1) // self.k + 1
        x = self.linear2(self.relu(self.linear1(F.pad(x, (0, 0, 0, chunk_num * self.k - seq_len)).unflatten(1, (chunk_num, self.k)).flatten(2))))
        if mask is not None: x = x * mask.unsqueeze(-1)
        if self.blocks is not None:
            for block in self.blocks: x, _ = block(x, mask)
        return x

class CTC(nn.Module):
    def __init__(self, odim, encoder_output_size):
        super().__init__()
        self.ctc_lo = nn.Linear(encoder_output_size, odim)
    def forward(self, hs_pad):
        return self.ctc_lo(hs_pad)

class HybridSenseVoice(nn.Module):
    def __init__(self, encoder_dim=512, llm_dim=1024, vocab_size=60515):
        super().__init__()
        self.audio_encoder = SenseVoiceEncoderSmall()
        self.audio_adaptor = CorrectTransformerAdaptor(1, 512, 1024, 2048, 2)
        self.ctc_decoder = CorrectTransformerAdaptor(1, 512, 512, 2048, 5)
        self.ctc_proj = CTC(vocab_size, 512)
        
    def load_weights(self, path):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in sd: sd = sd["state_dict"]
        nsd = {}
        for k, v in sd.items():
            if k.startswith("audio_encoder.") or k.startswith("audio_adaptor.") or k.startswith("ctc_decoder."): nsd[k] = v
            elif k.startswith("ctc.ctc_lo."): nsd[k.replace("ctc.ctc_lo", "ctc_proj.ctc_lo")] = v
        self.load_state_dict(nsd, strict=False)

# ============================================================================
# STFT Component
# ============================================================================

class STFT_Process(nn.Module):
    def __init__(self, n_fft=400, win_length=400, hop_len=160):
        super().__init__()
        self.n_fft, self.hop_len, self.half_n_fft = n_fft, hop_len, n_fft // 2
        window = torch.hamming_window(win_length, periodic=True).float()
        if win_length < n_fft: window = F.pad(window, ((n_fft - win_length) // 2, n_fft - win_length - (n_fft - win_length) // 2))
        t, f = torch.arange(n_fft).float().unsqueeze(0), torch.arange(self.half_n_fft + 1).float().unsqueeze(1)
        omega = 2 * torch.pi * f * t / n_fft
        self.register_buffer('cos_kernel', (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1))
        self.register_buffer('sin_kernel', (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1))
    def forward(self, x):
        xp = F.pad(x, (self.half_n_fft, self.half_n_fft))
        return F.conv1d(xp, self.cos_kernel, stride=self.hop_len), F.conv1d(xp, self.sin_kernel, stride=self.hop_len)

# ============================================================================
# Optimized Paddable Wrapper
# ============================================================================

class EncoderExportWrapperPaddable(nn.Module):
    def __init__(self, hybrid_model, stft_model, fbank, pre_emphasis=0.97, lfr_m=7, lfr_n=6):
        super().__init__()
        self.hybrid_model, self.stft_model, self.fbank = hybrid_model, stft_model, fbank
        self.pre_emphasis_val, self.lfr_m, self.lfr_n = float(pre_emphasis), lfr_m, lfr_n
        self.register_buffer('pre_emphasis', torch.tensor(pre_emphasis, dtype=torch.float32).view(1, 1, -1))

    def forward(self, audio, ilens):
        # 0. Initial Input
        valid_samples = ilens[0]

        # 1. Length-Aware Normalization & AUDIO HARD-ZEROING
        mean_val = torch.mean(audio[0, 0, :valid_samples])
        audio = audio - mean_val
        # Crucial Fix: Wipe out the background noise introduced by mean subtraction
        audio[:, :, valid_samples:] = 0.0
        
        # Pre-emphasis
        if self.pre_emphasis_val > 0:
            audio = torch.cat([audio[..., :1], audio[..., 1:] - self.pre_emphasis * audio[..., :-1]], dim=-1)
            # Re-apply hard-zeroing
            audio[:, :, valid_samples:] = 0.0
            
        # 2. STFT & Mel
        real, imag = self.stft_model(audio)
        T_mel_valid = (valid_samples // 160) + 1
        mel = (torch.matmul(self.fbank, real * real + imag * imag).transpose(1, 2) + 1e-7).log()
        
        # 3. Corrected LFR Logic with Replicate Padding
        T_lfr_valid = (T_mel_valid + self.lfr_n - 1) // self.lfr_n
        T_phys = mel.shape[1]
        T_lfr_phys = (T_phys + self.lfr_n - 1) // self.lfr_n
        
        # --- REPLICATE STRATEGY ---
        valid_part = mel[:, :T_mel_valid, :]
        last_frame = mel[:, [T_mel_valid - 1]]
        padding_fill = last_frame.repeat(1, T_phys - T_mel_valid, 1)
        mel_consistent = torch.cat([valid_part, padding_fill], dim=1)
        
        left_pad = mel_consistent[:, [0]].repeat(1, (self.lfr_m - 1) // 2, 1)
        right_pad = last_frame.repeat(1, (T_lfr_phys * self.lfr_n + self.lfr_m) - T_phys, 1)
        padded = torch.cat([left_pad, mel_consistent, right_pad], dim=1)
        
        lfr_list = []
        for i in range(self.lfr_m): lfr_list.append(padded[:, i : i + T_lfr_phys * self.lfr_n : self.lfr_n][:, :T_lfr_phys, :])
        x = torch.cat(lfr_list, dim=-1)
        
        # 4. Masking
        m = (torch.arange(T_lfr_phys, device=x.device).unsqueeze(0) < T_lfr_valid).float()
        x = x * m.unsqueeze(-1)
        
        # 5. Model
        enc = self.hybrid_model.audio_encoder(x, m)
        adapt = self.hybrid_model.audio_adaptor(enc, m)
        
        # Target length calculation
        olens_1 = 1 + (T_lfr_valid - 3 + 2) // 2
        target_len = (1 + (olens_1 - 3 + 2) // 2 - 1) // 2 + 1
        
        return enc, adapt[:, :target_len, :]
