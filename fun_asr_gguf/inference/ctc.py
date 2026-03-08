import onnxruntime
import numpy as np
import base64
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from . import logger

@dataclass
class Token:
    text: str
    start: float
    is_hotword: bool = False

class CTCTokenizer:
    """
    适配器模式：将 Nano 的 Base64 词表包装成满足 HotwordRadar 要求的接口
    """
    def __init__(self, id2token, encode_fn=None):
        self.id2token = id2token
        # 预构建反向查表字典，加速 encode()
        self.token2id = {v: k for k, v in id2token.items()}
        self._piece_size = len(id2token) if id2token else 0
        
    def get_piece_size(self):
        return self._piece_size
        
    def id_to_piece(self, i):
        # 兼容 SentencePiece 接口
        return self.id2token.get(i, f"<{i}>")
        
    def encode(self, text):
        """
        将文本编码为 CTC token ID 列表。
        按字符遍历，在 CTC 词表中查找对应 ID（精确匹配）。
        """
        result = []
        for char in text:
            tid = self.token2id.get(char)
            if tid is not None:
                result.append(tid)
        return result

    def encode_as_pieces(self, text):
        ids = self.encode(text)
        return [self.id_to_piece(i) for i in ids]

class CTCDecoder:
    """FunASR CTC 推理与解码器 (多阶段内部流水线)"""
    def __init__(self, model_path: str, tokens_path: str, onnx_provider: str = 'CPU', dml_pad_to: int = 30, corrector: Optional[Any] = None):
        self.model_path = model_path
        self.tokens_path = tokens_path
        self.onnx_provider = onnx_provider.upper()
        self.dml_pad_to = dml_pad_to
        self.corrector = corrector
        
        self.sess = None
        self.id2token = {}
        self.input_dtype = np.float32
        self.tokenizer = None   # CTCTokenizer 包装器
        self.radar = None       # HotwordRadar（由 ModelManager 注入）
        self.integrator = None  # ResultIntegrator（由 ModelManager 注入）
        
        self._initialize_session()
        self._load_tokens()
        self.warmup()

    def _initialize_session(self):
        session_opts = onnxruntime.SessionOptions()
        session_opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session_opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        # session_opts.enable_profiling = True
        
        available_providers = onnxruntime.get_available_providers()
        providers = ['CPUExecutionProvider']
        
        if self.onnx_provider in ('TENSORRT', 'TRT') and 'TensorrtExecutionProvider' in available_providers:
            providers.insert(0, ('TensorrtExecutionProvider', {
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': Path(self.model_path).parent / 'trt_cache',
            }))
        elif self.onnx_provider == 'DML' and 'DmlExecutionProvider' in available_providers:
            providers.insert(0, 'DmlExecutionProvider') 
        elif self.onnx_provider == 'CUDA' and 'CUDAExecutionProvider' in available_providers:
            providers.insert(0, 'CUDAExecutionProvider')
            
        logger.info(f"[CTC] 加载模型: {os.path.basename(self.model_path)} (Providers: {providers})")
        
        self.sess = onnxruntime.InferenceSession(
            self.model_path, 
            sess_options=session_opts, 
            providers=providers
        )
        
        # 检测模型输入精度
        in_type = self.sess.get_inputs()[0].type
        self.input_dtype = np.float16 if 'float16' in in_type else np.float32

    def _load_tokens(self):
        self.id2token = load_ctc_tokens(self.tokens_path)
        self.tokenizer = CTCTokenizer(self.id2token)
        
        # 精准寻找 Blank ID：优先匹配包含关键标识的符号
        self.blank_id = None
        for tid, token_text in self.id2token.items():
            clean_text = token_text.lower().strip()
            if clean_text in ("<blk>", "<blank>", "<pad>"):
                self.blank_id = tid
                break
        if self.blank_id is None:
            self.blank_id = max(self.id2token.keys()) if self.id2token else 0
            

    def warmup(self):
        if self.dml_pad_to <= 0:
            return
        target_t_lfr = int((self.dml_pad_to * 100 + 5) // 6) + 1
        dummy_enc = np.zeros((1, target_t_lfr, 512), dtype=self.input_dtype)
        in_name = self.sess.get_inputs()[0].name
        logger.info(f"[CTC] 正在预热 (固定形状: {self.dml_pad_to}s)...")
        self.sess.run(None, {in_name: dummy_enc})

    # ================================================================
    # 对外唯一入口：decode()
    # 返回三元组 (ctc_results, hotwords, t_stats)
    # ================================================================

    def decode(self, enc_output: np.ndarray, enable_ctc: bool, max_hotwords: int = 10, top_k: int = 10) -> Tuple[List[Token], List[str], Dict[str, float]]:
        """
        完整解码流水线（黑箱）。
        内部按顺序执行：ONNX推理 → 贪婪解码 → 雷达扫描 → 整合 → 拼音纠错
        
        Returns:
            ctc_results: 贪婪解码或整合后的 Token 列表
            hotwords:    综合检测到的热词文本列表
            t_stats:     各阶段耗时字典
        """
        t_stats = {"infer": 0.0, "decode": 0.0, "radar": 0.0, "integrate": 0.0, "hotword": 0.0}
        if not enable_ctc or self.sess is None:
            return [], [], t_stats

        # ---- 阶段 1: ONNX 推理 (获取 Top-K) ----
        t0 = time.perf_counter()
        topk_log_probs, topk_indices = self._infer(enc_output)
        t_stats["infer"] = time.perf_counter() - t0
        print(f"    [CTC] 1. Infer: {t_stats['infer']*1000:.2f}ms")
        
        # ---- 阶段 2: 贪婪解码 (Top-1) ----
        t0 = time.perf_counter()
        indices_2d = topk_indices[0]        # [T, K]
        top1_indices = indices_2d[:, 0]     # [T]
        ctc_text, ctc_results = self._greedy_decode(top1_indices)
        t_stats["decode"] = time.perf_counter() - t0
        print(f"    [CTC] 2. Decode: {t_stats['decode']*1000:.2f}ms | Text: '{ctc_text}'")
        
        # ---- 阶段 3: 雷达扫描 (Top-K 空间) ----
        t0 = time.perf_counter()
        topk_probs = np.exp(topk_log_probs[0])
        detected_hotwords = self._radar_scan(indices_2d, topk_probs, top1_indices, top_k=top_k)
        t_stats["radar"] = time.perf_counter() - t0
        if detected_hotwords:
            # 增加时间轴回显，方便排查误触发
            radar_info = [f"{h['text']}({h['start']:.1f}s)" for h in detected_hotwords]
            print(f"    [CTC] 3. Radar: {t_stats['radar']*1000:.2f}ms | Detected: {radar_info}")
        
        # ---- 阶段 4: 整合 (Greedy + 热词 → 替换) ----
        t0 = time.perf_counter()
        if detected_hotwords and ctc_results:
            ctc_text, ctc_results = self._integrate(ctc_results, detected_hotwords)
            print(f"    [CTC] 4. Integrate: {t_stats['integrate']*1000:.2f}ms | {''.join([r.text for r in ctc_results])}")
        t_stats["integrate"] = time.perf_counter() - t0
        
        # ---- 阶段 5: 拼音纠错 (补充热词) ----
        t0 = time.perf_counter()
        hotwords = [h["text"] for h in detected_hotwords]
        if self.corrector and self.corrector.hotwords and ctc_text:
            corrected_text, extra_hotwords = self._correct(ctc_text, max_hotwords)
            hotwords = list(set(hotwords) | set(extra_hotwords))
            t_stats["hotword"] = time.perf_counter() - t0
            if extra_hotwords:
                print(f"    [CTC] 5. Corrector: {t_stats['hotword']*1000:.2f}ms | Extra: {extra_hotwords}")
        else:
            t_stats["hotword"] = time.perf_counter() - t0
            
        return ctc_results, hotwords, t_stats

    # ================================================================
    # 内部阶段方法
    # ================================================================

    def _infer(self, enc_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """阶段 1: ONNX 推理，返回 (topk_log_probs, topk_indices)"""
        outputs = self.sess.run(None, {"enc_output": enc_output})
        return outputs[0], outputs[1]

    def _greedy_decode(self, top1_indices: np.ndarray) -> Tuple[str, List[Token]]:
        """阶段 2: 基于 Top-1 Index 的贪婪解码"""
        ctc_text, ctc_results, _ = decode_ctc_indices(top1_indices, self.id2token)
        return ctc_text, ctc_results

    def _radar_scan(self, indices_2d: np.ndarray, topk_probs: np.ndarray, top1_indices: np.ndarray, top_k: int = 10) -> List[Dict]:
        """阶段 3: 热词雷达扫描"""
        if self.radar is None:
            return []
        
        # 缩减搜索空间并确定网格打印深度
        sliced_ids = indices_2d[:, :top_k]
        sliced_probs = topk_probs[:, :top_k]
        
        # 仅在非预热/正常解码时通过 display_top_k 触发打印 (如果需要常驻显示可直接设为 top_k)
        # 显式传递 blank_id，防止雷达误判实音间隙
        return self.radar.scan(sliced_ids, sliced_probs, top1_indices, blank_id=self.blank_id)

    def _integrate(self, ctc_results: List[Token], detected_hotwords: List[Dict]) -> Tuple[str, List[Token]]:
        """阶段 4: 将雷达命中的热词整合进贪婪结果"""
        if self.integrator is None:
            return "".join([r.text for r in ctc_results]), ctc_results
        
        greedy_fmt = [{"text": r.text, "start": r.start} for r in ctc_results]
        integrated_list = self.integrator.integrate(greedy_fmt, detected_hotwords)
        
        # 将整合结果转回 Token 列表
        new_results = [
            Token(text=r["text"], start=r["start"], is_hotword=r.get("is_hotword", False))
            for r in integrated_list
        ]
        new_text = "".join([r.text for r in new_results])
        return new_text, new_results

    def _correct(self, text: str, max_hotwords: int) -> Tuple[str, List[str]]:
        """阶段 5: 拼音纠错，返回 (纠错后文本, 额外发现的热词列表)"""
        res = self.corrector.correct(text, k=max_hotwords)
        candidates = set()
        for _, hw, _ in res.matchs: candidates.add(hw)
        for _, hw, _ in res.similars: candidates.add(hw)
        return res.text, list(candidates)


def load_ctc_tokens(filename):
    """加载 CTC 词表"""
    id2token = dict()
    if not os.path.exists(filename):
        return id2token
    with open(filename, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            if len(parts) == 1:
                t, i = " ", parts[0]
            else:
                t, i = parts
            
            # Pre-decode base64 here to save time during inference
            try:
                # Some tokens might rely on being decoded, do it once
                token_text = base64.b64decode(t).decode("utf-8")
            except:
                token_text = t
                
            id2token[int(i)] = token_text
                
    return id2token

def decode_ctc_indices(indices, id2token):
    """
    Greedy search 贪心解码 (直接基于 Indices)。
    """
    t0 = time.perf_counter()
    blank_id = max(id2token.keys()) if id2token else 0
    
    frame_shift_ms = 60
    
    # 1. Collapse repeats
    collapsed = []
    if len(indices) > 0:
        current_id = indices[0]
        start_idx = 0
        for i in range(1, len(indices)):
            if indices[i] != current_id:
                collapsed.append((current_id, start_idx))
                current_id = indices[i]
                start_idx = i
        collapsed.append((current_id, start_idx))

    results = []

    # 2. Filter blanks and decode text
    for token_id, start in collapsed:
        if token_id == blank_id:
            continue

        token_text = id2token.get(token_id, "")
        if not token_text: continue

        # Calculate time (只计算起始位置)
        t_start = max((start * frame_shift_ms) / 1000.0, 0.0)

        results.append(Token(
            text=token_text,
            start=t_start
        ))
                
    full_text = "".join([r.text for r in results])
    t_loop = time.perf_counter() - t0
    
    timings = {
        "cast": 0.0,
        "argmax": 0.0,
        "loop": t_loop
    }
    return full_text, results, timings

def align_timestamps(ctc_results, llm_text):
    """
    使用 Needleman-Wunsch 算法对齐 CTC 结果和 LLM 文本
    只使用起始位置进行匹配
    """
    if not ctc_results or not llm_text:
        return []

    # 1. 展开 CTC 结果为字符级别（只保留起始位置）
    ctc_chars = []
    for item in ctc_results:
        text = item.text
        start = item.start

        if len(text) > 0:
            # 假设每个字符占用相同时间间隔
            char_duration = 0.08  # 默认每个字符约 80ms
            for i, char in enumerate(text):
                c_start = start + i * char_duration
                ctc_chars.append({"char": char, "start": c_start})

    llm_chars = list(llm_text)

    n = len(ctc_chars) + 1
    m = len(llm_chars) + 1

    # Core DP Matrix
    score = np.zeros((n, m), dtype=np.float32)
    # trace: 1=diag, 2=up, 3=left
    trace = np.zeros((n, m), dtype=np.int8)

    gap_penalty = -1.0
    match_score = 1.0
    mismatch_score = -1.0

    # Init margins
    for i in range(n): score[i][0] = i * gap_penalty
    for j in range(m): score[0][j] = j * gap_penalty

    # Fill DP
    for i in range(1, n):
        for j in range(1, m):
            char_ctc = ctc_chars[i-1]["char"]
            char_llm = llm_chars[j-1]

            s_diag = score[i-1][j-1] + (match_score if char_ctc.lower() == char_llm.lower() else mismatch_score)
            s_up = score[i-1][j] + gap_penalty
            s_left = score[i][j-1] + gap_penalty

            best = max(s_diag, s_up, s_left)
            score[i][j] = best

            if best == s_diag: trace[i][j] = 1
            elif best == s_up: trace[i][j] = 2
            else: trace[i][j] = 3

    # Traceback
    llm_alignment = [None] * len(llm_chars)
    i, j = n - 1, m - 1

    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i][j] == 1:
            llm_alignment[j-1] = ctc_chars[i-1]
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or trace[i][j] == 2):
            i -= 1
        elif j > 0 and (i == 0 or trace[i][j] == 3):
            llm_alignment[j-1] = None
            j -= 1

    # 插值填充未对齐的字符
    anchors = []
    for idx, item in enumerate(llm_alignment):
        if item is not None:
            anchors.append((idx, item["start"]))

    final_chars = []

    def get_interpolated_start(target_idx):
        """插值计算起始位置"""
        prev_a, next_a = None, None
        for a in anchors:
            if a[0] < target_idx:
                prev_a = a
            elif a[0] > target_idx:
                next_a = a
                break

        if prev_a and next_a:
            p_idx, p_start = prev_a
            n_idx, n_start = next_a

            # 线性插值
            total_gap = n_idx - p_idx
            time_gap = n_start - p_start
            step = time_gap / total_gap

            relative_step = target_idx - p_idx
            return p_start + relative_step * step
        elif prev_a:
            return prev_a[1] + 0.05  # 向后推一点
        elif next_a:
            return max(0, next_a[1] - 0.05)  # 向前推一点
        else:
            return 0.0

    for idx, char in enumerate(llm_chars):
        if llm_alignment[idx]:
            s = llm_alignment[idx]["start"]
        else:
            s = get_interpolated_start(idx)
        final_chars.append({"char": char, "start": s})

    return final_chars


