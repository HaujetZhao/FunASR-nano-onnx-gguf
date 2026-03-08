import time
import re
import ctypes
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from . import logger
from .. import llama
from ..ctc import align_timestamps, CTCDecoder
from ..utils import vprint
from ..schema import DecodeResult, Timings, RecognitionStream, LLMDecodeResult
from ..display import DisplayReporter
from .model_manager import ModelManager

# 全局静默 Reporter，用于默认参数，避免重复创建线程
_SILENT_REPORTER = DisplayReporter(verbose=False)

class LLMDecoder:
    """负责 LLM 推理循环"""
    def __init__(self, models: ModelManager):
        self.models = models
        self.stop_tokens = [151643, 151645]

    def decode(
        self,
        full_embd: np.ndarray,
        n_input_tokens: int,
        n_predict: int,
        stream_output: bool = False,
        reporter: Optional[DisplayReporter] = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        top_k: int = 50
    ) -> LLMDecodeResult:
        
        res = LLMDecodeResult()
        t_inject_start = time.perf_counter()
        
        # 1. Inject
        self.models.ctx.clear_kv_cache()
        
        batch_embd = llama.LlamaBatch(n_input_tokens, full_embd.shape[1], 1)
        batch_embd.set_embd(full_embd)
        batch_embd.struct.token = ctypes.cast(None, ctypes.POINTER(llama.llama_token))

        ret = self.models.ctx.decode(batch_embd)
        if ret != 0: raise RuntimeError(f"Decode failed (ret={ret})")
        
        res.t_inject = time.perf_counter() - t_inject_start

        # 2. Generation Loop
        t_gen_start = time.perf_counter()
        batch_text = llama.LlamaBatch(1, 0, 1)

        current_pos = n_input_tokens
        asr_decoder = llama.ASRStreamDecoder(self.models.vocab, reporter if stream_output else None)
        
        seed = int(np.random.randint(0, 2**31 - 1))
        with llama.LlamaSampler(temperature=temperature, top_k=top_k, top_p=top_p, seed=seed) as smpl:
            for _ in range(n_predict):
                # 采样
                token_id = smpl.sample(self.models.ctx, -1)

                # 提交异步解码任务
                if self.models.ctx.decode_token(token_id) != 0:
                    break
                current_pos += 1

                # 检查 token id 是否为中止符
                if token_id == self.models.eos_token or token_id in self.stop_tokens:
                    break
                
                # 解码 token id
                asr_decoder.push(token_id)
                
                # 熔断检查
                if len(asr_decoder.tokens) <= 30: 
                    continue
                
                # 尾部无限循环，熔断
                if len(set(asr_decoder.tokens[-30:])) <= 3:
                    res.is_aborted = True
                    break

                # 达到30个token时还没生成标点，熔断
                if len(asr_decoder.tokens) == 30: 
                    if not re.search(r'[，。？！、；：,\.?!;:]', asr_decoder.generated_text):
                        res.is_aborted = True
                        break


        
        asr_decoder.flush()

        # batch_text 会由 __del__ 自动释放
        res.text = asr_decoder.generated_text
        res.n_gen = asr_decoder.tokens_generated
        res.t_gen = time.perf_counter() - t_gen_start
        
        return res


class StreamDecoder:
    """协调完整流程的解码器"""
    def __init__(self, models: ModelManager):
        self.models = models
        self.llm_decoder = LLMDecoder(models)

    def decode_stream(
        self,
        stream: RecognitionStream,
        language: Optional[str] = None,
        context: Optional[str] = None,
        verbose: bool = True,
        reporter: Optional[DisplayReporter] = None,
        temperature: float = 0.3,
        top_p: float = 1.0,
        top_k: int = 50,
        timestamp_offset: float = -0.24
    ) -> DecodeResult:
        
        reporter = reporter or _SILENT_REPORTER
        
        timings = Timings()
        
        # 1. Encode
        reporter.print("\n[2] 音频编码...")
        t_s = time.perf_counter()
        audio_embd, enc_output = self.models.encoder.encode(stream.audio_data)
        timings.encode = time.perf_counter() - t_s
        reporter.print(f"    耗时: {timings.encode*1000:.2f}ms")

        reporter.print("\n[3] CTC 解码...")
        t_s = time.perf_counter()
        ctc_results, hotwords, ctc_times = self.models.ctc_decoder.decode(
            enc_output, 
            self.models.config.enable_ctc, 
            self.models.config.max_hotwords, 
            top_k = 10
        )
        timings.ctc = time.perf_counter() - t_s
        
        if reporter.verbose and ctc_results:
            ctc_text = "".join([r.text for r in ctc_results])
            reporter.print(f"    CTC: {ctc_text}")
            if hotwords: reporter.print(f"    热词: {hotwords}")
        reporter.print(f"    耗时: {timings.ctc*1000:.2f}ms")

        # 3. Prompt
        reporter.print("\n[4] 准备 Prompt...")
        
        t_s = time.perf_counter()
        p_embd, s_embd, n_p, n_s, p_text = self.models.prompt_builder.build_prompt(hotwords, language, context)
        
        # 确保属性已初始化
        if not hasattr(timings, 'prepare'): timings.prepare = 0.0
        timings.prepare += (time.perf_counter() - t_s)
        
        if reporter.verbose and reporter.skip_technical is False:
            reporter.print("-" * 15 + " Prefix Prompt " + "-" * 15 + "\n" + p_text + "\n" + "-" * 40)
        
        reporter.print(f"    Prefix: {n_p} tokens")
        reporter.print(f"    Suffix: {n_s} tokens")

        # 4. LLM
        reporter.print("\n[5] LLM 解码...")
        reporter.print("=" * 70)
        
        full_embd = np.concatenate([p_embd, audio_embd.astype(np.float32), s_embd], axis=0)

        # LLM 解码循环：若熔断则加温重试（最多重试 3 次）
        text = ""
        for _ in range(4):
            llm_res = self.llm_decoder.decode(
                full_embd, full_embd.shape[0], self.models.config.n_predict, 
                stream_output=verbose, reporter=reporter,
                temperature=temperature, top_p=top_p, top_k=top_k
            )
            if not llm_res.is_aborted: break
            temperature += 0.3
            llm_res.text += "====解码有误，强制熔断===="
            print(f"\n\n[!] 解码有误，熔断重试 (温度设为 {temperature:.1f})\n")

        text = llm_res.text.strip()
        timings.inject = llm_res.t_inject
        timings.llm_generate = llm_res.t_gen
        
        if reporter: reporter.print("\n" + "=" * 70)
        reporter.print("\n" + "=" * 70)


        # 5. Align
        reporter.print("\n[6] 时间戳对齐")
        t_s = time.perf_counter()
        aligned = None
        timestamps = []
        tokens = []
        if ctc_results:
            aligned = align_timestamps(ctc_results, text)
            if aligned:
                tokens = [seg['char'] for seg in aligned]
                timestamps = [seg['start'] for seg in aligned]
        timings.align = time.perf_counter() - t_s
        
        # 应用时间戳偏移（只影响字幕输出，不影响 integrate 和 radar）
        if aligned and timestamp_offset != 0.0:
            for seg in aligned:
                seg['start'] = max(seg['start'] + timestamp_offset, 0.0)
        
        if aligned:
            reporter.print(f"    对齐耗时: {timings.align*1000:.2f}ms")
            preview = " ".join([f"{r['char']}({r['start']:.2f}s)" for r in aligned[:10]])
            if len(aligned) > 10: preview += " ..."
            reporter.print(f"    结果预览: {preview}")

        # Set stream result
        stream.set_result(text=text, timestamps=timestamps, tokens=tokens)
        
        return DecodeResult(
            text=text, ctc_results=ctc_results, aligned=aligned,
            audio_embd=audio_embd, n_prefix=n_p, n_suffix=n_s,
            n_gen=llm_res.n_gen, timings=timings, hotwords=hotwords,
            is_aborted=llm_res.is_aborted
        )


