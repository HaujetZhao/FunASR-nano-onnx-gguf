import torch
import os
import json
import shutil
import subprocess
import sys

# =========================================================================
# 配置部分
# =========================================================================

from export_config import MODEL_DIR, EXPORT_DIR

# 源模型路径
SOURCE_MODEL_PATH = str(MODEL_DIR)
CONFIG_PATH = f'{SOURCE_MODEL_PATH}/Qwen3-0.6B/config.json'

# 中间产物 (HF 格式) 输出路径
OUTPUT_HF_DIR = str(EXPORT_DIR / 'Qwen3-0.6B')
# Tokenizer 输出路径
OUTPUT_TOKENIZER_DIR = str(EXPORT_DIR / 'Qwen3-0.6B')

# 最终 GGUF 输出文件
OUTPUT_GGUF_FILE_FP16 = str(EXPORT_DIR / 'Fun-ASR-Nano-Decoder.fp16.gguf')
OUTPUT_GGUF_FILE_INT8 = str(EXPORT_DIR / 'Fun-ASR-Nano-Decoder.q8_0.gguf')

# 转换脚本路径 (使用 fun_asr_gguf 目录下的 convert_hf_to_gguf.py)
CONVERT_SCRIPT = './fun_asr_gguf/convert_hf_to_gguf.py'


def main():
    # ---------------------------------------------------------------------
    # 1. 提取 LLM 并保存为 Hugging Face 格式
    # ---------------------------------------------------------------------
    print("\n[Stage 1] Checking/Extracting LLM Decoder to Hugging Face format...")
    
    # 检查是否已存在 HF 模型，存在则跳过提取
    if os.path.exists(os.path.join(OUTPUT_HF_DIR, "model.safetensors")):
         print(f"HF model appears to exist in {OUTPUT_HF_DIR}. Skipping extraction.")
    else:
        # 尝试导入 Qwen3 类 (参考 save_standard_hf_model.py)
        try:
            from transformers import Qwen3ForCausalLM, Qwen3Config
            print("Successfully imported Qwen3ForCausalLM and Qwen3Config")
        except ImportError:
            print("Warning: Qwen3 classes not found in transformers, falling back to Qwen2 or AutoClasses.")
            try:
                from transformers import Qwen2ForCausalLM as Qwen3ForCausalLM
                from transformers import Qwen2Config as Qwen3Config
            except ImportError:
                from transformers import AutoModelForCausalLM as Qwen3ForCausalLM
                from transformers import AutoConfig as Qwen3Config

        # 加载完整 PyTorch 模型 (FunASR 格式)
        model_pt_path = f'{SOURCE_MODEL_PATH}/model.pt'
        print(f"Loading full model from {model_pt_path} ...")
        full_model = torch.load(model_pt_path, map_location='cpu')

        # 提取 LLM 权重
        llm_weights = {}
        print("Extracting LLM weights...")
        for key in full_model.keys():
            if key.startswith('llm.'):
                # 将键名从 llm.model.xxx 转换为 model.xxx (HF 标准格式)
                hf_key = key.replace('llm.', '')
                llm_weights[hf_key] = full_model[key]
        
        print(f"Extracted {len(llm_weights)} weight keys.")
        del full_model
        
        # 加载配置
        print(f"Loading config from {CONFIG_PATH} ...")
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            
        config = Qwen3Config(**config_dict)
        
        # 初始化空模型
        print("Initializing empty Qwen3ForCausalLM...")
        qwen_model = Qwen3ForCausalLM(config)

        # 加载权重
        print("Loading state dict into LLM...")
        qwen_model.load_state_dict(llm_weights, strict=True)
        
        # 保存 HF 模型 (Safetensors)
        os.makedirs(OUTPUT_HF_DIR, exist_ok=True)
        print(f"Saving HF model to {OUTPUT_HF_DIR} ...")
        qwen_model.save_pretrained(OUTPUT_HF_DIR, safe_serialization=True)
        
        # 复制 tokenizer 文件到单独目录
        print(f"Copying tokenizer files to {OUTPUT_TOKENIZER_DIR} ...")
        os.makedirs(OUTPUT_TOKENIZER_DIR, exist_ok=True)
        original_tokenizer_dir = f'{SOURCE_MODEL_PATH}/Qwen3-0.6B'
        files_to_copy = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'generation_config.json']
        for file in files_to_copy:
            src = os.path.join(original_tokenizer_dir, file)
            dst = os.path.join(OUTPUT_TOKENIZER_DIR, file)
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"  Copied {file}")
        
        print("HF Model and Tokenizer saved successfully.")

    # ---------------------------------------------------------------------
    # 2. 转换为 GGUF 格式 (FP16 & Int8)
    # ---------------------------------------------------------------------
    print("\n[Stage 2] Converting HF model to GGUF...")

    if not os.path.exists(CONVERT_SCRIPT):
        print(f"Error: Conversion script not found at {CONVERT_SCRIPT}")
        print("Please ensure convert_hf_to_gguf.py is in the lib directory.")
        return

    con_script = CONVERT_SCRIPT

    # 定义转换任务列表: (输出路径, 量化类型)
    tasks = [
        (OUTPUT_GGUF_FILE_FP16, 'f16'),
        (OUTPUT_GGUF_FILE_INT8, 'q8_0')
    ]

    for output_file, out_type in tasks:
        print(f"\n---> Exporting {out_type} model to {output_file} ...")
        cmd = [
            sys.executable,
            con_script,
            OUTPUT_HF_DIR,
            '--outfile', output_file,
            '--outtype', out_type,
        ]
        
        print(f"Executing: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ GGUF {out_type} conversion successful! Output: {output_file}")
        except subprocess.CalledProcessError as e:
            print(f"❌ GGUF {out_type} conversion failed with error: {e}")

if __name__ == "__main__":
    main()
