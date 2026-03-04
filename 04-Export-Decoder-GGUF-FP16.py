import torch
import os
import json
import shutil
import subprocess
import sys
from export_config import MODEL_DIR, EXPORT_DIR

# 源模型路径
SOURCE_MODEL_PATH = str(MODEL_DIR)
CONFIG_PATH = f'{SOURCE_MODEL_PATH}/Qwen3-0.6B/config.json'

# 中间产物 (HF 格式) 输出路径
OUTPUT_HF_DIR = str(EXPORT_DIR / 'Qwen3-0.6B')

# 最终 GGUF 输出文件
OUTPUT_GGUF_FILE_FP16 = str(EXPORT_DIR / 'Fun-ASR-Nano-Decoder.fp16.gguf')

# 转换脚本路径
CONVERT_SCRIPT = './fun_asr_gguf/export/convert_hf_to_gguf.py'

def main():
    print("\n[Step 04] Exporting Decoder GGUF FP16...")
    
    # 1. 提取 LLM 并保存为 Hugging Face 格式
    if os.path.exists(os.path.join(OUTPUT_HF_DIR, "model.safetensors")):
         print(f"HF model appears to exist in {OUTPUT_HF_DIR}. Skipping extraction.")
    else:
        try:
            from transformers import Qwen3ForCausalLM, Qwen3Config
        except ImportError:
            try:
                from transformers import Qwen2ForCausalLM as Qwen3ForCausalLM
                from transformers import Qwen2Config as Qwen3Config
            except ImportError:
                from transformers import AutoModelForCausalLM as Qwen3ForCausalLM
                from transformers import AutoConfig as Qwen3Config

        model_pt_path = f'{SOURCE_MODEL_PATH}/model.pt'
        print(f"Loading full model from {model_pt_path} ...")
        full_model = torch.load(model_pt_path, map_location='cpu')

        llm_weights = {}
        for key in full_model.keys():
            if key.startswith('llm.'):
                hf_key = key.replace('llm.', '')
                llm_weights[hf_key] = full_model[key]
        
        print(f"Extracted {len(llm_weights)} weight keys.")
        del full_model
        
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            
        config = Qwen3Config(**config_dict)
        qwen_model = Qwen3ForCausalLM(config)
        qwen_model.load_state_dict(llm_weights, strict=True)
        
        os.makedirs(OUTPUT_HF_DIR, exist_ok=True)
        qwen_model.save_pretrained(OUTPUT_HF_DIR, safe_serialization=True)
        
        original_tokenizer_dir = f'{SOURCE_MODEL_PATH}/Qwen3-0.6B'
        files_to_copy = ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'generation_config.json']
        for file in files_to_copy:
            src = os.path.join(original_tokenizer_dir, file)
            dst = os.path.join(OUTPUT_HF_DIR, file)
            if os.path.exists(src): shutil.copy(src, dst)
        
        print("HF Model saved successfully.")

    # 2. 转换为 GGUF 格式 (仅 FP16)
    if not os.path.exists(CONVERT_SCRIPT):
        print(f"Error: Conversion script not found at {CONVERT_SCRIPT}")
        return

    print(f"\n---> Converting to FP16 GGUF: {OUTPUT_GGUF_FILE_FP16} ...")
    cmd = [
        sys.executable, CONVERT_SCRIPT,
        OUTPUT_HF_DIR,
        '--outfile', OUTPUT_GGUF_FILE_FP16,
        '--outtype', 'f16',
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ GGUF FP16 conversion successful!")
    except subprocess.CalledProcessError as e:
        print(f"❌ GGUF FP16 conversion failed: {e}")

if __name__ == "__main__":
    main()
