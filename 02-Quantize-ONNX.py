import os
import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.quantization import quantize_dynamic, QuantType
from export_config import EXPORT_DIR

# Configuration
MODEL_DIR = str(EXPORT_DIR)
FP32_MODELS = [
    f"{MODEL_DIR}/Fun-ASR-Nano-Encoder-Adaptor.fp32.onnx",
    f"{MODEL_DIR}/Fun-ASR-Nano-CTC.fp32.onnx"
]

def convert_to_fp16(input_path):
    output_path = input_path.replace(".fp32.onnx", ".fp16.onnx")
    print(f"\n[FP16] Converting {os.path.basename(input_path)} -> {os.path.basename(output_path)}...")
    
    try:
        model = onnx.load(input_path)
        # Use ORT Transformers conversion for better DML compatibility
        # Block ops that are sensitive to precision changes or shape calculation
        model_fp16 = convert_float_to_float16(
            model,
            keep_io_types=False,
            min_positive_val=1e-7,
            max_finite_val=65504,
            op_block_list=['LayerNormalization']
        )
        onnx.save(model_fp16, output_path)
        print(f"   [Success] Saved FP16 model.")
    except Exception as e:
        print(f"   [Failed] FP16 conversion error: {e}")

def convert_to_int8(input_path):
    output_path = input_path.replace(".fp32.onnx", ".int8.onnx")
    print(f"\n[INT8] Quantizing {os.path.basename(input_path)} -> {os.path.basename(output_path)}...")
    
    try:
        quantize_dynamic(
            input_path,
            output_path,
            op_types_to_quantize=["MatMul"], # Primary target for weight compression
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QUInt8
        )
        print(f"   [Success] Saved INT8 model.")
    except Exception as e:
        print(f"   [Failed] INT8 quantization error: {e}")

def main():
    print("--- Starting Batch Conversion ---")
    
    if not os.path.exists(MODEL_DIR):
        print(f"Error: Directory {MODEL_DIR} not found.")
        return

    for model_path in FP32_MODELS:
        if not os.path.exists(model_path):
            print(f"\n[Skip] Model not found: {model_path}")
            continue
            
        # 1. Convert to FP16
        convert_to_fp16(model_path)
        
        # 2. Convert to INT8
        convert_to_int8(model_path)

    print("\n--- All Conversions Complete ---")

if __name__ == "__main__":
    main()
