import os
import sys

# 确保能导入 fun_asr_gguf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_simplification():
    print("Testing Architecture Phase 2: Facade and OOP refactoring...")
    
    try:
        from fun_asr_gguf.inference.asr_engine import FunASREngine
        print("✓ FunASREngine exists")
        from fun_asr_gguf.inference.pipeline import InferencePipeline
        print("✓ InferencePipeline exists (renamed from ASRDecoder)")
        from fun_asr_gguf.inference.transcriber import AudioTranscriber
        print("✓ AudioTranscriber exists")
        
        # 检查是否成功移除了 create_asr_engine
        try:
            from fun_asr_gguf.inference.asr_engine import create_asr_engine
            print("✗ create_asr_engine still exists in asr_engine.py!")
        except ImportError:
            print("✓ create_asr_engine successfully removed from asr_engine.py")

        try:
            from fun_asr_gguf import create_asr_engine
            print("✗ create_asr_engine still exists in __init__.py!")
        except ImportError:
            print("✓ create_asr_engine successfully removed from __init__.py")

        # 检查是否还能导入已删除的文件（应该报错）
        try:
            from fun_asr_gguf.inference.core.decoder import ASRDecoder
            print("✗ decoder.py still exists!")
        except ImportError:
            print("✓ decoder.py successfully removed (renamed to pipeline.py)")
            
    except Exception as e:
        print(f"✗ Component check failed: {e}")
        return

    print("\nSimplification verification completed (Structural check).")

if __name__ == "__main__":
    test_simplification()
