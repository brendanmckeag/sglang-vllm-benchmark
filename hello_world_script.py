import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

import gc
import torch
import time
import uuid
import requests

import logging
logging.getLogger("sglang").setLevel(logging.ERROR)
logging.getLogger("vllm").setLevel(logging.ERROR)
os.environ['NCCL_DEBUG'] = 'ERROR'

# Shared configuration - change prompts here for both engines
SHARED_CONFIG = {
    "prompt": "Hello, I am a language model and I can",
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "max_tokens": 1024,
    "min_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "debug": False  # Set to True for debugging output
}

def clear_gpu_memory():
    """Clear GPU memory to ensure clean state between engines"""
    print("🧹 Clearing GPU memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("✓ GPU memory cleared")

def check_gpu_memory():
    """Check and display current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"📊 GPU Memory - Used: {allocated:.1f}GB, Reserved: {reserved:.1f}GB, Total: {total:.1f}GB")
        return allocated, reserved, total
    else:
        print("⚠ No CUDA available")
        return 0, 0, 0

def vllm_hello_world():
    """vLLM Hello World Example with full GPU resource management"""
    print("=== vLLM Hello World ===")
    
    clear_gpu_memory()
    check_gpu_memory()
    
    llm = None
    
    try:
        from vllm import LLM, SamplingParams
        print("✓ vLLM imported successfully")
        
        model_name = SHARED_CONFIG["model_name"]
        prompt = SHARED_CONFIG["prompt"]
        
        print(f"Loading model: {model_name}")
        
        llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            swap_space=0,
            max_model_len=2048,
            enforce_eager=True,
        )
        print("✓ Model loaded successfully!")
        check_gpu_memory()
        
        sampling_params = SamplingParams(
            temperature=SHARED_CONFIG["temperature"],
            top_p=SHARED_CONFIG["top_p"],
            max_tokens=SHARED_CONFIG["max_tokens"],
            min_tokens=SHARED_CONFIG["min_tokens"],
            repetition_penalty=SHARED_CONFIG["repetition_penalty"]
        )
        
        print(f"Input prompt: '{prompt}'")
        print(f"Generation params: min_tokens={SHARED_CONFIG['min_tokens']}, max_tokens={SHARED_CONFIG['max_tokens']}")
        
        print("Generating response...")
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        end_time = time.time()
        
        for output in outputs:
            generated_text = output.outputs[0].text
            token_count = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else len(generated_text.split())
            
            print(f"vLLM Generated: '{generated_text}'")
            print(f"📊 Tokens generated: ~{token_count}")
            print(f"⏱️ Generation time: {end_time - start_time:.2f} seconds")
            if token_count > 0:
                print(f"🚀 Speed: ~{token_count / (end_time - start_time):.1f} tokens/second")
        
        print("✓ vLLM Hello World completed successfully!")
        return True, end_time - start_time, token_count
        
    except Exception as e:
        print(f"✗ Error during vLLM Hello World: {e}")
        return False, 0, 0
    
    finally:
        if llm is not None:
            print("🧹 Cleaning up vLLM resources...")
            try:
                del llm
                llm = None
                
                for i in range(3):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    time.sleep(0.5)
                
                check_gpu_memory()
                print("✓ vLLM resources cleaned up")
                
            except Exception as e:
                print(f"⚠ Warning during vLLM cleanup: {e}")
                clear_gpu_memory()

def run_comparison():
    """Run both vLLM and SGLang examples with simple timing"""
    print("🔥 Running vLLM + SGLang Comparison (SIMPLIFIED TIMING)")
    print("=" * 50)
    print(f"Shared prompt: '{SHARED_CONFIG['prompt']}'")
    print(f"Model: {SHARED_CONFIG['model_name']}")
    print(f"Max tokens: {SHARED_CONFIG['max_tokens']}")
    print("=" * 50)
    
    # Run vLLM
    print("\n🚀 Running vLLM...")
    vllm_success, vllm_time, vllm_tokens = vllm_hello_world()
    
    time.sleep(2)
    
    # Try direct approach first
    print("\n🚀 Running SGLang with direct approach...")
    sglang_success, sglang_time, sglang_tokens = sglang_hello_world_direct()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 COMPARISON SUMMARY")
    print("=" * 50)
    if vllm_success and sglang_success:
        print(f"vLLM: {vllm_time:.2f}s, ~{vllm_tokens} tokens, {vllm_tokens/vllm_time:.1f} tok/s")
        print(f"SGLang: {sglang_time:.2f}s, ~{sglang_tokens} tokens, {sglang_tokens/sglang_time:.1f} tok/s")
        
        if sglang_time > 0.1:  # Only compare if timing seems valid
            if vllm_time < sglang_time:
                print(f"🥇 vLLM was {sglang_time/vllm_time:.1f}x faster")
            else:
                print(f"🥇 SGLang was {vllm_time/sglang_time:.1f}x faster")
        else:
            print("⚠ SGLang timing still seems invalid")
            print("💡 Run test_sglang_only() to try all timing approaches")

def sglang_hello_world_direct():
    """
    Most direct approach - use runtime's text generation
    """
    print("\n=== SGLang Hello World (DIRECT) ===")
    
    clear_gpu_memory()
    check_gpu_memory()
    
    runtime = None
    
    try:
        from sglang import Runtime
        print("✓ SGLang imported successfully")
        
        model_name = SHARED_CONFIG["model_name"]
        unique_id = str(uuid.uuid4())[:8]
        prompt = f"{SHARED_CONFIG['prompt']} (ID: {unique_id})"
        
        print(f"Starting SGLang runtime with model: {model_name}")
        runtime = Runtime(
            model_path=model_name,
            tp_size=1,
            mem_fraction_static=0.7,
        )
        print("✓ SGLang runtime started successfully!")
        
        print("🚀 Using direct text generation...")
        
        # Most direct timing possible
        start_time = time.time()
        import sglang as sgl
        sgl.set_default_backend(runtime)
        
        # Create a simple function
        @sgl.function
        def generate_text(s):
            s += sgl.user(prompt)
            s += sgl.assistant(sgl.gen("output", 
                max_tokens=SHARED_CONFIG["max_tokens"],
                temperature=SHARED_CONFIG["temperature"],
                top_p=SHARED_CONFIG["top_p"]
            ))
        
        # Run synchronously
        state = generate_text.run()
        outputs = state["output"]                    
        end_time = time.time()
        
        # Extract text from outputs
        if isinstance(outputs, str):
            generated_text = outputs
        elif isinstance(outputs, dict):
            generated_text = outputs.get('text', outputs.get('content', str(outputs)))
        elif isinstance(outputs, list) and len(outputs) > 0:
            output = outputs[0]
            if hasattr(output, 'text'):
                generated_text = output.text
            elif isinstance(output, dict):
                generated_text = output.get('text', str(output))
            else:
                generated_text = str(output)
        else:
            generated_text = str(outputs)
        
        # Remove prompt if it's included
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        elapsed_time = end_time - start_time
        token_count = len(generated_text.split())
        
        print(f"SGLang Generated: '{generated_text}'")
        print(f"📊 Tokens generated: ~{token_count}")
        print(f"⏱️ Generation time: {elapsed_time:.2f} seconds")
        
        if elapsed_time > 0:
            print(f"🚀 Speed: ~{token_count / elapsed_time:.1f} tokens/second")
        
        return True, elapsed_time, token_count
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0
    
    finally:
        if runtime is not None:
            try:
                if hasattr(runtime, 'shutdown'):
                    runtime.shutdown()
                del runtime
                clear_gpu_memory()
            except:
                pass
 
def change_prompt(new_prompt):
    """Helper function to easily change the prompt for both engines"""
    SHARED_CONFIG["prompt"] = new_prompt
    print(f"✓ Prompt changed to: '{new_prompt}'")

def change_tokens(min_tokens=None, max_tokens=None):
    """Helper function to change token limits"""
    if min_tokens is not None:
        SHARED_CONFIG["min_tokens"] = min_tokens
    if max_tokens is not None:
        SHARED_CONFIG["max_tokens"] = max_tokens
    print(f"✓ Token limits updated: min={SHARED_CONFIG['min_tokens']}, max={SHARED_CONFIG['max_tokens']}")

if __name__ == "__main__":
    print("🔧 Available functions:")
    print("  - run_comparison() : Compare vLLM and SGLang")
    print("  - test_sglang_only() : Test all SGLang timing approaches")
    print("  - set_debug(True) : Enable debug output")
    print("  - change_prompt('...') : Change the test prompt")
    print("  - change_tokens(min=X, max=Y) : Change token limits")
    print("\nRunning default comparison...\n")
    
    #run_comparison()