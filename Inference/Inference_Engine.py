import torch
import os
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import psutil
import argparse
import os
os.environ["FLASH_ATTENTION_FORCE_DISABLED"] = "1"



# Set thread settings globally (good practice)
torch.set_num_threads(12)
torch.set_num_interop_threads(4)
# Alternative: Using Accelerate library for more efficient offloading
class AccelerateOffloadInference:
    def __init__(self, model_name: str, max_new_tokens: int = 100, temperature: float = 0.7):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
  
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Loading model with Accelerate device mapping...")
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print("Total Memory",torch.cuda.get_device_properties(0).total_memory)
            max_gpu_memory = f"7GB" 
        else:
            max_gpu_memory = "0GB"
            
        bnb_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            # torch_dtype=torch.bfloat16 , 
            trust_remote_code=True,
            device_map="auto",
            max_memory={0: "0GB" , 1: "7GB","cpu": "0GB"},
            offload_folder="/offload_nvm",
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            attn_implementation="eager",
            # attn_implementation="eager"
        )
        self.model.gradient_checkpointing_enable=True
        self.model.eval()
        print("Model loaded successfully with Accelerate!")
        if hasattr(self.model, 'hf_device_map'):
            print("Device mapping:")
            for module, device in self.model.hf_device_map.items():
                print(f"  {module}: {device}")
    
    def generate_response(self, prompt: str, do_sample: bool = True, top_p: float = 0.95, top_k: int = 50):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        message=[
            {"role":"system",
             "content":"You are a helpful assistant help user with their query"},
            {
                "role":"user",
                "content":prompt
            }
        ]
        inputs = self.tokenizer(
            self.tokenizer.apply_chat_template(message,tokenize=False),
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
            
            generation_time = time.time() - start_time
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response.strip(), generation_time
    
    def print_memory_usage(self):
        """Print current memory usage"""
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"CPU Memory Usage: {memory_info.rss / 1024**3:.2f} GB")

def main():
    USE_ACCELERATE = True
    parser = argparse.ArgumentParser(description="Run Llama 3.1 8B inference with Llama.cpp offloading.")
    
    
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    if USE_ACCELERATE:
        print("Using Accelerate library for smart offloading (including NVMe)...")
        inference = AccelerateOffloadInference(
            model_name=model_name,
            max_new_tokens=100,
            temperature=0.7
        )
    try:
        inference.load_model()
        print("\nMemory usage after loading:")
        inference.print_memory_usage()
        prompts = [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "Write a short story about a robot.",
        ]
        
        print("\n" + "="*60)
        print("Starting inference with smart offloading...")
        print("="*60)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\nPrompt {i}: {prompt}")
            print("-" * 50)
            
            response, gen_time = inference.generate_response(prompt)
            
            print(f"Response: {response}")
            print(f"Generation time: {gen_time:.2f} seconds")
            print("\nMemory usage after generation:")
            inference.print_memory_usage()
            print("-" * 50)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    offload_path = "/offload_nvm"
    if not os.path.exists(offload_path):
        try:
            os.makedirs(offload_path)
            print(f"Created NVMe offload directory: {offload_path}")
        except OSError as e:
            print(f"Warning: Could not create offload directory {offload_path}. Make sure it's writable: {e}")
    else:
        print(f"NVMe offload directory already exists: {offload_path}")

    main()