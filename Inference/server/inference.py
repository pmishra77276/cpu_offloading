import torch
import os
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
import psutil
from transformers import TextIteratorStreamer
import threading
from fastapi.responses import StreamingResponse
os.environ["FLASH_ATTENTION_FORCE_DISABLED"] = "1"
torch.set_num_threads(6)
torch.set_num_interop_threads(3)


class AccelerateOffloadInference:
    def __init__(self, nvme_path, max_gpu1_memory, max_gpu2_memory, max_cpu_memory,
                 model_name: str, max_new_tokens: int = 100, temperature: float = 0.7):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.nvme_path = nvme_path
        self.max_gpu1_memory = max_gpu1_memory
        self.max_gpu2_memory = max_gpu2_memory
        self.max_cpu_memory = max_cpu_memory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None
        self.text_streamer = None

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
        self.text_streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            max_memory={0: self.max_gpu1_memory, "cpu": self.max_cpu_memory},
            offload_folder=self.nvme_path,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            use_cache=True
        )

        self.model.gradient_checkpointing_enable = True
        self.model.eval()
        print("✅ Model loaded successfully with Accelerate!")
    def generate_stream(self, prompt: str, do_sample=True, top_p=0.95, top_k=50):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer
    def generate_response(self, prompt: str, do_sample: bool = True,
                          top_p: float = 0.95, top_k: int = 50):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        inputs = self.tokenizer(
            prompt,
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
                streamer=self.text_streamer,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=None
            )
            generation_time = time.time() - start_time

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return response.strip(), generation_time

    def print_memory_usage(self):
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
            print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"CPU Memory Usage: {memory_info.rss / 1024 ** 3:.2f} GB")
