import os
import gc
import time
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM


class DeepSeekOffloadInference:
    def __init__(self, model_name, max_new_tokens=100, temperature=0.7):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = None
        self.model = None
        self.offload_dir = "./offload_cache"
        os.makedirs(self.offload_dir, exist_ok=True)

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_folder=self.offload_dir,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            max_memory={0: "2GB", "cpu": "8GB"}  # Adjust based on GPU RAM
        )

        self.model.eval()

    def generate_response(self, prompt, do_sample=True, top_p=0.95, top_k=50):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, padding=True)
        # Instead of forcing to self.device, detect from model
        first_param_device = next(self.model.parameters()).device
        inputs = {k: v.to(first_param_device) for k, v in inputs.items()}


        with torch.no_grad():
            start = time.time()
            output_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            duration = time.time() - start

        response = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip(), duration

    def print_memory(self):
        if torch.cuda.is_available():
            print(f"GPU: allocated={torch.cuda.memory_allocated()/1e9:.2f}GB, reserved={torch.cuda.memory_reserved()/1e9:.2f}GB")
        mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
        print(f"CPU RAM: {mem:.2f} GB")


def main():
    model_name = "google/gemma-2-9b-it"

    inference = DeepSeekOffloadInference(model_name)
    inference.load_model()

    prompts = [
        "Explain gradient descent in simple terms.",
        "Write a Python function to generate Fibonacci numbers.",
        "What are the differences between supervised and unsupervised learning?",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        response, t = inference.generate_response(prompt)
        print(f"Response: {response}\nTime: {t:.2f}s")
        inference.print_memory()


if __name__ == "__main__":
    main()
