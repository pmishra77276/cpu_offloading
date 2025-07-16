import torch
import os
import time
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import psutil

# Set thread settings globally (good practice)
torch.set_num_threads(12)
torch.set_num_interop_threads(4)

class SmartCPUOffloadInference:
    # ... (Your existing manual CPU offloading class, no changes needed here) ...
    def __init__(self, model_name: str, max_new_tokens: int = 100, temperature: float = 0.7):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")
        
        self.tokenizer = None
        self.model = None
        self.transformer_layers = None
        self.embeddings = None
        self.layer_norm = None
        self.lm_head = None
        
        # Track which layers are currently on GPU
        self.gpu_layers = set()
        
    def load_model(self):
        """Load model entirely on CPU first, then setup offloading"""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model entirely on CPU
        print("Loading model on CPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=None,  # Don't use device_map, load on CPU
            low_cpu_mem_usage=True
        )
        
        # Move entire model to CPU explicitly
        self.model = self.model.cpu()
        self.model.eval()
        
        # Setup layer references for efficient offloading
        self.setup_layer_references()
        
        # Keep only embeddings and head on GPU permanently
        self.setup_permanent_gpu_layers()
        
        print("Model loaded successfully with smart CPU offloading!")
    
    def setup_layer_references(self):
        """Get references to different parts of the model"""
        print("Setting up layer references...")
        
        # Detect model architecture and get layer references
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama/Mistral/Qwen style
            self.transformer_layers = self.model.model.layers
            self.embeddings = self.model.model.embed_tokens
            self.layer_norm = getattr(self.model.model, 'norm', None)
            self.lm_head = self.model.lm_head
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT style
            self.transformer_layers = self.model.transformer.h
            self.embeddings = self.model.transformer.wte
            self.layer_norm = getattr(self.model.transformer, 'ln_f', None)
            self.lm_head = self.model.lm_head
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
            # OPT style
            self.transformer_layers = self.model.model.decoder.layers
            self.embeddings = self.model.model.decoder.embed_tokens
            self.layer_norm = getattr(self.model.model.decoder, 'layer_norm', None)
            self.lm_head = self.model.lm_head
        else:
            raise ValueError("Unsupported model architecture")
        
        print(f"Found {len(self.transformer_layers)} transformer layers")
    
    def setup_permanent_gpu_layers(self):
        """Move embeddings and head to GPU permanently for efficiency"""
        if torch.cuda.is_available():
            print("Moving embeddings and LM head to GPU...")
            self.embeddings.to(self.device)
            if self.layer_norm:
                self.layer_norm.to(self.device)
            self.lm_head.to(self.device)
    
    def move_layer_to_gpu(self, layer_idx):
        """Move a specific layer to GPU"""
        if layer_idx not in self.gpu_layers:
            self.transformer_layers[layer_idx].to(self.device)
            self.gpu_layers.add(layer_idx)
    
    def move_layer_to_cpu(self, layer_idx):
        """Move a specific layer to CPU"""
        if layer_idx in self.gpu_layers:
            self.transformer_layers[layer_idx].to(self.cpu_device)
            self.gpu_layers.remove(layer_idx)
    
    def clear_gpu_layers(self):
        """Move all transformer layers back to CPU"""
        for layer_idx in list(self.gpu_layers):
            self.move_layer_to_cpu(layer_idx)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def forward_with_offloading(self, input_ids, attention_mask, **kwargs):
        """Custom forward pass with layer-by-layer GPU offloading"""
        # Start with embeddings (already on GPU)
        hidden_states = self.embeddings(input_ids)
        
        # Process each transformer layer
        for layer_idx, layer in enumerate(self.transformer_layers):
            # Move current layer to GPU
            self.move_layer_to_gpu(layer_idx)
            
            # Move hidden states to GPU
            hidden_states = hidden_states.to(self.device)
            attention_mask_gpu = attention_mask.to(self.device)
            
            # Forward pass through current layer
            if hasattr(layer, '__call__'):
                # For most transformer layers
                layer_output = layer(hidden_states, attention_mask=attention_mask_gpu)
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]
                else:
                    hidden_states = layer_output
            else:
                # Fallback
                hidden_states = layer(hidden_states)
            
            # Move layer back to CPU immediately after use
            self.move_layer_to_cpu(layer_idx)
            
            # Keep hidden states on GPU for next layer
            # (will be moved in next iteration)
            
        # Apply final layer norm (already on GPU)
        if self.layer_norm:
            hidden_states = hidden_states.to(self.device)
            hidden_states = self.layer_norm(hidden_states)
        
        # Apply LM head (already on GPU)
        hidden_states = hidden_states.to(self.device)
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def generate_response(self, prompt: str, do_sample: bool = True, top_p: float = 0.95, top_k: int = 50):
        """Generate response with smart CPU offloading"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Clear GPU cache
        self.clear_gpu_layers()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )
        
        # Move inputs to CPU initially
        input_ids = inputs['input_ids'].to(self.cpu_device)
        attention_mask = inputs['attention_mask'].to(self.cpu_device)
        
        # Generate response with custom generation loop
        with torch.no_grad():
            start_time = time.time()
            
            generated_tokens = input_ids.clone()
            
            for step in range(self.max_new_tokens):
                # Forward pass with offloading
                logits = self.forward_with_offloading(
                    generated_tokens, 
                    attention_mask
                )
                
                # Get next token
                next_token_logits = logits[:, -1, :] / self.temperature
                
                if do_sample:
                    # Apply top-k and top-p filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits[next_token_logits < top_k_logits[:, -1:]] = -float('inf')
                    
                    if top_p > 0.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float('inf')
                    
                    probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Append token
                generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device)
                ], dim=-1)
                
                # Clear GPU cache after each step
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            generation_time = time.time() - start_time
        
        # Final cleanup
        self.clear_gpu_layers()
        
        # Decode response
        response = self.tokenizer.decode(
            generated_tokens[0][input_ids.shape[1]:],
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
        print(f"Layers currently on GPU: {len(self.gpu_layers)}")

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
        """Load model using Accelerate for better offloading"""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with smart device mapping
        # This will automatically determine optimal layer placement
        print("Loading model with Accelerate device mapping...")
        
        # Calculate available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            # Use 80% of GPU memory as a target, ensure at least 1GB
            max_gpu_memory = f"8GB" 
        else:
            max_gpu_memory = "0GB" # No GPU memory if CUDA not available
        
        # **** MODIFIED SECTION FOR NVMe OFFLOADING ****
        # Reduced CPU max_memory to force more layers to disk
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            # For Gemma 2, bfloat16 is often preferred if your GPU supports it (Ampere or newer)
            torch_dtype=torch.bfloat16 , 
            trust_remote_code=True,
            device_map="auto",
            # Define max memory for GPU and CPU. Anything beyond this will go to `offload_folder`.
            # Setting CPU limit lower than total CPU offloadable model size to force disk offload.
            max_memory={0: max_gpu_memory, "cpu": "0GB"}, # REDUCED CPU MEMORY HERE
            offload_folder="/offload_nvm",  # THIS IS WHERE YOUR NVMe IS MOUNTED
            offload_state_dict=True,         # Crucial: Enables disk-based sharded weight loading during initial load
            low_cpu_mem_usage=True           # Helps prevent full model load to CPU initially
        )
        # **** END MODIFIED SECTION ****
        
        self.model.eval()
        print("Model loaded successfully with Accelerate!")
        
        # Print device mapping
        if hasattr(self.model, 'hf_device_map'):
            print("Device mapping:")
            for module, device in self.model.hf_device_map.items():
                print(f"  {module}: {device}")
    
    def generate_response(self, prompt: str, do_sample: bool = True, top_p: float = 0.95, top_k: int = 50):
        """Generate response using Accelerate's automatic offloading"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Clear GPU cache (Accelerate handles layer movement, but clearing cache can still help)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        )
        
        # **** MODIFIED SECTION FOR INPUT DEVICE PLACEMENT ****
        # Move input_ids and attention_mask to the model's primary device (GPU if available)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        # **** END MODIFIED SECTION ****

        # Generate response (Accelerate handles device placement automatically)
        with torch.no_grad():
            start_time = time.time()
            
            outputs = self.model.generate(
                input_ids, # Use the moved input_ids
                attention_mask=attention_mask, # Use the moved attention_mask
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True # Good for inference, combined with auto device mapping
            )
            
            generation_time = time.time() - start_time
        
        # Clear GPU cache after generation (good practice)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Decode response
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
    # Choose approach
    USE_ACCELERATE = True   # Set to False for manual smart offloading
    
    model_name = "google/gemma-2-9b-it" # Changed to 9B model
    
    if USE_ACCELERATE:
        print("Using Accelerate library for smart offloading (including NVMe)...")
        inference = AccelerateOffloadInference(
            model_name=model_name,
            max_new_tokens=100,
            temperature=0.7
        )
    else:
        print("Using manual smart CPU offloading...")
        inference = SmartCPUOffloadInference(
            model_name=model_name,
            max_new_tokens=100,
            temperature=0.7
        )
    
    try:
        # Load model
        inference.load_model()
        
        # Print memory usage after loading
        print("\nMemory usage after loading:")
        inference.print_memory_usage()
        
        # Example prompts
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
            
            # Print memory usage after each generation
            print("\nMemory usage after generation:")
            inference.print_memory_usage()
            print("-" * 50)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure the offload directory exists and is writable
    # It's good practice to create this directory explicitly.
    # If your NVMe is mounted at /, then /offload_nvm will be on NVMe.
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
