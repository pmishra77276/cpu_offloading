import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist 
from torch.utils.data import DataLoader

from functools import partial # Still needed for some partial applications, but not FSDP policy directly now

# Removed FSDP imports as we are doing pure GPU training
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# from torch.distributed.fsdp import CPUOffload
# from torch.distributed.fsdp import ShardingStrategy 

from transformers import AutoTokenizer, AutoModelForCausalLM 
from datasets import load_dataset

def setup():
    """
    Initializes the distributed environment.
    It checks for CUDA availability and sets up the process group accordingly.
    Returns the torch.device object to be used for computation (GPU if available, else CPU).
    """
    rank = 0  # For single process, rank is always 0
    world_size = 1 # For single process, world size is always 1

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # Use a unique port

    if torch.cuda.is_available():
        # If CUDA is available, use GPU and NCCCL backend
        compute_device = torch.device("cuda", rank)
        torch.cuda.set_device(compute_device)
        backend = "nccl"
        print(f"Using CUDA device for computation: {compute_device} with backend: {backend}")
    else:
        # Fallback to CPU and Gloo backend if CUDA is not available
        compute_device = torch.device("cpu")
        backend = "gloo"
        print(f"Using CPU device for computation with backend: {backend}")

    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    return compute_device

def cleanup():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()

def run_fine_tuning():
    # Manual distributed setup
    compute_device = setup()

    # --- 1. Model and Tokenizer ---
    model_name = "Qwen/Qwen1.5-0.5B-Chat" # Using Qwen-0.5B-Chat
    # Add trust_remote_code=True and use_fast=False for Qwen tokenizers
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    # Set pad_token for Qwen tokenizer if it doesn't have one.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer pad_token set to: {tokenizer.pad_token}")

    # --- 2. Load Model Directly to GPU ---
    # Load AutoModelForCausalLM and move it directly to the GPU.
    # Use torch_dtype=torch.bfloat16 for memory efficiency.
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.to(compute_device) # CRITICAL CHANGE: Move model to GPU directly
    print(f"Model loaded directly to device: {next(model.parameters()).device}")

    # --- Enable Activation Checkpointing ---
    # This reduces memory usage by recomputing activations during the backward pass
    # instead of storing them. This is very effective for large models.
    model.gradient_checkpointing_enable()
    print("Activation Checkpointing enabled.")

    # --- FSDP-related policy and wrapping are removed for pure GPU training ---
    # my_auto_wrap_policy = partial(
    #     transformer_auto_wrap_policy,
    #     transformer_layer_cls={transformer_layer_class}, 
    # )
    # cpu_offload = CPUOffload(offload_params=True) 
    # fsdp_model_args = { ... }
    # model = FSDP(**fsdp_model_args)

    # --- 3. Dataset Preparation ---
    print("Loading and preparing dataset...")
    raw_datasets = load_dataset("glue", "mrpc")
    
    max_sequence_length = 64 
    
    def tokenize_function(examples):
        text = [f"{s1} {s2} Label: {label}" for s1, s2, label in zip(examples["sentence1"], examples["sentence2"], examples["label"])]
        return tokenizer(text, truncation=True, padding="max_length", max_length=max_sequence_length)
        
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx", "label"])
    tokenized_datasets.set_format("torch") 

    train_loader = DataLoader(tokenized_datasets["train"], batch_size=1, shuffle=True, pin_memory=False)
    print(f"DataLoader batch size set to: {train_loader.batch_size}")
    print(f"Max sequence length set to: {max_sequence_length}")
    print("Dataset prepared.")

    # --- 4. Optimizer and Training Loop ---
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    
    print("Starting pure GPU fine-tuning...")
    model.train() # Set model to training mode
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            # Manually move batch to compute device
            batch = {k: v.to(compute_device) for k, v in batch.items()}
            
            # For Causal LM, the labels are typically the input_ids shifted.
            # The AutoModelForCausalLM handles this internally if 'labels' are provided.
            # So, we set labels to input_ids.
            batch["labels"] = batch["input_ids"].clone()

            optimizer.zero_grad() # Clear gradients
            outputs = model(**batch) # Forward pass 
            loss = outputs.loss # Get the loss from model outputs
            
            loss.backward() # Standard backward pass
            optimizer.step() # Update model parameters
            
            # Print loss periodically only on rank 0
            if dist.get_rank() == 0 and i % 10 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {i}, Loss: {loss.item():.4f}")

    print("Fine-tuning finished.")

    # --- 5. Saving the Model ---
    # Only save from rank 0
    if dist.get_rank() == 0:
        print("Saving model...")
        # Save the model directly. It's already on GPU, but save_pretrained handles moving to CPU if needed.
        save_directory = "./fine-tuned-qwen-0.5b-causal-lm-pure-gpu" # Updated save directory name
        os.makedirs(save_directory, exist_ok=True) # Ensure directory exists
        model.save_pretrained(save_directory) # Save the model directly
        tokenizer.save_pretrained(save_directory)
        print(f"Model saved successfully to {save_directory}.")

    # Manual cleanup
    cleanup() 

if __name__ == '__main__':
    run_fine_tuning()
