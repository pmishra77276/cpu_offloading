import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from functools import partial
# import optimum
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from transformers import AutoTokenizer, AutoModelForCausalLM 
from datasets import load_dataset

def run_fine_tuning():
    # --- 1. Model and Tokenizer ---
    model_name = "Qwen/Qwen1.5-0.5B-Chat" # Using Qwen1.5-0.5B-Chat
    # CRITICAL FIX: Add trust_remote_code=True and use_fast=False for Qwen tokenizers
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    # Set pad_token for Qwen tokenizer if it doesn't have one.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer pad_token set to: {tokenizer.pad_token}")

    # --- 2. Load Model and Dynamically Determine Transformer Layer Class ---
    # Load AutoModelForCausalLM onto the CPU initially.
    # CRITICAL FIX: Add trust_remote_code=True and torch_dtype=torch.bfloat16 for memory efficiency
    # Ensure your GPU supports bfloat16 (NVIDIA Ampere architecture or newer).
    # If not, use torch.float16 instead, but be aware of potential precision issues.
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    print(f"Initial model device (should be CPU): {next(model.parameters()).device}")

    # --- Dynamically determine the transformer layer class ---
    transformer_layer_class = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        transformer_layer_class = model.model.layers[0].__class__
    elif hasattr(model, 'layers'): # Some models might have layers directly under model
        transformer_layer_class = model.layers[0].__class__
    
    if transformer_layer_class:
        print(f"Dynamically detected transformer layer class: {transformer_layer_class.__name__}")
    else:
        raise ValueError("Could not dynamically determine transformer layer class. Please specify manually.")

    # --- Enable Activation Checkpointing ---
    # This reduces memory usage by recomputing activations during the backward pass
    # instead of storing them. This is very effective for large models.
    model.gradient_checkpointing_enable()
    print("Activation Checkpointing enabled.")

    # --- 3. Accelerate and FSDP Configuration ---
    # Define FSDP configuration using FullyShardedDataParallelPlugin
    fsdp_plugin = FullyShardedDataParallelPlugin(
        cpu_offload=True, # Enable CPU offloading
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_class},
        ),
        sharding_strategy="NO_SHARD", # For single GPU, NO_SHARD is effectively DDP-like behavior with offload
    )

    # Initialize Accelerator with FSDPPlugin
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    
    # Get the compute device from Accelerator
    compute_device = accelerator.device
    print(f"Accelerator initialized. Computations will run on: {compute_device}")

    # --- 4. Dataset Preparation ---
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

    # --- 5. Prepare Model, Optimizer, and DataLoader with Accelerator ---
    # Accelerate handles moving to device and FSDP wrapping
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    print(f"Model, Optimizer, DataLoader prepared by Accelerator. FSDP-wrapped model device (should reflect CPU for parameters, but computations on {accelerator.device}): {next(model.parameters()).device}")


    # --- 6. Training Loop ---
    num_epochs = 3
    
    print("Starting fine-tuning with Accelerate and FSDP (parameters on CPU, computations on GPU)...")
    model.train() # Set model to training mode
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            # No need to manually move batch to device, Accelerate handles it
            
            # For Causal LM, the labels are typically the input_ids shifted.
            # The AutoModelForCausalLM handles this internally if 'labels' are provided.
            # So, we set labels to input_ids.
            batch["labels"] = batch["input_ids"].clone()

            optimizer.zero_grad() # Clear gradients
            outputs = model(**batch) # Forward pass 
            loss = outputs.loss # Get the loss from model outputs
            
            # CRITICAL CHANGE: Use accelerator.backward() for backward pass
            accelerator.backward(loss) 
            optimizer.step() # Update model parameters
            
            if accelerator.is_main_process and i % 10 == 0:
                # Print loss periodically only on the main process
                print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {i}, Loss: {loss.item():.4f}")

    print("Fine-tuning finished.")

    # --- 7. Saving the Model ---
    # Only save from the main process
    if accelerator.is_main_process:
        print("Saving model...")
        # Use accelerator.save_model for FSDP-aware saving
        # It will unwrap the FSDP model and save the full state dict
        save_directory = "./fine-tuned-qwen-0.5b-causal-lm-accelerate-fsdp-cpu-offload" 
        accelerator.save_model(model, save_directory)
        tokenizer.save_pretrained(save_directory)
        print(f"Model saved successfully to {save_directory}.")

    # Accelerate handles distributed cleanup automatically on exit

if __name__ == '__main__':
    run_fine_tuning()
