import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader

# Import 'partial' to correctly create the wrap policy
from functools import partial

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import ShardingStrategy 

# Import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM 
# No longer need to import specific transformer layer class here, it will be detected dynamically
# from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer 
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
    # Setup distributed environment and get the device where computations will occur
    compute_device = setup()

    # --- 1. Model and Tokenizer ---
    model_name = "Qwen/Qwen1.5-0.5B-Chat" # Changed to Qwen-0.5B-Chat
    # CRITICAL FIX: Add trust_remote_code=True and use_fast=False for Qwen tokenizers
    # use_fast=False bypasses issues with the Rust-based fast tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    # Set pad_token for Qwen tokenizer if it doesn't have one.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer pad_token set to: {tokenizer.pad_token}")

    # --- 2. Load Model and Dynamically Determine Transformer Layer Class ---
    # CPUOffload(offload_params=True) means that the model parameters will primarily
    # reside in CPU memory and be moved to GPU only when needed for computation.
    cpu_offload = CPUOffload(offload_params=True) 
    
    # Load AutoModelForCausalLM onto the CPU initially.
    # CRITICAL FIX: Add trust_remote_code=True for Qwen models
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,trust_remote_code=True).to('cuda')
    print(f"Initial model device (should be CPU): {next(model.parameters()).device}")

    # --- Dynamically determine the transformer layer class ---
    # This makes the auto_wrap_policy general for different transformer models.
    # We assume the main transformer layers are within model.model.layers or model.layers
    # and we take the class of the first layer.
    transformer_layer_class = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        transformer_layer_class = model.model.layers[0].__class__
    elif hasattr(model, 'layers'): # Some models might have layers directly under model
        transformer_layer_class = model.layers[0].__class__
    
    if transformer_layer_class:
        print(f"Dynamically detected transformer layer class: {transformer_layer_class.__name__}")
    else:
        # Fallback if detection fails, though it should work for most HuggingFace models
        # You might need to add specific fallback logic for unusual architectures
        raise ValueError("Could not dynamically determine transformer layer class. Please specify manually.")

    # --- Create the FSDP Auto Wrap Policy with the dynamically detected class ---
    my_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_class}, 
    )

    # --- 3. FSDP Configuration with CPU Offloading ---
    # When cpu_offload is enabled, FSDP will manage parameters on CPU.
    # Re-added `cpu_offload` and set `device_id` to `compute_device`.
    fsdp_model_args = {
        "module": model,
        "auto_wrap_policy": my_auto_wrap_policy,
        "cpu_offload": cpu_offload, 
        "sharding_strategy": ShardingStrategy.FULL_SHARD  , 
        "device_id": compute_device # FSDP will use this for computations
    }

    # model = FSDP(**fsdp_model_args)
    model=torch.compile(model)
    # After FSDP wrapping with CPUOffload, the model parameters should still reflect CPU,
    # but computations will occur on the specified compute_device (GPU).
    print(f"FSDP-wrapped model device (should still reflect CPU for parameters, but computations on {compute_device.type}): {next(model.parameters()).device}")

    # --- Enable Activation Checkpointing ---
    # This reduces memory usage by recomputing activations during the backward pass
    # instead of storing them. This is very effective for large models.
    model.gradient_checkpointing_enable()
    print("Activation Checkpointing enabled.")

    # --- 4. Dataset Preparation ---
    print("Loading and preparing dataset...")
    # Using 'glue', 'mrpc' which is a relatively small dataset suitable for fine-tuning.
    raw_datasets = load_dataset("glue", "mrpc")
    
    # Reduce max_length to save GPU memory for activations and gradients.
    max_sequence_length = 64 
    
    def tokenize_function(examples):
        # For causal LM, we concatenate the sentences and the label into a single sequence.
        # The model will then learn to predict the next token in this combined sequence.
        # We convert the label (0 or 1) to a string for concatenation.
        text = [f"{s1} {s2} Label: {label}" for s1, s2, label in zip(examples["sentence1"], examples["sentence2"], examples["label"])]
        return tokenizer(text, truncation=True, padding="max_length", max_length=max_sequence_length)
        
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    # Remove original text columns and index.
    # For causal LM, the 'labels' for loss calculation will be derived from 'input_ids'.
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx", "label"])
    tokenized_datasets.set_format("torch") # Set format to PyTorch tensors

    # Create DataLoader for the training set
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=1, shuffle=True, pin_memory=False)
    print(f"DataLoader batch size set to: {train_loader.batch_size}")
    print(f"Max sequence length set to: {max_sequence_length}")
    print("Dataset prepared.")

    # --- 5. Optimizer and Training Loop ---
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    
    print("Starting fine-tuning with FSDP (parameters on CPU, computations on GPU)...")
    model.train() # Set model to training mode
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            # Move each tensor in the batch to the correct computation device (GPU if available)
            batch = {k: v.to(compute_device) for k, v in batch.items()}
            
            # For Causal LM, the labels are typically the input_ids shifted.
            # The AutoModelForCausalLM handles this internally if 'labels' are provided.
            # So, we set labels to input_ids.
            batch["labels"] = batch["input_ids"].clone()

            optimizer.zero_grad() # Clear gradients
            outputs = model(**batch) # Forward pass (FSDP moves necessary params to GPU here)
            loss = outputs.loss # Get the loss from model outputs
            loss.backward() # Backward pass to compute gradients
            optimizer.step() # Update model parameters
            
            if i % 10 == 0:
                # Print loss periodically
                print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {i}, Loss: {loss.item():.4f}")

    print("Fine-tuning finished.")

    # --- 6. Saving the Model ---
    # Only save from rank 0 in a distributed setting
    if dist.get_rank() == 0:
        print("Saving model...")
        # When calling state_dict() on an FSDP-wrapped model, it automatically
        # gathers the full state dictionary from all shards (even if NO_SHARD here)
        # and makes it available on the rank 0 process.
        full_state_dict = model.state_dict()
        
        # Load AutoModelForCausalLM for saving.
        # CRITICAL FIX: Add trust_remote_code=True for Qwen models
        cpu_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        cpu_model.load_state_dict(full_state_dict)
        
        # Save the model and tokenizer
        save_directory = "./fine-tuned-qwen-0.5b-causal-lm-fsdp-cpu-offload" # Updated save directory name
        os.makedirs(save_directory, exist_ok=True) # Ensure directory exists
        cpu_model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print(f"Model saved successfully to {save_directory}.")

    cleanup() # Clean up the distributed environment

if __name__ == '__main__':
    run_fine_tuning()
