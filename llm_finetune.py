import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from functools import partial
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import ShardingStrategy 
from transformers import AutoTokenizer, AutoModelForCausalLM  
from datasets import load_dataset

def setup():
    rank = 0
    world_size = 1
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if torch.cuda.is_available():
        compute_device = torch.device("cuda", rank)
        torch.cuda.set_device(compute_device)
        backend = "nccl"
        print(f"Using CUDA device for computation: {compute_device} with backend: {backend}")
    else:
        compute_device = torch.device("cpu")
        backend = "gloo"
        print(f"Using CPU device for computation with backend: {backend}")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
    return compute_device

def cleanup():
    """Cleans up the distributed environment."""
    dist.destroy_process_group()

def run_fine_tuning():
    compute_device = setup()
    model_name = "Qwen/Qwen1.5-0.5B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer pad_token set to: {tokenizer.pad_token}")
    cpu_offload = CPUOffload(offload_params=True) 
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,trust_remote_code=True)
    print(f"Initial model device (should be CPU): {next(model.parameters()).device}")
    transformer_layer_class = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        transformer_layer_class = model.model.layers[0].__class__
    elif hasattr(model, 'layers'):
        transformer_layer_class = model.layers[0].__class__
    
    if transformer_layer_class:
        print(f"Dynamically detected transformer layer class: {transformer_layer_class.__name__}")
    else:
        raise ValueError("Could not dynamically determine transformer layer class. Please specify manually.")
    my_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_class}, 
    )
    fsdp_model_args = {
        "module": model,
        "auto_wrap_policy": my_auto_wrap_policy,
        "cpu_offload": cpu_offload, 
        "sharding_strategy": ShardingStrategy.FULL_SHARD  , 
        "device_id": compute_device
    }

    model = FSDP(**fsdp_model_args)
    print(f"FSDP-wrapped model device (should still reflect CPU for parameters, but computations on {compute_device.type}): {next(model.parameters()).device}")
    model.gradient_checkpointing_enable()
    print("Activation Checkpointing enabled.")
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
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    print("Starting fine-tuning with FSDP (parameters on CPU, computations on GPU)...")
    model.train()
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(compute_device) for k, v in batch.items()}
            batch["labels"] = batch["input_ids"].clone()

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {i}, Loss: {loss.item():.4f}")

    print("Fine-tuning finished.")
    if dist.get_rank() == 0:
        print("Saving model...")
        full_state_dict = model.state_dict()
        cpu_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        cpu_model.load_state_dict(full_state_dict)
        save_directory = "./fine-tuned-qwen-0.5b-causal-lm-fsdp-cpu-offload"
        os.makedirs(save_directory, exist_ok=True)
        cpu_model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print(f"Model saved successfully to {save_directory}.")
    cleanup()

if __name__ == '__main__':
    run_fine_tuning()
