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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.bert.modeling_bert import BertLayer 
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
    dist.destroy_process_group()

def run_fine_tuning():
    compute_device = setup()
    model_name = "bert-large-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer pad_token set to: {tokenizer.pad_token}")
    my_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={BertLayer}, # Updated for BERT models checkpoint to divide for chunks
    )
    cpu_offload = CPUOffload(offload_params=True) 
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    print(f"Initial model device (should be CPU): {next(model.parameters()).device}")

    # --- CRITICAL CHANGE: Enable Activation Checkpointing ---
    # This reduces memory usage by recomputing activations during the backward pass
    # instead of storing them. This is very effective for large models.
    model.gradient_checkpointing_enable()
    print("Activation Checkpointing enabled.")

    # Wrap the model with FSDP.
    # When cpu_offload is enabled, the `device_id` for FSDP should be the compute device (GPU).
    # FSDP's internal mechanisms, along with CPUOffload, will handle moving parameters
    # to/from the CPU as needed for memory efficiency, while computations happen on `compute_device`.
    fsdp_model_args = {
        "module": model,
        "auto_wrap_policy": my_auto_wrap_policy,
        "cpu_offload": cpu_offload,
        "sharding_strategy": ShardingStrategy.NO_SHARD, 
        # Set FSDP's device_id to the actual compute_device (GPU).
        "device_id": compute_device 
    }

    model = FSDP(**fsdp_model_args)
    print(f"FSDP-wrapped model device (should still reflect CPU for parameters, but computations on {compute_device.type}): {next(model.parameters()).device}")


    print("Loading and preparing dataset...")
    raw_datasets = load_dataset("glue", "mrpc")
    
    max_sequence_length = 64
    
    def tokenize_function(examples):

        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length", max_length=max_sequence_length)
        
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=1, shuffle=True, pin_memory=False)
    print(f"DataLoader batch size set to: {train_loader.batch_size}")
    print(f"Max sequence length set to: {max_sequence_length}")
    print("Dataset prepared.")
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    
    print("Starting fine-tuning with FSDP + CPU Offload (parameters on CPU)...")
    model.train()
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(compute_device) for k, v in batch.items()}
            
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
        cpu_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        cpu_model.load_state_dict(full_state_dict)
        save_directory = "./fine-tuned-bert-fsdp-cpu-offload"
        os.makedirs(save_directory, exist_ok=True)
        cpu_model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print(f"Model saved successfully to {save_directory}.")

    cleanup()

if __name__ == '__main__':
    run_fine_tuning()
