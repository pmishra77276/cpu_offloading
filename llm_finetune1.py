import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch, StateDictType, FullStateDictConfig, CPUOffload
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import gc
def setup():
    """Initializes the distributed environment."""
    # Read rank, local_rank, and world_size from environment variables set by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # MASTER_ADDR and MASTER_PORT are typically set by torchrun, but ensure defaults
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500') 

    print(f"Rank {rank}/{world_size} (Local Rank {local_rank}): Initializing process group with MASTER_ADDR={os.environ['MASTER_ADDR']} and MASTER_PORT={os.environ['MASTER_PORT']}...")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank}/{world_size} (Local Rank {local_rank}): Process group initialized!")

    # Set the device for the current process based on local_rank
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        compute_device = torch.device("cuda", local_rank)
        print(f"Rank {rank}: Using CUDA device {local_rank}")
    else:
        compute_device = torch.device("cpu")
        print(f"Rank {rank}: CUDA not available, using CPU.")

    return compute_device, rank, world_size

def cleanup():
    """Cleans up the distributed environment."""
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: Destroying process group.")
        dist.destroy_process_group()

def run_fine_tuning():
    compute_device, rank, world_size = setup()
    
    model_name = "google/gemma-2-2b-it" 
    
    if rank == 0:
        print(f"Rank {rank}: Downloading model and tokenizer for {model_name}...")
    
    # Use a barrier to ensure all processes start downloading/loading at roughly the same time
    dist.barrier() 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if rank == 0:
        print(f"Rank {rank}: Tokenizer pad_token set to: {tokenizer.pad_token}")
        
    # Configure CPU offload for FSDP parameters
    cpu_offload = CPUOffload(offload_params=True)
    
    if rank == 0:
        print(f"Rank {rank}: Loading model from pre-trained (will use CPU offload for parameters)...")
    
    # Load model on CPU first if offloading params, then FSDP moves parts to GPU as needed
    # Using torch.float16 for broader compatibility
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, trust_remote_code=True)
    
    # Identify the transformer layer class for FSDP auto-wrapping
    # For Gemma, it's typically GemmaDecoderLayer
    transformer_layer_class = model.model.layers[0].__class__
    
    my_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_class},
    )

    if rank == 0:
        print(f"Rank {rank}: Wrapping model with FSDP on device {compute_device}...")
        
    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        cpu_offload=cpu_offload,
        sharding_strategy=ShardingStrategy.FULL_SHARD, # Shards params, grads, optimizer state
        device_id=compute_device, # Assign FSDP instance to specific GPU for this rank
        # Removed backward_prefetch to simplify for now, can add back if stable
        # backward_prefetch=BackwardPrefetch.BACKWARD_PRE 
    )
    
    model.gradient_checkpointing_enable() # Enable gradient checkpointing to save VRAM

    if rank == 0:
        print(f"Rank {rank}: Loading dataset 'glue', 'mrpc'...")
    
    raw_datasets = load_dataset("glue", "mrpc")
    max_sequence_length = 64

    def tokenize_function(examples):
        text = [f"{s1} {s2}" for s1, s2 in zip(examples["sentence1"], examples["sentence2"])]
        return tokenizer(text, truncation=True, padding="max_length", max_length=max_sequence_length)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["sentence1", "sentence2", "idx", "label"])
    tokenized_datasets.set_format("torch")

    # Use DistributedSampler to correctly partition data across GPUs
    train_sampler = DistributedSampler(tokenized_datasets["train"], rank=rank, num_replicas=world_size, shuffle=True)
    
    # *** CRITICAL CHANGE: Set pin_memory=False ***
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=1, pin_memory=False, sampler=train_sampler, shuffle=False)

    if rank == 0:
        print("Dataset prepared with DistributedSampler.")
        
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    
    if rank == 0:
        print("Starting fine-tuning...")
        
    for epoch in range(num_epochs):
        # Set the epoch for the sampler to ensure proper shuffling across epochs
        train_sampler.set_epoch(epoch)
        model.train()
        for i, batch in enumerate(train_loader):
            # Clear CUDA cache and collect garbage at the beginning of each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Move batch to compute device (GPU for this rank)
            batch = {k: v.to(compute_device) for k, v in batch.items()}
            batch["labels"] = batch["input_ids"].clone()

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            
            # Print loss before backward pass to help debug hangs in backward
            if rank == 0 and i % 10 == 0:
                print(f"Rank {rank}, Epoch: {epoch+1}/{num_epochs}, Batch: {i}, Loss (pre-backward): {loss.item():.4f}")

            loss.backward()
            optimizer.step()
            
            # Print loss after optimizer step
            if rank == 0 and i % 10 == 0:
                print(f"Rank {rank}, Epoch: {epoch+1}/{num_epochs}, Batch: {i}, Loss (post-step): {loss.item():.4f}")
                # You might also want to print memory usage here for debugging
                # print(f"Rank {rank}, GPU {compute_device.index} memory: {torch.cuda.memory_allocated(compute_device.index) / (1024**3):.2f} GB")


    # Ensure all processes complete training before attempting to save
    dist.barrier()
    if rank == 0:
        print("Fine-tuning finished. All ranks synchronized.")

    # Save the full model only from rank 0
    if rank == 0:
        print(f"Rank {rank}: Saving full model state_dict...")
        
        # Configure saving to offload to CPU and only save from rank 0
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            # This gathers the sharded state dict parts from all ranks and offloads to CPU
            cpu_state_dict = model.state_dict()

        # Load an uninitialized model and then load the gathered state dict
        # This ensures the saved model is a standard HuggingFace model, not an FSDP wrapped one
        print(f"Rank {rank}: Loading initial model for state_dict transfer...")
        cpu_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        cpu_model.load_state_dict(cpu_state_dict)
        
        save_directory = "./fine-tuned-gemma-2-2b-it-fsdp-2gpu" # Changed save directory name for clarity
        os.makedirs(save_directory, exist_ok=True)
        
        print(f"Rank {rank}: Saving model to {save_directory}...")
        cpu_model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print(f"Rank {rank}: Model saved successfully to {save_directory}.")
        
    cleanup()

if __name__ == '__main__':
    run_fine_tuning()
