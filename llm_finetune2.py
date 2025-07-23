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
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch, StateDictType, FullStateDictConfig 
# Removed CPUOffload from this import as it will no longer be used in FSDP constructor
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Added BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import gc

def setup():
    """Initializes the distributed environment."""
    # Read rank, local_rank, and world_size from environment variables set by torchrun
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"]) # Corrected: Reverted to standard LOCAL_RANK
    world_size = int(os.environ["WORLD_SIZE"])

    # MASTER_ADDR and MASTER_PORT are typically set by torchrun, but ensure defaults
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500') 

    # Set PYTORCH_CUDA_ALLOC_CONF to help with memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
        
    # --- CRITICAL CHANGE: Configure BitsAndBytes for 4-bit quantization ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16, # Using float16 for compute
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4', # NormalFloat4 quantization
    )

    if rank == 0:
        print(f"Rank {rank}: Loading model with 4-bit quantization and preparing for LoRA...")
    
    # Load base model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, # Use bnb_config here
        torch_dtype=None, # Let bitsandbytes handle the dtype for quantized layers
        trust_remote_code=True,
        attn_implementation='eager' # Recommended for training Gemma2 models
    )
    
    # Prepare model for k-bit training (required for LoRA with quantization)
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    if rank == 0:
        print(f"Rank {rank}: Configuring LoRA...")
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,   # Rank of adaptation
        lora_alpha=32,  # LoRA scaling parameter
        lora_dropout=0.1,   # LoRA dropout
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],   # Target modules for Gemma
        bias="none",    # No bias training
        inference_mode=False,   # Training mode
    )
    
    model = get_peft_model(model, lora_config)
    
    if rank == 0:
        print(f"Rank {rank}: LoRA configuration applied. Trainable parameters:")
        model.print_trainable_parameters()
    
    # Identify the transformer layer class for FSDP auto-wrapping
    def find_transformer_layer_class(model):
        """Find the transformer layer class for FSDP wrapping"""
        # Try different possible paths for transformer layers
        # For PEFT models, the base model is usually accessed via model.base_model.model
        possible_paths = [
            lambda m: m.base_model.model.model.layers[0].__class__, # Common for PEFT
            lambda m: m.base_model.layers[0].__class__,
            lambda m: m.model.layers[0].__class__,
        ]
        
        for path_func in possible_paths:
            try:
                return path_func(model)
            except (AttributeError, IndexError):
                continue
        
        # If none of the paths work, print model structure for debugging
        print("Model structure for debugging:")
        print(f"Model type: {type(model)}")
        if hasattr(model, 'base_model'):
            print(f"Base model type: {type(model.base_model)}")
            if hasattr(model.base_model, 'model'):
                print(f"Base model.model type: {type(model.base_model.model)}")
        
        raise AttributeError("Could not find transformer layer class for FSDP wrapping")
    
    transformer_layer_class = find_transformer_layer_class(model)
    
    my_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_class},
    )

    if rank == 0:
        print(f"Rank {rank}: Wrapping model with FSDP on device {compute_device}...")
        
    # --- CRITICAL CHANGE: Removed cpu_offload from FSDP constructor ---
    # FSDP's CPUOffload(offload_params=True) is incompatible with 4-bit quantized base models.
    # The 4-bit quantization itself provides memory savings for the base model.
    # FSDP will still shard the trainable LoRA parameters and optimizer state.
    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        # cpu_offload=cpu_offload, # This line is intentionally removed/commented out
        sharding_strategy=ShardingStrategy.FULL_SHARD, # Shards params, grads, optimizer state
        device_id=compute_device, # Assign FSDP instance to specific GPU for this rank
        use_orig_params=True,   # CRITICAL: This allows non-uniform requires_grad
        # Re-enabling backward_prefetch as it can help performance
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE 
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
    
    # Set pin_memory=False to avoid potential issues
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=1, pin_memory=False, sampler=train_sampler, shuffle=False)

    if rank == 0:
        print("Dataset prepared with DistributedSampler.")
        
    # Only optimize LoRA parameters (automatically handled by PEFT)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    
    if rank == 0:
        print("Starting fine-tuning with LoRA...")
        
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

    # Ensure all processes complete training before attempting to save
    dist.barrier()
    if rank == 0:
        print("Fine-tuning finished. All ranks synchronized.")

    # Save the LoRA adapter only from rank 0
    if rank == 0:
        print(f"Rank {rank}: Saving LoRA adapter...")
        
        save_directory = "./fine-tuned-gemma-2-2b-it-lora-fsdp-2gpu"
        os.makedirs(save_directory, exist_ok=True)
        
        # Save LoRA adapter
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        
        print(f"Rank {rank}: LoRA adapter saved successfully to {save_directory}.")
        print(f"Rank {rank}: To use the fine-tuned model, load the base model and then load the LoRA adapter.")
        
    cleanup()

if __name__ == '__main__':
    run_fine_tuning()
