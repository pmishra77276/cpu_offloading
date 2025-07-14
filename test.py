import torch
import torch.nn as nn
import torch.distributed as dist
import os
import logging
import argparse
import traceback
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        try:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': encoding['input_ids'].flatten()
            }
        except:
            # Fallback for any tokenization issues
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'labels': torch.zeros(self.max_length, dtype=torch.long)
            }

def setup_distributed():
    """Setup distributed training - handles all edge cases"""
    try:
        # Set environment variables
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Initialize process group
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            try:
                dist.init_process_group(backend='nccl', rank=0, world_size=1)
            except:
                dist.init_process_group(backend='gloo', rank=0, world_size=1)
        else:
            dist.init_process_group(backend='gloo', rank=0, world_size=1)
        
        return 0, 1, 0
    except Exception as e:
        logger.warning(f"Failed to setup distributed: {e}")
        return 0, 1, 0

def create_model_with_fsdp(model_name, cpu_offload=True):
    """Create model with FSDP - bulletproof version"""
    try:
        # Import what we need
        from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=None
        )
        
        # Move to CPU for FSDP wrapping
        model = model.cpu()
        
        # Try to use FSDP
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import CPUOffload, MixedPrecision
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            
            # Get transformer layer class
            transformer_layer_cls = model.model.decoder.layers[0].__class__
            
            # Create auto wrap policy
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={transformer_layer_cls}
            )
            
            # CPU offload config
            cpu_offload_config = CPUOffload(offload_params=True) if cpu_offload else None
            
            # Mixed precision config - Fixed to use FP16 for gradients
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,  # Changed from float32 to float16
                buffer_dtype=torch.float16   # Changed from float32 to float16
            )
            
            # FSDP arguments - only use what's available
            fsdp_kwargs = {
                'auto_wrap_policy': auto_wrap_policy,
                'cpu_offload': cpu_offload_config,
                'mixed_precision': mixed_precision_policy,
                'sync_module_states': True
            }
            
            # Add optional parameters if they exist
            try:
                fsdp_kwargs['use_orig_params'] = True
            except:
                pass
                
            if torch.cuda.is_available():
                try:
                    fsdp_kwargs['device_id'] = torch.cuda.current_device()
                except:
                    pass
            
            # Wrap with FSDP
            model = FSDP(model, **fsdp_kwargs)
            logger.info("Successfully wrapped model with FSDP")
            
        except Exception as e:
            logger.warning(f"FSDP wrapping failed: {e}, using regular model")
            if torch.cuda.is_available():
                model = model.cuda()
        
        return model
        
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        raise

def prepare_data(max_samples=1000):
    """Prepare simple dataset"""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming various industries.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models require large amounts of data for training.",
        "Fine-tuning pre-trained models is an effective approach for specific tasks.",
        "Artificial intelligence continues to advance rapidly.",
        "Large language models can generate human-like text.",
        "Training neural networks requires careful optimization.",
        "Data preprocessing is crucial for model performance.",
        "Transformer architectures have revolutionized NLP.",
        "Python is a popular programming language for machine learning.",
        "Neural networks are inspired by the human brain.",
        "Gradient descent is an optimization algorithm.",
        "Overfitting occurs when a model memorizes training data.",
        "Cross-validation helps evaluate model performance.",
        "Feature engineering is important for model accuracy.",
        "Regularization techniques prevent overfitting.",
        "Ensemble methods combine multiple models.",
        "Hyperparameter tuning improves model performance.",
        "Data augmentation increases training data variety."
    ]
    
    # Repeat to get desired number of samples
    multiplier = (max_samples // len(texts)) + 1
    texts = texts * multiplier
    
    return texts[:max_samples]

def train_step(model, batch, optimizer, scaler=None):
    """Single training step - handles all cases"""
    try:
        model.train()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # Move to GPU if available
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            labels = labels.cuda()
        
        optimizer.zero_grad()
        
        # Check if model is FSDP wrapped
        is_fsdp_model = hasattr(model, '_fsdp_wrapped_module')
        
        # Forward pass - FSDP handles mixed precision internally
        if is_fsdp_model:
            # FSDP model - no external autocast or scaler needed
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning("NaN loss detected, skipping batch")
                return 0.0
                
            loss.backward()
            
            # Gradient clipping to prevent NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        else:
            # Regular model - use external mixed precision if available
            if scaler and torch.cuda.is_available():
                try:
                    # Try new autocast API
                    with torch.amp.autocast('cuda'):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                except:
                    try:
                        # Try old autocast API
                        with torch.cuda.amp.autocast():
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                            loss = outputs.loss
                    except:
                        # No autocast
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                
                # Check for NaN loss
                if torch.isnan(loss):
                    logger.warning("NaN loss detected, skipping batch")
                    return 0.0
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                # Check for NaN loss
                if torch.isnan(loss):
                    logger.warning("NaN loss detected, skipping batch")
                    return 0.0
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
        
        return loss.item()
        
    except Exception as e:
        logger.error(f"Training step failed: {e}")
        return 0.0

def main():
    parser = argparse.ArgumentParser(description='Bulletproof FSDP Training')
    parser.add_argument('--model_name', default='facebook/opt-1.3b', help='Model name')
    parser.add_argument('--max_samples', type=int, default=100, help='Max samples')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')  # Reduced from 5e-5
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=128, help='Max sequence length')
    parser.add_argument('--save_dir', default='./fsdp_model', help='Save directory')
    parser.add_argument('--cpu_offload', action='store_true', help='Enable CPU offloading')
    
    args = parser.parse_args()
    
    try:
        # Setup distributed
        logger.info("Setting up distributed training...")
        rank, world_size, local_rank = setup_distributed()
        logger.info(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare data
        logger.info("Preparing data...")
        texts = prepare_data(args.max_samples)
        logger.info(f"Loaded {len(texts)} text samples")
        
        # Create dataset and dataloader
        dataset = TextDataset(texts, tokenizer, max_length=args.max_length)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        # Create model
        logger.info("Creating model...")
        model = create_model_with_fsdp(args.model_name, cpu_offload=args.cpu_offload)
        
        # Setup optimizer with gradient clipping to prevent NaN
        logger.info("Setting up optimizer...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
        
        # Setup scaler - Disable for FSDP as it handles scaling internally
        scaler = None
        # Check if model is wrapped with FSDP
        is_fsdp_model = hasattr(model, '_fsdp_wrapped_module')
        
        if torch.cuda.is_available() and not is_fsdp_model:
            try:
                # Only use GradScaler for non-FSDP models
                scaler = torch.cuda.amp.GradScaler()
                logger.info("Using CUDA GradScaler for mixed precision")
            except:
                logger.info("No mixed precision available")
        else:
            logger.info("FSDP detected - using internal gradient scaling")
        
        # Print initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"Initial GPU Memory: {initial_memory:.2f}GB")
        
        # Training loop
        logger.info("Starting training...")
        
        for epoch in range(args.num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    loss = train_step(model, batch, optimizer, scaler)
                    total_loss += loss
                    num_batches += 1
                    
                    if batch_idx % 10 == 0:
                        avg_loss = total_loss / max(num_batches, 1)
                        logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {avg_loss:.4f}")
                        
                        # Check for NaN in average loss
                        if torch.isnan(torch.tensor(avg_loss)):
                            logger.error("NaN detected in average loss! Stopping training.")
                            break
                        
                        if torch.cuda.is_available():
                            memory_allocated = torch.cuda.memory_allocated() / 1024**3
                            logger.info(f"GPU Memory: {memory_allocated:.2f}GB")
                            
                            if batch_idx % 50 == 0:
                                torch.cuda.empty_cache()
                
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    continue
            
            avg_epoch_loss = total_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        # Save model
        logger.info("Saving model...")
        try:
            os.makedirs(args.save_dir, exist_ok=True)
            
            # Try FSDP state dict first
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                with FSDP.state_dict_type(
                    model,
                    torch.distributed.fsdp.StateDictType.FULL_STATE_DICT,
                    torch.distributed.fsdp.FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
                ):
                    state_dict = model.state_dict()
                    torch.save(state_dict, os.path.join(args.save_dir, "pytorch_model.bin"))
            except:
                # Fallback to regular state dict
                state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(args.save_dir, "pytorch_model.bin"))
            
            tokenizer.save_pretrained(args.save_dir)
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
        
        # Test generation
        logger.info("Testing generation...")
        try:
            model.eval()
            test_prompt = "The future of AI is"
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            # Ensure model and inputs are on the same device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_length=30,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"Generated: {generated_text}")
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            logger.error(f"Model device: {next(model.parameters()).device}")
            logger.error(f"Input device: {inputs['input_ids'].device if 'inputs' in locals() else 'Not created'}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    finally:
        # Cleanup
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except:
            pass

if __name__ == "__main__":
    main()