
from trl import DPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
import os,json
from datasets import load_dataset, Dataset
from utils.data_utils import get_dataset
from utils.data_utils import long_form_confidence_prompt
from utils.training_utils import load_config, save_config
from peft import PeftModel
import pandas as pd
torch.manual_seed(42)

def train(args, cfg):
    dtypes = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    
    dtype = dtypes[cfg.dtype]
    
    with open(cfg.dataset_name, "r") as f:
        data = json.load(f)
    
    if 'gemma' in cfg.model_name:
        df = pd.DataFrame([{
            "chosen": [
                {'role': 'user', 'content': long_form_confidence_prompt + d['prompt']},
                {'role': 'assistant', 'content': d['chosen']}
            ],
            "rejected": [
                {'role': 'user', 'content': long_form_confidence_prompt + d['prompt']},
                {'role': 'assistant', 'content': d['rejected']}
            ]
        } for d in data])
    else:
        df = pd.DataFrame([{
            "chosen": [
                {'role': 'system', 'content': long_form_confidence_prompt},
                {'role': 'user', 'content': d['prompt']},
                {'role': 'assistant', 'content': d['chosen']}
            ],
            "rejected": [
                {'role': 'system', 'content': long_form_confidence_prompt},
                {'role': 'user', 'content': d['prompt']},
                {'role': 'assistant', 'content': d['rejected']}
            ]
        } for d in data])
        
    train_dataset = Dataset.from_pandas(df)
    
    
    if args.unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = cfg.model_name,
            max_seq_length = cfg.max_seq_length,
            dtype = dtype,
            load_in_4bit = cfg.load_in_4bit,
        )
    else:
        if 'gemma' in cfg.model_name:
            model = AutoModelForCausalLM.from_pretrained(cfg.model_name, attn_implementation='eager', torch_dtype=dtype, device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        
    if 'Llama-3' in cfg.model_name:
        tokenizer.pad_token_id = 128001
    else:
        tokenizer.pad_token = tokenizer.eos_token
        
    if cfg.lora_rank is not None:
        if args.unsloth:
            model = FastLanguageModel.get_peft_model(
                model,
                r = cfg.lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = cfg.target_modules if hasattr(cfg, 'target_modules') else ["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_alpha = cfg.lora_alpha if hasattr(cfg, 'lora_alpha') else 16,
                lora_dropout = cfg.lora_dropout if hasattr(cfg, 'lora_dropout') else 0.05, # Supports any, but = 0 is optimized
                bias = "none",    # Supports any, but = "none" is optimized
                # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                random_state = 42,
            )
        from peft import LoraConfig, get_peft_model, TaskType
    
        # Configure LoRA
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha if hasattr(cfg, 'lora_alpha') else 16,
            lora_dropout=cfg.lora_dropout if hasattr(cfg, 'lora_dropout') else 0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=cfg.target_modules if hasattr(cfg, 'target_modules') else ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        
        # Apply LoRA to the model
        if hasattr(cfg, 'lora_path') and cfg.lora_path is not None:
            print(f"Loading LoRA weights from: {cfg.lora_path}")
            model = PeftModel.from_pretrained(model, cfg.lora_path)
        else:
            model = get_peft_model(model, lora_config)
            
        model.print_trainable_parameters()
        # Check number of trainable vs total parameters
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.4f}% of total)")
    print(f"Total parameters: {total_params:,}")
    

    if args.rl_mode == 'DPO':
        train_config = DPOConfig(
            learning_rate=cfg.learning_rate,
            adam_beta1=cfg.adam_beta1,
            adam_beta2=cfg.adam_beta2,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler_type,
            optim=cfg.optim,
            logging_steps=cfg.logging_steps,
            bf16= True if dtype == torch.bfloat16 else False,
            fp16= True if dtype == torch.float16 else False,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            max_steps=cfg.max_steps if cfg.num_train_epochs == None else -1,
            num_train_epochs=cfg.num_train_epochs if cfg.num_train_epochs is not None else -1,
            save_steps=cfg.save_steps,
            max_grad_norm=cfg.max_grad_norm,
            report_to=cfg.report_to,
            output_dir=cfg.output_dir,
            beta=cfg.beta,
        )
        
        trainer = DPOTrainer(
            model=model,
            args=train_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
    elif args.rl_mode == 'ORPO':
        train_config = ORPOConfig(learning_rate=cfg.learning_rate,
            adam_beta1=cfg.adam_beta1,
            adam_beta2=cfg.adam_beta2,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            lr_scheduler_type=cfg.lr_scheduler_type,
            optim=cfg.optim,
            logging_steps=cfg.logging_steps,
            bf16= True if dtype == torch.bfloat16 else False,
            fp16= True if dtype == torch.float16 else False,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            max_steps=cfg.max_steps if cfg.num_train_epochs == None else -1,
            num_train_epochs=cfg.num_train_epochs if cfg.num_train_epochs is not None else -1,
            save_steps=cfg.save_steps,
            max_grad_norm=cfg.max_grad_norm,
            report_to=cfg.report_to,
            output_dir=cfg.output_dir,
        )
        trainer = ORPOTrainer(
            model=model,
            args=train_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
        )
        
    # Train model
    trainer.train()
    # Model checkpoints should be saved automatically by the trainer
    
    if args.merge_lora and cfg.lora_rank is not None:
        print("Merging LoRA weights with base model...")
        # Get the base model from the PeftModel
        merged_model = model.merge_and_unload()

        # Save the merged model to the output directory
        merged_output_path = os.path.join(cfg.output_dir, "merged_model")
        merged_model.save_pretrained(merged_output_path)
        tokenizer.save_pretrained(merged_output_path)
        print(f"Merged model saved to {merged_output_path}")
    
def parse_args():
    parser = argparse.ArgumentParser(description="RL for long-form confidence prediction")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--unsloth", default=False, action="store_true", help="Use unsloth for training")
    parser.add_argument("--chat_format", action="store_true", help="Use chat format for training")
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA weights into the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--rl_mode", default="DPO", type=str, help="RL mode: DPO or ORPO")
    parser.add_argument("--dataset_name", type=str, default="ecqa", help="Dataset name")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize wandb
    cfg = load_config(args.config)
    
    if cfg.report_to == 'wandb':
        import wandb

        os.environ['WANDB_API_KEY'] = 'your_wandb_api_key_here'
        wandb.init(project="your_project_name", config=cfg.__dict__, name=cfg.wandb_name)
        
    # Save configuration
    save_config(cfg, cfg.output_dir)
    
    if not args.eval:
        # Start training
        train(args, cfg)
    else:
        evaluate(args, cfg)
    
    wandb.finish()

if __name__ == "__main__":
    main()

