from trl import GRPOConfig, GRPOTrainer
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
import wandb
from evaluating.grpo_reward_evaluator import grpo_confidence_reward, bleu_regularisation_reward
from evaluating.vllm_evaluator import ServerJudge
from functools import partial
os.environ["WANDB_PROJECT"]="your_project_name"
torch.manual_seed(42)

def train(args, cfg):
    dtypes = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    
    dtype = dtypes[cfg.dtype]
    judge = ServerJudge()
    
    grpo_reward_func = partial(grpo_confidence_reward, judge=judge, evaluate_mode=args.evaluate_mode)
    grpo_reward_func.__name__ = "grpo_reward_func"

    reward_funcs = [grpo_reward_func, bleu_regularisation_reward]
    
    if args.unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = cfg.model_name,
            max_seq_length = cfg.max_seq_length,
            dtype = dtype,
            load_in_4bit = cfg.load_in_4bit,
        )
    else:
        if 'gemma' in cfg.model_name:
            model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype, device_map='auto', attn_implementation='eager')
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


    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if 'Llama-3' in cfg.model_name:
        tokenizer.pad_token_id = 128001
    else:
        tokenizer.pad_token = tokenizer.eos_token
    

    # Configure training arguments
    training_args = GRPOConfig(
        use_vllm=cfg.use_vllm,
        vllm_dtype=dtype,
        vllm_gpu_memory_utilization=cfg.gpu_memory_utilization,
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
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        max_steps=cfg.max_steps if cfg.num_train_epochs == None else -1,
        num_train_epochs=cfg.num_train_epochs if cfg.num_train_epochs is not None else -1,
        save_steps=cfg.save_steps,
        max_grad_norm=cfg.max_grad_norm,
        report_to=cfg.report_to,
        run_name=cfg.wandb_name,
        output_dir=cfg.output_dir,
    )
    
    if cfg.num_train_epochs is not None:
        training_args.num_train_epochs = cfg.num_train_epochs

    # Initialize dataset
    train_dataset, validation_dataset, test_dataset = get_dataset(cfg.dataset_name, tokenizer=tokenizer, chat_format=args.chat_format, with_instruction=args.with_instruction)
    #print(len(train_dataset))
    #breakpoint()
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Start training
    trainer.train() # Model checkpoints should be saved automatically by the trainer
    
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
    parser.add_argument("--with_instruction", action="store_true", help="Use instruction format for training")
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA weights into the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--rl_mode", default="DPO", type=str, help="RL mode: DPO or ORPO")
    parser.add_argument("--dataset_name", type=str, default="ecqa", help="Dataset name")
    parser.add_argument("--evaluate_mode", type=str, default="numerical", help="GRPO evaluation mode: numerical or binary")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize wandb
    cfg = load_config(args.config)
        
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

