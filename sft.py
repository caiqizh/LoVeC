#from unsloth import FastLanguageModel
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import torch
import os, json, argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM, setup_chat_format
from utils.data_utils import long_form_confidence_prompt
from utils.training_utils import load_config, save_config
from peft import PeftModel
from datasets import Dataset
import pandas as pd

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    combined = [item["combined"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "combined": combined
    }
    
    
def train(args, cfg):
    
    dtypes = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    
    dtype = dtypes[cfg.dtype]
    
    sft_config = SFTConfig(
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
        max_seq_length=cfg.max_seq_length,
    )
    
    if args.unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = cfg.model_name,
            max_seq_length = cfg.max_seq_length,
            dtype = dtype,
            load_in_4bit = cfg.load_in_4bit,
        )
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
    
    # Load the data
    with open(cfg.dataset_name, "r") as f:
        data = json.load(f)
    
    if args.chat_format:
        if 'gemma-2' in cfg.model_name and 'it' in cfg.model_name: 
            #For some reason, the gemme-2-it model does not support 'system' role
            df = pd.DataFrame([{
                "messages": [
                    {'role': 'user', 'content': long_form_confidence_prompt + d['prompt']},
                    {'role': 'assistant', 'content': d['rated_output_openai']}
                ]
            } for d in data if 'rated_output_openai' in d.keys()])
        else:
            df = pd.DataFrame([{
                "messages": [
                    {'role': 'system', 'content': long_form_confidence_prompt},
                    {'role': 'user', 'content': d['prompt']},
                    {'role': 'assistant', 'content': d['rated_output_openai']}
                ]
            } for d in data if 'rated_output_openai' in d.keys()])
    else: #Still chat format but train on completion only
        new_data = []
        for d in data:
            if 'gemma-2' in cfg.model_name and 'it' in cfg.model_name:
                # gemme-2-it model does not support 'system' role
                prompt_message = [
                    {'role': 'user', 'content': long_form_confidence_prompt + d['prompt']}]
                
                prompt_mesage = tokenizer.apply_chat_template(prompt_message, tokenize=False)
                
                # gemma-2 is giving me an error when I try to use the 'assistant' role alone
                full_message = [
                    {'role': 'user', 'content': long_form_confidence_prompt + d['prompt']},
                    {'role': 'assistant', 'content': d['rated_output_openai']}]
                
                completion_message = tokenizer.apply_chat_template(full_message, tokenize=False)
                # Need to remove the prompt part from the completion message
                # Find the assistant part in the completion message
                assistant_start = completion_message.find('<start_of_turn>model\n')
                completion_message = completion_message[assistant_start:]
            else:
                prompt_message = [
                        {'role': 'system', 'content': long_form_confidence_prompt},
                        {'role': 'user', 'content': d['prompt']}]
                prompt_mesage = tokenizer.apply_chat_template(prompt_message, tokenize=False)
                
                completion_message = [{'role': 'assistant', 'content': d['rated_output_openai']}]
                completion_message = tokenizer.apply_chat_template(completion_message, tokenize=False)
            
            new_data.append({'instruction': prompt_mesage, 'output': completion_message})
        df = pd.DataFrame(new_data)
    
    # Convert pandas DataFrame to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(df)
    print("Training set sample:")
    print(train_dataset[0])
    # Set up training arguments

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = example['instruction'][i] + example['output'][i]
            output_texts.append(text)
        return output_texts

    if 'gemma' in cfg.model_name:
        response_template = '<start_of_turn>model\n'
    else:
        response_template = '<|start_header_id|>assistant<|end_header_id|>\n\n'
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # Initialize standard trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
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
    parser = argparse.ArgumentParser(description="SFT model for long-form confidence prediction")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--unsloth", default=False, action="store_true", help="Use unsloth for training")
    parser.add_argument("--chat_format", action="store_true", help="Use chat format for training")
    parser.add_argument("--merge_lora", action="store_true", help="Merge LoRA weights into the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    # parser.add_argument("--dataset_name", type=str, default="ecqa", help="Dataset name")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize wandb
    cfg = load_config(args.config)
    
    if cfg.report_to == 'wandb':
        import wandb

        os.environ['WANDB_API_KEY'] = 'your_wandb_api_key_here'
        wandb.init(project="your_wandb_project_name", config=cfg.__dict__, name=cfg.wandb_name)
        
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
