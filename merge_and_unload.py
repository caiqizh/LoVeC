import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Merge and unload PEFT adapter weights into base model")
    parser.add_argument("--model_name", type=str, required=True, help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the PEFT adapter weights")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged model")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], 
                       help="Data type for model loading")
    parser.add_argument("--device_map", type=str, default="auto", help="Device map for model loading")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading base model: {args.model_name}")
    
    # Convert dtype string to torch dtype
    dtypes = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    dtype = dtypes[args.dtype]
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=args.device_map
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Handle special tokens for different model families
    if 'Llama-3' in args.model_name:
        tokenizer.pad_token_id = 128001
    else:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading PEFT adapter from: {args.adapter_path}")
    # Load the PEFT model with adapter weights
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    
    print("Merging adapter weights with base model...")
    # Merge weights
    merged_model = model.merge_and_unload()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    print(f"Saving merged model to: {args.output_path}")
    # Save the merged model
    merged_model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    print("Merge and save completed successfully!")

if __name__ == "__main__":
    main()