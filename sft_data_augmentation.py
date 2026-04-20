from vllm import LLM, SamplingParams
import json, os
from argparse import ArgumentParser
from utils.data_utils import get_dataset
from transformers import AutoTokenizer
import torch

sampling_params = SamplingParams( # Greedy decoding
    max_tokens=512,  # Maximum number of tokens to generate
    temperature=0,  # Temperature of 0 means no randomness (deterministic)
    top_p=1.0,      # Consider all tokens in the distribution
    top_k=1         # Only select the most likely token at each step
)

def generate_predictions(model_name, dataset_name, dev_mode, dtype, output_dir, apply_chat_template):
    # Load the dataset
    train_dataset, _, _ = get_dataset(dataset_name)
    
    dtypes = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    
    dtype = dtypes[dtype]
    # Initialize the LLM model
    llm = LLM(model=model_name, dtype=dtype)
    
    # Prepare the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through the dataset and generate predictions
    vllm_prompts = [x['prompt'] for x in train_dataset]
    
    if apply_chat_template:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        vllm_prompts = [tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a helpful assistant. You will provide information upon request."},
            {"role": "user", "content": test_question}], tokenize=False) for test_question in vllm_prompts]
        
    if dev_mode:
        vllm_prompts = vllm_prompts[:100]
        
    # Generate predictions
    outputs = llm.generate(
        vllm_prompts,
        sampling_params=sampling_params)
    
    results = []
    
    for output, x in zip(outputs, train_dataset):
        result = x.copy()  # Copy everything from x
        result["model_output"] = output.outputs[0].text  # Add model output
        results.append(result)

    with open(os.path.join(output_dir, "predictions.json"), "w") as f:
        f.write(json.dumps(results, indent=4, ensure_ascii=False))


if __name__ ==  '__main__':
    
    args = ArgumentParser()
    args.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    args.add_argument("--dataset_name", type=str, default="wildhallucination")
    args.add_argument("--dev_mode", default=False, action="store_true", help="Development mode")
    args.add_argument("--dtype", type=str, default="bfloat16", help="Data type for the model")
    args.add_argument("--output_dir", type=str, default="sft_data_augmentation_output")
    args.add_argument("--apply_chat_template", default=False, action="store_true", help="Apply chat template to the dataset")
    args = args.parse_args()
    
    generate_predictions(
        model_name=args.model,
        dataset_name=args.dataset_name,
        dev_mode=args.dev_mode,
        dtype=args.dtype,
        output_dir=args.output_dir,
        apply_chat_template=args.apply_chat_template
    )
    