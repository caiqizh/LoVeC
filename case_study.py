from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import GuidedDecodingParams
import json

from utils.inference_utils import long_form_confidence_prompt

def main():
    
    prompts = [
        'In a paragraph, tell me about mRNA vaccines.',
        'In a paragraph, tell me about Data Protection Act 1998.']
    
    model_path = '/home/spacehunter/shorter-cot/outputs/manual_sft/checkpoint-750/merged_model'
    llm = LLM(model_path, dtype='bfloat16', max_model_len=512, seed=42, max_lora_rank=64, enable_lora=True)
    lora_path = 'outputs/llama-checkpoints/sft_epoch_2'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    vllm_prompts = [tokenizer.apply_chat_template([
        {'role': 'system', 'content': long_form_confidence_prompt},
        {'role': 'user', 'content': prompt},
    ], tokenize=False) for prompt in prompts]
    
    sampling_params = SamplingParams(
        max_tokens=512,  # Maximum number of tokens to generate
        temperature=0,  # Temperature of 0 means no randomness (deterministic)
        top_p=1.0,      # Consider all tokens in the distribution
        top_k=-1,         # Only select the most likely token at each step
        n=1,
    )
    
    outputs = llm.generate(vllm_prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest('test-lora', 1, lora_path=lora_path)
    )
    
    for i, output in enumerate(outputs):
        print(f"Prompt: {prompts[i]}")
        print(f"Output: {output}")
    
    
    fill_prompts = [{'prompt': 'In a paragraph, could you tell me what you know about King\'s College, Cambridge?',
                     'model_outputs':'<|start_header_id|>assistant<|end_header_id|>\n\nKing\'s College, Cambridge is a constituent college of the University of Cambridge, one of the world\'s oldest and most prestigious universities.'},
                    {'prompt':'In a paragraph, could you tell me what you know about MiniGPT4?',
                     'model_outputs':'<|start_header_id|>assistant<|end_header_id|>\n\nMiniGPT4 is a lightweight and efficient variant of the popular GPT-4 language model, designed to be more accessible and easier to deploy in resource-constrained environments.'}]
    
    #choices = [' ' + str(i) + ' </confidence>' for i in range(0, 11)]
    
    vllm_prompts = [tokenizer.apply_chat_template([
        {'role': 'system', 'content': long_form_confidence_prompt},
        {'role': 'user', 'content': item['prompt']},
        {'role': 'assistant', 'content': item['model_outputs']},
    ], tokenize=False) + ' <confidence> ' for item in fill_prompts]
    
    sampling_params = SamplingParams(
        max_tokens=1,  # Maximum number of tokens to generate
        temperature=0,  # Temperature of 0 means no randomness (deterministic)
        top_p=1.0,      # Consider all tokens in the distribution
        top_k=-1,         # Only select the most likely token at each step
        n=1,
        logprobs=15
    )
    
    outputs = llm.generate(vllm_prompts,
        sampling_params=sampling_params,
        lora_request=LoRARequest('test-lora', 1, lora_path=lora_path)
    )
    for i, output in enumerate(outputs):
        print(f"Prompt: {fill_prompts[i]['prompt']}")
        print(f"Output: {output}")
        
        
if __name__ == "__main__":
    main()