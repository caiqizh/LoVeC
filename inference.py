from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import json
from argparse import ArgumentParser
from utils.data_utils import get_dataset
from transformers import AutoTokenizer
import os
from utils.data_utils import long_form_confidence_prompt
from utils.inference_utils import confidence_tagging
import torch

def inference(args):
    
    dtypes = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    
    dtype = dtypes[args.dtype]
    
    if 'gemma' in args.model_name:
        llm = LLM(args.model_name, dtype=dtype, max_model_len=args.max_seq_len, seed=args.seed,
                max_lora_rank=args.max_lora_rank, enable_lora=args.lora_path is not None, enforce_eager=True)
    else:
        llm = LLM(args.model_name, dtype=dtype, max_model_len=args.max_seq_len, seed=args.seed,
                max_lora_rank=args.max_lora_rank, enable_lora=args.lora_path is not None)
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    lora = LoRARequest('test-lora', 1, lora_path=args.lora_path) if args.lora_path is not None else None
    
    if args.mode == 'inference':
    
        _, _, test_dataset = get_dataset(args.dataset_name)
        
        sampling_params = SamplingParams(temperature=args.temperature,
                                        top_p=args.top_p, 
                                        max_tokens=args.max_seq_len, 
                                        top_k=args.top_k,
                                        n=args.n)
        
        prompts = [x['prompt'] for x in test_dataset]
        
        if args.dev_mode:
            prompts = prompts[:10]
        
        if args.chat_format:
            new_prompts = []
            for prompt in prompts:
                message = []
                
                if 'gemma' in args.model_name:
                    if args.with_instruction:
                        message.append({'role': 'user', 'content': long_form_confidence_prompt + prompt})
                    else:
                        message.append({'role': 'user', 'content': prompt})
                else:
                    if args.with_instruction:
                        message.append({'role': 'system', 'content': long_form_confidence_prompt})
                    message.append({'role': 'user', 'content': prompt})
                new_prompts.append(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))
            prompts = new_prompts
        
        print('Running inference on', len(prompts), 'prompts')
        print('Prompt Example:', prompts[0])
        
        #breakpoint()
        
        outputs = llm.generate(prompts, sampling_params,
                            lora_request=lora)
        
        results = []
        for x, output in zip(test_dataset, outputs):
            prompt = output.prompt
            generated_texts = [x.text for x in output.outputs]
            new_x = dict(x)
            new_x['model_outputs'] = generated_texts
            results.append(new_x)
            
        save_path = os.path.join(args.save_dir, args.model_name.replace('/','-'), f'results_t-{args.temperature}_p-{args.top_p}_k-{args.top_k}-n-{args.n}.json')

        print('Saving results to', save_path)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
    elif args.mode == 'tagging':

        with open(args.dataset_name, 'r') as f:
            data = json.load(f)
            
        if args.dev_mode:
            data = data[:10]
        
        results = confidence_tagging(data, llm, mode=args.tagging_mode, chat_format=args.chat_format,
                                               with_instruction=args.with_instruction, 
                                               tokenizer=tokenizer, lora_request=lora)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            
        if args.lora_path is not None:
            save_path = os.path.join(args.output_dir, f'results_tagged_{args.lora_path.split('-')[-1]}.json')
        else:
            save_path = os.path.join(args.output_dir, 'results_tagged.json')
    
    with open(save_path, 'w') as f:
        f.write(json.dumps(results, indent=4, ensure_ascii=False))
    

if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--model_name', type=str, required=True)
    args.add_argument('--temperature', type=float, default=0)
    args.add_argument('--top_p', type=float, default=1)
    args.add_argument('--top_k', type=int, default=-1)
    args.add_argument('--n', type=int, default=1)
    args.add_argument('--dev_mode', action='store_true', default=False)
    
    args.add_argument('--chat_format', action='store_true', default=False)
    args.add_argument('--with_instruction', action='store_true', default=False)
    
    args.add_argument('--max_seq_len', type=int, default=512)
    args.add_argument('--dtype', type=str, default='bfloat16')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--dataset_name', type=str, default='MATH-500')
    args.add_argument('--save_dir', type=str, default='.')
    args.add_argument('--lora_path', type=str, default=None)
    args.add_argument('--max_lora_rank', type=int, default=None)
    args.add_argument('--mode', type=str, default='inference')
    args.add_argument('--output_dir', type=str, default='outputs')
    args.add_argument('--tagging_mode', type=str, default='single')

    
    args = args.parse_args()
    
    inference(args)
    