from nltk.tokenize import sent_tokenize
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import GuidedDecodingParams
import nltk
from tqdm import tqdm
long_form_confidence_prompt = """
You are a helpful assistant. Your task is to provide accurate and informative answers to user queries.

For each sentence in your response:
- Include a confidence score from 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10.
- The score indicates how certain you are that the sentence is factually correct:
    - 0 = very low confidence (likely incorrect)
    - 10 = very high confidence (very likely correct)

Append the confidence score at the end of each sentence using the format: <confidence> X </confidence>, 
where X is a number from 0 to 10.
"""

old_prompt = '''You are a helpful assistant. You will provide information upon request. Additionally provide a confidence between 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, of how sure you are the answer is correct. A value close to 0 means you think there is a high probability that the answer is wrong. The closer the value is to 10, the higher you think is the probability that the answer is correct. You should add the confidence level at the end of every sentence in the form of <confidence> level </confidence>.'''


def group_sentences_to_paragraphs(sentences, indices):
    # Create a dictionary to store sentences for each paragraph
    paragraph_dict = {}
    
    # Group sentences by their paragraph number
    for sent, idx in zip(sentences, indices):
        if idx not in paragraph_dict:
            paragraph_dict[idx] = []
        paragraph_dict[idx].append(sent)
    
    # Convert grouped sentences to paragraphs
    paragraphs = []
    for idx in sorted(paragraph_dict.keys()):
        paragraph = ' '.join(paragraph_dict[idx])
        paragraphs.append(paragraph)
    
    return paragraphs

def confidence_tagging(data, llm, mode='single', chat_format=False, with_instruction=False, tokenizer=None, lora_request=None):
    """
    Function to tag the confidence of the model's output.
    
    Args:
        data (list): List of dictionaries containing model outputs.
        mode (str): Mode of tagging ('single' or 'concatenate').
        
    Returns:
        tagged_data (list): List of dictionaries with tagged confidence.
    """

    confidence_scores = [' ' + str(i) + ' </confidence>' for i in range(0, 11)]
    
    if 'gemma' in tokenizer.name_or_path:
        if chat_format:
            eos_token = '<end_of_turn>\n'
    else:
        if chat_format:
            eos_token = tokenizer.eos_token
            
        
    if mode == 'single':
        #deprecated don't use this
        input_prompts = []
        all_sentences = []
        indicies = []
        
        for i, item in enumerate(data):
            if '\n\n' in item['model_outputs'][0]:
                model_outputs = item['model_outputs'][0].split('\n\n')[1] #take out the template head
            else:
                model_outputs = item['model_outputs']
            
            no_confidence_sentences = nltk.sent_tokenize(model_outputs)
            
            for s in no_confidence_sentences:
                if chat_format:
                    message = []
                    
                    if 'gemma' in tokenizer.name_or_path:
                        if with_instruction:
                            message.append({'role': 'user', 'content': long_form_confidence_prompt + item['prompt']})
                        else:
                            message.append({'role': 'user', 'content': item['prompt']})
                    else:
                        if with_instruction:
                            message.append({'role': 'system', 'content': long_form_confidence_prompt})
                        message.append({'role': 'user', 'content': item['prompt']})
                    message.append({'role': 'assistant', 'content': s})
                    message = tokenizer.apply_chat_template(message, tokenize=False)
                    
                    if message.endswith(eos_token):
                        message = message[:-len(eos_token)]
                else:
                    if with_instruction:
                        message = long_form_confidence_prompt + ' ' + item['prompt'] + ' ' + s

                message = message + ' <confidence> '
                input_prompts.append(message)
                indicies.append(i)
                all_sentences.append(s + ' <confidence> ')
                
        outputs = llm.generate(
            input_prompts,
            sampling_params= SamplingParams(top_k=-1, top_p=1, temperature=0, guided_decoding=GuidedDecodingParams(choice=confidence_scores)),
            lora_request= lora_request,
        )
        
        for i, output in enumerate(outputs):
            score_tag = output.outputs[0].text
            all_sentences[i] = all_sentences[i] + score_tag
        
        paragraphs = group_sentences_to_paragraphs(all_sentences, indicies)

    elif mode == 'concatenate':
        #BFS batch decoding
        
        item_prompts = [item['prompt'] for item in data]
        paragraphs = ["" for _ in range(len(data))]
        frontier = [i for i in range(len(data))]
        remaining_sentences = {}
        
        for i, item in enumerate(data):
            if item['model_outputs'][0].startswith('\n\n') or item['model_outputs'][0].startswith('<|start_header_id|>assistant<|end_header_id|>\n\n'): 
                # I know this is a hack, but I don't have time to fix it
                # this is a hack to remove the template head
                model_outputs = item['model_outputs'][0].split('\n\n')[1] #take out the template head
            else:
                model_outputs = item['model_outputs'][0]
            
            try:
                no_confidence_sentences = nltk.sent_tokenize(model_outputs)
            except:
                print(model_outputs)
                breakpoint()
            remaining_sentences[i] = no_confidence_sentences
        
        while frontier != []:
            new_frontier = []
            input_prompts = []
            
            for idx in frontier:
                current_prompt = item_prompts[idx]
                try:
                    current_fact = remaining_sentences[idx].pop(0)
                except:
                    breakpoint()
                paragraphs[idx] = paragraphs[idx] + current_fact + ' <confidence>'
                
                if chat_format:
                    message = []
                    if 'gemma' in tokenizer.name_or_path:
                        if with_instruction:
                            message.append({'role': 'user', 'content': long_form_confidence_prompt + current_prompt})
                        else:
                            message.append({'role': 'user', 'content': current_prompt})
                    else:
                        if with_instruction:
                            message.append({'role': 'system', 'content': long_form_confidence_prompt})
                        message.append({'role': 'user', 'content': current_prompt})
                    
                    message.append({'role': 'assistant', 'content': paragraphs[idx]})
                    message = tokenizer.apply_chat_template(message, tokenize=False)
                    
                    if message.endswith(eos_token):
                        message = message[:-len(eos_token)]
                else:
                    if with_instruction:
                        message = long_form_confidence_prompt + ' ' + current_prompt + ' ' + paragraphs[idx]
                        
                input_prompts.append(message)

            print('Running inference on', len(input_prompts), 'prompts')
            print('Prompt Example:', input_prompts[0])
            outputs = llm.generate(
                input_prompts,
                sampling_params=SamplingParams(top_k=-1, top_p=1, temperature=0, guided_decoding=GuidedDecodingParams(choice=confidence_scores)),
                lora_request=lora_request,
            )
            
            for idx, output in zip(frontier, outputs):
                score_tag = output.outputs[0].text
                paragraphs[idx] = paragraphs[idx] +  f'{score_tag} '
                
                if len(remaining_sentences[idx]) > 0:
                    new_frontier.append(idx)
            
            frontier = new_frontier
    else:
        raise NotImplementedError("Only single mode is implemented")

    return paragraphs