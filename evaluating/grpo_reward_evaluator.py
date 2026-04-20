import math
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import wandb

def extract_sentences_with_confidence(text):
    # Split by closing confidence tag
    parts = text.split('</confidence>')
    sentences = []
    confidences = []
    
    for part in parts[:-1]:  # Last split will be empty
        # Split each part into sentence and confidence
        sentence_conf = part.split('<confidence>')
        if len(sentence_conf) == 2:
            sentence = sentence_conf[0].strip()
            try:
                confidence = int(sentence_conf[1].strip())
            except:
                confidence = None
            sentences.append(sentence)
            confidences.append(confidence)
        else:
            sentence = part.strip()
            sentences.append(sentence)
            confidences.append(None)
    
    return sentences, confidences

def binary_reward_function(confidence: int, is_answer_correct: bool) -> float:

    scale = 10.0
    max_reward = -0.0010005003335835344
    min_reward = -6.907755278982137 / 2
    wrong_format_penalty = -scale * 3.0

    if confidence == None or confidence > 10 or confidence < 0:
        return wrong_format_penalty
    
    normalized_confidence = min(0.999, max(0.001, confidence / 10))

    if is_answer_correct:
        score = math.log(normalized_confidence)
    else: 
        score = math.log(1 - normalized_confidence)

    norm_score = (score - min_reward) / (max_reward - min_reward)
    if is_answer_correct:
        norm_score += 0.25
        
    return float(scale * norm_score)

def numerical_reward_function(confidence: int, answer_correct_level: int) -> float:
    scale = 10.0
    max_reward = -0.0010005003335835344
    min_reward = -6.907755278982137 / 2
    wrong_format_penalty = -scale * 3.0

    if confidence == None or confidence > 10 or confidence < 0:
        return wrong_format_penalty
    
    normalized_confidence = min(0.999, max(0.001, confidence / 10))
    normalized_correctness = answer_correct_level / 10

    if normalized_correctness > 0.5:
        score = math.log(normalized_confidence) * normalized_correctness
    else:
        score = math.log(1 - normalized_confidence) * (1 - normalized_correctness)

    norm_score = (score - min_reward) / (max_reward - min_reward)
   
    norm_score += 0.25 * normalized_correctness
        
    return float(scale * norm_score)

def proper_log_reward(confidence: int, correctness: int, scale=10.0):

    # Given by GPT
    if confidence is None or not (0 <= confidence <= 10):
        return -3 * scale  # malformed input penalty
    
    p = np.clip(confidence / 10, 1e-3, 1 - 1e-3)
    y = correctness / 10
    nll = -(y * math.log(p) + (1 - y) * math.log(1 - p))
    best_nll = 0
    worst_nll = -(math.log(1e-3) + math.log(1 - 1e-3)) / 2
    reward = scale * (1 - (nll - best_nll) / (worst_nll - best_nll))

    reward += 0.075 * (correctness)  # add a small bonus for correctness

    return float(reward)

def improved_log_reward(confidence: int, correctness: int, scale=10.0, gamma=1.5, penalty_strength=5.0):
    if confidence is None or not (0 <= confidence <= 10):
        return -3 * scale  # malformed input penalty

    # Core log-likelihood reward
    p = np.clip(confidence / 10, 1e-6, 1 - 1e-6)
    y = correctness / 10
    nll = -(y * math.log(p) + (1 - y) * math.log(1 - p))

    best_nll = 0
    worst_nll = -(math.log(1e-6) + math.log(1 - 1e-6)) / 2

    reward = scale * (1 - (nll - best_nll) / (worst_nll - best_nll))

    # Stretch reward to amplify good/bad
    reward = np.sign(reward) * (abs(reward) ** gamma)

    # Asymmetric overconfidence penalty
    if confidence >= 7:
        if confidence - correctness > 2:
            reward -= penalty_strength * ((confidence - 6) / 4)  # stronger penalty as confidence rises

    # Correctness bonus (small)
    reward += 0.15 * correctness

    return float(reward)

def quadratic_reward(confidence: int, correctness: int, scale=20.0):
    if confidence is None or not (0 <= confidence <= 10):
        return -3 * scale
    
    y = correctness / 10
    c = confidence / 10
    # Quadratic penalty for deviation from correctness
    penalty = (c - y) ** 2
    reward = scale * (1 - penalty)
    reward += 0.3 * correctness
    return float(reward) - 10

def grpo_confidence_reward(completions, evidence, judge, evaluate_mode='binary', **kwargs):

    # Warning, weird naming pattern, this evidence matches the GRPO trainer, but it's actually a list of evidences
    
    
    #breakpoint()
    list_of_sentences = []
    list_of_confidences = []

    for output in completions:
        if output.startswith('\n\n') or output.startswith('"<|start_header_id|>assistant<|end_header_id|>\n\n'): 
                # I know this is a hack, but I don't have time to fix it
                # this is a hack to remove the template head
                output = output.split('\n\n')[1] #take out the template head
        else:
            pass

        sentences, confidences = extract_sentences_with_confidence(output)
        
        if len(sentences) == 0:
            print("Warning: No sentences found in the completion. Setting reward to -60.")
            print("Completion:", output)
            list_of_sentences.append(['no output'])
            list_of_confidences.append([None])
        else:
            list_of_sentences.append(sentences)
            list_of_confidences.append(confidences)
        
    
    evaluations = judge.evaluate_correctness(paragraphs=[' '.join(sentences) for sentences in list_of_sentences], evidences=evidence, tagging_method=evaluate_mode)

    if 'wandb' in globals() or 'wandb' in locals():
        try:
            if wandb.run is not None:
                # Create a summary of evaluations for logging
                eval_summary = []
                for i, eval_list in enumerate(evaluations):
                    sentences_eval = {}
                    for j, (sentence, correctness) in enumerate(eval_list):
                        sentences_eval[f"sentence_{j}"] = sentence
                        sentences_eval[f"correctness_{j}"] = correctness
                    eval_summary.append(sentences_eval)
                
                wandb.log({"evaluations": eval_summary})
        except Exception as e:
            print(f"Failed to log evaluations to wandb: {str(e)}")
    
    list_of_rewards = []
    #breakpoint()
    for confidences, sentences, evaluation in zip(list_of_confidences, list_of_sentences, evaluations):
        total_reward = 0

        for confidence, (_, correctness) in zip(confidences, evaluation):
            if evaluate_mode == 'binary':
                reward = binary_reward_function(confidence, correctness)
                total_reward += reward
            elif evaluate_mode == 'numerical':
                reward = improved_log_reward(confidence, correctness)
                total_reward += reward
            else:
                raise ValueError("Invalid evaluate_mode. Choose either 'binary' or 'numerical'.")
        
        if len(sentences) == 1 and sentences[0] == 'no output':
            print("Warning: No sentences found in the completion. Setting reward to -60.")
            total_reward = -60
            print("Completion:", output)
        else:
            total_reward /= len(sentences)
        list_of_rewards.append(total_reward)

    return list_of_rewards

def bleu_regularisation_reward(completions, base_prediction, scale=1.5, **kwargs):
    """
    This function computes the BLEU score between the completions and the base prediction.
    The BLEU score is a measure of how similar the completions are to the base prediction.
    """
    list_of_rewards = []
    for completion, base in zip(completions, base_prediction):
        # Tokenize the completion and base prediction
        if '\n\n' in completion:
            completion = completion.split('\n\n')[-1]
        
        sentences, confidences = extract_sentences_with_confidence(completion)
        completion = ' '.join(sentences)

        completion_tokens = completion.split()
        base_prediction_tokens = base.split()

        # Compute the BLEU score
        bleu_score = sentence_bleu([base_prediction_tokens], completion_tokens)

        # Convert BLEU score to reward
        reward = 1 - bleu_score
        list_of_rewards.append(scale * reward)

    return list_of_rewards


if __name__ == "__main__":
    # Example usage
    model_name = "google/gemma-2-9b-it"
    import json
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import os
    from peft import AutoPeftModelForCausalLM
    from vllm_evaluator import ServerJudge
    long_form_confidence_prompt = '''You are a helpful assistant. You will provide information upon request. Additionally provide a confidence between 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, of how sure you are the answer is correct. A value close to 0 means you think there is a high probability that the answer is wrong. The closer the value is to 10, the higher you think is the probability that the answer is correct. You should add the confidence level at the end of every sentence in the form of <confidence> level </confidence>.'''
    wildhallucination_prompt = '''In a paragraph, could you tell me what you know about {entity}?'''

    model = AutoPeftModelForCausalLM.from_pretrained('path', attn_implementation='eager', device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    with open('datapath', 'r') as f:
        data = json.load(f)

    completions = []
    evidences = []
    base_predictions = []

    for i in range(5):
        if 'gemma' in model_name:
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': long_form_confidence_prompt + wildhallucination_prompt.format(entity=data[i]['entity'])}], return_tensors='pt', add_generation_prompt=True)
        else:
            prompt = tokenizer.apply_chat_template([{'role': 'system', 'content': long_form_confidence_prompt},
                                                    {'role': 'user', 'content': wildhallucination_prompt.format(entity=data[i]['entity'])}], return_tensors='pt', truncation=True)
        prompt = prompt.to('cuda')
        #print(tokenizer.decode(prompt[0], skip_special_tokens=False))
        evidence = data[i]['evidence']
        tokenized_evidence = tokenizer(evidence, truncation=True, max_length=7000, return_tensors='pt')
        evidence = tokenizer.decode(tokenized_evidence['input_ids'][0], skip_special_tokens=True)

        evidences.append(evidence)
        base_predictions.append(data[i]['base_prediction'])

        #breakpoint()
        #bios = data[0]['evidence']
        
        
        answer = model.generate(input_ids=prompt, max_new_tokens=512)
        #breakpoint()
        if 'gemma' in model_name:
            # Gemma is outputting the input prompt in the output, so we need to remove it
            # This is unconventional, no wonder people are not using gemma.
            answer = tokenizer.decode(answer[0][len(prompt[0]):], skip_special_tokens=True)
        else:
            answer = tokenizer.decode(answer[0], skip_special_tokens=True)
        #answer = answer.split('\n\n')[-1]
        completions.append(answer)
        #print(answer)

        #breakpoint()
    evidences.append('no evidence')
    completions.append('')
    base_predictions.append('no base prediction')
    evaluator = ServerJudge()
    print('completions:' + str(completions))
    #print('evidences:' + str(evidences))
    print('binary:' + str(grpo_confidence_reward(completions, evidences, evaluator, evaluate_mode='binary')))
    print('numerical:' + str(grpo_confidence_reward(completions, evidences, evaluator, evaluate_mode='numerical')))
    print('bleu:' + str(bleu_regularisation_reward(completions, base_predictions)))