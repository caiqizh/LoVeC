import re
import math
import os, sys, json, time
import importlib
import torch
import torch.nn as nn
from functools import partial

def ecqa_correctness_eval(extracted, answer):
    masks = [True if a.lower() in e.lower() else False for e, a in zip(extracted, answer)]
    return masks

def trivia_qa_correctness_eval(extracted, answers, threshold=0.5):
    masks = []
    
    for e, candidates in zip(extracted, answers):
        # https://arxiv.org/pdf/2503.02623 threshold > 0.5 as correct
        if any([compute_f1(e, a) > threshold for a in candidates]):
            masks.append(True)
        else:
            masks.append(False)
    
    return masks

def compute_f1(a, b):
    """
    Calculate F1 score between candidate string 'a' and gold string 'b'.
    
    Args:
        a: Candidate string
        b: Gold standard string
    
    Returns:
        F1 score as a float
    """
    # Convert to lowercase and tokenize by splitting on whitespace
    a_tokens = a.lower().split()
    b_tokens = b.lower().split()
    
    # Empty strings edge case
    if len(a_tokens) == 0 or len(b_tokens) == 0:
        return 1.0 if len(a_tokens) == len(b_tokens) else 0.0
    
    # Find common tokens
    common = set(a_tokens) & set(b_tokens)
    
    # Calculate precision, recall and F1
    precision = len(common) / len(a_tokens)
    recall = len(common) / len(b_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_xml_confidence(text: str) -> float:
    confidence = text.split("<confidence>")[-1]
    confidence = confidence.split("</confidence>")[0]
    return confidence.strip()

def correctness_reward_func(completions, answer, eval_func, **kwargs) -> list[float]:
    
    '''
    eval_function should take extracted_responses and answer as input and return a list of rewards
    answer : might be a list of answers or a single answer, depending on your dataset.
    '''
    
    responses = ['<answer> ' + completion for completion in completions]
    extracted_responses = [extract_xml_answer(r).lower() for r in responses]
    correctness = eval_func(extracted_responses, answer)
    return [0.25 if c else 0 for c in correctness]


def confidence_reward_func(completions, answer, eval_func, **kwargs) -> list[float]:
    confidences = [extract_xml_confidence(completion) for completion in completions]
    responses = ['<answer> ' + completion for completion in completions]
    predicted = [extract_xml_answer(r).lower() for r in responses]
    
    final_rewards = []
    correctness = eval_func(predicted, answer)
    for conf, correct in zip(confidences, correctness):
        if conf.isdigit():
            normalized_confidence = torch.clamp(torch.tensor(float(conf)), min=0.001, max=10)
            if correct:
                current_reward = torch.clamp(torch.log10(normalized_confidence),  min=-1, max=1).item()
            else:
                current_reward = torch.clamp(torch.log10(1 - normalized_confidence + 9),  min=-1, max=1).item()
        else:
            current_reward = -1.0
        
        final_rewards.append(current_reward)
        #print(c,p,a,current_reward)
    
    #print([(gold, pred, conf, mask, reward) for gold, pred, conf, mask, reward in zip(answer, predicted, confidences, mask, final_rewards)])
    return final_rewards

def int_reward_func(completions, **kwargs) -> list[float]:
    confidences = [extract_xml_confidence(completion) for completion in completions]
    return [0.5 if r.isdigit() else 0.0 for r in confidences]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<answer>.*?</answer>\s*<confidence>.*?</confidence>"
    responses = ['<answer> ' + completion for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.0 if match else -3.0 for match in matches]


def get_reward_functions(reward_func_names, dataset_name):
    """Map reward function names to actual functions"""
    
    if dataset_name == 'ecqa':
        # Create partial functions with proper names
        correctness_with_eval = partial(correctness_reward_func, eval_func=ecqa_correctness_eval)
        correctness_with_eval.__name__ = "correctness_reward_func"
        
        confidence_with_eval = partial(confidence_reward_func, eval_func=ecqa_correctness_eval)
        confidence_with_eval.__name__ = "confidence_reward_func"
        
        reward_funcs_map = {
            "correctness_reward_func": correctness_with_eval,
            "confidence_reward_func": confidence_with_eval,
            "int_reward_func": int_reward_func,
            "soft_format_reward_func": soft_format_reward_func
        }
    elif dataset_name == 'triviaqa':
        # Create partial functions with proper names
        if "reward_doubt_baseline" in reward_func_names:
            reward_funcs_map = {
                "reward_doubt_baseline": reward_doubt_baseline
            }
        else:
            correctness_with_eval = partial(correctness_reward_func, eval_func=trivia_qa_correctness_eval)
            correctness_with_eval.__name__ = "correctness_reward_func"
            
            confidence_with_eval = partial(confidence_reward_func, eval_func=trivia_qa_correctness_eval)
            confidence_with_eval.__name__ = "confidence_reward_func"
            
            reward_funcs_map = {
                "correctness_reward_func":  correctness_with_eval,
                "confidence_reward_func": confidence_with_eval,
                "soft_format_reward_func": soft_format_reward_func
            }
    
    return [reward_funcs_map[name] for name in reward_func_names]


def reward_doubt_baseline(completions, answer, **kwargs) -> list[float]:
    
    #They hardcoded these things in this way, I don't know why, and I dont
    #want to understand why. I will just copy the code and make it work.
    scale = 10.0
    max_reward = -0.0010005003335835344
    min_reward = -6.907755278982137 / 2
    wrong_format_penalty = -scale * 3.0

    def reward_function(confidence: int, is_answer_correct: bool) -> float:
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
    
    confidences = [extract_xml_confidence(completion) for completion in completions]
    confidences = [int(conf) if conf.isdigit() else None for conf in confidences]
    responses = ['<answer> ' + completion for completion in completions]
    predicted = [extract_xml_answer(r).lower() for r in responses]
    
    #print(responses, predicted, confidences, answer)
    
    correctness = trivia_qa_correctness_eval(predicted, answer, threshold=0.5)
    rewards = [reward_function(conf, c) for conf, c in zip(confidences, correctness)]
    
    return rewards


if __name__ == '__main__':
    
    print(compute_f1("hello world", "hello world"))
    print(compute_f1("hello world", "hello"))
    print(compute_f1("hello world", "new york"))