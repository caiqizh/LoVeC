
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Set, Tuple, Union
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import pytz
import argparse
import nltk
from nltk.tokenize import sent_tokenize
import re
nltk.download('punkt_tab')

def remove_confidence_tags(text):
    return re.sub(r"<confidence>.*?</confidence>", "", text)

def remove_header(text):
    # Remove <|start_header_id|>assistant<|end_header_id|>\n\n
    return text.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "")

class LUQ_vllm:

    def __init__(
        self,
        nli_model: str,
        method: str,
        cuda_devices: str = "0",
        gpu_memory_utilization: float = 0.9,
    ):
        """
        nli_model: str - the name of the model to do NLI
        method: str - the method to use for the task, either "binary" or "multiclass"
        """

        if nli_model == "llama3.1-8b-instruct":
            model_path = "/home/cz391/rds/hpc-work/huggingface/meta-llama-Llama-3.1-8B-Instruct"
        else:
            raise ValueError("Model not supported")

        self.method = method
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(
            n=1,
            temperature=0,
            top_p=0.9,
            max_tokens=5,
            stop_token_ids=[self.tokenizer.eos_token_id],
            skip_special_tokens=True,
        )

        self.llm = LLM(
            model=model_path, 
            tensor_parallel_size=len(cuda_devices.split(",")),
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=2048, # For input prompts 
            )
        
        if self.method == "binary":
            self.prompt_template = (
                "Context: {context}\n\n"
                "Sentence: {sentence}\n\n"
                "Is the sentence entirely supported by the context above?\n\n"
                "You should answer the question purely based on the given context and not your own knowledge. "
                "Do not output the explanations.\n\n"
                "Your answer should be within \"yes\" or \"no\".\n\n"
                "Answer: "
            )
            self.text_mapping = {'yes': 1, 'no': 0, 'n/a': 0.5}

        elif self.method == "multiclass":
            self.prompt_template = (
                "Context: {context}\n\n"
                "Sentence: {sentence}\n\n"
                "Is the sentence supported, refuted or not mentioned by the context above?\n\n"
                "You should answer the question purely based on the given context and not your own knowledge. "
                "Do not output the explanations.\n\n"
                "Your answer should be within \"supported\", \"refuted\", or \"not mentioned\".\n\n"
                "Answer: "
            )
            self.text_mapping = {'supported': 1, 'refuted': 0, 'not mentioned': -1, 'n/a': 0.5}

        self.not_defined_text = set()

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template
    
    def completion(self, prompts: str):
        outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)
        return outputs

    def predict(
        self,
        sentences: List[str],
        sampled_passages: List[str],
    ):
        num_sentences = len(sentences)
        num_samples = len(sampled_passages)
        raw_scores = np.zeros((num_sentences, num_samples))
        
        for sent_i in range(num_sentences):
            prompts = []
            sentence = sentences[sent_i]
            for sample_i, sample in enumerate(sampled_passages):
                # this seems to improve performance when using the simple prompt template
                sample = sample.replace("\n", " ") 
                prompt_text = self.prompt_template.format(context=sample, sentence=sentence)
                # print(prompt_text)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_text}
                ]

                prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

                prompts.append(prompt)

            outputs = self.completion(prompts)

            for sample_i, output in enumerate(outputs):
                generate_text = output.outputs[0].text
                # print(generate_text)
                score_ = self.text_postprocessing(generate_text)
                # print(score_)
                raw_scores[sent_i, sample_i] = score_

        scores_filtered = np.ma.masked_equal(raw_scores, -1)
        scores_per_sentence = scores_filtered.mean(axis=-1)
        scores_per_sentence = np.where(scores_per_sentence.mask, 0, scores_per_sentence)
        # print(scores_per_sentence)
        return scores_per_sentence, raw_scores
        

    def text_postprocessing(
        self,
        text,
    ):
        """
        To map from generated text to score
        """

        if self.method == "binary":
            text = text.lower().strip()
            if text[:3] == 'yes':
                text = 'yes'
            elif text[:2] == 'no':
                text = 'no'
            else:
                if text not in self.not_defined_text:
                    print(f"warning: {text} not defined")
                    self.not_defined_text.add(text)
                text = 'n/a'
            return self.text_mapping[text]
        
        elif self.method == "multiclass":
            text = text.lower().strip()
            if text[:7] == 'support':
                text = 'supported'
            elif text[:5] == 'refut':
                text = 'refuted'
            elif text[:3] == 'not':
                text = 'not mentioned'
            else:
                if text not in self.not_defined_text:
                    print(f"warning: {text} not defined")
                    self.not_defined_text.add(text)
                text = 'n/a'
            return self.text_mapping[text]
            

if __name__ == "__main__":

    answer_path = "checkpoint-750-greedy.json"
    with open(answer_path, "r") as f:
        model_answers = json.load(f)
    
    samples_path = "results_t-1.0_p-1_k--1-n-10.json"
    with open(samples_path, "r") as f:
        model_samples = json.load(f)
    
    # for debugging
    # model_answers = model_answers[:10]
    # model_samples = model_samples[:10]
                            
    LUQ_vllm = LUQ_vllm(nli_model = "llama3.1-8b-instruct", method = "binary")

    # Load previously saved results if they exist
    temp_save_path = answer_path.replace(".json", f"-luq-abridged-temp.json")
    if os.path.exists(temp_save_path):
        with open(temp_save_path, "r") as f:
            answers_with_luq_scores = json.load(f)
        start_idx = len(answers_with_luq_scores)
        print(f"Resuming from index {start_idx}")
    else:
        answers_with_luq_scores = []
        start_idx = 0

    for idx, (answer, samples) in enumerate(tqdm(zip(model_answers[start_idx:], model_samples[start_idx:]), total=len(model_answers[start_idx:]))):

        # Remove the headers
        sentences = sent_tokenize(remove_confidence_tags(remove_header(answer["model_outputs"][0])))
        
        samples = [remove_header(sample) for sample in samples["model_outputs"]]

        try:
            scores_per_sentence, _ = LUQ_vllm.predict(sentences, samples)
        except Exception as e:
            print(f"Error at index {idx + start_idx}: {e}")
            scores_per_sentence = [None] * len(sentences)

        answer["luq_scores"] = scores_per_sentence.tolist()
        answers_with_luq_scores.append(answer)

        # Save progress after every iteration
        with open(temp_save_path, "w") as f:
            json.dump(answers_with_luq_scores, f, indent=4)

    # Final save
    with open(answer_path.replace(".json", "-luq-abridged.json"), "w") as f:
        json.dump(answers_with_luq_scores, f, indent=4)

    # Remove temporary file after completion
    if os.path.exists(temp_save_path):
        os.remove(temp_save_path)