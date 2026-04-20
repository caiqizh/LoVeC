import os
import json
from datasets import load_dataset, Dataset
import pandas as pd

ecqa_prompt = '''Answer the following question: {question} Choices: {choices}. Report confidence level on a scale of 0 (uncertain) to 10 (extremely confident). Answer in the format: <answer> choice </answer> <confidence> level </confidence>. 
<answer>'''

# triviaqa_prompt = '''Answer the following question: {question}. Report confidence level on a scale of 0 (uncertain) to 10 (extremely confident). Answer in the format: <answer> choice </answer> <confidence> level </confidence>.
# <answer>'''

triviaqa_prompt = '''You will get questions. Answer with the correct answer. Additionally provide a confidence between 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, of how sure you are the answer is correct. A value close to 0 means you think there is a high probability that the answer is wrong. The closer the value is to 10, the higher you think is the probability that the answer is correct. The output should have the format Answer in the format: <answer> choice </answer> <confidence> level </confidence> and nothing else. {question} <answer>'''

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

wildhallucination_prompt = '''In a paragraph, could you tell me what you know about {entity}?'''


def get_dataset(file_path, tokenizer=None, chat_format=False, with_instruction=False):
    
    if 'ecqa' in file_path:
        ds = load_dataset("yangdong/ecqa")

        # Function to add prompt to each example
        def add_prompt(example):
            # Construct choices string
            choices = [example['q_op{i}'.format(i=i)] for i in range(1, 6)]
            
            # Format the prompt
            example['prompt'] = ecqa_prompt.format(
                question=example['q_text'],
                choices=", ".join(choices)
            )
            example['answer'] = example['q_ans']
            return example

        # Process each split
        train_dataset = ds['train'].map(add_prompt)
        validation_dataset = ds['validation'].map(add_prompt)
        test_dataset = ds['test'].map(add_prompt)

        # Preview a sample from the training set
        print("Training set sample:")
        print(train_dataset[0])

        # Return the processed datasets
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(validation_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
            
    elif ('bios' in file_path) or ('popqa' in file_path): #test only dataset
        with open(file_path, "r") as f:
            data = json.load(f)
        
        for d in data:

            if 'popqa' in file_path:
                message = wildhallucination_prompt.format(entity=d['wikipedia_page'])
            else:
                message = wildhallucination_prompt.format(entity=d['entity'])
            d['prompt'] = message
        
        df = pd.DataFrame([{
            "entity": d['wikipedia_page'] if 'popqa' in file_path else d['entity'],
            "prompt": d['prompt'],
        } for d in data])

        # Convert the DataFrame to a Dataset
        train_dataset = Dataset.from_dict({'prompt': [], 'entity': []})
        # Create empty datasets for validation and test
        validation_dataset = Dataset.from_dict({'prompt': [], 'entity': []})
        test_dataset = Dataset.from_pandas(df)
        
        print("Test set sample:")
        print(test_dataset[0])

        # Print dataset sizes
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(validation_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
            
        return train_dataset, validation_dataset, test_dataset
    
    elif 'triviaqa' in file_path:
        ds = load_dataset("mandarjoshi/trivia_qa", "rc")
        
        def add_prompt(example):
            # Format the prompt
            example['prompt'] = triviaqa_prompt.format(
                question=example['question'],
            )
            example['answer'] = example['answer']['normalized_aliases']
            return example
        
        # Process each split
        columns_to_keep = ['question', 'answer', 'prompt', 'question_id']
        remove_columns = [col for col in ds['train'].column_names if col not in columns_to_keep]
        
        train_dataset = ds['train'].map(add_prompt, remove_columns=remove_columns)
        validation_dataset = ds['validation'].map(add_prompt, remove_columns=remove_columns)
        test_dataset = ds['test'].map(add_prompt, remove_columns=remove_columns)
        
    elif 'wildhallucination' in file_path:
        ds = load_dataset("wentingzhao/WildHallucinations")
        
        def add_prompt(example):
            # Format the prompt
            example['prompt'] = wildhallucination_prompt.format(
                entity=example['entity'],
            )
            return example
        
        columns_to_keep = ['entity', 'prompt']
        # Process the dataset and keep only required columns
        remove_columns = [col for col in ds['train'].column_names if col not in columns_to_keep]
        processed_dataset = ds['train'].map(add_prompt, remove_columns=remove_columns)

        # Split the dataset into train, validation, and test with 8:1:1 ratio
        dataset_dict = processed_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset_dict['train']
        temp_dataset = dataset_dict['test'].train_test_split(test_size=0.5, seed=42)
        validation_dataset = temp_dataset['train']
        test_dataset = temp_dataset['test']

    elif 'grpo' in file_path:
        # Let's just suppose give me a json file with the format:
        # [{'entity': '...', 'evidence': '...', ...]
        # and I don't know why he is putting test split in a separate file, and there is no validation split
        # So I will just load the json file and return empty list for validation and test
        # If you want to use the test split, you can just load it separately

        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Create a DataFrame from the data
        if 'bios' in file_path: # temporary fix for bios dataset, should give me evidence in sane length < (8192 - 5) : (tokenizer.model_max_length - evaluation_max_length)
            for d in data:
                evidence = d['wikipedia_page']
                tokenized_evidence = tokenizer(evidence, truncation=True, max_length=4096, return_tensors='pt')
                evidence = tokenizer.decode(tokenized_evidence['input_ids'][0], skip_special_tokens=True)
                d['evidence'] = evidence
                d.pop('wikipedia_page', None)
        elif 'cleaned' in file_path: # temporary fix for cleaned dataset
            pass # cleaned

        # add additional key as prompt for each entity in the dataframe
        for d in data:
            if chat_format:
                message = []
                
                if 'gemma' in tokenizer.name_or_path:
                    if with_instruction:
                        message.append({'role': 'user', 'content': long_form_confidence_prompt + wildhallucination_prompt.format(entity=d['entity'])})
                    else:
                        message.append({'role': 'user', 'content': wildhallucination_prompt.format(entity=d['entity'])})
                    d['prompt'] = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                else:
                    if with_instruction:
                        message.append({'role': 'system', 'content': long_form_confidence_prompt})
                    message.append({'role': 'user', 'content': wildhallucination_prompt.format(entity=d['entity'])})

                    d['prompt'] = tokenizer.apply_chat_template(message, tokenize=False)
            else:
                message = []
                if with_instruction:
                    message.append(long_form_confidence_prompt)
                message.append(wildhallucination_prompt.format(entity=d['entity']))
                message = ' '.join(message)
                d['prompt'] = message

        df = pd.DataFrame([{
            "entity": d['entity'],
            "evidence": d['evidence'],
            "prompt": d['prompt'],
            "base_prediction": d['base_prediction'],
        } for d in data])

        # Convert the DataFrame to a Dataset
        train_dataset = Dataset.from_pandas(df)
        # Create empty datasets for validation and test
        validation_dataset = Dataset.from_dict({'prompt': [], 'entity': [], 'evidence': [], 'base_prediction': []})
        test_dataset = Dataset.from_dict({'prompt': [], 'entity': [], 'evidence': [], 'base_prediction': []})

    # Preview a sample from the training set
    print("Training set sample:")
    print(train_dataset[0])

    # Print dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
        
    return train_dataset, validation_dataset, test_dataset

if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    file_path = "your_dataset_path_here.json"  # Replace with your dataset path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, validation_dataset, test_dataset = get_dataset(file_path, tokenizer=tokenizer)