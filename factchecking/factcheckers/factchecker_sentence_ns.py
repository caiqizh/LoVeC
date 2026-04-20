import os
import time
from openai import OpenAI
import vertexai
from vertexai.generative_models import GenerativeModel
import nltk
from nltk.tokenize import sent_tokenize
from transformers import GPT2Tokenizer

# Download the NLTK tokenizer data (if not already downloaded)
nltk.download('punkt', quiet=True)

# Modified fact-checking prompt for sentences
INSTUCT_FACTCHECK = """Your task is to fact-check the following sentences extracted from a passage about a specific subject (e.g., a person, a place, an event, etc.).

For each sentence, if all the details in the sentence are supported by your knowledge and are factually correct, assign the veracity label 'S'. However, if even one detail is unfactual or you are unsure, assign the label 'NS'.

To assess a sentence, first analyze it and then append the veracity label enclosed in dollar signs ($). Use the following format:

### [Sentence]. Analysis: [Your detailed analysis]. $[S/NS]$

For example:
### Lebron James is a basketball player. Analysis: Lebron James is an American basketball player, so this is correct. $S$
### Obama was the 46th president of the United States. Analysis: While Obama is well-known, he was the 44th president, not the 46th, so this is incorrect. $NS$
### Jackie Chan was born on April 7, 1955. Analysis: Jackie Chan was born on April 7, 1954, so this is incorrect. $NS$
### ECC9876 is a great place to visit. Analysis: There is no reliable information confirming that ECC9876 is a great place to visit. $NS$

The sentences to evaluate are as follows:
{atomic_facts_string}
"""

def estimate_cost(input_tokens, output_tokens, model):
    if model == "gpt-4o":
        INPUT_COST_PER_MILLION = 2.5  # USD
        OUTPUT_COST_PER_MILLION = 10.00  # USD
    elif model == "gpt-4o-mini":
        INPUT_COST_PER_MILLION = 0.15
        OUTPUT_COST_PER_MILLION = 0.60
    elif model == "gemini-2.0-flash-001":
        INPUT_COST_PER_MILLION = 0.15
        OUTPUT_COST_PER_MILLION = 0.60
    elif model == "gemini-1.5-flash-002":
        INPUT_COST_PER_MILLION = 1.25
        OUTPUT_COST_PER_MILLION = 5.00
    else:
        raise ValueError(f"Unknown model: {model}")
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_MILLION
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_MILLION
    return input_cost + output_cost


def estimate_overall_cost(text, model):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    total_input_tokens = len(tokenizer.encode(text))
    total_output_tokens = total_input_tokens  # assuming output ≈ input
    return estimate_cost(total_input_tokens, total_output_tokens, model)


def get_completion_openai(user_prompt, retries=5, delay=2, backoff_factor=2):
    """
    Calls the OpenAI API with exponential backoff retry mechanism.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"OpenAI Error on attempt {attempt+1}: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= backoff_factor
    return None

def get_completion_gemini(user_prompt, retries=5, delay=2, backoff_factor=2):
    """
    Calls the Gemini API with exponential backoff retry mechanism.
    """
    PROJECT_ID = os.getenv("PROJECT_ID")
    vertexai.init(project=PROJECT_ID, location="us-central1")
    model = GenerativeModel("gemini-1.5-flash-002")
    for attempt in range(retries):
        try:
            response = model.generate_content(user_prompt)
            return response.text
        except Exception as e:
            print(f"Gemini Error on attempt {attempt+1}: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= backoff_factor
    return None

def get_factcheck_results(sentences_string, provider="openai"):
    """
    Formats the fact-checking prompt with the given sentences string and retrieves the fact-checking output.
    """
    prompt = INSTUCT_FACTCHECK.format(atomic_facts_string=sentences_string)
    if provider == "openai":
        raw_output = get_completion_openai(prompt)
    elif provider == "gemini":
        raw_output = get_completion_gemini(prompt)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    try:
        fact_check_results = [line.strip() for line in raw_output.split("###") if line.strip()]
        fact_check_labels = [line.split("$")[1] for line in fact_check_results]
    except Exception as e:
        print(f"Error processing output: {e}")
        fact_check_labels = []
    
    return raw_output, fact_check_labels

def main():
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
    os.environ["PROJECT_ID"] = "your_gemini_project_id_here"


    # Example paragraph input to fact-check
    paragraph = (
        "Bill Gates is born in 1955 in California. "
        "University of Cambridge is founded in 1208. "
        "The Great Wall of China is built in 7th century BC."
    )

    # Use NLTK to split the paragraph into sentences
    sentences = sent_tokenize(paragraph)

    # Assemble the sentences string ensuring each sentence begins with "###"
    sentences_str = "\n".join(["### " + sentence for sentence in sentences])
    
    print("\n--- Fact-checking Sentences using OpenAI ---")
    raw_output_openai, factcheck_results_openai = get_factcheck_results(sentences_str, provider="openai")
    print(raw_output_openai)
    print(factcheck_results_openai)

    print("\n--- Fact-checking Sentences using Gemini ---")
    raw_output_gemini, factcheck_results_gemini = get_factcheck_results(sentences_str, provider="gemini")
    print(raw_output_gemini)
    print(factcheck_results_gemini)

if __name__ == "__main__":
    main()
