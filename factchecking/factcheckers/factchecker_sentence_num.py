import os
import re
import time
from openai import OpenAI
import vertexai
from vertexai.generative_models import GenerativeModel
import nltk
from nltk.tokenize import sent_tokenize
from transformers import GPT2Tokenizer
from openai import AzureOpenAI

# Download the NLTK tokenizer data (if not already downloaded)
nltk.download('punkt', quiet=True)

# Modified fact-checking prompt for sentences with numeric ratings
INSTUCT_FACTCHECK = """Your task is to fact-check each of the following sentences.

Each sentence may contain multiple factual claims. For each one:
1. Break down and assess the factual accuracy of each individual detail.
2. Consider names, dates, locations, statistics, events, and attributions.
3. Assign a numeric **veracity rating** from 0 to 10 based on this scale:

0: Completely incorrect — entirely false or misleading.  
1–3: Mostly incorrect — several key inaccuracies.  
4–6: Partially correct — mix of accurate and inaccurate information.  
7–9: Mostly correct — generally accurate with minor issues.  
10: Completely correct — all facts are verifiably accurate.

Use the following format for your output (do **not** repeat the sentence):

**Analysis:** [Your detailed factual analysis]  
**Rating:** $[0–10]$

---

**Example Inputs:**  
### Marie Curie won two Nobel Prizes, one in Physics in 1903 and another in Chemistry in 1911 for her work on radioactivity.  
### The Great Fire of London occurred in 1666 and destroyed nearly half of the city’s modern skyscrapers.  
### Albert Einstein developed the theory of relativity while working as a professor at the University of Zurich and received the Nobel Prize in Physics in 1921 for this work.  
### Mount Everest, located on the border between Nepal and India, is the second-highest mountain in the world after K2.

**Example Outputs:**  
**Analysis:** Marie Curie received the Nobel Prize in Physics in 1903 (shared with Pierre Curie and Henri Becquerel) and the Nobel Prize in Chemistry in 1911 for discovering polonium and radium. The statement is entirely accurate.  
**Rating:** $10$

**Analysis:** While the date of the fire is correct, the mention of "modern skyscrapers" is anachronistic and factually incorrect. Skyscrapers did not exist in 1666.  
**Rating:** $2$

**Analysis:** Einstein did work at the University of Zurich and received the Nobel Prize in 1921, but it was awarded for his explanation of the photoelectric effect, not for the theory of relativity.  
**Rating:** $6$

**Analysis:** Mount Everest is located between Nepal and the Tibet Autonomous Region of China, not India. Additionally, it is the highest mountain in the world, not the second-highest.  
**Rating:** $1$

---

Now evaluate the following sentences. You should only output the analysis and rating for each sentence:

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
    return input_cost + output_cost * 1.5  # 50% extra for output

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


def get_completion_openai_tencent(user_prompt, retries=5, delay=2, backoff_factor=2):
    """
    Calls the Tencent variant of OpenAI (AzureOpenAI) API with an exponential backoff retry mechanism.
    """
    client = AzureOpenAI(
        azure_endpoint="https://text-embedding-3-small-ailab.openai.azure.com/", 
        api_key=os.getenv("OPENAI_API_KEY_TENCENT"),  
        api_version="2024-02-01"
    )
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # model or deployment name
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Tencent OpenAI Error on attempt {attempt+1}: {e}. Retrying in {delay} seconds...")
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
    Formats the fact-checking prompt with the given sentences and retrieves the fact-checking output.
    Parses the output to extract numeric ratings for each sentence.
    
    Args:
        sentences_string (str): Sentences formatted with "### " prefixes.
        provider (str): 'openai' or 'gemini'.
    
    Returns:
        tuple: (raw_output, list of extracted integer ratings or None)
    """
    prompt = INSTUCT_FACTCHECK.format(atomic_facts_string=sentences_string)

    # Get model output
    if provider == "openai":
        raw_output = get_completion_openai(prompt)
    elif provider == "gemini":
        raw_output = get_completion_gemini(prompt)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not raw_output:
        print("No output received from model.")
        return "", []

    # Split into analyses
    fact_check_results = raw_output.split("**Analysis:**")[1:]
    fact_check_ratings = []

    for result in fact_check_results:
        try:
            # Match the pattern for **Rating:** $[0–10]$
            match = re.search(r"\*\*Rating:\*\*\s*\$?(\d{1,2})\$?", result)
            if match:
                rating = int(match.group(1))
                fact_check_ratings.append(rating)
            else:
                print(f"Could not extract rating from result:\n{result}")
                fact_check_ratings.append(None)
        except Exception as e:
            print(f"Error processing result:\n{result}\nException: {e}")
            fact_check_ratings.append(None)

    return raw_output, fact_check_ratings


def main():
    # Set your API keys and project id
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
    os.environ["PROJECT_ID"] = "your_gemini_project_id_here"


    # Example paragraph input to fact-check
    paragraph = """The University of Cambridge was founded in 1209 and is the second-oldest university in the English-speaking world.
    Cambridge is located in Oxford, England, and has produced over 50 Nobel Prize winners.
    Stephen Hawking served as the Lucasian Professor of Mathematics at Cambridge from 1979 to 2009.
    The university’s main language of instruction is Latin.
    Cambridge University Press is one of the oldest and largest academic publishers in the world."
    """

    # Use NLTK to split the paragraph into sentences
    sentences = sent_tokenize(paragraph)

    # Assemble the sentences string ensuring each sentence begins with "###"
    sentences_str = "\n".join(["### " + sentence for sentence in sentences])
    
    print("\n--- Fact-checking Sentences using OpenAI ---")
    raw_output_openai, factcheck_ratings_openai = get_factcheck_results(sentences_str, provider="openai")
    print(raw_output_openai)
    print("Ratings:", factcheck_ratings_openai)

    # print("\n--- Fact-checking Sentences using Gemini ---")
    # raw_output_gemini, factcheck_ratings_gemini = get_factcheck_results(sentences_str, provider="gemini")
    # print(raw_output_gemini)
    # print("Ratings:", factcheck_ratings_gemini)

    print("\n--- Fact-checking Sentences using Tencent OpenAI ---")
    raw_output_tencent, factcheck_ratings_tencent = get_factcheck_results(sentences_str, provider="tencent")
    print(raw_output_tencent)
    print("Ratings:", factcheck_ratings_tencent)

if __name__ == "__main__":
    main()
