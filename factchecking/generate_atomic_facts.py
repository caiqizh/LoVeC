import os
import json
import time
from openai import OpenAI
from transformers import GPT2Tokenizer
import vertexai
from vertexai.generative_models import GenerativeModel



prompt_template = """
Please breakdown the following passage into independent fact pieces. 

Step 1: For each sentence, you should break it into several fact pieces. Each fact piece should ONLY contain ONE single independent fact. For example, "Bill Gates is born in 1955 in Seattle, Washington." should be broken down into "Bill Gates is born in 1955." and "Bill Gates is born in Seattle, Washington."

Step 2: Do this for all the sentences. Output each piece of fact in one single line starting with ###. Do not include other formatting. 

Step 3: Each atomic fact should be self-contained. Do not use pronouns as the subject of a piece of fact, such as he, she, it, this that, use the original subject whenever possible.

Here are some examples:

Example 1:
Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969.
### Michael Collins was born on October 31, 1930.
### Michael Collins is retired.
### Michael Collins is an American.
### Michael Collins was an astronaut.
### Michael Collins was a test pilot.
### Michael Collins was the Command Module Pilot.
### Michael Collins was the Command Module Pilot for the Apollo 11 mission.
### Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969.

Example 2:
League of Legends (often abbreviated as LoL) is a multiplayer online battle arena (MOBA) video game developed and published by Riot Games. 
### League of Legends is a video game.
### League of Legends is often abbreviated as LoL.
### League of Legends is a multiplayer online battle arena.
### League of Legends is a MOBA video game.
### League of Legends is developed by Riot Games.
### League of Legends is published by Riot Games.

Example 3:
Emory University has a strong athletics program, competing in the National Collegiate Athletic Association (NCAA) Division I Atlantic Coast Conference (ACC). The university's mascot is the Eagle.
### Emory University has a strong athletics program.
### Emory University competes in the National Collegiate Athletic Association Division I.
### Emory University competes in the Atlantic Coast Conference.
### Emory University is part of the ACC.
### Emory University's mascot is the Eagle.

Now it's your turn. Here is the passage: 

{}

You should only return the final answer. Now your answer is:
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
                model="gpt-4o-mini",
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

def get_atomic_facts_from_text(text, provider="openai"):
    prompt = prompt_template.format(text)

    if provider == "openai":
        raw_output = get_completion_openai(prompt)
    elif provider == "gemini":
        raw_output = get_completion_gemini(prompt)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if raw_output is None:
        return []

    facts = [line.strip() for line in raw_output.split("###") if line.strip()]
    return facts



def main():
    os.environ["OPENAI_API_KEY"] = "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"
    os.environ["PROJECT_ID"] = "quixotic-galaxy-430215-t9"

    test_input = (
        "The University of Cambridge is a collegiate research university in Cambridge, United Kingdom. "
        "Founded in 1209 and granted a royal charter by King Henry III in 1231, Cambridge is the second-oldest university in the English-speaking world and the world's fourth-oldest surviving university. "
        "The university grew out of an association of scholars who left the University of Oxford after a dispute with the townspeople. "
    )

    print("\n--- Using OpenAI ---")
    estimated_cost = estimate_overall_cost(test_input, "gpt-4o-mini")
    print(f"Estimated cost: USD{estimated_cost:.4f}")
    atomic_facts_openai = get_atomic_facts_from_text(test_input, provider="openai")
    for fact in atomic_facts_openai:
        print(f"### {fact}")

    print("\n--- Using Gemini ---")
    estimated_cost = estimate_overall_cost(test_input, "gemini-2.0-flash-001")
    print(f"Estimated cost: USD{estimated_cost:.4f}")
    atomic_facts_gemini = get_atomic_facts_from_text(test_input, provider="gemini")
    for fact in atomic_facts_gemini:
        print(f"### {fact}")



if __name__ == "__main__":
    main()
