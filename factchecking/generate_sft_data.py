import os
import json
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from factchecker_sentence_num import get_factcheck_results, estimate_overall_cost
from abstrain_detection import is_response_abstained

os.environ["OPENAI_API_KEY"] = ""

# Load the data
# input_path = "sft-data/predictions.json"
# output_path = "sft-data/predictions_with_ratings.json"

input_path = "../baselines/checkpoint-750-greedy.json"
output_path = "../baselines/checkpoint-750-greedy-with-ratings.json"

with open(input_path, "r") as file:
    data = json.load(file)

# Estimate the cost
all_text = "\n".join([item["model_output"] for item in data])
cost = estimate_overall_cost(all_text, model="gpt-4o")
print(f"Estimated cost: {cost}")

# Ask for confirmation
response = input("Do you want to continue? (y/n): ")
if response.lower() != "y":
    print("Exiting...")
    exit()

sft_data = []
abstained_count = 0

# Load previous progress if exists
if os.path.exists(output_path):
    with open(output_path, "r") as file:
        sft_data = json.load(file)

# Keep track of already processed items by some ID (fallback to index if no ID)
processed_indices = set(range(len(sft_data)))

for idx, item in enumerate(tqdm(data)):
    if idx in processed_indices:
        continue

    paragraph = item["model_output"].replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "")

    if is_response_abstained(paragraph, "generic"):
        abstained_count += 1
        print(f"Response abstained, {abstained_count} in total")
        item["is_abstained"] = True
    else:
        item["is_abstained"] = False
        sentences = sent_tokenize(paragraph)
        sentences_str = "\n".join(["### " + sentence for sentence in sentences])

        raw_output_openai, factcheck_results_openai = get_factcheck_results(sentences_str, provider="openai")
        item["ratings_openai"] = factcheck_results_openai
        item["raw_output_openai"] = raw_output_openai

        # Combine the rated sentences back into a paragraph
        rated_sentences = [
            f"{sentence} <confidence> {rating} </confidence>"
            for sentence, rating in zip(sentences, factcheck_results_openai)
        ]
        item["rated_output_openai"] = " ".join(rated_sentences)

    # Save progress after each item
    sft_data.append(item)
    with open(output_path, "w") as f:
        json.dump(sft_data, f, indent=4)

print("All results saved.")
