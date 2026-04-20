from utils.data_utils import get_dataset
from utils.training_utils import load_config, save_config, extract_xml_answer, extract_xml_confidence
from argparse import ArgumentParser
import os, sys
import torch
from vllm import SamplingParams
from unsloth import FastLanguageModel, is_bfloat16_supported
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
import matplotlib.pyplot as plt

torch.manual_seed(42)  # You can choose any integer as your seed
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def compute_ece_metrics(confidences, all_correctness, name):
    # Define a metric name
    metric = name

    confidences = np.array(confidences)/10
    all_correctness = np.array(all_correctness)
    # Display the generated data

    df = pd.DataFrame({"Confidence": confidences, "Correctness": all_correctness})
    print(df.head())  # Print the first few rows for reference

    # Compute Mean Confidence
    mean_confidence = np.mean(confidences)
    print(f"\nMetric: {metric}")
    print(f"  Mean Confidence: {mean_confidence:.4f}")

    # Calculate Brier Score
    brier_score = np.mean((all_correctness - confidences) ** 2)
    print(f"  Brier Score: {brier_score:.4f}")

    # Calculate AUROC, handling potential errors
    try:
        auroc = roc_auc_score(all_correctness, confidences)
        print(f"  AUROC: {auroc:.4f}")
    except ValueError as e:
        print(f"  AUROC: Error ({e})")

    # Calculate PR-AUC, handling potential errors
    try:
        pr_auc = average_precision_score(all_correctness, confidences)
        print(f"  PR-AUC: {pr_auc:.4f}")
    except ValueError as e:
        print(f"  PR-AUC: Error ({e})")

    # Calculate Spearman Correlation
    spearman_corr, _ = spearmanr(all_correctness, confidences)
    print(f"  Spearman Correlation: {spearman_corr:.4f}")

    # Calculate Expected Calibration Error (ECE)
    ece = ECE(bins=10)
    ece_value = ece.measure(confidences.astype(np.float64), np.array(all_correctness, dtype=np.float64))
    print(f"  ECE: {ece_value:.4f}")

    # Generate Reliability Diagram
    diagram = ReliabilityDiagram(n_bins=10)
    fig = diagram.plot(confidences, all_correctness)  # visualize miscalibration
    fig.suptitle(f"Reliability Diagram for {metric}", y=1.05)

    # Save the plot and the metrics
    plt.savefig(f"{metric}_reliability_diagram.png")

def main():
    
    args = ArgumentParser()
    args.add_argument("--config", type=str, default=None, help="Path to config file")
    args.add_argument("--lora_path", type=str, default=None, help="Path to LoRA dataset")
    
    args = args.parse_args()
    
    cfg = load_config(args.config)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        fast_inference=cfg.fast_inference,
        max_lora_rank=cfg.lora_rank,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
    )
    lora = None
    
    if args.lora_path is not None:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_rank,
            target_modules=cfg.target_modules,
            lora_alpha=cfg.lora_rank,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        lora = model.load_lora(args.lora_path)
        print('using lora ' + args.lora_path)
            
    _, _, test_dataset = get_dataset("ecqa")
    
    sampling_params = SamplingParams( # Greedy decoding
        max_tokens=32,  # Maximum number of tokens to generate
        temperature=0,  # Temperature of 0 means no randomness (deterministic)
        top_p=1.0,      # Consider all tokens in the distribution
        top_k=1         # Only select the most likely token at each step
    )
    
    vllm_prompts = [x['prompt'] for x in test_dataset]
    
    outputs = model.fast_generate(
        vllm_prompts,
        sampling_params = sampling_params,
        lora_request = lora
    )
    
    all_correctness = []
    confidences = []
    wrong = 0
    for out, x in zip(outputs, test_dataset):
        response = '<answer> ' + out.outputs[0].text
        predicted = extract_xml_answer(response).lower()
        all_correctness.append(1 if x['answer'].lower() in predicted else 0)
        try:
            confidences.append(float(extract_xml_confidence(response)))
        except:
            
            confidences.append(0)
            wrong += 1
            print(extract_xml_confidence(response))
    print(sum(all_correctness) / len(all_correctness))
    print(wrong)
    
    output_name = cfg.model_name.replace("/", "_")
    if args.lora_path is not None:
        output_name += "_lora_" + os.path.basename(args.lora_path).split("/")[-1]
    
    compute_ece_metrics(confidences, all_correctness, output_name)
    
if __name__ == "__main__":
    main()