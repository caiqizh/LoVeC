import re
from typing import List, Tuple, Union
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import nltk
from nltk.tokenize import sent_tokenize
from openai import OpenAI

class ServerJudge:
    def __init__(self, api_key="EMPTY", api_base="http://localhost:8000/v1",):
        """
        Initializes the ServerJudge with the OpenAI client.

        Parameters:
          api_key: The API key for the OpenAI server.
          api_base: The base URL for the OpenAI server.
        """
        # Modify OpenAI's API key and API base to use vLLM's API server.
        self.client = OpenAI(
            # defaults to os.environ.get("OPENAI_API_KEY")
            api_key=api_key,
            base_url=api_base,
        )
        self.model_name = self.client.models.list().data[0].id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def create_completion(self, prompt, max_tokens: int = 512, temperature: float = 0, top_p: float = 1, stream: bool = False):
        """
        Creates a completion using the OpenAI client.

        Parameters:
          model: The model ID to use for the completion.
          prompt: The prompt to send to the model.
          max_tokens: The maximum number of tokens to generate.
          temperature: Sampling temperature.
          top_p: Nucleus sampling probability.
          stream: Whether to stream the response.

        Returns:
          The completion response.
        """
        return self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            echo=False,
            stream=stream,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def evaluate_correctness(
        self,
        paragraphs: List[str],
        evidences: List[str],
        tagging_method: str = "binary"  # "binary" or "numerical"
    ) -> List[List[Tuple[str, Union[bool, int]]]]:
        """
        Evaluates each sentence in a batch of paragraphs for its support by the corresponding evidence.
        
        Parameters:
        llm: An instance of the vLLM model.
        model_path: Path to the LLM model (used to initialize the tokenizer).
        paragraphs: A batch (list) of paragraphs; each paragraph is a string.
        evidences: A batch (list) of evidence passages; each string corresponds to a paragraph.
        tagging_method: "binary" for yes/no tagging (returns bool),
                        "numerical" for degree tagging (returns int on a scale from 0 to 10).
        
        Returns:
        A list of lists; each inner list consists of tuples (sentence, correctness).
        For binary tagging, correctness is True/False.
        For numerical tagging, correctness is an integer (0, 5, or 10 ideally).
        
        Notes:
        - All prompts are batched into one single call to llm.generate for efficiency.
        - The context is clearly separated using delimiters, which is especially useful for long contexts.
        """
        # Initialize tokenizer.

        # Define prompt templates for binary and numerical tagging.
        prompt_template_binary = (
            "Task: Determine whether the following sentence is entirely supported by the given context. "
            "Answer with \"yes\" if it is fully supported, or \"no\" if it is not. "
            "Do not provide any additional explanation.\n\n"
            "You should answer the question purely based on the given context and not your own knowledge.\n\n"
            "===== CONTEXT START =====\n"
            "{context}\n"
            "===== CONTEXT END =====\n\n"
            "===== SENTENCE START =====\n"
            "{sentence}\n"
            "===== SENTENCE END =====\n\n"
            "Answer: "
        )

        prompt_template_numerical = (
            "Task: Evaluate how well the following sentence is supported by the provided context.\n"
            "Assign a rating from 0 to 10 based solely on the context, using the following scale:\n"
            "  0   = Not supported at all (no relevant evidence or context contradicts the sentence)\n"
            " 1–3  = Weakly supported (very little or indirect support)\n"
            " 4–6  = Moderately supported (some relevant support, but with gaps or ambiguity)\n"
            " 7–9  = Strongly supported (mostly well-supported with minor uncertainty)\n"
            " 10   = Fully supported (clear, direct, and complete support)\n\n"
            "Important: Respond with only the numerical rating—no explanation or extra text.\n\n"
            "----- CONTEXT -----\n"
            "{context}\n"
            "-------------------\n\n"
            "----- SENTENCE -----\n"
            "{sentence}\n"
            "--------------------\n\n"
            "Answer: "
        )

        if tagging_method == "binary":
            prompt_template = prompt_template_binary
        elif tagging_method == "numerical":
            prompt_template = prompt_template_numerical
        else:
            raise ValueError("Unsupported tagging method. Use 'binary' or 'numerical'.")

        # Tokenize the paragraphs into sentences.
        tokenized_paragraphs = []
        for paragraph in paragraphs:
            sentences = sent_tokenize(paragraph)
            tokenized_paragraphs.append(sentences)

        # Accumulate all prompts and record the count per paragraph.
        all_prompts = []
        prompt_counts = []  # Tracks number of prompts (sentences) per paragraph.
        for paragraph_sentences, evidence in zip(tokenized_paragraphs, evidences):
            count = 0
            clean_evidence = evidence.replace("\n", " ")
            for sentence in paragraph_sentences:
                prompt = prompt_template.format(context=clean_evidence, sentence=sentence)
                all_prompts.append(prompt)
                count += 1
            prompt_counts.append(count)

        # For debugging: Uncomment the following line to see all prompts.
        # print(all_prompts)
        
        # Generate responses in one single call.
        outputs = self.create_completion(all_prompts, max_tokens=5, temperature=0, top_p=1, stream=False)
        generated_texts = [o.text for o in outputs.choices]
        #outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

        # Reassemble outputs into the paragraph structure.
        results = []
        current_index = 0
        for count, sentences in zip(prompt_counts, tokenized_paragraphs):
            paragraph_results = []
            for sentence in sentences:
                #output = outputs[current_index]
                #generated_text = output.outputs[0].text
                generated_text = generated_texts[current_index]
                answer = generated_text.lower().strip()
                if tagging_method == "binary":
                    if answer.startswith("yes"):
                        correctness = True
                    elif answer.startswith("no"):
                        correctness = False
                    else:
                        correctness = False
                elif tagging_method == "numerical":
                    # Attempt to extract a number from the answer.
                    match = re.search(r'\d+', answer)
                    if match:
                        try:
                            value = int(match.group())
                        except ValueError:
                            value = 0
                    else:
                        value = 0
                    correctness = value
                paragraph_results.append((sentence, correctness))
                current_index += 1
            results.append(paragraph_results)
        
        return results

# ===========================
# Test cases for the function
# ===========================
if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    # Test Case 1: Geographical facts.
    paragraphs1 = [
        "The capital of France is Paris. The Eiffel Tower is in Berlin."
    ]
    evidences1 = [
        "Paris is the capital of France and the Eiffel Tower is located in Paris."
    ]
    
    # Test Case 2: Programming language and comedy group.
    paragraphs2 = [
        "Python is a programming language. Monty Python is a comedy group."
    ]
    evidences2 = [
        "Python is a high-level programming language. Monty Python is a British surreal comedy group."
    ]

    # Test Case 3: Counterfactual statements.
    paragraphs3 = [
        "The capital of France is Paris. The Eiffel Tower is in Berlin."
    ]
    evidences3 = [
        "The capital of France is London. The Eiffel Tower is located in Berlin."
    ]
    
    # Combine test cases into batches.
    all_paragraphs = paragraphs1 + paragraphs2 + paragraphs3
    all_evidences = evidences1 + evidences2 + evidences3

    # print("=== Binary Tagging ===")
    # try:
    #     binary_results = evaluate_correctness(llm_instance, model_path, all_paragraphs, all_evidences, tagging_method="binary")
    #     for idx, paragraph in enumerate(binary_results):
    #         print(f"Paragraph {idx+1}:")
    #         for sentence, correctness in paragraph:
    #             print(f"  Sentence: \"{sentence}\" -> Correctness: {correctness}")
    # except Exception as e:
    #     print("Error (binary tagging):", str(e))
    evaluator = ServerJudge()
    print("\n=== Numerical Tagging ===")
    try:
        numerical_results = evaluator.evaluate_correctness(all_paragraphs, all_evidences, tagging_method="numerical")
        for idx, paragraph in enumerate(numerical_results):
            print(f"Paragraph {idx+1}:")
            for sentence, support_level in paragraph:
                print(f"  Sentence: \"{sentence}\" -> Support Level: {support_level}")
    except Exception as e:
        print("Error (numerical tagging):", str(e))
