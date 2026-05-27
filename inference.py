"""
Multilingual LLM Reasoning Evaluation Pipeline.
Loads GPQA and MGSM datasets from Hugging Face and performs batched inference using vLLM.
"""

import os
import json
from datasets import load_dataset
from vllm import LLM, SamplingParams

def initialize_llm():
    """
    Initializes the vLLM engine with the Qwen2.5-7B-Instruct-AWQ model.
    Returns the initialized LLM and SamplingParams objects configured for zero-shot CoT.
    """
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct-AWQ",
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.9
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["</s>", "<|im_end|>"]
    )
    return llm, sampling_params

def evaluate_gpqa(llm, sampling_params, output_base_dir):
    """
    Evaluates the Multilingual GPQA dataset across 17 language splits.
    Saves output JSON files containing the question, true answer, and model generation.
    """
    repo_id = "suryaat19/Multilingual_gpqa"
    splits = ['en', 'es', 'fr', 'de', 'ru', 'zh_cn', 'ja', 'th', 'sw', 'bn', 'te', 'ar', 'ko', 'sr', 'hu', 'vi', 'cs']
    output_dir = os.path.join(output_base_dir, "GPQA_Eval_Results")
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        dataset = load_dataset(repo_id, split=split)
        prompts = []
        
        for row in dataset:
            prompt = f"System prompt: Always think step by step and give your final choice among (A), (B), (C) and (D) by Answer: [Your Choice] in a single last line.\n\nUser prompt: What is the correct answer to this question: {row['Question']}\nChoices:\n(A) {row['Correct Answer']}\n(B) {row['Incorrect Answer 1']}\n(C) {row['Incorrect Answer 2']}\n(D) {row['Incorrect Answer 3']}\nLet's think step by step:\n"
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)
        results = []
        
        for i, output in enumerate(outputs):
            results.append({
                "question": dataset[i]['Question'],
                "correct_answer_text": dataset[i]['Correct Answer'],
                "generated_text": output.outputs[0].text
            })

        file_path = os.path.join(output_dir, f"{split}_results.json")
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

def evaluate_mgsm(llm, sampling_params, output_base_dir):
    """
    Evaluates the Multilingual MGSM dataset across 17 language splits.
    Saves output JSON files containing the question, numerical true answer, and model generation.
    """
    repo_id = "suryaat19/Multilingual_mgsm"
    splits = ['en', 'bn', 'de', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh', 'ar', 'ko', 'sr', 'hu', 'vi', 'cs']
    output_dir = os.path.join(output_base_dir, "MGSM_Eval_Results")
    os.makedirs(output_dir, exist_ok=True)

    for split in splits:
        dataset = load_dataset(repo_id, split=split)
        prompts = []
        
        for row in dataset:
            prompt = f"System prompt: You are a mathematical reasoning assistant. Solve the problem step by step. Always output your final numerical answer on the last line in the format 'Final Answer: [number]'.\n\nUser prompt: {row['question']}\n\nLet's think step by step:\n"
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)
        results = []
        
        for i, output in enumerate(outputs):
            results.append({
                "question": dataset[i]['question'],
                "true_answer_number": dataset[i]['answer_number'],
                "generated_text": output.outputs[0].text
            })

        file_path = os.path.join(output_dir, f"{split}_results.json")
        with open(file_path, "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    """
    Main execution pipeline. Sets output directory to Google Drive, 
    initializes the model, and sequentially runs GPQA and MGSM evaluations.
    """
    output_base_dir = "/content/drive/MyDrive"
    llm, sampling_params = initialize_llm()
    
    evaluate_gpqa(llm, sampling_params, output_base_dir)
    evaluate_mgsm(llm, sampling_params, output_base_dir)

if __name__ == "__main__":
    main()