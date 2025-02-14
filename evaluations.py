import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from peft import get_peft_model, LoraConfig, TaskType
import re
import json
from torch.cuda.amp import autocast
from utils.metrics import get_hp_accuracy, get_mmlu_accuracy  
import lm_eval
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from peft import PeftModel
from main_hp import calculate_perplexity
import logging
from datasets import load_dataset
import pickle
import pandas as pd
import zlib
import argparse
import datasets
os.environ['GIT_DISCOVERY_ACROSS_FILESYSTEM'] = '1'
logging.getLogger("transformers").setLevel(logging.ERROR)

seq_length = 128
stride = 127

# DMRA
def compress_list(original_list, target_size=500):
    n = len(original_list)
    group_size = n // target_size
    new_list = ["".join(map(str, original_list[i:i + group_size])) for i in range(0, n, group_size)]
    # Ensure the new list size is exactly target_size (handle remainder cases)
    return new_list[:target_size]

def compute_DM_document(documents, model, tokenizer, context_length, device):
    """
    Calculates the DRMA score for a given list of documents.

    Parameters:
    - documents: a list of strings representing the document content
    - model: a pre-trained model
    - tokenizer: a tokenizer
    - device: a PyTorch device (e.g., 'cuda' or 'cpu')
    - context_length: the maximum sequence length supported by the model
    """
    model.eval()
    num_docs = len(documents)
    all_prob_list = []
    for document in documents:
        chunks = [document[i:i + context_length] for i in range(0, len(document), context_length)]
        all_prob = []
        for chunk in chunks:
            input_ids = tokenizer.encode(chunk, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            input_ids_processed = input_ids[0][1:]
            for i, token_id in enumerate(input_ids_processed):
                probability = probabilities[0, i, token_id].item()
                all_prob.append(probability)
        all_prob_list.append(sum(all_prob))
    DM = sum(all_prob_list) / num_docs
    return DM

def compute_DM_file_document(folder_path, model, tokenizer, context_length, device):
    """
    Calculates the DRMA score for all txt files in a given folder.

    Parameters:
    - folder_path: string, the folder path containing multiple txt files
    - model: a pre-trained model
    - tokenizer: a tokenizer
    - context_length: the maximum sequence length supported by the model
    - device: a PyTorch device (e.g., 'cuda' or 'cpu')
    """
    model.eval()
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    all_prob_list = []
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            document = f.read()
        chunks = [document[i:i + context_length] for i in range(0, len(document), context_length)]
        all_prob = []
        for chunk in chunks:
            input_ids = tokenizer.encode(chunk, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            input_ids_processed = input_ids[0][1:]
            for i, token_id in enumerate(input_ids_processed):
                probability = probabilities[0, i, token_id].item()
                all_prob.append(probability)
        all_prob_list.append(sum(all_prob))
    DM = sum(all_prob_list) / len(txt_files)
    return DM

def load_documents(document_file_path):
    """Load document data."""
    documents = []
    for filename in os.listdir(document_file_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(document_file_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(content)
    return documents

def calculatePerplexity(sentence, model, tokenizer, device):
    """
    Calculates the perplexity of a sentence.
    """
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    logits = outputs.logits
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()

def compute_baselines(text, model1, model2, tokenizer1, tokenizer2, device1, device2):
    """
    Computes baseline metrics.
    """
    baseline_scores = {}
    p1, all_prob, p1_likelihood = calculatePerplexity(text, model1, tokenizer1, device1)
    p_ref, all_prob_ref, p_ref_likelihood = calculatePerplexity(text, model2, tokenizer2, device2)
    baseline_scores["ppl"] = p1
    baseline_scores["ppl/Ref_ppl"] = p1_likelihood - p_ref_likelihood
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    baseline_scores["ppl/zlib"] = np.log(p1) / zlib_entropy
    for ratio in [0.2]:
        k_length = int(len(all_prob) * ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        baseline_scores[f"Min_{ratio * 100}% Prob"] = -np.mean(topk_prob).item()
    return baseline_scores



def process_documents(document_list, output_dir, target_model, reference_model, target_tokenizer, reference_tokenizer, device1, device2, seq_length=128, stride=127):
    """
    Processes a list of documents, computes baseline metrics, and saves results.

    Parameters:
    - document_list: List of documents. Each element can be:
        - A string representing the document text (doc_id will be generated as 'doc_{index}')
        - A tuple (doc_id, text)
        - A dictionary with keys 'doc_id' and 'text'
    - output_dir: Directory to save the output .pkl files
    - target_model: The target model to evaluate
    - reference_model: The reference model to compare against
    - target_tokenizer: Tokenizer corresponding to the target model
    - reference_tokenizer: Tokenizer corresponding to the reference model
    - device1: Device for the target model (e.g., 'cuda:0')
    - device2: Device for the reference model (e.g., 'cuda:1')
    - seq_length: The sequence length for processing (default: 128)
    - stride: The stride size for moving the window (default: 127)
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for idx, document in enumerate(document_list):
        # Extract doc_id and text based on the type of 'document'
        if isinstance(document, tuple):
            doc_id, text = document
        elif isinstance(document, dict):
            doc_id = document.get('doc_id', f"doc_{idx}")
            text = document['text']
        else:
            # If it's just text, generate a doc_id based on the index
            text = document
            doc_id = f"doc_{idx}"
        
        output_path = os.path.join(output_dir, f"{doc_id}_baseline_results.pkl")
        
        print(f"Processing document: {doc_id}")
        
        # Tokenization
        tokens = target_tokenizer.encode(text)
        book_len = len(tokens)
        
        doc_ids = []
        all_seqs = []
        for begin_loc in range(0, book_len, stride):
            end_loc = min(begin_loc + seq_length, book_len)
            input_ids = tokens[begin_loc:end_loc]
            all_seqs.append(input_ids)
            doc_ids.append(doc_id)

        baseline_results = []
        for seq_tokens in all_seqs:
            text_segment = target_tokenizer.decode(seq_tokens)
            result = compute_baselines(text_segment, target_model, reference_model, target_tokenizer, reference_tokenizer, device1, device2)
            baseline_results.append(result)

        # Prepare data for DataFrame
        data = {"doc_id": doc_ids, "sequence": all_seqs}
        for key in baseline_results[0].keys():
            data[key] = [seq_results[key] for seq_results in baseline_results]
        
        df = pd.DataFrame(data=data)
        with open(output_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"Results saved to {output_path}")

def compute_average_metrics(output_dir):
    """
    Computes the average of selected metrics from .pkl files in the specified directory.

    Parameters:
    - output_dir: Directory containing .pkl files
    """
    # List to hold DataFrames from all files
    data_frames = []
    metrics = [
        'ppl',
        'ppl/Ref_ppl',
        'ppl/zlib',
        'Min_20.0% Prob'
    ]
    # Read each .pkl file and collect data
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
                # Check if all required metrics are in the DataFrame
                missing_metrics = [metric for metric in metrics if metric not in df.columns]
                if missing_metrics:
                    print(f"Warning: File {file_name} is missing metrics: {missing_metrics}")
                    # You can choose to skip this file or handle missing metrics
                    continue
                # Ensure that only the desired metrics are included
                df_metrics = df[metrics]
                data_frames.append(df_metrics)

    # Check if any data was collected
    if not data_frames:
        print("No data collected. Please check your files and directory.")
        return None

    # Concatenate all DataFrames
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Compute the average of each metric
    average_metrics = combined_df.mean()

    # Print the average values
    print("Average values for each metric:")
    for metric in metrics:
        avg_value = average_metrics[metric]
        print(f"{metric}: {avg_value}")

# Fluency
def load_questions(file_path):
    """
    Load question dataset.

    Parameters:
    - file_path: Path to the question file

    Returns:
    - A list of questions
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
    return questions

def generate_answers(model, tokenizer, questions):
    """
    Generate answers using a LoRA fine-tuned model.

    Parameters:
    - model: The fine-tuned model
    - tokenizer: The tokenizer for the model
    - questions: A list of questions

    Returns:
    - A list of generated answers
    """
    answers = []
    model.eval()
    for question in questions:
        # Encode the question as model input
        inputs = tokenizer(
            question,
            return_tensors='pt'
        ).to(model.device)
        
        # Generate the answer using the model
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        # Decode the generated output
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # If the generated answer contains the input question, remove the question part
        if answer.startswith(question):
            answer = answer[len(question):].strip()
        answers.append(answer)
    return answers


if __name__ == "__main__": 
    # Use argparse to parse hyperparameters
    parser = argparse.ArgumentParser(description="Evaluate LoRA model and compute average metrics.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the base model.")
    parser.add_argument('--peft_path', type=str, required=True, help="Path to the LoRA fine-tuned model.")
    parser.add_argument('--device', type=str, default="cuda:0", help="Device to run the evaluation on (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument('--dataset', type=str, required=True, choices=["Harry Potter", "WMDP", "TOFU"], help="Dataset to evaluate on.")
    parser.add_argument('--output_dir', type=str, required=False, help="Directory containing .pkl files for metrics computation.")
    parser.add_argument('--metrics', type=str, nargs='+', default=['ppl', 'ppl/Ref_ppl', 'ppl/zlib', 'Min_20.0% Prob'], help="Metrics to compute the average for.")
    parser.add_argument('--output_name', type=str, required=False, help="Directory containing answers.")

    args = parser.parse_args()

    # Load the model and tokenizer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # Load the LoRA fine-tuned model
    lora_model = PeftModel.from_pretrained(model, args.peft_path)
    lora_model.to(device)
    lora_model.eval()
    path_to_reference_model = "openai-community/gpt2"  # Replace with your reference model path
    path_to_reference_tokenizer = "openai-community/gpt2"  # Replace with your reference tokenizer path
    reference_model = AutoModelForCausalLM.from_pretrained(path_to_reference_model).to(device)
    reference_tokenizer = AutoTokenizer.from_pretrained(path_to_reference_tokenizer)

    if args.dataset == "Harry Potter":
        # Accuracy evaluation
        HP_Generalization = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={args.model_name},peft={args.peft_path},device={device}",
            tasks=["mmlu"],
            batch_size=32
        )
        HP_mmlu_acc = HP_Generalization['results']['mmlu']['acc,none']
        print('HP_llm_mmlu_eva', HP_mmlu_acc)
        HP_whp_acc = get_hp_accuracy(lora_model, tokenizer, network=None, batch_size=5, dtype=torch.bfloat16, verbose=True, data_path='data/HP/harrypotter_eva/hp-questions-dual.json')
        print(f"HP_ACC-dual: {HP_whp_acc}")
        HP_whp_acc = get_hp_accuracy(lora_model, tokenizer, network=None, batch_size=5, dtype=torch.bfloat16, verbose=True, data_path='data/HP/harrypotter_eva/hp-questions.json')
        print(f"HP_ACC-four: {HP_whp_acc}")

        # DMRA
        unseen_documents_file_path = 'data/HP/Divide_doc'
        context_length = 1024
        drma_Df = compute_DM_file_document(unseen_documents_file_path, lora_model, tokenizer, context_length, device)
        print(f"HP_DRMA score: {drma_Df}")

        # MIAs
        hp_documents = load_documents('data/HP/Divide_doc')
        process_documents(hp_documents, args.output_dir, lora_model, reference_model, tokenizer, reference_tokenizer, device, device, seq_length=128, stride=127)
        compute_average_metrics(args.output_dir)

        # Fluency
        forget_questions = load_questions('data/HP/fluency_questions/forget_set.txt')
        retain_questions = load_questions('data/HP/fluency_questions/common_questions.txt')
        all_questions = forget_questions + retain_questions
        answers = generate_answers(lora_model, tokenizer, all_questions)
        with open(f'{args.output_name}/answers.txt', 'w', encoding='utf-8') as f:
            for idx, answer in enumerate(answers, start=1):
                f.write(f"Answer{idx}: {answer}\n")
        print("Answers saved to answers.txt.")

    elif args.dataset == "WMDP":
        # Accuracy evaluation
        wmdp_eval_results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={args.model_name},peft={args.peft_path},device={device}",
            tasks=["wmdp_bio", "wmdp_cyber"]
        )
        wmdp_bio_acc = wmdp_eval_results['results']['wmdp_bio']['acc,none']
        wmdp_cyber_acc = wmdp_eval_results['results']['wmdp_cyber']['acc,none']
        print('WMDP_bio_eva', wmdp_bio_acc)
        print('WMDP_cyber_eva', wmdp_cyber_acc)
        WMDP_Generalization = lm_eval.simple_evaluate(
                        model="hf",
                        model_args=f"pretrained={args.model_name},peft={args.peft_path},device={device}",
                        tasks=["mmlu"],
                        batch_size=32
                    )
        WMDP_mmlu_acc = WMDP_Generalization['results']['mmlu']['acc,none']
        print('WMDP_llm_mmlu_eva', WMDP_mmlu_acc)
        # DMRA
        forget_prompts = {}
        wmdp_corpora_path = "cais/wmdp-corpora"
        bio_corpus_path = 'data/WMDP/bio_remove_dataset.jsonl'
        forget_prompts[0] = [str(json.loads(line)['text'][:80]) for line in open(bio_corpus_path, "r") if len(json.loads(line)['text']) > 50]
        forget_prompts[1] = datasets.load_dataset(wmdp_corpora_path, 'cyber-forget-corpus', split="train")['text']
        forget_prompts[1] = [p[:700] for p in forget_prompts[1] if len(p) > 50]
        forget_list = compress_list(forget_prompts[0], target_size=350) + compress_list(forget_prompts[1], target_size=50)
        context_length = 1024
        drma_wmdp = compute_DM_document(forget_list, lora_model, tokenizer, context_length, device)
        print(f"WMDP_DRMA score: {drma_wmdp}")

        # MIAs and Fluency
        process_documents(forget_list, args.output_dir, lora_model, reference_model, tokenizer, reference_tokenizer, device, device, seq_length=128, stride=127)
        compute_average_metrics(args.output_dir)
        forget_questions = load_questions('data/WMDP/fluency_questions/forget_set.txt')
        retain_questions = load_questions('data/WMDP/fluency_questions/common_questions.txt')
        answers = generate_answers(lora_model, tokenizer, forget_questions + retain_questions)
        with open(f'{args.output_name}/answers.txt', 'w', encoding='utf-8') as f:
            for idx, answer in enumerate(answers, start=1):
                f.write(f"Answer{idx}: {answer}\n")
        print("Answers saved to answers.txt.")

    elif args.dataset == "TOFU":
        # Prepare TOFU dataset
        forget_10 = load_dataset("locuslab/TOFU", "forget10")
        train_dataset = forget_10["train"]
        document_list = compress_list([f"{q}: {a}" for q, a in zip(train_dataset["question"], train_dataset["answer"])], target_size=len(train_dataset))
        # # MIAs and Fluency
        process_documents(document_list, args.output_dir, lora_model, reference_model, tokenizer, reference_tokenizer, device, device, seq_length=128, stride=127)
        compute_average_metrics(args.output_dir)
        forget_questions = load_questions('data/TOFU/fluency_questions/forget_set.txt')
        retain_questions = load_questions('data/TOFU/fluency_questions/common_questions.txt')
        answers = generate_answers(lora_model, tokenizer, forget_questions + retain_questions)
        with open(f'{args.output_name}/answers.txt', 'w', encoding='utf-8') as f:
            for idx, answer in enumerate(answers, start=1):
                f.write(f"Answer{idx}: {answer}\n")
