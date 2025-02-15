import torch 
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.nn.functional import cross_entropy
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
import re
import json
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import math
import sys
import gc
from utils.metrics import get_hp_accuracy, get_mmlu_accuracy  # Ensure this module exists and is importable
from torch.optim.lr_scheduler import _LRScheduler
import datasets
import random
from tqdm import tqdm
import time
import argparse

# Use argparse to parse hyperparameters
parser = argparse.ArgumentParser(description="Evaluate LoRA model and compute average metrics.")
parser.add_argument('--model_name', type=str, required=True, help="Name of the base model.")
parser.add_argument('--model_file_name', type=str, required=True, help="Name of the model file.")
args = parser.parse_args()

# Load model and tokenizer
model_name = args.model_name
model_namelist = args.model_file_name

def prepare_prompts(verbose=False, min_len=50, max_len=700):
    """
    Initialize and filter retain prompts.
    """
    retain_prompts = []
    retain_prompts = datasets.load_dataset(
        "philschmid/easyrag-mini-wikipedia",
        "documents",
        split="full"
    )['document']
    # Filter out texts not within the specified length range
    retain_prompts = [p[:max_len] for p in retain_prompts if len(p) > min_len]

    if verbose:
        print(f"Loaded {len(retain_prompts)} retain prompts for dataset")
    return retain_prompts

def load_sensitive_words(file_path):
    """
    Load sensitive words from a file.
    """
    with open(file_path, 'r') as file:
        sensitive_words = [line.strip() for line in file if line.strip()]
    return sensitive_words

# Define a custom learning rate scheduler
class CosineAnnealingWithWarmupLR(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, eta_min=0.0, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.eta_min = eta_min
        super(CosineAnnealingWithWarmupLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        current_step = max(0, self.last_epoch)
        if current_step < self.num_warmup_steps:
            # Warm-up phase
            lr_scale = float(current_step) / float(max(1, self.num_warmup_steps))
        else:
            # Cosine annealing phase
            progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr_scale = (1 - self.eta_min) * cosine_decay + self.eta_min

        return [base_lr * lr_scale for base_lr in self.base_lrs]

def calculate_perplexity(text, tokenizer, model):
    """
    Calculate the perplexity of the given text.

    Parameters:
    - text (str): The text for which to calculate perplexity.
    - tokenizer: Tokenizer for the pre-trained model.
    - model: Pre-trained language model.

    Returns:
    - perplexity (float): The perplexity value of the text.
    """
    # Tokenize the text and return PyTorch tensors
    tokens = tokenizer(text, return_tensors='pt')
    # Move tokens to the same device as the model
    tokens = {key: value.to(model.device) for key, value in tokens.items()}

    # Unpack tokens as keyword arguments and set labels to input_ids
    outputs = model(**tokens, labels=tokens['input_ids'])
    loss = outputs.loss

    # Calculate perplexity
    perplexity = torch.exp(loss).item()
    return perplexity

# Define a function to log results
def log_results(log_file, Lambda_1, Lambda_2, max_seq_length, total_loss, kl_loss, distillation_loss, retain_loss, innocence_acc, innocence_acc_before_dual, Generalization, prompt, before_unlearning, after_unlearning, Fluency):
    """
    Log results to a file.
    """
    log_data = {
        "Lambda_1": Lambda_1,
        "Lambda_2": Lambda_2,
        "max_seq_length": max_seq_length,
        "total_loss": total_loss,
        "kl_loss": kl_loss,
        "distillation_loss": distillation_loss,
        "retain_loss": retain_loss,
        "innocence_acc": innocence_acc,
        "innocence_acc_dual": innocence_acc_before_dual,
        "Generalization": Generalization,
        "input prompt": prompt,
        "before unlearning output": before_unlearning,
        "after unlearning output": after_unlearning,
        "Fluency": Fluency
    }
    
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump([log_data], f, indent=4, ensure_ascii=False)
    else:
        with open(log_file, 'r+') as f:
            existing_data = json.load(f)
            existing_data.append(log_data)
            f.seek(0)
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

# Custom dataset class
def custom_collate_fn(batch):
    """
    Custom collate function to process batch data.
    Ensure that returned batch data does not affect DataLoader's batch_size.
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded
    }

class DocumentDataset(Dataset):
    def __init__(self, documents, tokenizer, padding=True, truncation=True, max_length=None):
        """
        :param documents: List of document data, each document is a piece of text.
        :param tokenizer: Tokenizer for the pre-trained model.
        :param padding: Whether to pad samples to align to the same length.
        :param truncation: Whether to truncate samples exceeding max_length.
        :param max_length: Maximum length, if None there is no limit.
        """
        self.documents = documents
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        # Get the idx-th document
        document = self.documents[idx]

        # Tokenize the document into input_ids and attention_mask
        inputs = self.tokenizer(
            document,
            return_tensors="pt",  # Return PyTorch tensors
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length
        )

        input_ids = inputs['input_ids'].squeeze(0) if 'input_ids' in inputs else None
        attention_mask = inputs['attention_mask'].squeeze(0) if 'attention_mask' in inputs else None

        if input_ids is None or attention_mask is None:
            raise ValueError("The tokenizer did not return valid input_ids or attention_mask.")

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


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

def prepare_lora_model(model_name):
    """Configure and return the LoRA fine-tuned model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Avoid using device_map='auto' to prevent issues in distributed training
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16  # Use 16-bit precision to reduce memory usage
    )

    lora_config = LoraConfig(
        r=256,
        lora_alpha=16,
        layers_to_transform=list(range(4, 8)),
        target_modules=[
            "q_proj",  # Includes MLP and attention layers as they are significant
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    return model, tokenizer

# Load sensitive words
def prepare_dict(filename):
    def parse_dict(s):
        s = s.replace("\n", "")
        # Use regex to extract dictionary from string
        match = re.search(r'translations\s*=\s*({.*?})', s)

        if match:
            dict_str = match.group(1)
            try:
                dict_str = re.sub(r',\s*([}\]])', r'\1', dict_str)
                dict_str = re.sub(r'#.*?(,|})', r'\1', dict_str)
                my_dict = json.loads(dict_str)

                if my_dict is None:
                    my_dict = {}

                return my_dict

            except:
                print(f"Couldn't parse the string: {dict_str}")
                return {}
        else:
            return {}

    def consolidate_dicts(dict_list):
        consolidated = {}

        for d in dict_list:
            for key, value in d.items():
                if key not in consolidated:
                    consolidated[key] = []
                if value not in consolidated[key]:  # Ensure uniqueness
                    consolidated[key].append(value)

        return consolidated

    dicts = np.load(filename, allow_pickle=True)
    dicts = [parse_dict(dict_str) for dict_str in dicts]
    consolidated_dict = consolidate_dicts(dicts)

    def splittable_key(dict_obj, key):
        # Remove entries like "Harry's" if "Harry" exists in the dictionary
        if key[-2:] == "'s" and key[:-2] in dict_obj.keys():
            return True

        words = key.split()
        if len(words) == 1:
            return False

        return all([word in dict_obj.keys() for word in words])

    consolidated_dict = {k: v for k, v in consolidated_dict.items() if not splittable_key(consolidated_dict, k)}

    return consolidated_dict

def calculate_drma(folder_path, model, tokenizer, accelerator, max_seq_length=1024):
    """
    Calculate the DRMA score for all txt files in a given folder.

    Parameters:
    - folder_path: String, path to the folder containing txt files
    - model: Pre-trained model
    - tokenizer: Tokenizer
    - accelerator: Accelerator instance
    - max_seq_length: Maximum sequence length supported by the model
    """
    # Get all txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    # Initialize RMA list
    total_rmas = []
    doc_num = len(txt_files)
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            document = f.read()
        # Encode the document as token IDs and truncate to max_seq_length
        input_ids = tokenizer.encode(
            document,
            add_special_tokens=True,
            max_length=max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = input_ids[0]  # Remove batch dimension
        input_ids = input_ids.to(accelerator.device)  # Ensure input data is on the correct device
        input_ids = input_ids.unsqueeze(0)
        # Model prediction
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits  # Shape: (1, seq_len, vocab_size)
        # Calculate current chunk's RMA
        probs = F.softmax(logits, dim=-1)  # Calculate probability distribution
        probs = probs[:, 1:, :]  # Skip the first token, shape: (1, seq_len - 1, vocab_size)
        token_ids = input_ids[:, 1:]  # Skip the first token, shape: (1, seq_len - 1)
        # Get probabilities of the true tokens
        token_probs = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # Shape: (1, seq_len - 1)
        rmas = sum((token_probs.cpu().tolist()[0]))
        # Add current document's RMA to the total RMA list
        total_rmas.append(rmas)

    # Calculate sum of RMAs
    sum_rmas = sum(total_rmas)
    rma_score = sum_rmas / doc_num
    return rma_score

def compress_list(original_list, target_size=500):
    """
    Compress a list to a target size by grouping elements.

    Parameters:
    - original_list: The list to be compressed
    - target_size: The desired size of the compressed list
    """
    n = len(original_list)
    group_size = n // target_size
    new_list = ["".join(map(str, original_list[i:i + group_size])) for i in range(0, n, group_size)]
    
    # Ensure the new list size is exactly target_size (handle remainders)
    return new_list[:target_size]


# Model Training
def model_train_with_lora(dataloader, generic_documents, sensitive_token, teacher_model, approximate_documents, model, tokenizer, num_epochs, accelerator, Lambda_1, Lambda_2, max_seq_length):
    """
    :dataloader : Dataset for the forget set
    :generic_documents: Replacement documents, 500 documents similar to HP documents, used to calculate enhancement loss and retain other capabilities
    :unseen_documents_text: Unseen documents text, a complete text file, used to calculate unseen DRMA to determine training stop threshold
    :sensitive_token: Sensitive tokens provided by WHP authors, specific terms in HP, used to calculate masking loss
    :raw_data: Complete HP documents used to calculate Df DRMA to determine training stop threshold
    :teacher_model: Teacher model used to calculate distillation loss
    :approximate_documents: Approximation documents to simulate the distribution of forget set
    :model: Input model (e.g., Llama-2-7b-chat-hf)
    :tokenizer: Tokenizer for the input model (e.g., Llama-2-7b-chat-hf)
    :num_epochs: Number of training epochs
    Three types of loss: distillation loss, masking loss, retain loss
    """
    print('########load model ########')
    # Assume generic_documents contains 500 replacement documents
    assert len(generic_documents) == len(dataloader.dataset)  # Ensure replacement documents and original documents correspond one-to-one
    
    # Prepare optimizer and loss functions
    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    distillation_loss_fn = torch.nn.MSELoss()  # Used to calculate the difference from teacher model output
    
    # Calculate total training steps
    total_steps = num_epochs * len(dataloader)
    # Define warm-up steps
    warmup_steps = int(0.1 * total_steps)  # For example, warm up 10% of total steps
    
    # Define learning rate scheduler
    scheduler = CosineAnnealingWithWarmupLR(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        eta_min=0.1  # Minimum learning rate is 10% of initial learning rate
    )
    
    # Prepare optimizer, dataloader, and scheduler with accelerator
    optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
    
    # Define the maximum norm for gradient clipping
    max_grad_norm = 1.0

    # Initialize lists for plotting
    epoch_steps = []
    total_losses = []
    kl_losses = []
    distillation_losses = []
    retain_losses = []
    iteration = 0  # Iteration counter

    # Initialize control variables
    retain_softloss = True
    loss_fun_to_use = 'cross'
    retain_prompts = prepare_prompts()
    total_retain_prompts = len(retain_prompts)

    # Compute sensitive_token_ids
    sensitive_token_ids = []
    for token in sensitive_token:
        token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
        sensitive_token_ids.extend(token_ids)  # Add all token IDs to a list
    
    print('#######model train######')

    for epoch in range(num_epochs):
        dataset_cntr = 0  # Counter for traversing retain_prompts
        total_loss = 0.0
        total_kl_loss = 0.0
        total_distillation_loss = 0.0
        total_retain_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            optimizer.zero_grad()
            
            with accelerator.autocast():
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                # Generate labels
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = -100  # Ignore the loss of the last token

                # Calculate KL divergence loss
                vocabulary_mask = torch.ones(logits.size(-1), device=logits.device)
                for token_id in sensitive_token_ids:
                    if token_id < vocabulary_mask.size(0):  # Ensure token_id is within bounds
                        vocabulary_mask[token_id] = 0
                logits_masked = logits * vocabulary_mask
                logits_probs = F.log_softmax(logits, dim=-1)
                target_probs = F.softmax(logits_masked, dim=-1)
                kl_loss = kl_loss_fn(logits_probs, target_probs)
                
                # Calculate distillation loss
                # For different styles of novels (batch_idx)
                inputs_others = tokenizer(approximate_documents[batch_idx], return_tensors="pt", padding='longest', truncation=True, max_length=max_seq_length)
                inputs_others = {k: v.to(accelerator.device) for k, v in inputs_others.items()}
                logits_others = model(**inputs_others).logits
                
                with torch.no_grad():
                    teacher_others_outputs = teacher_model(**inputs_others)
                    teacher_others_logits = teacher_others_outputs.logits
                distillation_others_loss = distillation_loss_fn(
                    logits_others.view(-1, logits_others.size(-1)),
                    teacher_others_logits.view(-1, teacher_others_logits.size(-1))
                )
                
                # For same styles of novels (batch_idx)
                inputs_same = tokenizer(generic_documents[batch_idx], return_tensors="pt", padding='longest', truncation=True, max_length=max_seq_length)
                inputs_same = {k: v.to(accelerator.device) for k, v in inputs_same.items()}
                logits_same = model(**inputs_same).logits
                
                with torch.no_grad():
                    teacher_same_outputs = teacher_model(**inputs_same)
                    teacher_same_logits = teacher_same_outputs.logits
                distillation_same_loss = distillation_loss_fn(
                    logits_same.view(-1, logits_same.size(-1)),
                    teacher_same_logits.view(-1, teacher_same_logits.size(-1))
                )
                distillation_loss = distillation_same_loss + distillation_others_loss

                # Calculate retain loss
                retain_prompt = retain_prompts[dataset_cntr % total_retain_prompts:dataset_cntr % total_retain_prompts+3]
                dataset_cntr += 4
                
                inputs_retain = tokenizer(retain_prompt, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    retain_vector = teacher_model(**inputs_retain).logits.softmax(dim=-1)
                    retain_vector = retain_vector.contiguous()
                
                activations_retain = model(**inputs_retain).logits
                activations_retain = activations_retain.contiguous()
                
                if retain_softloss and loss_fun_to_use == 'kld':
                    activations_retain_log_softmax = torch.nn.functional.log_softmax(activations_retain, dim=-1)
                    retain_loss_value = torch.nn.functional.kl_div(
                        activations_retain_log_softmax,
                        retain_vector.detach(),
                        reduction='batchmean'
                    )
                else:
                    retain_targets = retain_vector.detach().argmax(dim=-1)
                    retain_loss_value = torch.nn.functional.cross_entropy(
                        activations_retain.view(-1, activations_retain.size(-1)),
                        retain_targets.view(-1),
                        ignore_index=tokenizer.pad_token_id
                    )
                
                # Total loss
                loss = kl_loss + Lambda_1 * distillation_loss + Lambda_2 * retain_loss_value
                progress_bar.set_postfix(
                    loss=loss.item(),
                    distillation_loss=distillation_loss.item(),
                    retain_loss=retain_loss_value.item(),
                    kl_loss=kl_loss.item()
                )
            
            # Backward pass
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            total_kl_loss += kl_loss.item()
            total_distillation_loss += distillation_loss.item()
            total_retain_loss += retain_loss_value.item()
            total_loss += loss.item()
            iteration += 1
        
        print(f"Epoch {epoch + 1}:")
        avg_loss = (total_loss / len(dataloader))
        avg_kl_loss = (total_kl_loss / len(dataloader))
        avg_distillation_loss = (total_distillation_loss / len(dataloader))
        avg_retain_loss = (total_retain_loss / len(dataloader))
        epoch_steps.append(epoch+1)
        total_losses.append(avg_loss)
        kl_losses.append(avg_kl_loss)
        distillation_losses.append(avg_distillation_loss)
        retain_losses.append(avg_retain_loss)
        print(f"Loss: {avg_loss:.6f}")
    
    save_dir = f'Lora_model/WMDP/lora_finetuned_{model_namelist}_model'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    accelerator.wait_for_everyone()


def main():
    # Train the model
    start_time = time.time()
    Lambda_2 = 0.7
    Lambda_1 = 0.2
    # Prepare WMDP dataset
    retain_prompts = {}
    forget_prompts = {}
    min_len = 50 
    max_len_retain_bio = 60
    max_len_retain_cyber = 170
    max_len_forget_bio = 80
    max_len_forget_cyber = 700
    wmdp_corpora_path = "cais/wmdp-corpora"
    bio_corpus_path = 'data/WMDP/bio_remove_dataset.jsonl'
    
    retain_prompts[0] = datasets.load_dataset(
        wmdp_corpora_path, 
        'bio-retain-corpus',
        split="train"
    )['text']
    retain_prompts[0] = [p[:max_len_retain_bio] for p in retain_prompts[0] if len(p) > min_len]
    forget_prompts[0] = []
    for line in open(bio_corpus_path, "r"):
        raw_text = json.loads(line)['text']
        if len(raw_text) > min_len:
            forget_prompts[0].append(str(raw_text[:max_len_forget_bio]))

    retain_prompts[1] = datasets.load_dataset(
        wmdp_corpora_path, 
        'cyber-retain-corpus',
        split="train"
    )['text']
    retain_prompts[1] = [p[:max_len_retain_cyber] for p in retain_prompts[1] if len(p) > min_len]
    forget_prompts[1] = datasets.load_dataset(
        wmdp_corpora_path, 
        'cyber-forget-corpus',
        split="train"
    )['text']
    forget_prompts[1] = [str(p[:max_len_forget_cyber]) for p in forget_prompts[1] if len(p) > min_len]

    retain_prompts_bio_list = compress_list(retain_prompts[0], target_size=350)
    retain_prompts_cyber_list = compress_list(retain_prompts[1], target_size=50)
    forget_prompts_bio_list = compress_list(forget_prompts[0], target_size=350)
    forget_prompts_cyber_list = compress_list(forget_prompts[1], target_size=50)
    forget_list = forget_prompts_bio_list + forget_prompts_cyber_list
    all_retain_list = retain_prompts_bio_list + retain_prompts_cyber_list  
    
    num_epochs = 4  # Number of training epochs
    batch_size = 1  # Adjust batch size to control memory usage
    max_seq_length = 1024  # Maximum truncation size
    accelerator = Accelerator()  # Distributed training
    
    # Load sensitive word dictionaries
    bio_path = "data/WMDP/sensitive_tokens_bio.txt"
    cyber_path = "data/WMDP/sensitive_tokens_cyber.txt"
    target_words_bio = load_sensitive_words(bio_path)
    target_words_cyber = load_sensitive_words(cyber_path)
    target_tokens = target_words_bio + target_words_cyber
    
    # Prepare documents; `documents` is a combination of two sets, `generic_documents` represents biological knowledge, `approximate_documents` represents cybersecurity knowledge
    documents = forget_list
    generic_documents = all_retain_list
    other_documents = all_retain_list
    random.shuffle(other_documents)
    
    # Prepare the model and tokenizer
    model, tokenizer = prepare_lora_model(model_name)
    
    # Prepare the teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    teacher_model.gradient_checkpointing_enable()
    teacher_model.eval()  # Set to evaluation mode
    model, teacher_model = accelerator.prepare(model, teacher_model)

    # Prepare dataset and dataloader
    print('########start load dataset##########')
    dataset = DocumentDataset(documents, tokenizer, padding='longest', truncation=True, max_length=max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    
    model_train_with_lora(
        dataloader,
        generic_documents,
        target_tokens,
        teacher_model,
        other_documents,
        model,
        tokenizer,
        num_epochs,
        accelerator,
        Lambda_1,
        Lambda_2,
        max_seq_length
    )   
    
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

