import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
from utils.metrics import get_hp_accuracy, get_mmlu_accuracy  # Ensure the module exists and is importable
from torch.optim.lr_scheduler import _LRScheduler
import datasets
from tqdm import tqdm
import time

# Distributed training

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
# Create a quantization configuration
quant_config = BitsAndBytesConfig(bits=8)  # Use bits=4 for more aggressive quantization

# Retain dataset
def prepare_prompts(verbose=False, min_len=50, max_len=700):
    # Initialize retain_prompts
    retain_prompts = datasets.load_dataset(
        "philschmid/easyrag-mini-wikipedia",
        "documents",
        split="full"
    )['document']
    # Filter out text that is not within the specified length range
    retain_prompts = [p[:max_len] for p in retain_prompts if len(p) > min_len]

    if verbose:
        print(f"Loaded {len(retain_prompts)} retain prompts for dataset")
    return retain_prompts


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
    Calculate the perplexity of a given text.

    Parameters:
    - text (str): Text to calculate perplexity for.
    - tokenizer: Tokenizer of the pre-trained model.
    - model: Pre-trained language model.

    Returns:
    - perplexity (float): The perplexity value of the text.
    """
    # Tokenize the text and return PyTorch tensors
    tokens = tokenizer(text, return_tensors='pt')
    # Move tokens to the same device as the model
    tokens = {key: value.to(model.device) for key, value in tokens.items()}

    # Unpack tokens as keyword arguments and set labels as input_ids
    outputs = model(**tokens, labels=tokens['input_ids'])
    loss = outputs.loss

    # Calculate perplexity
    perplexity = torch.exp(loss).item()
    return perplexity

# Define a function to log results
def log_results(log_file, Lambda_1, Lambda_2, max_seq_length, total_loss, kl_loss, distillation_loss, retain_loss, innocence_acc, innocence_acc_before_dual, Generalization, prompt, before_unlearning, after_unlearning, Fluency):
    # Create a dictionary to store data to be logged
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
        "before unlearning output:": before_unlearning,
        "after unlearning output": after_unlearning,
        "Fluency": Fluency
    }
    
    # If the file doesn't exist, create a new file and write the data
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump([log_data], f, indent=4, ensure_ascii=False)  # Write in list format for easy appending

    # If the file exists, append the data
    else:
        with open(log_file, 'r+') as f:
            # Read existing data
            existing_data = json.load(f)
            existing_data.append(log_data)  # Add new data
            f.seek(0)  # Move file pointer to the beginning
            json.dump(existing_data, f, indent=4, ensure_ascii=False)  # Rewrite the file

# Custom dataset class
def custom_collate_fn(batch):
    """
    Custom collate function for processing batch data.
    Ensures the returned batch data does not affect DataLoader's batch_size.
    """
    # Assume each sample is a dictionary containing 'input_ids' and 'attention_mask'
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    # Use pad_sequence to ensure consistent input sizes
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
    attention_masks_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True)

    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded
    }

class DocumentDataset(Dataset):
    def __init__(self, documents, tokenizer, padding=True, truncation=True, max_length=None):
        """
        :param documents: List of document data, each document is a piece of text
        :param tokenizer: Tokenizer of the pre-trained model
        :param padding: Whether to pad samples to align to the same length
        :param truncation: Whether to truncate samples exceeding max_length
        :param max_length: Maximum length, if None there is no restriction
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
            padding=self.padding,  # Whether to pad
            truncation=self.truncation,  # Whether to truncate
            max_length=self.max_length  # Max length, no limit if not set
        )

        # Ensure validity of the input
        input_ids = inputs['input_ids'].squeeze(0) if 'input_ids' in inputs else None
        attention_mask = inputs['attention_mask'].squeeze(0) if 'attention_mask' in inputs else None

        # Check if valid input_ids and attention_mask are returned
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

    # Do not use device_map='auto' to avoid issues during distributed training
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config  # Use 8-bit quantization to reduce memory usage
    )

    lora_config = LoraConfig(
        r=256,
        lora_alpha=16,
        layers_to_transform=list(range(4, 8)),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"],  # These layers are critical for MLP+Attention
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    # Enable gradient checkpointing to reduce memory usage
    model.gradient_checkpointing_enable()
    return model, tokenizer

# Load sensitive vocabulary
def prepare_dict(filename):
    def parse_dict(s):
        s = s.replace("\n", "")
        # Use regex to extract a dictionary from a string
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
        # If both "Harry's" and "Harry" are in the dictionary, remove the former
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
    Calculate the RMA score for all txt files in a given folder, max_seq_length=1024.

    Parameters:
    - folder_path: String, folder path containing multiple txt files.
    - model: Pre-trained model.
    - tokenizer: Tokenizer.
    - accelerator: Accelerator instance.
    - max_seq_length: Maximum sequence length supported by the model.
    """
    # Get all txt files in the folder
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    # Initialize the RMA list
    total_rmas = []
    doc_num = len(txt_files)
    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            document = f.read()
        # Encode the document as token IDs, truncate, and take only max_seq_length tokens
        input_ids = tokenizer.encode(document, 
                                     add_special_tokens=True,
                                     max_length=max_seq_length,
                                     padding='max_length',
                                     truncation=True,
                                     return_tensors='pt')
        input_ids = input_ids[0]  # Remove the batch dimension
        input_ids = input_ids.to(accelerator.device)  # Ensure input data is on the correct device
        # Get the input_ids for the current document, ensuring it does not exceed total length
        input_ids = input_ids.unsqueeze(0)
        # Model prediction
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits  # Shape: (1, seq_len, vocab_size)
        # Calculate the RMA for the current block
        probs = F.softmax(logits, dim=-1)  # Compute probability distribution
        probs = probs[:, 1:, :]  # Skip the first token, shape: (1, seq_len -1, vocab_size)
        token_ids = input_ids[:, 1:]  # Skip the first token, shape: (1, seq_len -1)
        # Get the probabilities of the real tokens
        token_probs = probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)  # Shape: (1, seq_len -1)
        rmas = sum((token_probs.cpu().tolist()[0]))
        # Add the RMA for the current document to the total RMA list
        total_rmas.append(rmas)

    # Calculate sum_rmas
    sum_rmas = sum(total_rmas)
    rma_score = sum_rmas / doc_num
    return rma_score



# Model training
def model_train_with_lora(dataloader, generic_documents, unseen_documents_file_path, sensitive_token, document_file_path, teacher_model, approximate_documents, model, tokenizer, num_epochs, accelerator, Lambda_1, Lambda_2, max_seq_length):
    """
    :dataloader : Forget set dataset.
    :generic_documents: Replacement documents (500 documents) similar to HP documents, used to calculate retention loss and preserve other capabilities.
    :unseen_documents_text: Unseen document text (a single text file) used to calculate unseen DRMA and determine the training stopping threshold (memory score).
    :sensitive_token: Sensitive tokens (provided by HP authors) specific to HP vocabulary, used to calculate masking loss.
    :raw_data: Complete HP documents used to calculate DRMA of Df and determine training stopping threshold (memory score).
    :teacher_model: Teacher model used to calculate distillation loss.
    :approximate_documents: Approximate documents used to approximate the distribution of the forget set dataset.
    :model: Input model (Llama-2-7b-chat-hf).
    :tokenizer: Tokenizer used with the model (Llama-2-7b-chat-hf).
    :num_epochs: Number of training epochs.

    Total losses: distillation loss, masking loss, retention loss.
    """
    print('######## Load model ########')
    # Ensure generic_documents corresponds one-to-one with the dataset
    assert len(generic_documents) == len(dataloader.dataset)

    # Prepare optimizer and loss functions
    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)  # Based on NLP best practices for large models
    kl_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')
    distillation_loss_fn = torch.nn.MSELoss()  # Used to calculate the difference with teacher model output
    # Total training steps
    total_steps = num_epochs * len(dataloader)
    # Warm-up steps (e.g., 10% of total steps)
    warmup_steps = int(0.1 * total_steps)
    # Learning rate scheduler
    scheduler = CosineAnnealingWithWarmupLR(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        eta_min=0.1  # Minimum learning rate is 10% of the initial learning rate
    )
    # Wrap optimizer, dataloader, and scheduler with accelerator
    optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
    # Max gradient norm for clipping
    max_grad_norm = 1.0

    # Initialize lists for plotting
    epoch_steps = []
    total_losses = []
    kl_losses = []
    distillation_losses = []
    retain_losses = []
    drma_Df_list = []
    drma_Dunseen_list = []
    iteration = 0  # Iteration counter

    # Control variables
    retain_softloss = True
    loss_fun_to_use = 'cross'
    retain_prompts = prepare_prompts()
    total_retain_prompts = len(retain_prompts)

    # Compute sensitive token IDs
    sensitive_token_ids = []
    for token in sensitive_token:
        token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
        sensitive_token_ids.extend(token_ids)
    print('####### Model training ######')

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
                
                # Calculate distillation loss (different styles)
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

                # Calculate distillation loss (similar style)
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
                distillation_loss = distillation_same_loss + distillation_others_loss  # Retain both styles

                # Calculate retain loss
                retain_prompt = retain_prompts[dataset_cntr % total_retain_prompts]
                dataset_cntr += 1
                inputs_retain = tokenizer(retain_prompt, return_tensors="pt", padding=True)
                with torch.no_grad():
                    retain_vector = teacher_model(**inputs_retain).logits.softmax(dim=-1).contiguous()
                activations_retain = model(**inputs_retain).logits.contiguous()
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
            # Clip gradients
            accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            # Optimizer step
            optimizer.step()
            # Update learning rate
            scheduler.step()
            total_kl_loss += kl_loss.item()                                       
            total_distillation_loss += distillation_loss.item()
            total_retain_loss += retain_loss_value.item()
            total_loss += loss.item()
            iteration += 1
                      
        # Log loss values
        print(f"Epoch {epoch + 1}:")
        avg_loss = total_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        avg_distillation_loss = total_distillation_loss / len(dataloader)
        avg_retain_loss = total_retain_loss / len(dataloader)
        epoch_steps.append(epoch+1)
        total_losses.append(avg_loss)
        kl_losses.append(avg_kl_loss)
        distillation_losses.append(avg_distillation_loss)
        retain_losses.append(avg_retain_loss)
        print(f"Loss: {avg_loss:.6f}")

    # Save LoRA weights (if needed)
    save_dir = f'Lora_model/HP/model'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    accelerator.wait_for_everyone()

def main():
    # Train the model
    start_time = time.time()
    Lambda_2 = 0.7
    Lambda_1 = 0.2
    document_file_path = 'data/HP/Divide_doc'
    generic_document_file_path = 'data/HP/sim_outpt'
    other_documents_file_path = 'data/HP/approximate_documents'
    num_epochs = 4
    batch_size = 1
    max_seq_length = 1024
    accelerator = Accelerator()
    documents = load_documents(document_file_path)
    generic_documents = load_documents(generic_document_file_path)
    other_documents = load_documents(other_documents_file_path)
    sensitive_token_file_path = 'data/HP/sensitive_hp.npy'
    anchored_expressions_dictionary = prepare_dict(sensitive_token_file_path)
    sensitive_tokens = list(anchored_expressions_dictionary.keys())
    model, tokenizer = prepare_lora_model(model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config
    )
    teacher_model.gradient_checkpointing_enable()
    teacher_model.eval()
    model, teacher_model = accelerator.prepare(model, teacher_model)
    dataset = DocumentDataset(documents, tokenizer, padding='longest', truncation=True, max_length=max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    model_train_with_lora(
        dataloader,
        generic_documents,
        generic_document_file_path,
        sensitive_tokens,
        document_file_path,
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

