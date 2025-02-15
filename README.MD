# OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models

This is the implementation for the paper **Revisiting **OBLIVIATE**: Robust and Practical Machine Unlearning for Large Language Models**.

We propose **OBLIVIATE**, a robust and practical unlearning framework that removes targeted data while preserving model utility. 
It employs a structured approach by first ``extracting'' target tokens and retain sets (from forget sets), and then applying our tailored unlearning loss, decomposed to mask, distillation, and world fact losses. Leveraging low-rank adapters (LoRA), our approach ensures efficiency without compromising unlearning quality.

In order to reproduce our results, take the following steps:

### 1. Create conda environment and install requirements
```
conda create -n unlearn python=3.10.15
conda activate unlearn
conda install pytorch==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn==2.5.3 --no-build-isolation
python -m spacy download en_core_web_sm
```

### 2. Preparing the dataset for fine-tuning and evaluation
We collect and provide the "generic and other styles documents," "fluency_questions," and "target tokens," which are used for fine-tuning and evaluation. The forget sets, including Harry Potter, WMDP, TOFU are collected from other papers (**Who's Harry Potter? Approximate Unlearning in LLMs**), (**TOFU: A Task of Fictitious Unlearning for LLMs**), (**The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning**)

### 3. Use the following command to Reproduce Experimental Results
The model LoRA weights are stored in Lora_model/HP/, Lora_model/WMDP/ or Lora_model/TOFU/.
**For Harry Potter**
```
accelerate launch --gpu_ids='all' main_hp.py
```
**For WMDP**
```
accelerate launch --gpu_ids='all' main_wmdp.py \
 --model_name "HuggingFaceH4/zephyr-7b-beta" \  
 --model_file_name "zephyr-7b-beta"
```
**For TOFU**
```
accelerate launch --num_processes=1 main_tofu.py
```

### 4. Eval unlearned model
Before evaluation, You need to create two folders, answers and MIAs_Results, to store the results.
```
python evaluations.py \
    --model_name "meta-llama/Llama-2-7b-chat-hf" \
    --peft_path "Lora_model/HP/lora_finetuned_epoch_final_model" \
    --device "cuda:0" \
    --dataset "Harry Potter" \
    --output_dir "MIAs_Results/HP" \
    --output_name "answers/HP"
```


