# !pip install -U torch trl datasets peft transformers bitsandbytes accelerate
# python 1-train.py

import os
import time
import torch
from trl import SFTTrainer
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments


# -----------------------------------------------------------------------------------------------
#
# Settings for base model, trained adapter and path for merged new model
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
basemodel_name = 'Rijgersberg/GEITje-7B-chat'
adapter_path = './model/adapter'
newmodel_path = './model/full'

# Chat template for tokenizer prep function.
template_string = """
{% if messages[0]['role'] == 'system' %}
    {% set system_message = messages[0]['content'] | trim + '\n\n' %}
    {% set messages = messages[1:] %}
{% else %}
    {% set system_message = '' %}
{% endif %}

{{ bos_token + system_message}}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ '[INST] ' + message['content'] | trim + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' ' + message['content'] | trim + eos_token }}
    {% endif %}
{% endfor %}
"""

def load_model(basemodel_name):
    # Load base model
    step_time = time.time()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        basemodel_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(basemodel_name, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    tokenizer.bos_token, tokenizer.eos_token

    print(f"Loading base model and tokenizer time taken: {round(time.time() - step_time)}s")
    return model, tokenizer

def format_for_trainer(conversations):
    return [tokenizer.apply_chat_template(conversation, chat_template=template_string, tokenize=False, return_tensors="pt") for conversation in conversations['messages']]


# -----------------------------------------------------------------------------------------------
#
# Load model and tokenizer
start_time = time.time()
model, tokenizer = load_model(basemodel_name)
dataset = load_dataset("json", data_files={"train": "dataset/train.json", "test": "dataset/test.json"})


# Adding the adapters in the layers
adapter = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)
adapter = get_peft_model(adapter, peft_config)


# Training arguments for adapter
training_arguments = TrainingArguments(
    output_dir = './train',
    num_train_epochs = 7, # 1,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 8, # 1,
    optim = "paged_adamw_32bit",
    save_steps = 50,
    learning_rate = 2e-4, # 1e-5
    weight_decay = 0.001,
    fp16 = False,
    bf16 = False,
    max_grad_norm = 0.3,
    max_steps = -1,
    warmup_ratio= 0.1,
    group_by_length = True,
    lr_scheduler_type = "constant",
    logging_steps = 1,
    logging_first_step = True,
    logging_dir = './train/logs',
)

trainer = SFTTrainer(
    model=adapter,
    peft_config=peft_config,
    max_seq_length= None,
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
    train_dataset = dataset['train'],
    eval_dataset = dataset['test'],
    formatting_func = format_for_trainer,
)

# Training adapter.
step_time = time.time()
train_result = trainer.train()
print(f"Training adapter time taken: {round(time.time() - step_time)}s")

# Save the adapter model.
train_result.metrics["train"] = len(dataset['train'])
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.model.save_pretrained(adapter_path)
trainer.tokenizer.save_pretrained(adapter_path)
trainer.model.config.use_cache = True
trainer.model.eval()


# -----------------------------------------------------------------------------------------------
#
# Merge the adapter with base model.
step_time = time.time()
base_model_reload = AutoModelForCausalLM.from_pretrained(
        basemodel_name,
        torch_dtype=torch.bfloat16,
        return_dict=True,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
)
model = PeftModel.from_pretrained(basemodel_name, adapter_path)
model = model.merge_and_unload()

# Reload tokenizer, save merged 'model.safetensors' and 'tokenizer.model' to ./model/full directory.
tokenizer = AutoTokenizer.from_pretrained(basemodel_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.save_pretrained(newmodel_path)
model.save_pretrained(newmodel_path)
model.eval()


print(f"Merging model time taken: {round(time.time() - step_time)}s")
print(f"Total time taken: {round(time.time() - start_time)}s")
