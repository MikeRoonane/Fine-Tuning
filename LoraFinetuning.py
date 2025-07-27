import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,BitsAndBytesConfig,DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel
import os


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model_name = "EleutherAI/pythia-410m"  
tokenizer = AutoTokenizer.from_pretrained(model_name) 
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(
    tokenizer = tokenizer,
    mlm=False,  
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query_key_value", "dense"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

output_dir = "./lora_finetuned_model"
checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint")]
if checkpoints:
    latest_checkpoint = sorted(checkpoints)[-1]
    model = PeftModel.from_pretrained(model, latest_checkpoint);
else:
    print("No checkpoints found, starting from scratch.")
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(use_reentrant=False)

def preprocess(example):
    if example["context"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Context:\n{example['context']}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    return {"text": prompt + example["response"] + tokenizer.eos_token}

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=1048, padding="max_length")

dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
dataset = dataset.select(range(2000,5000))  
dataset = dataset.map(preprocess, remove_columns=["instruction", "response", "category", "context"])
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.add_column("labels", tokenized_dataset["input_ids"])
split_data = tokenized_dataset.train_test_split(test_size=0.1)

training_args = TrainingArguments(
    output_dir="./lora_finetuned_model",
    auto_find_batch_size=True,
    gradient_accumulation_steps=32,  
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    label_names=["labels"],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_data["train"],
    eval_dataset=split_data["test"],
    data_collator=data_collator,
)
    
trainer.train()

model.save_pretrained("lora_finetuned_model")
tokenizer.save_pretrained("lora_finetuned_model")
