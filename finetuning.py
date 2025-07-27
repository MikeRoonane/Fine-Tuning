import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

def preprocess(example, tokenizer):
    if example["context"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Context:\n{example['context']}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    text = prompt + example["response"] + tokenizer.eos_token
    return {"text": text}

def tokenize(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=2048)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_name = "EleutherAI/pythia-70m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    dataset = dataset.map(
        preprocess,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["instruction", "response", "category", "context"]
    )

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        fn_kwargs={"tokenizer": tokenizer}
    )

    split_data = tokenized_dataset.train_test_split(test_size=0.1)

    training_args = TrainingArguments(
        output_dir="C:/Users/miker/OneDrive/Desktop/finetuning",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=True,
        logging_steps=50,
        report_to="none",
    )
    os.makedirs(training_args.output_dir, exist_ok=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split_data["train"],
        eval_dataset=split_data["test"],
        data_collator=data_collator,
    )
    trainer.train()

    save_path = "./fine_tuned_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    def inference(prompt, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
        input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=max_input_tokens, truncation=True).to(device)
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_output_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)[len(prompt):].strip()

    ft_tokenizer = AutoTokenizer.from_pretrained(save_path)
    ft_model = AutoModelForCausalLM.from_pretrained(save_path).to(device)

    prompt = "Explain what is fine tuning?"
    print("Base output:\n", inference(prompt, model, tokenizer))
    print("\nFine Tuned:\n", inference(prompt, ft_model, ft_tokenizer))