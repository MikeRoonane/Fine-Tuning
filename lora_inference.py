import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model_name = "EleutherAI/pythia-410m"
adapter_path = "./lora_finetuned_model/checkpoint-42"

base_tok = AutoTokenizer.from_pretrained(base_model_name)
base_tok.pad_token = base_tok.eos_token

tokenizer = AutoTokenizer.from_pretrained(adapter_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

def inference(prompt, model, tokenizer, max_input_tokens=1000, max_output_tokens=80):
    encoding = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=max_input_tokens)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_output_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt):].strip()

# === Prompt input ===
prompt = input("Enter a prompt: ")
prompt_template = f"""### Instruction:
{prompt}
### Response:
"""

print("\n========================================")
print("LoRA Fine-tuned Model Output:")
print(inference(prompt_template, model, tokenizer))
print("========================================")

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
base_model.eval() 

print("Base Model Output:\n", inference(prompt_template, base_model, base_tok))
