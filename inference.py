from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

base_model=AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m")
base_tokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
base_tokenizer.pad_token = base_tokenizer.eos_token
base_model.config.pad_token_id = base_tokenizer.pad_token_id
model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model").to(device)
tokenizer= AutoTokenizer.from_pretrained("./fine_tuned_model")

def inference(prompt, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
    encoding = tokenizer(prompt, return_tensors="pt", max_length=max_input_tokens, truncation=True, padding=True)
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding["attention_mask"].to(model.device)

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

prompt=input("Enter a prompt: ")
prompt_template = """### Instruction:
{prompt}

### Response:
"""
print("=======================================================================")
print("Fine-tuned model output:\n", inference(prompt_template.format(prompt=prompt),model,tokenizer))
print("=======================================================================")
print("Base output:\n", inference(prompt_template.format(prompt=prompt),base_model,base_tokenizer))
print("=======================================================================")
