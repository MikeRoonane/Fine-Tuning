## Fine Tuning

This project fine-tunes a causal language model (EleutherAI/pythia-410m) using LoRA (Low-Rank Adaptation) and 4-bit quantization on the [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.

##  Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/lora-finetuning.git
cd lora-finetuning

# Create and activate environment
python -m venv myenv
source myenv/bin/activate  # or `myenv\Scripts\activate` on Windows

# Install requirements
pip install -r requirements.txt
