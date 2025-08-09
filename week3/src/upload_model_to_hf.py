import os
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import *
import dotenv
dotenv.load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HF_TOKEN is None:
        raise ValueError("⚠️ HUGGINGFACE_TOKEN env var not found")

HF_USERNAME = os.getenv("HUGGINGFACE_USERNAME")
if HF_USERNAME is None:
        raise ValueError("⚠️ HUGGINGFACE_USERNAME env var not found")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, help="Directory path for the model checkpoint")
    parser.add_argument("--model_name", required=True, help="Model name to use when uploading the model checkpoint to HF")
    args = parser.parse_args()
    
    print("🔥 Loading & Converting from Checkpoint -> Model")
    model = AutoModelForCausalLM.from_pretrained(args.ckpt_dir)

    print("🔥 Loading & Converting from Checkpoint -> Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)

    model_identifier = f"{HF_USERNAME}/{args.model_name}"
    print(f"🚀 Uploading Model Checkpoint to Huggingface as: {model_identifier}")
    model.push_to_hub(model_identifier, token=HF_TOKEN)
    tokenizer.push_to_hub(model_identifier, token=HF_TOKEN)
    
    print("✅ Upload Successful")

if __name__ == "__main__":
    main()