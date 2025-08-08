import os
import argparse
from datasets import Dataset
from util import *
from pathlib import Path
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
    parser.add_argument("--jsonl_file", required=True)
    parser.add_argument("--hf_dataset_name", required=True)
    args = parser.parse_args()

    
    print("📂 Loading JSONL file")
    data = load_jsonl(args.jsonl_file)
    print("🔥 Converting JSONL -> HF Dataset")
    dataset = Dataset.from_list(data)

    dataset_identifier = f"{HF_USERNAME}/{args.hf_dataset_name}"
    print(f"🚀 Uploading the Dataset to Huggingface as: {dataset_identifier}")
    dataset.push_to_hub(repo_id=dataset_identifier, token=os.getenv("HUGGINGFACE_TOKEN"))
    print("✅ Upload Successful")

if __name__ == "__main__":
    main()