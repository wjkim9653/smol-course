import os
from pathlib import Path
import argparse
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import wandb
import dotenv
dotenv.load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

LETTER_INDICES = ["A", "B", "C", "D"]

def format_example(example):
    question = example["question"]
    choices = example["choices"]
    answer = example["answer"]

    prompt = f"Q: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{LETTER_INDICES[i]}. {choice}\n"
    prompt += "Answer:"

    return {"prompt": prompt, "label": answer.strip().upper()}

def tokenize_example(example, tokenizer):
    formatted = format_example(example)
    prompt = formatted["prompt"]
    label = formatted["label"]

    tokenized = tokenizer(
        prompt + " " + label,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_identifier", required=True, default="wjkim9653/highschool_chemical_train_wonjin_10000_v1")
    parser.add_argument("--base_model", required=True, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--trained_model_dir", required=True, default="../model")
    args = parser.parse_args()

    # ✅ 고유한 run 식별자 생성
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"SFT-{args.base_model.split('/')[-1]}-RUN-{timestamp}"
    output_dir = os.path.join(args.trained_model_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # ✅ WANDB 초기화
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        name=run_id
    )

    # ✅ HF에서 데이터셋 불러오기
    dataset = load_dataset(args.dataset_identifier)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    # ✅ 전처리
    tokenized_dataset = dataset["train"].map(
        lambda x: tokenize_example(x, tokenizer),
        batched=False
    )

    # ✅ 학습 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        learning_rate=1e-5,
        save_strategy="steps",
        save_steps=250,
        fp16=True,
        logging_steps=10,
        report_to="wandb",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ✅ Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

if __name__ == "__main__":
    main()