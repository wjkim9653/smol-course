import os
from pathlib import Path
import argparse
import torch
from datetime import datetime
from datasets import load_dataset
from prompts import *
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import wandb
import dotenv
dotenv.load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

def format_example(example):
    prompt, gold = mmlu_prompt_builder(example)

    return {"prompt": prompt, "label": gold}

def tokenize_example(example, tokenizer):
    formatted = format_example(example)
    prompt = formatted["prompt"]
    label = formatted["label"]
    
    full_text = prompt + " " + label

    tokenized = tokenizer(
        full_text,
        truncation=True,
        padding="max_length",
        max_length=512,
    )

    labels = [-100] * len(tokenized["input_ids"])  # labels를 모두 -100으로 초기화 (loss 계산 미반영하기 위해)
    prompt_len = len(tokenizer(prompt)["input_ids"])  # 입력부에 해당하는 토큰들의 길이 계산
    generation_label_ids = tokenizer(label)["input_ids"]  # 실제 생성하는 정답부분만 (loss 계산 반영하기 위해) (마스킹 되지 않은) 정상 토큰 사용
    generation_len = len(generation_label_ids)  # 생성부에 해당하는 토큰들의 길이 계산
    labels[prompt_len:prompt_len+generation_len] = generation_label_ids

    tokenized["labels"] = labels
    return tokenized

def main():
    if not torch.cuda.is_available():
        raise ValueError("No CUDA available")

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
    dataset = load_dataset(args.dataset_identifier).remove_columns(["difficulty"])
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    # ✅ 전처리
    tokenized_dataset = dataset["train"].map(
        lambda x: tokenize_example(x, tokenizer),
        batched=False
    )

    print("==== Example tokenized input and labels ====")
    for i in range(3):  # 원하는 개수만큼
        sample = tokenized_dataset[i]
        input_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        label_ids = sample["labels"]
        # labels는 -100이 있는 자리 제외하고 실제 정답 토큰만 디코딩
        label_text = tokenizer.decode([id for id in label_ids if id != -100], skip_special_tokens=True)

        print(f"Sample {i+1}:")
        print("Input tokens:", sample["input_ids"])
        print("Input text:", input_text)
        print("Labels:", label_ids)
        print("Label text (masked tokens 제외):", label_text)
        print("="*50)

    # ✅ 학습 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=15,
        # learning_rate=1e-5,
        # learning_rate=3e-5,
        learning_rate=4e-5,
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        save_strategy="steps",
        save_steps=125,
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