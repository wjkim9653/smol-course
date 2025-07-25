import json
from pathlib import Path
import re
from datasets import Dataset, DatasetDict
from huggingface_hub import login, create_repo, HfApi
from tqdm import tqdm
from pprint import pprint

# ✅ 설정
INPUT_JSONL = "../data/mmlu_generated_qa.jsonl"  # 사용자 jsonl 파일 경로
# HF_REPO_NAME = "your-username/high-school-chemistry"  # 업로드할 HF repo 이름
# HF_TOKEN = "your_hf_token"  # Hugging Face access token
SUBJECT = "high_school_chemistry"

def parse_qa_item(raw_text: str) -> dict:
    """
    다양한 포맷을 대응하는 robust QA 파서
    """
    # 중괄호 블럭 추출
    match = re.search(r"\{(.*)\}", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"Cannot extract main block from:\n{raw_text[:200]}")

    content = match.group(1).strip()

    # 시도 1️⃣: JSON 형태 시도
    try:
        normalized = re.sub(r'(\b\w+\b):', r'"\1":', content)  # "question": ...
        obj = json.loads(normalized)
        question = obj.get("question", "").strip()
        choices = obj.get("choices", [])
        answer = obj.get("answer", "").strip()

        if isinstance(choices, list) and all(isinstance(c, str) for c in choices):
            if answer in ["A", "B", "C", "D"]:
                answer_letter = answer
            elif answer in choices:
                answer_letter = chr(ord("A") + choices.index(answer))
            else:
                raise ValueError(f"Answer '{answer}' not found in choices: {choices}")
            
            return {
                "question": question,
                "choices": choices,
                "answer": answer_letter,
                "subject": "high_school_chemistry"
            }
    except Exception:
        pass  # fallthrough

    # 시도 2️⃣: YAML-like with - A) choices
    question = None
    choices = []
    answer = None

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]

        # question
        if "question" in line:
            q_match = re.match(r'"?question"?\s*:\s*(.*)', line)
            if q_match:
                question = q_match.group(1).strip()

        # choices
        elif "choices" in line:
            i += 1
            while i < len(lines) and lines[i].startswith("-"):
                choice_match = re.match(r"-\s*[A-D]\)\s*(.*)", lines[i])
                if choice_match:
                    choices.append(choice_match.group(1).strip())
                i += 1
            continue  # already incremented

        # answer
        elif "answer" in line:
            a_match = re.match(r'"?answer"?\s*:\s*("?)([A-D]|.+?)\1$', line)
            if a_match:
                answer_raw = a_match.group(2).strip()
                if answer_raw in ["A", "B", "C", "D"]:
                    answer = answer_raw
                elif answer_raw in choices:
                    answer = chr(ord("A") + choices.index(answer_raw))
        i += 1

    # 결과 확인
    if not (question and choices and answer):
        raise ValueError(f"Unparsed QA item:\n{raw_text[:300]}")
    
    return {
        "question": question,
        "choices": choices,
        "answer": answer,
        "subject": "high_school_chemistry"
    }

def load_and_parse_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Parsing"):
            entry = json.loads(line)
            try:
                parsed = parse_qa_item(entry["qa_item"])
                data.append(parsed)
            except Exception as e:
                print(f"⚠️ Parse error: {e}")
                continue
    return data

def save_to_jsonl(data: list, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

def push_to_hub(dataset: Dataset, repo_name: str, hf_token: str):
    login(token=hf_token)
    
    # Repo가 없다면 생성
    api = HfApi()
    if repo_name not in [d.repo_id for d in api.list_datasets()]:
        create_repo(repo_id=repo_name, repo_type="dataset", token=hf_token)

    # push to Hub
    dataset.push_to_hub(repo_name)

def main():
    print("🔍 JSONL 파싱 중...")
    parsed_data = load_and_parse_jsonl(INPUT_JSONL)
    pprint(parsed_data)
    print(f"변환된 MMLU Format Dataset 개수: {len(parsed_data)}")
    save_to_jsonl(parsed_data, "../data/mmlu_official_format.jsonl")
    
    print("📦 Datasets 포맷으로 변환 중...")
    dataset = Dataset.from_list(parsed_data)

    # print("🚀 Hugging Face Hub로 업로드 중...")
    # push_to_hub(dataset, HF_REPO_NAME, HF_TOKEN)
    # print("✅ 업로드 완료!")

if __name__ == "__main__":
    main()
