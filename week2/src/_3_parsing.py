import argparse
import json
from pathlib import Path
import re
from datasets import Dataset, DatasetDict
from huggingface_hub import login, create_repo, HfApi
from tqdm import tqdm
from pprint import pprint
from util import *

# ✅ 설정
# HF_REPO_NAME = "your-username/high-school-chemistry"  # 업로드할 HF repo 이름
# HF_TOKEN = "your_hf_token"  # Hugging Face access token

def load_and_parse_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Parsing"):
            entry = json.loads(line)
            try:
                # jsonrepair 기반 파싱
                parsed = parse_qa_item_list_w_jsonrepair(entry["qa_item"])  # list of dicts
                for candidate in parsed:
                    
                    # 파싱한 json의 형태 문제 있는지 체크 및 정상화
                    choices_flag = True  # 정상 형식 플래그
                    if (type(candidate["choices"]) is list):  # choices가 리스트 타입인 경우
                        if len(candidate['choices']) == 0:  # 길이 0 -> 비정상으로 간주
                            choices_flag = False
                        elif not all(type(elem) is str for elem in candidate["choices"]):  # choices가 리스트 타입이나 choices 중 특정 elem이 str 타입이 아닌 경우 -> str으로 각 elem 변환작업 거친 후 정상으로 간주
                            tmp = []
                            for elem in candidate["choices"]:
                                tmp.append(str(elem))
                            candidate["choices"] = tmp
                            choices_flag = True
                        elif len(candidate["choices"]) == 1 and isinstance(candidate["choices"][0], str) and ',' in candidate["choices"][0]:
                            split_choices = [c.strip() for c in candidate["choices"][0].split(',')]
                            if all(isinstance(choice, str) for choice in split_choices):
                                candidate["choices"] = split_choices
                            else:
                                choices_flag = False
                    
                    answer_flag = True
                    if type(candidate["answer"]) is int:  # answer가 int 타입인 경우
                        candidate["answer"] = str(candidate["answer"])
                    if type(candidate["answer"]) is list:  # answer가 list 타입인 경우
                        if len(candidate["answer"]) == 1:  # answer list의 길이가 1인 경우 -> 정상/비정상 체크할 것
                            if type(candidate["answer"][0]) is int:  # answer list 안의 유일요소 타입이 int인 경우 -> str 변환
                                candidate["answer"] = str(candidate["answer"][0])
                            else:
                                if type(candidate["answer"][0]) is str:  # answer list 안의 유일요소 타입이 str인 경우
                                    candidate["answer"] = candidate["answer"][0].upper()
                                else:  # answer list 안의 유일요소 타입이 int, str 모두 해당되지 않을 경우 -> 비정상
                                    answer_flag = False
                        else:  # answer list의 길이가 0이거나 2 이상인 경우 -> 비정상으로 간주
                            answer_flag = False
                    if (answer_flag is not False) and (type(candidate["answer"]) is str):  # answer가 str 타입인 경우
                        if candidate["answer"] in ["a", "b", "c", "d", "e", "f", "A", "B", "C", "D", "E", "F"]:  # 단일 알파벳 선지인 경우 중 소문자인 경우 -> 대문자로 변환
                            candidate["answer"] = candidate["answer"].upper()
                        elif any(candidate["answer"].upper() == choice.upper() for choice in candidate["choices"]):  # A, B, C, D, E, F 대신 Choices 내 특정 선지와 동일한 텍스트인 경우 변환해주기
                            for idx, choice in enumerate(candidate["choices"]):
                                if candidate["answer"].upper() == choice.upper():
                                    candidate["answer"] = chr(ord("A") + idx)
                        elif candidate["answer"].upper() in ["T", "F"]:  # T, F 양식의 answer인 경우
                            if (len(candidate["choices"]) == 2) and all(elem.upper() in ["TRUE", "FALSE"] for elem in candidate["choices"]):  # 정답이 T, F 이며 choices가 True, False인 경우
                                if candidate["answer"].upper() == "T":  # 정답이 T인 경우
                                    idx = [i for i, c in enumerate(candidate["choices"]) if c.upper() == "TRUE"] # find the index of "TRUE" from choices
                                    if idx:
                                        candidate["answer"] = chr(ord("A") + idx[0])
                                    else:
                                        answer_flag = False
                                else:  # 정답이 F인 경우
                                    idx = [i for i, c in enumerate(candidate["choices"]) if c.upper() == "FALSE"]
                                    if idx:
                                        candidate["answer"] = chr(ord("A") + idx[0])
                                    else:
                                        answer_flag = False
                            else:  # choices가 True, False 중 최소 1개 이상을 누락하고 있는 경우 -> 비정상
                                answer_flag = False
                        else:  # 상기 answer 정상화 로직으로 해결되지 않는 예외사례의 경우 -> 비정상
                            answer_flag = False
                    if not (choices_flag and answer_flag):
                        print("⚠️ 비정상 candidate:", candidate)
                    # 정상 간주 파싱 딕셔너리만 추가
                    if choices_flag and answer_flag:
                        data.append(candidate)
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

def main(input_path, output_path):
    print("🔍 JSONL 파싱 중...")
    parsed_data = load_and_parse_jsonl(input_path)
    # pprint(parsed_data)
    print(f"변환된 MMLU Format Dataset 개수: {len(parsed_data)}")
    save_to_jsonl(parsed_data, output_path)
    
    # print("📦 Datasets 포맷으로 변환 중...")
    # dataset = Dataset.from_list(parsed_data)

    # print("🚀 Hugging Face Hub로 업로드 중...")
    # push_to_hub(dataset, HF_REPO_NAME, HF_TOKEN)
    # print("✅ 업로드 완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse the Generated QAs and Create Final MMLU-format File")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file dir for pipeline output(generated QAs)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file dir for parsed final MMLU format dataset file")
    args = parser.parse_args()
    main(args.input, args.output)