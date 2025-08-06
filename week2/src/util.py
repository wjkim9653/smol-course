import os
import re
import json
import logging
import pythonmonkey

def parse_qa_item_list_w_jsonrepair(text):
    """
    parse llm generations in `list of json` format into serialized `list of dict`.
    input:
        (string) llm-generated text
    output:
        (list of dict) parsed QAs
    """
    jsonrepair = pythonmonkey.require('jsonrepair').jsonrepair
    
    generated_json_blocks = []  # a list to store each blocks of json-format(begins w/ {, ends w/ } strings)
    while True:
        match = re.search(r"\{[\s\S]*?\}", text)
        if not match:
            break
        json_str = match.group(0)
        try:
            repaired_str = jsonrepair(json_str)
            if isinstance(repaired_str, str): 
                dict_obj = json.loads(repaired_str)
            else:
                dict_obj = repaired_str
            generated_json_blocks.append(dict_obj)
        except Exception as e:
            logging.error(f"❌ JSON parsing error: {e}")
            logging.error(f"Content was:\n{json_str}")
            pass
        text = text[match.end():]
    return generated_json_blocks

def parse_qa_item_w_jsonrepair(text):
    jsonrepair = pythonmonkey.require('jsonrepair').jsonrepair

    match = re.search(r"\{[\s\S]*?\}", text)  # {로 시작해 }로 끝나는 전체 블록 매칭, 중간에는 공백문자(\s) 및 문자(\S) 모두 포함되므로 줄바꿈도 포함
    if not match:
        return None
    content = match.group(0)

    fixed_text = jsonrepair(content)
    try:
        return json.loads(fixed_text)
    except Exception as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Content was:\n{fixed_text}")
        return None

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

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")