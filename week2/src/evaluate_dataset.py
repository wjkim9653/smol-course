import json
import re
from lexical_diversity import lex_div as ld
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

def parse_qa_item(text):
    match = re.search(r"\{(.|\n)*\}", text)
    if not match:
        return None

    content = match.group(0)

    # 키에 쌍따옴표 추가
    content = re.sub(r'(\bquestion\b)\s*:', r'"\1":', content)
    content = re.sub(r'(\bchoices\b)\s*:', r'"\1":', content)
    content = re.sub(r'(\banswer\b)\s*:', r'"\1":', content)

    # question 값 쌍따옴표 추가 (없으면)
    content = re.sub(
        r'("question"\s*:\s*)([^"\{\}\[\]\n][^"\n]*)',
        r'\1"\2"',
        content
    )

    # answer 값 쌍따옴표 추가 (없으면)
    content = re.sub(
        r'("answer"\s*:\s*)([^\s",\}\]]+)',
        r'\1"\2"',
        content
    )

    # choices 배열 변환
    lines = content.splitlines()
    new_lines = []
    choices = []
    in_choices = False

    for line in lines:
        if '"choices":' in line:
            in_choices = True
            new_lines.append('"choices": [')
            continue
        if in_choices:
            if re.match(r'\s*-\s*[A-Z]\)', line):
                choice_text = re.sub(r'\s*-\s*[A-Z]\)\s*', '', line).strip()
                choices.append(f'"{choice_text}"')
            else:
                in_choices = False
                new_lines.append(", ".join(choices) + "],")
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Insert missing comma between question and choices if necessary
    for i in range(len(new_lines) - 1):
        if '"question"' in new_lines[i] and not new_lines[i].strip().endswith(','):
            if '"choices"' in new_lines[i + 1]:
                new_lines[i] = new_lines[i] + ','

    fixed_text = "\n".join(new_lines)

    try:
        return json.loads(fixed_text)
    except Exception as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Content was:\n{fixed_text}")
        return None

jsonl_path = "../data/mmlu_generated_qa.jsonl"

num_samples = 0
input_lengths = []
output_lengths = []
outputs = []

with open(jsonl_path, "r") as f:
    for line in tqdm(f, desc="Reading"):
        obj = json.loads(line.strip())
        qa = None
        if "qa_item" in obj and isinstance(obj["qa_item"], str):
            qa = parse_qa_item(obj["qa_item"])

        if qa and all(k in qa for k in ("question", "choices", "answer")):
            num_samples += 1

            # question 길이 (토큰 수)
            q_tokens = word_tokenize(qa["question"])
            input_lengths.append(len(q_tokens))

            # choices + answer 길이
            output_text = " ".join(qa["choices"]) + " " + qa["answer"]
            o_tokens = word_tokenize(output_text)
            output_lengths.append(len(o_tokens))

            outputs.extend(q_tokens + o_tokens)
        else:
            print("⚠️ Invalid or unparsable qa_item:", obj.get("qa_item", "")[:100])

avg_input_len = sum(input_lengths) / num_samples if num_samples else 0
avg_output_len = sum(output_lengths) / num_samples if num_samples else 0

mattr = ld.mattr(outputs, window_length=50) if outputs else 0
hdd = ld.hdd(outputs) if outputs else 0
mtld = ld.mtld(outputs) if outputs else 0

print("\n📊 Evaluation Results")
print(f"# Samples             : {num_samples}")
print(f"Avg Input Length     : {avg_input_len:.2f} tokens")
print(f"Avg Output Length    : {avg_output_len:.2f} tokens")
print(f"Lexical Diversity (MATTR): {mattr:.4f}")
print(f"Lexical Diversity (HD-D) : {hdd:.4f}")
print(f"Lexical Diversity (MTLD) : {mtld:.4f}")