LETTER_INDICES = ["A", "B", "C", "D"]

def mmlu_prompt_builder(line, topic="high_school_chemistry"):
    """
    input:
        line: (jsonl, str) dict로 binarize되지 않은 상태인, jsonl 파일 한 행을 읽어들인 str을 받음
    output:
        prompt: (str) 질문, 선지를 포함하고 답변 자리만 비워둔 형태의 문자열 반환
        gold: (str) 정답 선지 알파벳
    """
    prompt = f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n"
    prompt += line["question"] + "\n"
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["choices"])])
    prompt += "Answer:"
    
    gold = line["answer"]
    if isinstance(gold, int):
        gold = LETTER_INDICES[gold]
    gold = gold.strip().upper()
    # gold_ix = LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]
    
    return prompt, gold