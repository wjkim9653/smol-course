LETTER_INDICES = ["A", "B", "C", "D"]
def mmlu_query_builder(line, topic="high_school_chemistry"):
    """
    input:
        line: (jsonl, str) dict로 binarize되지 않은 상태인, jsonl 파일 한 행을 읽어들인 str을 받음
    output:
        query: (str) 질문, 선지를 포함하고 답변 자리만 비워둔 형태의 문자열 반환
        gold_ix: (str) 정답 선지에 해당하는 정수 인덱스 값을 str으로 변환하여 반환
    """
    query = f"The following are multiple choice questions (with answers) about  {topic.replace('_', ' ')}.\n\n"
    query += line["question"] + "\n"
    query += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["choices"])])
    query += "Answer:"

    # gold_ix = LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return query, line["answer"]