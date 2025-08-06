import argparse
import os
import json
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import faiss

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from util import *

# ✅ 기본 설정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CHOICE_LETTERS = ["A", "B", "C", "D"]

# ✅ STEP 1: 선지 4개만 추려내고 정답 분포 균형화 (A~D 4000개씩 = 16,000)
def step1_filter_and_balance(input_data):
    buckets = defaultdict(list)  # 처음 키 지정할 때 값을 주지 않으면 해당 키에 대한 값을 빈 리스트로 초기화

    for sample in input_data:
        if len(sample["choices"]) != 4:
            continue  # 선지 4개짜리만 유지, 그 외의 경우 -> 스킵
        try:
            answer_idx = CHOICE_LETTERS.index(sample["answer"].upper())  # 샘플 원본에서 정답의 선지 내 인덱스(A:0, B:1, C:2, D:3...) 체크
        except ValueError:
            continue  # 정답이 A, B, C, D 중 존재하지 않는 예외 경우 -> 스킵

        # 선지 순서 랜덤 섞기
        paired = list(zip(CHOICE_LETTERS, sample["choices"]))  # zip() -> ("A", sample["choices"][0]), ("B", sample["choices"][1]), ("C", sample["choices"][2]), ("D", sample["choices"][3])
        random.shuffle(paired)

        new_choices = [c for _, c in paired]  # 순서 랜덤으로 섞은 choice들 다시 리스트로 담기
        correct_text = sample["choices"][answer_idx]  # 정답 선지 텍스트
        new_answer_idx = new_choices.index(correct_text)  # 정답 선지 텍스트의 새로운 (랜덤화 이후) 인덱스
        sample["choices"] = new_choices  # 랜덤화한 선지들로 리매핑
        sample["answer"] = CHOICE_LETTERS[new_answer_idx]  # 신규 인덱스로 정답 리매핑
        buckets[sample["answer"]].append(sample)  # bucket에 (신규 인덱스에 해당하는) 정답 선지(A,B,C,D 중) 키의 밸류 리스트에 샘플 추가

    # 각 정답 문자별 4000개까지만 유지
    balanced = []
    for letter in CHOICE_LETTERS:
        samples = buckets[letter]
        random.shuffle(samples)
        balanced.extend(samples[:4000])  # 랜덤화한 각 선지(A,B,C,D) 별 4000개씩 담아 총 16000개 짜리 밸런스드 샘플 리스트 반환
    return balanced

# ✅ STEP 2: 질문 중복 제거 (임베딩 + FAISS + 유사도 기준)
def step2_deduplicate(samples, model):
    questions = [s["question"] for s in samples]
    embeddings = model.encode(questions, show_progress_bar=True, normalize_embeddings=True)  # normalize하여 단위벡터화 -> 코사인 유사도 계산을 단순 내적으로 간소화 가능
    # embeddings는 N개 질문들을 각각 D 차원으로 임베딩한 Numpy 배열이며, (N, D)의 shape을 가짐

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 벡터 내적 기반 유사도 검색 인덱스 생성
    index.add(embeddings)

    _, neighbors = index.search(embeddings, 2)  # 자기 자신 + 가장 가까운 이웃 (top-2) 서치
    # neighbors -> 각 질문 임베딩벡터(N개) 별로 가장 유사한 2개 벡터를 찾음 -> (N, 2)의 shape을 가짐 (neighbors[i][0] -> 자기자신, neighbors[i][1] -> 최근접 이웃의 index)
    to_remove = set()

    for i, n in enumerate(neighbors):  # neighbors는 (N, 2)의 shape을 가지는 Numpy 배열임을 기억 (neighbors[i][0] -> 자기자신, neighbors[i][1] -> 최근접 이웃의 index)
        if n[1] == i:  # 최근접 이웃 또한 자기 자신이라는 말은, 유효한 다른 질문이 없는 상태임을 의미 -> 비교 무의미하므로 스킵
            continue
        sim = np.dot(embeddings[i], embeddings[n[1]])  # 코사인 유사도 (단위 벡터 간 코사인 유사도이므로 단순 내적)
        if sim > 0.96:  # 유사 질문으로 간주
            to_remove.add(i)  # 유사 질문은 삭제목록에 추가

    filtered = [s for i, s in enumerate(samples) if i not in to_remove]  # 삭제 목록 인덱스 제외한 샘플들만 필터링
    return filtered

# ✅ STEP 3: 난이도 추정 (sLLM을 이용한 정답 예측 성공 여부)
def step3_estimate_difficulty(samples, model, tokenizer):
    annotated = []
    for sample in tqdm(samples, desc="Estimating difficulty"):
        prompt = f"Q: {sample['question']}\n"
        for i, c in enumerate(sample["choices"]):
            prompt += f"{CHOICE_LETTERS[i]}. {c}\n"
        prompt += "Answer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1)  # 허용 토큰 1개로 제한해 선지만 출력하도록 유도
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        guess = generated.strip()[-1].upper()  # 모델이 예측한 답변 선지

        # 난이도 간단히 매핑
        sample["difficulty"] = "hard"
        if guess == sample["answer"]:  # 모델이 예측한 답변 선지가 정답인 경우
            sample["difficulty"] = "easy"  # 난이도 낮음
        elif guess in CHOICE_LETTERS:  # 모델이 예측한 답변 선지가 오답이며 선지 내에 있는 경우
            sample["difficulty"] = "medium"  # 난이도 중간
        annotated.append(sample)
    return annotated

# ✅ STEP 4: distractor 품질 평가 (정답과 너무 떨어진 오답 제거)
def step4_filter_distractors(samples, model):
    filtered = []
    for s in samples:
        correct = s["choices"][CHOICE_LETTERS.index(s["answer"])]
        good_distractors = 0
        for i, c in enumerate(s["choices"]):
            if CHOICE_LETTERS[i] == s["answer"]:
                continue
            sim = model.similarity(correct, c)  # 실제 정답의 임베딩 값과 오답에 해당하는 distractor의 임베딩 값 간의 유사도
            if sim > 0.3:  # 정답과의 유사도가 높게 나오는 훌륭한 distractor인 경우
                good_distractors += 1
        if good_distractors >= 2:  # 최소 2개 distractor는 괜찮아야 통과
            filtered.append(s)  # 선지에 있는 오답 중 good distractor에 해당하는 것이 2개 미만인 경우는 필터링해 샘플에서 제외
    return filtered

# ✅ STEP 5: 다양성 기반 샘플링 (MMR 방식으로 10,000개 선택)
def step5_select_final(samples, model, k=10000):
    embeddings = model.encode([s["question"] for s in samples], normalize_embeddings=True)

    selected = []
    selected_set = set()
    sims = np.inner(embeddings, embeddings)  # cosine similarity matrix, (N, N) shape
    scores = sims.mean(axis=1)  # 각 질문이 전체 질문에 대해 갖는 유사도 평균 행렬, (N, 1) shape => 품질 점수 (Relevance Score in MMR)

    ranked = list(np.argsort(-scores))  # 점수 높은 순으로 내림차순 정렬
    selected.append(ranked[0])  # 첫 질문은 가장 높은 relevance score 가진 샘플
    selected_set.add(ranked[0])

    while len(selected) < k:  # k개 샘플까지 반복 샘플링
        mmr_scores = []
        for i in range(len(samples)):
            if i in selected_set:
                continue
            relevance = scores[i]
            diversity = max([sims[i][j] for j in selected])  # 이미 선택되어 있는 샘플들과의 유사도 중 가장 큰 값이 diversity값이 됨
            mmr = 0.7 * relevance - 0.3 * diversity  # MMR 가중치는 0.7로, MMR은 질문의 좋은 정도(relevance) - 질문이 기존 질문과 비슷한 정도(diversity)로 산출 (가중치는 0.7, (1-0.7)로 잡음)
            mmr_scores.append((mmr, i))
        mmr_scores.sort(reverse=True)  # 높은 MMR 점수 순으로 정렬
        _, best = mmr_scores[0]  # 가장 높은 MMR 점수를 받은 샘플 1개를 선정해 selected에 추가
        selected.append(best)
        selected_set.add(best)

    final = [samples[i] for i in selected]
    return final

# ✅ 임베딩 유틸 클래스
class EmbeddingModel:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, **kwargs):
        return self.model.encode(texts, **kwargs)

    def similarity(self, a, b):
        vecs = self.encode([a, b], normalize_embeddings=True)
        return np.dot(vecs[0], vecs[1])

# ✅ sLLM 로딩 함수
def load_sllm(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    return tokenizer, model

# ✅ MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--intermediate_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.makedirs(args.intermediate_dir, exist_ok=True)

    print("🔹 Loading input data...")
    data = load_jsonl(args.input)

    print("🔹 Step 1: Filtering 4-choice & balancing answers...")
    data1 = step1_filter_and_balance(data)
    save_jsonl(data1, os.path.join(args.intermediate_dir, "step1_balanced.jsonl"))

    print("🔹 Step 2: Removing duplicates...")
    embed_model = EmbeddingModel()
    data2 = step2_deduplicate(data1, embed_model)
    save_jsonl(data2, os.path.join(args.intermediate_dir, "step2_dedup.jsonl"))

    print("🔹 Step 3: Estimating difficulty using sLLM...")
    # tokenizer, sllm = load_sllm("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer, sllm = load_sllm("Qwen/Qwen3-0.6B")
    data3 = step3_estimate_difficulty(data2, sllm, tokenizer)
    save_jsonl(data3, os.path.join(args.intermediate_dir, "step3_difficulty.jsonl"))

    print("🔹 Step 4: Filtering low-quality distractors...")
    data4 = step4_filter_distractors(data3, embed_model)
    save_jsonl(data4, os.path.join(args.intermediate_dir, "step4_distractor_filtered.jsonl"))

    print("🔹 Step 5: Selecting final 10k with diversity...")
    data5 = step5_select_final(data4, embed_model, k=10000)
    save_jsonl(data5, args.output)

    print(f"✅ Done! Final dataset saved to: {args.output}")

if __name__ == "__main__":
    main()