import os
import json
import random
from tqdm import tqdm
from pprint import pprint
from distilabel.models import TransformersLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

with open("../data/mmlu_prompt_input.jsonl", "r") as f:
    # prompts = [json.loads(line) for line in f]
    # prompts = [json.loads(line) for _, line in zip(range(100), f)]
    
    lines = f.readlines()
    prompts = [json.loads(line) for line in random.sample(lines, min(2000, len(lines)))]  # 2000 randomly sampled prompts

with Pipeline() as pipeline:
    data = LoadDataFromDicts(data=prompts)
    llm = TransformersLLM(model="meta-llama/Llama-3.2-3B-Instruct", model_kwargs={"torch_dtype":"float16"})# , generation_kwargs={"max_new_tokens": 128, "temperature": 0.7})
    qa_gen = TextGeneration(llm=llm, output_mappings={"generation": "qa_item"}) # , verbose=True)
    data >> qa_gen


if __name__ == "__main__":
    distiset = pipeline.run(use_cache=False)
    print(distiset)
    examples = distiset["default"]["train"].to_list()
    with open("../data/mmlu_generated_qa.jsonl", "w") as f:
        for example in tqdm(examples, desc="🗂️ Saving..."):
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')
    print(f"✅ 총 {len(examples)}개의 문제 생성 완료")
