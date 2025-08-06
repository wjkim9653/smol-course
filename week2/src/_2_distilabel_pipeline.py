import os
import json
import random
import logging
import argparse
from tqdm import tqdm
from distilabel.models import TransformersLLM, OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration
from util import *


SYSTEM_PROMPT="""
You are a QA generator specialized in writing high-school level chemistry questions in the MMLU format.
Given a context, generate **as many diverse and relevant multiple-choice QA items as possible**.
Each QA must adhere to the following format:
```
    {
        "question": "Your question relevant to the given context",
        "choices": ["candidate A", "candidate B", "candidate C", "candidate D"]
        "answer": "(one of following) A, B, C, or D",
    }
```
Return the QA items as a JSON list.
"""
CUSTOM_TEMPLATE = "{{ instruction }}"


def rewrite_instruction(sample_dict):
    sample_dict['instruction'] = (
        f'Based on the provided context below, generate **as many high school-level multiple-choice chemistry questions as appropriate** in the MMLU format.\n'
        f'The output should be a list of JSONs, each following this structure:\n'
        f'{{"question": "...", "choices": ["A", "B", "C", "D"], "answer": "A/B/C/D"}}\n'
        f'Context:\n{sample_dict["context"]}\n'
    )
    return sample_dict


def main(input_path, output_path, model_name, sample_cnt):
    # Load Prompts
    with open(input_path, 'r') as f:
        lines = f.readlines()
        prompts = [rewrite_instruction(json.loads(line)) for line in random.sample(lines, min(sample_cnt, len(lines)))]  # randomly sampled prompts

    # Define Pipeline
    with Pipeline() as pipeline:
        data = LoadDataFromDicts(data=prompts)
        llm = TransformersLLM(
            model=model_name, 
            model_kwargs={
                "torch_dtype":"float16"
            },
            generation_kwargs={
                "max_new_tokens": 1024, 
                "temperature": 0.7, 
                "top_p": 0.92, 
                "do_sample": True, 
                "repetition_penalty": 1.1
            }
        )
        '''llm = OpenAILLM(
            model='gpt-4.1-mini',
            api_key=''
        )'''
        qa_gen = TextGeneration(
            system_prompt=SYSTEM_PROMPT,
            template=CUSTOM_TEMPLATE,
            llm=llm,
            output_mappings={"generation": "qa_item"}
        )
        data >> qa_gen

    # Run
    distiset = pipeline.run(use_cache=False)
    logging.info(distiset)

    # Save
    examples = distiset["default"]["train"].to_list()
    with open(output_path, "w") as f:
        for example in tqdm(examples, desc="🗂️ Saving..."):
            json.dump(example, f, ensure_ascii=False)
            f.write('\n')
    logging.info(f"✅ 총 {len(examples)}개의 Context Chunk에 대해 QA Pairs 생성 성공")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Distilabel Pipeline for QA Generation")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file dir w/ prompt data")
    parser.add_argument("--output", type=str, default="../data/mmlu_generated_qa.jsonl", help="Output JSONL path")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Hugging Face model name")
    parser.add_argument("--sample_cnt", type=int, default=100, help="# of sample context chunks to use for synthetic dataset generation")

    args = parser.parse_args()
    main(args.input, args.output, args.model, args.sample_cnt)