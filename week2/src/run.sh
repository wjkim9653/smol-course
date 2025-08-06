#!/bin/bash
#SBATCH -J WJK-Smol-Course        # 작업 이름
#SBATCH --output=../logs/%x_%j.out             # 표준 출력 로그 (%x=job name, %j=job id)
#SBATCH --error=../logs/%x_%j.err              # 표준 에러 로그
#SBATCH --gres=gpu:1                           # GPU 1개 요청 (필요 없으면 주석 처리)
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=24G
#SBATCH --nodelist=moana-y2  # 노드 특정 요청
#SBATCH -p batch_ce_ugrad
#SBATCH -t 1-0

# Conda 환경 활성화
# source ~/.bashrc
# conda activate smol-course

# 작업 디렉토리 이동
# cd /data/wjkim9653/repos/smol-course/week2/src

# # 파이프라인 실행
# python _2_distilabel_pipeline.py \
#   --input ../data/pipeline_input_prompts/pipeline_input_prompt_from_raw_pdf_paths_with_pypdf2.jsonl \
#   --output ../data/pipeline_output_generated_QA/mmlu_generated_qa_from_raw_pdf_paths_with_pypdf2_with_model_llama3_1_8B.jsonl \
#   --model meta-llama/Llama-3.1-8B-Instruct \
#   --sample_cnt 2000

# python _3_parsing.py \
#   --input ../data/pipeline_output_generated_QA/mmlu_generated_qa_from_raw_pdf_paths_with_pypdf2_with_model_llama3_1_8B.jsonl \
#   --output ../data/pipeline_output_parsed_MMLU/mmlu_official_format_dataset_from_raw_pdf_paths_with_pypdf2_with_model_llama3_1_8B.jsonl

# python _4_evaluate_dataset.py \
#   --input ../data/pipeline_output_parsed_MMLU/mmlu_official_format_dataset_from_raw_pdf_paths_with_pypdf2_with_model_llama3_1_8B.jsonl

python _5_enhance_dataset.py \
  --input ../data/pipeline_output_parsed_MMLU/mmlu_official_format_dataset_from_raw_pdf_paths_with_pypdf2_with_model_llama3_1_8B.jsonl \
  --intermediate_dir ../data/pipeline_output_filtered_MMLU/intermediate \
  --output ../data/pipeline_output_filtered_MMLU/final_10k_filtered.jsonl

exit 0