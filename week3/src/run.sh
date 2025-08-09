#!/bin/bash
#SBATCH --nodelist=moana-y4  # 노드 특정 요청
#SBATCH -J WJK-Smol-Course        # 작업 이름
#SBATCH --output=../logs/%x_%j.out             # 표준 출력 로그 (%x=job name, %j=job id)
#SBATCH --error=../logs/%x_%j.err              # 표준 에러 로그
#SBATCH --gres=gpu:1                           # GPU 1개 요청 (필요 없으면 주석 처리)
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=24G
#SBATCH -p batch_ce_ugrad
#SBATCH -t 1-0

# Conda 환경 활성화
# source ~/.bashrc
# conda activate smol-course

# 작업 디렉토리 이동
# cd /data/wjkim9653/repos/smol-course/week3/src

# HF에 Model Checkpoint 업로드
# python upload_model_to_hf.py \
#   --ckpt_dir ../model/SFT-SmolLM2-360M-Instruct-RUN-20250808-211417/checkpoint-3375 \
#   --model_name SmolLM2-360M-CheMMLU-v2

# 학습 진행
python sft.py \
  --dataset_identifier wjkim9653/highschool_chemical_train_wonjin_10000_v1 \
  --base_model HuggingFaceTB/SmolLM2-360M-Instruct \
  --trained_model_dir ../model

# 가장 최근 디렉토리 찾기
latest_ckpt=$(ls -td ../model/SFT-SmolLM2-360M-Instruct-RUN-* | head -n 1)
python evaluate.py \
  --checkpoint_dir "$latest_ckpt"

# HF에 Model Checkpoint 업로드
# python upload_model_to_hf.py \
#   --ckpt_dir ../model/SFT-SmolLM2-360M-Instruct-RUN-/checkpoint- \
#   --model_name SmolLM2-360M-CheMMLU-

exit 0