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

python sft.py \
  --dataset_identifier wjkim9653/highschool_chemical_train_wonjin_10000_v1 \
  --base_model HuggingFaceTB/SmolLM2-360M-Instruct \
  --trained_model_dir ../model

exit 0