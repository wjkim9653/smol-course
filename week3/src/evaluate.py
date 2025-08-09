import os
import argparse
import logging
logging.basicConfig(level=logging.INFO)
from datetime import timedelta
from pprint import pprint
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import lighteval
from transformers import AutoModelForCausalLM
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig


def glob_checkpoint_paths(dir):
    checkpoint_paths = {}
    checkpoint_tags = []
    for dirpath, dirnames, filenames in os.walk(dir):
        unsorted_checkpoint_tags = dirnames  # ['checkpoint-3000', 'checkpoint-5250', 'checkpoint-1250', 'checkpoint-750', 'checkpoint-1000', 'checkpoint-2500', 'checkpoint-3250', 'checkpoint-3750', 'checkpoint-4000', 'checkpoint-1500', 'checkpoint-1750', 'checkpoint-5500', 'checkpoint-2000', 'checkpoint-5000', 'checkpoint-4500', 'checkpoint-4250', 'checkpoint-2750', 'checkpoint-2250', 'checkpoint-3500', 'checkpoint-500', 'checkpoint-250', 'checkpoint-4750']
        sorted_checkpoint_tags = sorted(unsorted_checkpoint_tags, key=lambda x: int(x.split('-')[-1]))
        checkpoint_tags = sorted_checkpoint_tags
        break
    print(f"checkpoint tags: {checkpoint_tags}\n")

    for checkpoint_tag in checkpoint_tags:
        checkpoint_paths[checkpoint_tag] = os.path.join(dir, checkpoint_tag)
    print(f"checkpoint paths:")
    pprint(checkpoint_paths)

    return checkpoint_tags, checkpoint_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_tasks", default="leaderboard|mmlu:high_school_chemistry|5|0")
    parser.add_argument("--checkpoint_dir", default="../model/SFT-SmolLM2-360M-Instruct-RUN-20250808-155010", help="path to the directory where model checkpoints are saved")  # ../model/SFT-SmolLM2-360M-Instruct-RUN-20250808-155010
    args = parser.parse_args()

    domain_tasks = args.domain_tasks
    ckpt_tags, ckpt_paths = glob_checkpoint_paths(args.checkpoint_dir)
    ckpt_results = {}

    # 모델 체크포인트별 성능 평가
    for ckpt_tag in tqdm(ckpt_tags):
        print("="*100)
        print(f"🔥 Evaluating -> {ckpt_tag}")
        tag = ckpt_tag
        ckpt = ckpt_paths[tag]

        evaluation_tracker = EvaluationTracker(
            output_dir="~/tmp",
            save_details=False,
            push_to_hub=False,
            push_to_tensorboard=False,
            public=False,
            hub_results_org=False,
        )

        pipeline_params = PipelineParameters(
            launcher_type=ParallelismManager.ACCELERATE,
            job_id=1,
            num_fewshot_seeds=0,
        )

        config = TransformersModelConfig(
            model_name=ckpt, 
            device="cuda",
            # dtype="bfloat16",
            dtype="float32",
            batch_size=1,
        )
        
        model = TransformersModel(config=config)
        
        pipeline = Pipeline(
            tasks=domain_tasks,
            pipeline_parameters=pipeline_params,
            evaluation_tracker=evaluation_tracker,
            model=model,
            model_config=config
        )
        
        pipeline.evaluate()
        pipeline.show_results()
        
        result = pipeline.get_results()
        ckpt_results[tag] = result

        ####################################################################

        del model
        del pipeline
        torch.cuda.empty_cache()  # GPU 메모리 완전 해제
        print("="*100)

    # Acc, StdErr 추출
    acc = []
    stderr = []
    for ckpt_tag in ckpt_tags:  # already sorted
        result = ckpt_results[ckpt_tag]
        '''
        {
        "config_general": {...},
        "results": {
            "leaderboard:mmlu:high_school_chemistry:5|acc": 0.3206,
            "leaderboard:mmlu:high_school_chemistry:5|acc_stderr": 0.0067,
            ...
        },
        "versions": {...},
        "config_tasks": {...},
        "summary_tasks": {...},
        "summary_general": {...}
        }
        '''
        try:
            key = args.domain_tasks.replace("|", ":")[:-2]
            acc_key = "acc"
            stderr_key = "acc_stderr"
            results = result["results"].get(key, {})
            val = results.get(acc_key, np.nan)
            err = results.get(stderr_key, 0)
        except:
            logging.error(f"couldn't retrieve actual evaluation result accuracy from:\n{result}")
            val = float('nan')
            err = 0
        acc.append(val)
        stderr.append(err)


    # 플롯
    logging.info("🎨 Plotting...")
    plt.figure(figsize=(10, 6))
    plt.errorbar(ckpt_tags, acc, yerr=stderr, fmt='-o', ecolor='red', capsize=5)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Checkpoint")
    plt.ylabel("High School Chemistry Accuracy")
    plt.title("High School Chemistry Accuracy by Checkpoint with Std Err")
    plt.tight_layout()
    logging.info("📂 Saving Plot...")
    savedir = os.path.join(args.checkpoint_dir, "performance_plot.png")
    plt.savefig(savedir)
    logging.info("✅ Complete")
    
if __name__ == "__main__":
    main()