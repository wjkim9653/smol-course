import torch
print(torch.cuda.is_available())  # True여야 GPU 사용 가능
print(torch.cuda.device_count())  # 사용 가능한 GPU 개수
print(torch.cuda.get_device_name(0))  # 첫 번째 GPU 이름

import lighteval
import os
from datetime import timedelta
from transformers import AutoModelForCausalLM

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

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

domain_tasks = "leaderboard|mmlu:anatomy|5|0,leaderboard|mmlu:professional_medicine|5|0,leaderboard|mmlu:high_school_biology|5|0,leaderboard|mmlu:high_school_chemistry|5|0"

from lighteval.models.transformers.transformers_model import TransformersModel, TransformersModelConfig

import logging
logging.basicConfig(level=logging.INFO)

####################################################################

qwen3_model_id = "Qwen/Qwen3-4B"
qwen3_config = TransformersModelConfig(
    model_name=qwen3_model_id, 
    device="cuda",
    dtype="bfloat16",
    batch_size=1,
)
qwen3_model = TransformersModel(config=qwen3_config)
pipeline = Pipeline(
    tasks=domain_tasks,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    model=qwen3_model,
    model_config=qwen3_config
)
pipeline.evaluate()
qwen3_results = pipeline.get_results()
pipeline.show_results()

####################################################################

del qwen3_model
del pipeline
torch.cuda.empty_cache()  # GPU 메모리 완전 해제

####################################################################

llama3_1_model_id = "meta-llama/Llama-3.2-3B-Instruct"
llama3_1_config = TransformersModelConfig(
    model_name=llama3_1_model_id,
    device="cude",
    dtype="bfloat16",
    batch_size=1,
)
llama3_1_model = TransformersModel(config=llama3_1_config)

pipeline = Pipeline(
    tasks=domain_tasks,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    model=llama3_1_model,
    model_config=llama3_1_config
)
pipeline.evaluate()
llama3_1_results = pipeline.get_results()
pipeline.show_results()

#####################################################################

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame.from_records(llama3_1_results["results"]).T["acc"].rename("Llama-3.2-3B-Instruct")
_df = pd.DataFrame.from_records(qwen3_results["results"]).T["acc"].rename("Qwen3-4B")
df = pd.concat([df, _df], axis=1)

ax = df.plot(kind="barh", figsize=(10, 6))

plt.title("Accuracy Comparison")
plt.xlabel("Accuracy")
plt.tight_layout()  # 레이아웃 잘리거나 겹침 방지
plt.savefig("accuracy_comparison.png", dpi=300)
plt.show()  # 노트북에서 보기용