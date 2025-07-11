# lighteval==0.9.2, datasets==3.5.0
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters


evaluation_tracker = EvaluationTracker(
    output_dir="./tmp",
    save_details=False,
    push_to_hub=False,
    push_to_tensorboard=False,
    public=False,
    hub_results_org=False,
)

pipeline_params = PipelineParameters(
    launcher_type=ParallelismManager.ACCELERATE,
    # env_config=env_config,
    # job_id=1,
    # override_batch_size=1,
    # num_fewshot_seeds=0,
    # max_samples=10, # IMPORTANT!
    # use_chat_template=False,
)

domain_tasks = "leaderboard|mmlu:anatomy|5|0,leaderboard|mmlu:professional_medicine|5|0,leaderboard|mmlu:high_school_biology|5|0,leaderboard|mmlu:high_school_chemistry|5|0"

model_config = TransformersModelConfig(
        # model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        model_name="Qwen/Qwen2.5-0.5B",
        # dtype="float16",
        use_chat_template=True,
        # use_chat_template=False,
)
pipeline = Pipeline(
    tasks=domain_tasks,
    pipeline_parameters=pipeline_params,
    evaluation_tracker=evaluation_tracker,
    model_config=model_config
)

pipeline.evaluate()

smol_results = pipeline.get_results()

pipeline.show_results()