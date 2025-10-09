import argparse
import json
from typing import Any, Dict, Optional

from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session


DEFAULT_HYPERPARAMETERS: Dict[str, Any] = {
    # Model configuration (ModelArguments)
    "model_name_or_path": "facebook/mms-lid-512",
    "cache_dir": None,
    "initialize_label_prototypes": True,
    "language_mapping_strategy": "pure",  # pure or accented
    "train_classifier_head_only": False,

    # Data manifests and preprocessing (DataTrainingArguments)
    "train_manifest": None,
    "validation_manifest": None,
    "test_manifest": None,
    "audio_column": "wav",
    "label_column": "lang",
    "preprocessing_num_workers": 16,
    "audio_cache_dir": "/opt/ml/input/data/audio_cache",
    "class_weighted_loss": True,
    # "max_train_samples": 100,
    # "max_eval_samples": 100,

    # Core training run behaviour (TrainingArguments)
    "output_dir": "/opt/ml/model",
    "overwrite_output_dir": True,
    "seed": 42,
    "do_train": True,
    "do_eval": True,
    "do_predict": False,
    "disable_tqdm": True,
    "bf16": True,

    # Optimisation schedule (TrainingArguments)
    "num_train_epochs": 5,
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "learning_rate": 3e-5,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "linear",

    # Evaluation, logging, and checkpointing (TrainingArguments)
    "eval_strategy": "steps", "eval_steps": 500,
    "eval_on_start": True,
    "save_strategy": "steps", "save_steps": 500,
    "save_total_limit": 2,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "greater_is_better": True,
    "logging_strategy": "steps", "logging_steps": 100,
    "report_to": "wandb",
}


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_overrides(raw_overrides: Optional[str]) -> Dict[str, str]:
    if not raw_overrides:
        return {}
    overrides = json.loads(raw_overrides)
    return {str(key): _stringify(value) for key, value in overrides.items()}


def serialise_hyperparameters(
    defaults: Dict[str, Any], overrides: Dict[str, str]
) -> Dict[str, str]:
    hyperparameters: Dict[str, str] = {}
    for key, value in defaults.items():
        if value is None:
            continue
        hyperparameters[key] = _stringify(value)
    hyperparameters.update(overrides)
    return hyperparameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a SageMaker training job for MMS LID fine-tuning")
    # parser.add_argument("--role", required=True, help="IAM role ARN that SageMaker will assume.")
    # parser.add_argument("--job-name", required=True, help="Name of the SageMaker training job.")
    parser.add_argument(
        "--train-manifest-s3",
        default="s3://us-west-2-ehmli/lid-job/manifests/data_train_0930.s3.jsonl",
        help="S3 URI to the training manifest JSONL file.",
    )
    parser.add_argument(
        "--validation-manifest-s3",
        default="s3://us-west-2-ehmli/lid-job/manifests/data_valid_0930.s3.jsonl",
        help="S3 URI to the validation manifest JSONL file.",
    )
    parser.add_argument("--test-manifest-s3", help="Optional S3 URI to the test manifest JSONL file.")
    parser.add_argument(
        "--output-s3",
        default="s3://us-west-2-ehmli/lid-job/models/",
        help="S3 URI where model artifacts will be stored.")
    parser.add_argument("--instance-type", default="ml.g5.4xlarge", help="Instance type for training.")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances to launch.")
    parser.add_argument("--volume-size", type=int, default=300, help="EBS volume size in GB.")
    parser.add_argument(
        "--hyperparameters",
        help="JSON dictionary of hyperparameter overrides to merge with the defaults.",
    )
    parser.add_argument(
        "--image-uri",
        default="763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:2.5.1-transformers4.49.0-gpu-py311-cu124-ubuntu22.04",
        help="Container image URI to use for the training job.",
    )
    # parser.add_argument(
    #     "--disable-spot-instance",
    #     action="store_true",
    #     help="Disable SageMaker managed spot training (on-demand instances will be used).",
    # )
    # parser.add_argument(
    #     "--max-wait",
    #     type=int,
    #     default=None,
    #     help="When using spot instances, the maximum wait time (in seconds).",
    # )

    args = parser.parse_args()

    overrides = parse_overrides(args.hyperparameters)
    hyperparameters = serialise_hyperparameters(DEFAULT_HYPERPARAMETERS, overrides)

    inputs = {"train": args.train_manifest_s3}
    if args.validation_manifest_s3:
        inputs["validation"] = args.validation_manifest_s3
    if args.test_manifest_s3:
        inputs["test"] = args.test_manifest_s3

    session = Session()

    estimator = PyTorch(
        entry_point="train.py",
        source_dir="src",
        image_uri=args.image_uri,
        role=get_execution_role(),
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        volume_size=args.volume_size,
        # base_job_name=args.job_name,
        hyperparameters=hyperparameters,
        output_path=args.output_s3,
        sagemaker_session=session,
        # use_spot_instances=not args.disable_spot_instance,
        # max_wait=None if args.disable_spot_instance else args.max_wait,
        dependencies=["requirements.txt"],
        max_run=24*3600,
    )

    estimator.fit(inputs=inputs)


if __name__ == "__main__":
    main()
