import argparse
import json
from typing import Dict, Optional

from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session


DEFAULT_HYPERPARAMETERS: Dict[str, str] = {
    "model_name_or_path": "facebook/mms-lid-126",
    "do_train": "true",
    "do_eval": "true",
    "output_dir": "/opt/ml/model",
    "overwrite_output_dir": "true",
    "eval_strategy": "step", "eval_steps": "500",
    "save_strategy": "step", "save_steps": "500",
    "save_strategy": "step", "logging_steps": "100",
    "per_device_train_batch_size": "8",
    "per_device_eval_batch_size": "8",
    "gradient_accumulation_steps": "1",
    "learning_rate": "3e-5",
    "num_train_epochs": "5",
    "warmup_ratio": "0.1",
    "load_best_model_at_end": "true",
    "metric_for_best_model": "accuracy",
    "preprocessing_num_workers": "16",
    "initialize_label_prototypes": "true",
}


def parse_overrides(raw_overrides: Optional[str]) -> Dict[str, str]:
    if not raw_overrides:
        return {}
    overrides = json.loads(raw_overrides)
    return {str(key): str(value) for key, value in overrides.items()}


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

    hyperparameters = DEFAULT_HYPERPARAMETERS.copy()
    hyperparameters.update(parse_overrides(args.hyperparameters))

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
        max_run=3600,
    )

    estimator.fit(inputs=inputs)


if __name__ == "__main__":
    main()
