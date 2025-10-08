import argparse
import json
from typing import Dict, Optional

from sagemaker.huggingface import HuggingFace
from sagemaker.session import Session


DEFAULT_HYPERPARAMETERS: Dict[str, str] = {
    "model_name_or_path": "facebook/mms-lid-126",
    "do_train": "true",
    "do_eval": "true",
    "output_dir": "/opt/ml/model",
    "overwrite_output_dir": "true",
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "logging_steps": "50",
    "per_device_train_batch_size": "8",
    "per_device_eval_batch_size": "8",
    "gradient_accumulation_steps": "1",
    "learning_rate": "3e-5",
    "num_train_epochs": "5",
    "warmup_ratio": "0.1",
    "load_best_model_at_end": "true",
    "metric_for_best_model": "accuracy",
    "preprocessing_num_workers": "8",
}


def parse_overrides(raw_overrides: Optional[str]) -> Dict[str, str]:
    if not raw_overrides:
        return {}
    overrides = json.loads(raw_overrides)
    return {str(key): str(value) for key, value in overrides.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a SageMaker training job for MMS LID fine-tuning")
    parser.add_argument("--role", required=True, help="IAM role ARN that SageMaker will assume.")
    parser.add_argument("--job-name", required=True, help="Name of the SageMaker training job.")
    parser.add_argument("--train-manifest-s3", required=True, help="S3 URI to the training manifest JSONL file.")
    parser.add_argument("--validation-manifest-s3", help="Optional S3 URI to the validation manifest JSONL file.")
    parser.add_argument("--test-manifest-s3", help="Optional S3 URI to the test manifest JSONL file.")
    parser.add_argument("--output-s3", required=True, help="S3 URI where model artifacts will be stored.")
    parser.add_argument("--instance-type", default="ml.g5.2xlarge", help="Instance type for training.")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances to launch.")
    parser.add_argument("--volume-size", type=int, default=300, help="EBS volume size in GB.")
    parser.add_argument(
        "--hyperparameters",
        help="JSON dictionary of hyperparameter overrides to merge with the defaults.",
    )
    parser.add_argument(
        "--transformers-version",
        default="4.38",
        help="Transformers container version to use.",
    )
    parser.add_argument("--pytorch-version", default="2.1", help="PyTorch version to use.")
    parser.add_argument("--py-version", default="py310", help="Python version to use.")
    parser.add_argument(
        "--disable-spot-instance",
        action="store_true",
        help="Disable SageMaker managed spot training (on-demand instances will be used).",
    )
    parser.add_argument(
        "--max-wait",
        type=int,
        default=None,
        help="When using spot instances, the maximum wait time (in seconds).",
    )

    args = parser.parse_args()

    hyperparameters = DEFAULT_HYPERPARAMETERS.copy()
    hyperparameters.update(parse_overrides(args.hyperparameters))

    inputs = {"train": args.train_manifest_s3}
    if args.validation_manifest_s3:
        inputs["validation"] = args.validation_manifest_s3
    if args.test_manifest_s3:
        inputs["test"] = args.test_manifest_s3

    session = Session()

    estimator = HuggingFace(
        entry_point="train.py",
        source_dir="src",
        role=args.role,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        volume_size=args.volume_size,
        transformers_version=args.transformers_version,
        pytorch_version=args.pytorch_version,
        py_version=args.py_version,
        base_job_name=args.job_name,
        hyperparameters=hyperparameters,
        output_path=args.output_s3,
        sagemaker_session=session,
        use_spot_instances=not args.disable_spot_instance,
        max_wait=None if args.disable_spot_instance else args.max_wait,
        dependencies=["requirements.txt"],
    )

    estimator.fit(inputs=inputs, job_name=args.job_name)


if __name__ == "__main__":
    main()
