import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import boto3
import datasets
import evaluate
import numpy as np
from datasets import Audio, Dataset, DatasetDict
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_S3_CLIENT = None


def get_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        session = boto3.session.Session()
        _S3_CLIENT = session.client("s3")
    return _S3_CLIENT


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="facebook/mms-lid-126",
        metadata={"help": "Model identifier from huggingface.co/models"},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store pretrained models downloaded from huggingface.co"}
    )


@dataclass
class DataTrainingArguments:
    train_manifest: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a training manifest JSONL file. If not provided, the script will look for a manifest in the SageMaker training channel.",
        },
    )
    validation_manifest: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a validation manifest JSONL file. If not provided, the script will look for a manifest in the SageMaker validation channel.",
        },
    )
    test_manifest: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a test manifest JSONL file. If not provided, the script will look for a manifest in the SageMaker test channel.",
        },
    )
    audio_column: str = field(
        default="wav", metadata={"help": "Column name containing audio URIs."}
    )
    label_column: str = field(
        default="lang", metadata={"help": "Column name containing class labels."}
    )
    preprocessing_num_workers: int = field(
        default=4, metadata={"help": "Number of worker processes to use when preprocessing audio."}
    )
    audio_cache_dir: str = field(
        default="/opt/ml/input/data/audio_cache",
        metadata={"help": "Directory where S3 audio files will be cached locally."},
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "For debugging, truncate the number of training examples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None, metadata={"help": "For debugging, truncate the number of evaluation examples."}
    )


SAGEMAKER_CHANNEL_ENV_VARS = {
    "train": "SM_CHANNEL_TRAIN",
    "validation": "SM_CHANNEL_VALIDATION",
    "test": "SM_CHANNEL_TEST",
}


def resolve_manifest_path(user_defined: Optional[str], channel: str) -> Optional[str]:
    """Resolve the manifest path either from user input or from a SageMaker channel."""

    if user_defined:
        return user_defined

    channel_env = SAGEMAKER_CHANNEL_ENV_VARS[channel]
    channel_path = os.environ.get(channel_env)
    if not channel_path:
        return None

    path = Path(channel_path)
    if path.is_file():
        return str(path)

    jsonl_files = sorted(path.glob("*.jsonl"))
    if jsonl_files:
        return str(jsonl_files[0])

    json_files = sorted(path.glob("*.json"))
    if json_files:
        return str(json_files[0])

    logger.warning("No manifest file found in %s", channel_path)
    return None


def load_manifest_dataset(manifest_path: str) -> Dataset:
    data_files = manifest_path
    logger.info("Loading manifest from %s", manifest_path)
    dataset = datasets.load_dataset("json", data_files=data_files, split="train")
    return dataset


def build_dataset_dict(train_path: Optional[str], validation_path: Optional[str], test_path: Optional[str]) -> DatasetDict:
    dataset_dict: Dict[str, Dataset] = {}
    if train_path:
        dataset_dict["train"] = load_manifest_dataset(train_path)
    if validation_path:
        dataset_dict["validation"] = load_manifest_dataset(validation_path)
    if test_path:
        dataset_dict["test"] = load_manifest_dataset(test_path)

    if not dataset_dict:
        raise ValueError("At least one manifest must be provided for training, validation, or testing.")

    return DatasetDict(dataset_dict)


def ensure_local_audio(uri: str, cache_dir: Path) -> str:
    if uri.startswith("s3://"):
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        local_path = cache_dir / bucket / key
        if not local_path.exists():
            local_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug("Downloading %s to %s", uri, local_path)
            client = get_s3_client()
            client.download_file(bucket, key, str(local_path))
        return str(local_path)
    return uri


def cache_audio_paths(dataset: DatasetDict, data_args: DataTrainingArguments) -> DatasetDict:
    cache_dir = Path(data_args.audio_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _map_fn(example):
        return {"audio": ensure_local_audio(example[data_args.audio_column], cache_dir)}

    for split, ds in dataset.items():
        logger.info("Caching audio files for %s split", split)
        dataset[split] = ds.map(
            _map_fn,
            remove_columns=[],
            num_proc=data_args.preprocessing_num_workers,
            desc=f"Downloading audio for {split}",
        )

    return dataset


def prepare_label_mapping(dataset: DatasetDict, label_column: str) -> Dict[str, Dict]:
    if "train" not in dataset:
        raise ValueError("A training split is required to build the label set.")

    label_list = sorted(set(dataset["train"][label_column]))
    logger.info("Discovered %d labels: %s", len(label_list), ", ".join(label_list))

    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return {"label_list": label_list, "label2id": label2id, "id2label": id2label}


def preprocess_dataset(
    dataset: DatasetDict,
    feature_extractor: AutoFeatureExtractor,
    data_args: DataTrainingArguments,
    label2id: Dict[str, int],
) -> DatasetDict:
    target_sampling_rate = feature_extractor.sampling_rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))

    def _prepare_batch(batch):
        audio_arrays = [audio["array"] for audio in batch["audio"]]
        sampling_rates = [audio["sampling_rate"] for audio in batch["audio"]]
        if any(sr != target_sampling_rate for sr in sampling_rates):
            logger.warning("Resampling detected; Audio feature extractor will handle conversion.")
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=target_sampling_rate,
            return_attention_mask=True,
        )
        result = {
            "input_values": inputs["input_values"],
            "labels": [label2id[label] for label in batch[data_args.label_column]],
        }
        if "attention_mask" in inputs:
            result["attention_mask"] = inputs["attention_mask"]
        return result

    processed = DatasetDict()
    for split, ds in dataset.items():
        columns_to_keep = set()
        if "utt_id" in ds.column_names:
            columns_to_keep.add("utt_id")
        remove_columns = [col for col in ds.column_names if col not in columns_to_keep]
        logger.info("Preparing features for %s split", split)
        processed_split = ds.map(
            _prepare_batch,
            batched=True,
            batch_size=8,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=remove_columns,
            desc=f"Preprocessing {split}",
        )
        processed_split.set_format(
            type="torch",
            columns=[col for col in ["input_values", "attention_mask", "labels"] if col in processed_split.column_names],
        )
        processed[split] = processed_split

    return processed


def truncate_datasets(dataset: DatasetDict, train_max: Optional[int], eval_max: Optional[int]) -> DatasetDict:
    truncated = DatasetDict()
    for split, ds in dataset.items():
        if split == "train" and train_max is not None:
            truncated[split] = ds.select(range(min(train_max, len(ds))))
        elif split != "train" and eval_max is not None:
            truncated[split] = ds.select(range(min(eval_max, len(ds))))
        else:
            truncated[split] = ds
    return truncated


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False
    set_seed(training_args.seed)

    train_manifest = resolve_manifest_path(data_args.train_manifest, "train")
    eval_manifest = resolve_manifest_path(data_args.validation_manifest, "validation")
    test_manifest = resolve_manifest_path(data_args.test_manifest, "test")

    dataset_dict = build_dataset_dict(train_manifest, eval_manifest, test_manifest)

    if data_args.max_train_samples is not None or data_args.max_eval_samples is not None:
        dataset_dict = truncate_datasets(dataset_dict, data_args.max_train_samples, data_args.max_eval_samples)

    dataset_dict = cache_audio_paths(dataset_dict, data_args)

    test_utt_ids = None
    if "test" in dataset_dict and "utt_id" in dataset_dict["test"].column_names:
        test_utt_ids = list(dataset_dict["test"]["utt_id"])

    label_info = prepare_label_mapping(dataset_dict, data_args.label_column)
    label_list = label_info["label_list"]
    label2id = label_info["label2id"]
    id2label = label_info["id2label"]

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=len(label_list),
        label2id=label2id,
        id2label=id2label,
        cache_dir=model_args.cache_dir,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    dataset_dict = preprocess_dataset(dataset_dict, feature_extractor, data_args, label2id)

    model = AutoModelForAudioClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    data_collator = DataCollatorWithPadding(feature_extractor=feature_extractor, padding=True)

    metric = evaluate.load("accuracy")

    def compute_metrics(prediction):
        preds = np.argmax(prediction.predictions, axis=-1)
        return metric.compute(predictions=preds, references=prediction.label_ids)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict.get("train"),
        eval_dataset=dataset_dict.get("validation"),
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval and dataset_dict.get("validation") is not None:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    if training_args.do_predict and dataset_dict.get("test") is not None:
        predictions = trainer.predict(dataset_dict["test"])
        preds = np.argmax(predictions.predictions, axis=-1)
        output_path = Path(training_args.output_dir) / "test_predictions.json"
        with output_path.open("w") as f:
            for idx, pred in enumerate(preds):
                utt_id = test_utt_ids[idx] if test_utt_ids is not None else idx
                record = {
                    "utt_id": utt_id,
                    "prediction": id2label[pred],
                    "score_distribution": predictions.predictions[idx].tolist(),
                }
                f.write(json.dumps(record) + "\n")
        trainer.log_metrics("test", predictions.metrics)
        trainer.save_metrics("test", predictions.metrics)


if __name__ == "__main__":
    main()
