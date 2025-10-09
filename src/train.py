import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse
import wandb
wandb.login(key="019cb7049b90ead6ebf8c152ba16611436fcf45a")

import boto3
import datasets
import evaluate
import numpy as np
import torch
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

from language_groups import (
    CHINESE_LANGUAGE_CODES,
    EAST_AND_SOUTHEAST_ASIA_ENGLISH_CODES,
    EAST_AND_SOUTHEAST_ASIA_LANGUAGE_CODES,
    ENGLISH_LANGUAGE_CODES,
)


class WeightedLossTrainer(Trainer):
    """Trainer subclass that supports class-weighted cross-entropy loss."""

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.class_weights is None:
            return super().compute_loss(model, inputs, return_outputs)

        labels = inputs.get("labels")
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs)

        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**model_inputs)
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("Model outputs must contain `logits` when using class-weighted loss.")

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


_S3_CLIENT = None


def get_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        session = boto3.session.Session()
        _S3_CLIENT = session.client("s3")
    return _S3_CLIENT


def _build_target_to_sources(
    id2label: Dict[int, str], mapping_strategy: str, other_scope: str
) -> Dict[str, List[str]]:
    """Map MMS language codes to zh/en/other groups based on the chosen strategy."""

    normalized_codes = [label.lower() for label in id2label.values()]

    if mapping_strategy == "accented":
        zh_sources = sorted({code for code in normalized_codes if code in CHINESE_LANGUAGE_CODES})
        en_sources = sorted({code for code in normalized_codes if code in ENGLISH_LANGUAGE_CODES})
    elif mapping_strategy == "pure":
        zh_sources = [code for code in normalized_codes if code == "cmn"]
        en_sources = [code for code in normalized_codes if code == "eng"]
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"Unsupported language mapping strategy: {mapping_strategy}")

    used_sources = set(zh_sources) | set(en_sources)
    remaining_sources = {code for code in normalized_codes if code not in used_sources}

    if other_scope == "global":
        other_sources = sorted(remaining_sources)
    elif other_scope == "east_and_southeast_asia":
        regional_codes = set(EAST_AND_SOUTHEAST_ASIA_LANGUAGE_CODES)
        regional_english = set(EAST_AND_SOUTHEAST_ASIA_ENGLISH_CODES)
        regional_sources = remaining_sources & (regional_codes - regional_english)
        other_sources = sorted(regional_sources)
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"Unsupported other-language scope: {other_scope}")

    logger.info(
        "Language mapping strategy '%s' (other=%s) -> zh:%d en:%d other:%d",
        mapping_strategy,
        other_scope,
        len(zh_sources),
        len(en_sources),
        len(other_sources),
    )

    return {"zh": zh_sources, "en": en_sources, "other": other_sources}


def initialize_classifier_with_prototypes(model, model_args: "ModelArguments", label2id: Dict[str, int]) -> None:
    """Optionally initialize classifier weights from MMS-LID prototypes."""

    try:
        base_model = AutoModelForAudioClassification.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to load base model for prototype initialization: %s", exc)
        return

    if base_model.config.id2label:
        target_to_sources = _build_target_to_sources(
            base_model.config.id2label,
            model_args.language_mapping_strategy,
            model_args.other_language_scope,
        )
        base_label2id = {
            label.lower(): idx for idx, label in base_model.config.id2label.items()
        }
    elif base_model.config.label2id:
        target_to_sources = _build_target_to_sources(
            {idx: label for label, idx in base_model.config.label2id.items()},
            model_args.language_mapping_strategy,
            model_args.other_language_scope,
        )
        base_label2id = {
            label.lower(): idx for label, idx in base_model.config.label2id.items()
        }
    else:
        logger.warning(
            "Base model %s does not define label2id or id2label mapping; skipping prototype initialization",
            model_args.model_name_or_path,
        )
        return
    base_weight = base_model.classifier.weight.detach()
    base_bias = base_model.classifier.bias.detach()

    with torch.no_grad():
        for target_label, source_labels in target_to_sources.items():
            if target_label not in label2id:
                continue

            source_indices = [base_label2id.get(code.lower()) for code in source_labels]
            source_indices = [idx for idx in source_indices if idx is not None]

            if not source_indices:
                logger.warning("No matching source languages found for target label '%s'", target_label)
                continue

            target_idx = label2id[target_label]
            weight_mean = base_weight[source_indices].mean(dim=0)
            bias_mean = base_bias[source_indices].mean()

            model.classifier.weight.data[target_idx] = weight_mean.to(model.classifier.weight.device)
            model.classifier.bias.data[target_idx] = bias_mean.to(model.classifier.bias.device)
            logger.info(
                "Initialized classifier weights for '%s' using %d source languages", target_label, len(source_indices)
            )

    del base_model


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="facebook/mms-lid-126",
        metadata={"help": "Model identifier from huggingface.co/models"},
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where to store pretrained models downloaded from huggingface.co"}
    )
    initialize_label_prototypes: bool = field(
        default=False,
        metadata={
            "help": (
                "If true, initialize the classification head using averaged prototypes from the "
                "facebook/mms-lid-126 checkpoint for zh/en/other labels."
            )
        },
    )
    language_mapping_strategy: str = field(
        default="accented",
        metadata={
            "help": (
                "Controls how MMS language codes are grouped when initializing zh/en/other prototypes. "
                "Use 'accented' to map all Chinese varieties to zh and all English varieties to en. "
                "Use 'pure' to map only cmn to zh and eng to en while routing other varieties to other."
            )
        },
    )
    other_language_scope: str = field(
        default="global",
        metadata={
            "help": (
                "Controls which languages are grouped into the 'other' prototype when initializing from MMS models. "
                "Use 'global' to include every language that is not mapped to zh or en. "
                "Use 'east_and_southeast_asia' to include only East and Southeast Asian languages that are not mapped to zh or en."
            )
        },
    )
    train_classifier_head_only: bool = field(
        default=False,
        metadata={
            "help": "If true, freeze all model parameters except the classifier head so only it is updated during training.",
        },
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
    class_weighted_loss: bool = field(
        default=True,
        metadata={
            "help": "If true, compute class weights from the training split and use them for a weighted cross-entropy loss."
        },
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

    valid_strategies = {"accented", "pure"}
    if model_args.language_mapping_strategy not in valid_strategies:
        raise ValueError(
            f"language_mapping_strategy must be one of {sorted(valid_strategies)}; received "
            f"{model_args.language_mapping_strategy!r}."
        )

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
        ignore_mismatched_sizes=True,
    )

    if model_args.initialize_label_prototypes:
        initialize_classifier_with_prototypes(model, model_args, label2id)

    if model_args.train_classifier_head_only:
        classifier_head = getattr(model, "classifier", None)
        if classifier_head is None:
            raise ValueError(
                "Model does not expose a `classifier` attribute; cannot train only the classifier head."
            )

        for param in model.parameters():
            param.requires_grad = False
        for param in classifier_head.parameters():
            param.requires_grad = True
        logger.info("Training classifier head only; all other model parameters have been frozen.")

    data_collator = DataCollatorWithPadding(tokenizer=feature_extractor, padding=True)

    class_weights = None
    if data_args.class_weighted_loss and dataset_dict.get("train") is not None:
        train_labels = dataset_dict["train"]["labels"]
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.numpy()
        label_counts = np.bincount(train_labels, minlength=len(label_list)).astype(float)
        zero_mask = label_counts == 0
        if zero_mask.any():
            logger.warning(
                "Encountered %d classes with zero training examples; their weights will be set to zero.",
                int(zero_mask.sum()),
            )
        total = label_counts.sum()
        class_weight_values = np.zeros_like(label_counts)
        non_zero = ~zero_mask
        if non_zero.any():
            class_weight_values[non_zero] = total / (len(label_counts) * label_counts[non_zero])
        class_weights = torch.tensor(class_weight_values, dtype=torch.float)
        logger.info("Using class-weighted loss with weights: %s", class_weights.tolist())

    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    confusion_metric = evaluate.load("confusion_matrix")

    def compute_metrics(prediction):
        preds = np.argmax(prediction.predictions, axis=-1)
        references = prediction.label_ids

        accuracy = accuracy_metric.compute(predictions=preds, references=references)["accuracy"]
        precision = precision_metric.compute(
            predictions=preds,
            references=references,
            average=None,
            zero_division=0,
        )["precision"]
        recall = recall_metric.compute(
            predictions=preds,
            references=references,
            average=None,
            zero_division=0,
        )["recall"]
        confusion = confusion_metric.compute(
            predictions=preds,
            references=references,
            labels=list(range(len(label_list))),
        )["confusion_matrix"]

        confusion_matrix = np.asarray(confusion)
        per_class_accuracy = []
        for idx, label in enumerate(label_list):
            true_positives = confusion_matrix[idx, idx]
            total_actual = confusion_matrix[idx].sum()
            class_accuracy = (true_positives / total_actual) if total_actual else 0.0
            per_class_accuracy.append(class_accuracy)
            logger.info(
                "Evaluation metrics for class '%s': accuracy=%.4f precision=%.4f recall=%.4f",
                label,
                class_accuracy,
                precision[idx],
                recall[idx],
            )

        logger.info("Evaluation confusion matrix (rows=actual, cols=predicted):\n%s", confusion_matrix)

        metrics = {"accuracy": float(accuracy)}
        for idx, label in enumerate(label_list):
            metrics[f"accuracy_{label}"] = float(per_class_accuracy[idx])
            metrics[f"precision_{label}"] = float(precision[idx])
            metrics[f"recall_{label}"] = float(recall[idx])

        return metrics

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_dict.get("train"),
        eval_dataset=dataset_dict.get("validation"),
        tokenizer=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
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
