# MMS LID Fine-Tuning Utilities

This repository contains utilities to launch a SageMaker training job and fine-tune the [`facebook/mms-lid-126`](https://huggingface.co/facebook/mms-lid-126) model on audio classification data stored in Amazon S3. The data is described through JSON Lines manifest files whose entries look like:

```json
{"utt_id": "utt_7fa3a1d7ca9c", "wav": "s3://bucket/path/audio.wav", "lang": "zh", "length": 2.85}
```

## Training entry point (`src/train.py`)

The training script is compatible with SageMaker's HuggingFace containers and can also be run locally. Key features:

- Automatically resolves manifest locations either from explicit arguments (`--train_manifest`, etc.) or from SageMaker input channels (e.g., `/opt/ml/input/data/train`).
- Downloads referenced S3 audio files into a local cache directory (default `/opt/ml/input/data/audio_cache`).
- Uses Hugging Face `datasets` multiprocessing (`num_proc`) to parallelise both the S3 downloads and the feature extraction.
- Builds the label mapping from the training manifest and fine-tunes `facebook/mms-lid-126` with Hugging Face `Trainer`.
- Saves evaluation metrics and optional prediction outputs (`test_predictions.json`).

Example local launch:

```bash
python src/train.py \
  --model_name_or_path facebook/mms-lid-126 \
  --train_manifest /path/to/train.jsonl \
  --validation_manifest /path/to/validation.jsonl \
  --output_dir ./lid-checkpoints \
  --num_train_epochs 5 \
  --per_device_train_batch_size 8 \
  --preprocessing_num_workers 8 \
  --do_train --do_eval
```

## SageMaker launcher (`src/launch_sagemaker.py`)

`sagemaker_launch.py` automates the creation of a SageMaker training job that uses `src/train.py` as the entry point.

```bash
python src/launch_sagemaker.py \
  --role arn:aws:iam::123456789012:role/SageMakerExecutionRole \
  --job-name lid-mms-training \
  --train-manifest-s3 s3://my-bucket/manifests/train.jsonl \
  --validation-manifest-s3 s3://my-bucket/manifests/validation.jsonl \
  --output-s3 s3://my-bucket/lid-training-artifacts/
```

Hyperparameters default to a sensible configuration but can be overridden with `--hyperparameters '{"learning_rate": 5e-5, "num_train_epochs": 10}'`. Additional CLI options allow configuration of instance type/count, managed spot training, and container versions.

## Dependencies

All libraries required for model fine-tuning are listed in `requirements.txt`. Install them locally with:

```bash
pip install -r requirements.txt
```

The SageMaker launcher additionally requires the AWS SageMaker SDK (`pip install sagemaker`).
