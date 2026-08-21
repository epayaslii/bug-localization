"""Fine-tune a CodeBERT-style cross-encoder to classify (bug report, code chunk) pairs as
relevant/not-relevant -- the first gradient-based training job anywhere in this project
(confirmed via grep: zero prior Trainer/AutoModelForSequenceClassification/training-loop
code exists). See docs/relevance_classifier_scoping.md for design rationale.

Hyperparameters matched to IQLoc's own published spec where practical (LR 1e-4, batch size
8, 512 token limit, up to 100 epochs, threshold 0.5, 4:1 negative ratio already applied by
scripts/build_classifier_training_data.py) -- one disclosed deviation: EarlyStoppingCallback
(patience 2) on validation F1 as this project's practical HF-Trainer-native stand-in for
IQLoc's literal ReduceLROnPlateau scheduler (Trainer doesn't support that scheduler
natively).

Split is BUG-LEVEL, never chunk-level -- chunks from the same bug must not straddle
train/val, or validation "accuracy" would be inflated by memorized bug-specific vocabulary
rather than real generalization.

ALWAYS run with --smoke-test first (--max-train-examples 200 --epochs 1 --no-cuda,
runs on a laptop CPU in minutes) before any MN5 GPU submission -- catches data-shape bugs,
tokenizer/label-mapping mistakes, and Trainer API version mismatches for free.
"""

import os
import sys
import argparse
import json
import random
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.utils import setup_logging, get_logger

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


class ChunkPairDataset(Dataset):
    """(bug_report, chunk_text) -> label, tokenized as a sentence pair -- standard
    cross-encoder input shape. Pre-tokenizes eagerly in __init__ rather than per-__getitem__
    call: dataset sizes here (hundreds to low thousands of records) are small enough that
    the memory cost is trivial and it keeps __getitem__ simple."""

    def __init__(self, records, tokenizer, max_length=512):
        self.encodings = tokenizer(
            [r["bug_report"] for r in records], [r["chunk_text"] for r in records],
            truncation=True, max_length=max_length, padding="max_length",
        )
        self.labels = [int(r["label"]) for r in records]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def bug_level_split(records, val_fraction, seed):
    """Split by bug_instance_id, not by record -- every chunk from a given bug ends up
    entirely in train or entirely in val, never split across both."""
    bug_ids = sorted(set(r["bug_instance_id"] for r in records))
    rng = random.Random(seed)
    rng.shuffle(bug_ids)
    n_val = max(1, int(len(bug_ids) * val_fraction))
    val_ids = set(bug_ids[:n_val])
    train_records = [r for r in records if r["bug_instance_id"] not in val_ids]
    val_records = [r for r in records if r["bug_instance_id"] in val_ids]
    return train_records, val_records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--training-data', required=True, help='JSONL from scripts/build_classifier_training_data.py')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model-name', default='microsoft/codebert-base')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100, help='Upper bound -- EarlyStoppingCallback stops well before this in practice.')
    parser.add_argument('--early-stopping-patience', type=int, default=2)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-train-examples', type=int, default=None, help='Truncate the training set (smoke-test use).')
    parser.add_argument('--no-cuda', action='store_true', help='Force CPU even if a GPU is visible (smoke-test use).')
    args = parser.parse_args()

    records = [json.loads(line) for line in open(args.training_data)]
    logger.info(f"Loaded {len(records)} chunk records from {args.training_data}")

    train_records, val_records = bug_level_split(records, args.val_fraction, args.seed)
    if args.max_train_examples is not None:
        random.Random(args.seed).shuffle(train_records)
        train_records = train_records[: args.max_train_examples]
    logger.info(f"Train: {len(train_records)} records ({len(set(r['bug_instance_id'] for r in train_records))} bugs), "
                f"Val: {len(val_records)} records ({len(set(r['bug_instance_id'] for r in val_records))} bugs)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    train_dataset = ChunkPairDataset(train_records, tokenizer, max_length=args.max_length)
    val_dataset = ChunkPairDataset(val_records, tokenizer, max_length=args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        no_cuda=args.no_cuda,
        report_to="none",
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    trainer.train()
    final_metrics = trainer.evaluate()
    logger.info(f"Final validation metrics: {final_metrics}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Saved checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
