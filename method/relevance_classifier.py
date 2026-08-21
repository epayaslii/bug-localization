"""Loads and runs a fine-tuned CodeBERT-style relevance classifier (Phase B of the
fine-tuning/prompt-optimization plan) -- a cross-encoder scoring (bug report, code chunk)
pairs, trained by scripts/train_relevance_classifier.py. Mirrors
method/embedding_retriever.py's _load_model/_DEVICE caching pattern rather than inventing a
new one.

Unlike the embedding models in method/embedding_retriever.py (which embed each text
independently, then compare via cosine similarity), a cross-encoder scores a (query,
document) PAIR jointly in one forward pass -- no separate embedding step, no cosine
similarity, just a single relevance probability per pair. This is a fundamentally
different mechanism from both existing relevance-judgment modes (--relevance-mode llm:
zero-shot text generation; --relevance-mode embedding-cosine: independent embeddings +
similarity) -- see docs/relevance_classifier_scoping.md for why that matters (it isn't
subject to the output-token-budget ceiling that made the zero-shot LLM path miss most of
its candidate pool).
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dataset.utils import get_logger

logger = get_logger(__name__)

_DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

_CLASSIFIER_CACHE = {}


def load_classifier(checkpoint_path: str):
    """Load (and cache) a fine-tuned sequence-classification checkpoint. checkpoint_path
    is a local directory (the output of scripts/train_relevance_classifier.py's
    Trainer.save_model()), not a Hugging Face Hub model ID -- there's no pretrained
    checkpoint for this task to pull from the Hub."""
    if checkpoint_path not in _CLASSIFIER_CACHE:
        logger.info(f"Loading relevance classifier checkpoint: {checkpoint_path}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path).to(_DEVICE).eval()
        _CLASSIFIER_CACHE[checkpoint_path] = (tokenizer, model)
    return _CLASSIFIER_CACHE[checkpoint_path]


def score_chunks(bug_report: str, chunk_texts: list[str], checkpoint_path: str, batch_size: int = 8, max_length: int = 512) -> list[float]:
    """Score each chunk's relevance to bug_report, returning one probability per chunk in
    [0, 1] (P(relevant)) -- same batched-list-in, flat-list-out shape as
    method.embedding_retriever.embed_texts, for consistency with the rest of method/.
    max_length=512 matches IQLoc's own published classifier spec (see
    docs/relevance_classifier_scoping.md)."""
    if not chunk_texts:
        return []

    tokenizer, model = load_classifier(checkpoint_path)
    scores = []

    with torch.no_grad():
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i + batch_size]
            queries = [bug_report] * len(batch)
            inputs = tokenizer(
                queries, batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt",
            ).to(_DEVICE)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]  # P(label=1="relevant")
            scores.extend(probs.cpu().tolist())

    return scores
