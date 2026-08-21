"""One-off comparison: score the sparse-output chunk-relevance format (list only the
relevant chunks) against the same dev-set bugs as the dense-output baseline
(scripts/optimize_relevance_prompt.py), so the two are directly comparable -- same dev
set, same subsample selection (random.Random(seed=0), same --dev-bugs-subsample value),
same model/max_tokens/num_ctx. If this beats the dense format's F1=0.000 (5 bugs, tp=0
fp=49 fn=20), that's real evidence the output-token-budget bottleneck (not prompt wording,
not the model's judgment quality) was the dominant problem -- see
scripts/optimize_relevance_prompt.py's module docstring / the diagnosis that motivated
this script.
"""

import os
import sys
import argparse
import json
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.utils import setup_logging, get_logger
from method.models import SparseChunkRelevanceResponse
from method.ollama_localizer import OllamaLocalizer
from method.prompt import PromptGenerator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from optimize_relevance_prompt import load_dev_set  # noqa: E402

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-set-jsonl', required=True)
    parser.add_argument('--dev-bugs-subsample', type=int, default=5)
    parser.add_argument('--model', default='qwen2.5-coder:7b')
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--num-ctx', type=int, default=32768)
    parser.add_argument('--ollama-host', default=None)
    args = parser.parse_args()

    dev_by_bug = load_dev_set(args.dev_set_jsonl, dev_bugs_subsample=args.dev_bugs_subsample)
    logger.info(f"Scoring sparse format against {len(dev_by_bug)} dev bugs")

    localizer = OllamaLocalizer(model=args.model, host=args.ollama_host, max_tokens=args.max_tokens, num_ctx=args.num_ctx)
    prompt_gen = PromptGenerator()

    from types import SimpleNamespace
    tp = fp = fn = tn = 0
    n_calls_failed = 0
    per_bug_coverage = []

    for bug_id, bug_records in dev_by_bug.items():
        fake_bug = SimpleNamespace(
            repo=bug_records[0]["repo"], instance_id=bug_id,
            hints_text=bug_records[0]["hints_text"], bug_report=bug_records[0]["bug_report"],
        )
        chunks = [(r["file"], r["chunk_index"], r["chunk_text"]) for r in bug_records]
        prompt = prompt_gen.generate_sparse_chunk_relevance_feedback_prompt(fake_bug, chunks)

        try:
            response = localizer.invoke_structured(prompt, SparseChunkRelevanceResponse)
            relevant_set = {(c.file, c.chunk_index) for c in response.relevant_chunks}
        except Exception as e:
            logger.warning(f"Call failed for {bug_id}: {e}")
            n_calls_failed += 1
            continue

        per_bug_coverage.append((bug_id, len(relevant_set), len(chunks)))
        for r in bug_records:
            key = (r["file"], r["chunk_index"])
            predicted = key in relevant_set
            actual = r["label"]
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and actual:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    logger.info(f"tp={tp} fp={fp} fn={fn} tn={tn} n_calls_failed={n_calls_failed}")
    logger.info(f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    for bug_id, n_relevant, n_chunks in per_bug_coverage:
        logger.info(f"  {bug_id}: predicted {n_relevant} relevant out of {n_chunks} chunks")


if __name__ == "__main__":
    main()
