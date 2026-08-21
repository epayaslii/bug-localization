"""BRaIn/SAMMO-style beam search over chunk-relevance-judgment prompt variants, scored
against a cheap classification-accuracy dev set (scripts/build_relevance_dev_set.py)
rather than a full retrieval+rerank cycle per variant -- see method/prompt_optimizer.py
for the mutation step and method/prompt.py's DEFAULT_CHUNK_RELEVANCE_TEMPLATE for
generation 0 (the current, confirmed-negative production prompt).

Cost model, stated explicitly since Ollama calls are genuinely slow (90-460s/instance
observed elsewhere in this project): total scoring calls ~= dev_bugs * (1 + generations *
beam_width * children_per_survivor). Default flags below target ~70 calls (a tractable
first pass, likely tens of minutes to a few hours depending on instance size) -- widen
--dev-bugs-subsample/--generations/--beam-width for a real overnight run once the small
pass confirms the loop itself works.

ALWAYS run with --generations 0 first (scores only the production template, gen-0) as a
smoke test before spending any mutation budget -- this is the single scoring call shape
the whole search depends on, and it's the cheapest possible way to catch a bug in it.
"""

import os
import sys
import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.utils import setup_logging, get_logger
from method.models import ChunkRelevanceFeedbackResponse
from method.ollama_localizer import OllamaLocalizer
from method.prompt import PromptGenerator, DEFAULT_CHUNK_RELEVANCE_TEMPLATE
from method.prompt_optimizer import generate_prompt_variant, template_has_all_placeholders

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "prompt_optimization"
)


def load_dev_set(jsonl_path, dev_bugs_subsample=None, seed=0):
    records = [json.loads(line) for line in open(jsonl_path)]
    by_bug = defaultdict(list)
    for r in records:
        by_bug[r["bug_instance_id"]].append(r)
    bug_ids = sorted(by_bug.keys())
    if dev_bugs_subsample is not None and dev_bugs_subsample < len(bug_ids):
        import random
        random.Random(seed).shuffle(bug_ids)
        bug_ids = sorted(bug_ids[:dev_bugs_subsample])
    return {bid: by_bug[bid] for bid in bug_ids}


def score_template(template, dev_by_bug, localizer, prompt_gen):
    """One batched call per bug (same shape _relevance_feedback_chunked uses in
    production), scored against the dev set's derived ground-truth-file-implies-relevant
    labels. Returns (f1, precision, recall, n_calls_failed, failure_examples)."""
    tp = fp = fn = tn = 0
    n_calls_failed = 0
    failure_examples = []

    for bug_id, bug_records in dev_by_bug.items():
        fake_bug = SimpleNamespace(
            repo=bug_records[0]["repo"], instance_id=bug_id,
            hints_text=bug_records[0]["hints_text"], bug_report=bug_records[0]["bug_report"],
        )
        chunks = [(r["file"], r["chunk_index"], r["chunk_text"]) for r in bug_records]
        prompt = prompt_gen.generate_chunk_relevance_feedback_prompt(fake_bug, chunks, template_override=template)

        try:
            response = localizer.invoke_structured(prompt, ChunkRelevanceFeedbackResponse)
            judged = {(j.file, j.chunk_index): j.relevant for j in response.judgments}
        except Exception as e:
            logger.warning(f"Scoring call failed for {bug_id}: {e}")
            n_calls_failed += 1
            continue

        for r in bug_records:
            key = (r["file"], r["chunk_index"])
            predicted = judged.get(key, False)
            actual = r["label"]
            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
                if len(failure_examples) < 20:
                    failure_examples.append({"chunk_text": r["chunk_text"], "predicted": predicted, "actual": actual})
            elif not predicted and actual:
                fn += 1
                if len(failure_examples) < 20:
                    failure_examples.append({"chunk_text": r["chunk_text"], "predicted": predicted, "actual": actual})
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_calls_failed": n_calls_failed}, failure_examples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dev-set-jsonl', required=True)
    parser.add_argument('--dev-bugs-subsample', type=int, default=10, help='Score against a random subset of dev bugs (cost control). None-equivalent: pass a number >= dev set size.')
    parser.add_argument('--generations', type=int, default=3, help='0 = smoke test, scores only generation-0 (the current production prompt).')
    parser.add_argument('--beam-width', type=int, default=2)
    parser.add_argument('--children-per-survivor', type=int, default=1)
    parser.add_argument('--model', default='qwen2.5-coder:7b')
    parser.add_argument('--ollama-host', default=None)
    parser.add_argument('--max-tokens', type=int, default=8192, help='Matches run_relevance_feedback_test.py production default -- OllamaLocalizer\'s own class default (4096) truncates judgments on large chunk pools.')
    parser.add_argument('--num-ctx', type=int, default=32768, help='Ollama context window. OllamaLocalizer\'s own class default (16384) leaves most of a full candidate-pool chunk prompt (~211 chunks, ~23K tokens) outside the model\'s visible context -- confirmed empirically (only 10/211 chunks judged at 16384 vs. 37/211 at 32768). Raised beyond the production script\'s own default for that reason; this may itself be a bigger lever than prompt wording.')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    dev_by_bug = load_dev_set(args.dev_set_jsonl, dev_bugs_subsample=args.dev_bugs_subsample)
    n_chunks = sum(len(v) for v in dev_by_bug.values())
    est_scoring_calls = len(dev_by_bug) * (1 + args.generations * args.beam_width * args.children_per_survivor)
    logger.info(f"Dev set: {len(dev_by_bug)} bugs, {n_chunks} chunks. Estimated scoring calls this run: ~{est_scoring_calls}")

    localizer = OllamaLocalizer(model=args.model, host=args.ollama_host, max_tokens=args.max_tokens, num_ctx=args.num_ctx)
    prompt_gen = PromptGenerator()

    beam = [{"template": DEFAULT_CHUNK_RELEVANCE_TEMPLATE, "rationale": "generation 0: current production prompt"}]
    generations_log = []

    for gen in range(args.generations + 1):
        scored = []
        for candidate in beam:
            metrics, failures = score_template(candidate["template"], dev_by_bug, localizer, prompt_gen)
            scored.append({**candidate, "metrics": metrics, "failure_examples": failures})
            logger.info(f"gen {gen}: F1={metrics['f1']:.3f} P={metrics['precision']:.3f} R={metrics['recall']:.3f} ({candidate['rationale'][:60]})")

        scored.sort(key=lambda c: c["metrics"]["f1"], reverse=True)
        generations_log.append({
            "generation": gen,
            "candidates": [{"template": c["template"], "rationale": c["rationale"], "metrics": c["metrics"], "failure_examples": c["failure_examples"]} for c in scored],
        })

        if gen == args.generations:
            break

        survivors = scored[: args.beam_width]
        children = []
        for survivor in survivors:
            for _ in range(args.children_per_survivor):
                new_template, rationale = generate_prompt_variant(
                    survivor["template"], survivor["failure_examples"], localizer, prompt_gen
                )
                if template_has_all_placeholders(new_template):
                    children.append({"template": new_template, "rationale": rationale})
        beam = survivors + children if children else survivors

    best = generations_log[-1]["candidates"][0] if generations_log[-1]["candidates"] else generations_log[0]["candidates"][0]
    logger.info(f"Best F1 across all generations: {best['metrics']['f1']:.3f} (P={best['metrics']['precision']:.3f}, R={best['metrics']['recall']:.3f})")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = os.path.join(args.output_dir, f"relevance_prompt_search_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump({
            "dev_set_jsonl": args.dev_set_jsonl, "dev_bugs_used": len(dev_by_bug),
            "generations": args.generations, "beam_width": args.beam_width,
            "children_per_survivor": args.children_per_survivor, "model": args.model,
            "best": {"template": best["template"], "rationale": best["rationale"], "metrics": best["metrics"], "failure_examples": best["failure_examples"]},
            "all_generations": generations_log,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    logger.info(f"Full search log saved to {out_path}")


if __name__ == "__main__":
    main()
