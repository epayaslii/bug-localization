"""Local<->MN5 determinism check (Phase 5, docs/iqloc_replication_scoping.md): confirms the
hybrid-RRF retrieval code produces byte-identical rankings regardless of which machine runs
it, given identical inputs -- the same claim the co-intern's MN5 smoke test verifies for their
BM25-only pipeline (exact scientific match vs. local, per their slide 8).

Deliberately NOT run against this project's actual confirmed-best config (OpenAI
text-embedding-3-small + skeleton-BM25) -- MN5 has no outbound internet, so an OpenAI-API-
dependent pipeline literally cannot execute there at all, making a same-config comparison
impossible. Uses Qwen3-Embedding-0.6B instead (fully local/offline, runs identically on both
machines) to validate the actual code path -- repo content, BM25 skeleton generation,
chunking, RRF fusion -- rather than the specific embedding provider.

Prints a stable JSON summary (ranked file list + a hash per instance) so two runs (one per
machine) can be diffed byte-for-byte.
"""

import os
import sys
import json
import hashlib
import argparse
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from dataset.bench4bl import Bench4BL
from dataset.utils import setup_logging, get_logger
from method.hybrid_retriever import rank_files_hybrid

setup_logging(level=logging.WARNING)
logger = get_logger(__name__)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=2, help='Number of instances to check, sorted by instance_id for a stable choice.')
    parser.add_argument('--candidate-pool-size', type=int, default=30)
    parser.add_argument('--model', default='Qwen/Qwen3-Embedding-0.6B')
    parser.add_argument('--weights', default='1,5')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    weights = [float(w) for w in args.weights.split(',')]

    ds = Bench4BL()
    pool = ds.get_bug_instances(sample_size=4418, random_sample=True, random_seed=42)
    pool.sort(key=lambda b: b.instance_id)
    bugs = pool[:args.n]

    results = {}
    for bug in bugs:
        ranked, _timing = rank_files_hybrid(
            bug, top_k=None, candidate_pool_size=args.candidate_pool_size,
            embedding_model=args.model, weights=weights,
        )
        digest = hashlib.sha256(json.dumps(ranked).encode()).hexdigest()[:16]
        results[bug.instance_id] = {"ranked": ranked, "sha256_16": digest}
        print(f"{bug.instance_id}: {len(ranked)} ranked, sha256={digest}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
