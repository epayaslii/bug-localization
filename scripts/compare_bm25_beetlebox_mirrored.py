"""BM25 representation comparison for a BeetleBox manifest whose repos are only *partially*
mirrored (e.g. a small-repo-only manifest drawn from the 13-repo BeetleBox pool).

compare_bm25_representations.py samples `pool_size` instances across ALL repos in the
dataset before filtering down to the manifest's wanted instances -- for a small-repo-only
manifest that means live-fetching file trees for every unmirrored repo it happens to sample
along the way (huge repos like odoo/ClickHouse), which is slow/flaky and unnecessary. This
script instead builds the wanted instances directly, one already-mirrored repo at a time,
via BeetleBox(repo_filter=repo) -- the repo filter is applied before any file-content fetch
(dataset/beetlebox.py's per-bug loop `continue`s on a repo mismatch before calling
get_code_files()), so non-matching repos are never touched at all.
"""

import os
import sys
import argparse
import json
import logging
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from dataset.beetlebox import BeetleBox
from dataset.localizability import load_cache, save_cache
from dataset.utils import setup_logging, get_logger
from evaluation.manifest import load_manifest
from evaluation.screening import screen_manifest, summarize_screening
from method.bm25_retriever import rank_files_bm25, rank_files_bm25_with_skeleton, rank_files_bm25_with_symbols

setup_logging(level=logging.INFO)
logger = get_logger(__name__)

REPRESENTATIONS = {
    "path_only": lambda b: rank_files_bm25(b.bug_report, b.code_files, top_k=None),
    "skeleton": lambda b: rank_files_bm25_with_skeleton(b, top_k=None),
    "symbols_with_imports": lambda b: rank_files_bm25_with_symbols(b, top_k=None, include_imports=True),
    "symbols_no_imports": lambda b: rank_files_bm25_with_symbols(b, top_k=None, include_imports=False),
}


def _load_wanted_bugs(manifest):
    wanted_by_repo = defaultdict(set)
    for inst in manifest['instances']:
        wanted_by_repo[inst['repo']].add(inst['instance_id'])

    bugs = []
    for repo, wanted_ids in wanted_by_repo.items():
        logger.info(f"Loading repo-filtered instances for {repo} ({len(wanted_ids)} wanted)")
        repo_dataset = BeetleBox(repo_filter=repo)
        repo_bugs = repo_dataset.get_bug_instances(sample_size=None)
        found = [b for b in repo_bugs if b.instance_id in wanted_ids]
        missing = wanted_ids - {b.instance_id for b in found}
        if missing:
            logger.warning(f"{repo}: {len(missing)} wanted instance(s) not found: {sorted(missing)}")
        bugs.extend(found)
    return bugs


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--representations', nargs='+', choices=list(REPRESENTATIONS), default=None)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    bugs = _load_wanted_bugs(manifest)
    logger.info(f"Loaded {len(bugs)}/{manifest['size']} manifest instances (manifest {manifest['manifest_id']}), fully offline, mirrored repos only")

    names = args.representations or list(REPRESENTATIONS)
    token = os.getenv("GITHUB_TOKEN")
    cache = load_cache()

    results = {}
    for name in names:
        logger.info(f"--- Representation: {name} ---")
        report = screen_manifest(bugs, token=token, cache=cache, rank_fn=REPRESENTATIONS[name])
        summary = summarize_screening(report)
        results[name] = {"screening_report": report, "summary": summary}
        logger.info(
            f"  Hit@1={summary['macro_hit_at'][1]:.3f} Hit@5={summary['macro_hit_at'][5]:.3f} "
            f"Hit@10={summary['macro_hit_at'][10]:.3f} MRR={summary['mrr']:.4f} MAP={summary['map']:.4f}"
        )

    save_cache(cache)

    logger.info("=== Summary (macro, across all representations) ===")
    logger.info(f"{'representation':<22} {'Hit@1':>7} {'Hit@5':>7} {'Hit@10':>7} {'MRR':>8} {'MAP':>8}")
    for name in names:
        s = results[name]["summary"]
        logger.info(
            f"{name:<22} {s['macro_hit_at'][1]:>7.3f} {s['macro_hit_at'][5]:>7.3f} "
            f"{s['macro_hit_at'][10]:>7.3f} {s['mrr']:>8.4f} {s['map']:>8.4f}"
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({"manifest_id": manifest["manifest_id"], "representations": results}, f, indent=2)
        logger.info(f"Wrote full report to {args.output}")


if __name__ == "__main__":
    main()
