"""Phase 4.2 CLI: build (or rebuild) a persistent repository vector index and report
throughput/storage stats -- the actual point of Phase 4.2's "measure indexing throughput
and storage efficiency" ask, rather than those numbers only being visible via ad-hoc
Python calls or buried in log lines.

Run twice on the same repo@commit to see the incremental-indexing win directly: the second
run's cache_hits should equal its num_chunks (nothing to re-embed, same content already
cached from the first run).
"""

import os
import sys
import argparse
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.utils import setup_logging, get_logger
from method.repository_index import build_repository_index, DEFAULT_EMBEDDING_MODEL

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, help='e.g. django/django (must already be mirrored via scripts/mirror_repos.py)')
    parser.add_argument('--commit', required=True)
    parser.add_argument('--model', default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument('--force-recompute', action='store_true', help='Bypass the chunk cache, re-embed everything')
    args = parser.parse_args()

    stats = build_repository_index(args.repo, args.commit, model_name=args.model, force_recompute=args.force_recompute)
    if stats is None:
        logger.warning("Nothing indexed (no indexable Python files/chunks found)")
        return

    hit_rate = stats["cache_hits"] / stats["num_chunks"] if stats["num_chunks"] else 0.0
    chunks_per_sec = stats["num_chunks"] / stats["embed_elapsed_s"] if stats["embed_elapsed_s"] > 0 else float("inf")
    bytes_per_chunk = stats["index_bytes"] / stats["num_chunks"] if stats["num_chunks"] else 0

    logger.info("=== Indexing report ===")
    logger.info(f"  Files:              {stats['num_files']}")
    logger.info(f"  Chunks:             {stats['num_chunks']}")
    logger.info(f"  Cache hits/misses:  {stats['cache_hits']}/{stats['cache_misses']} ({hit_rate:.1%} hit rate)")
    logger.info(f"  Embed time:         {stats['embed_elapsed_s']:.1f}s ({chunks_per_sec:.1f} chunks/s for cache misses only)")
    logger.info(f"  Total time:         {stats['total_elapsed_s']:.1f}s")
    logger.info(f"  Index size:         {stats['index_bytes']:,} bytes ({bytes_per_chunk:.0f} bytes/chunk)")
    logger.info(f"  Chunk cache size:   {stats['cache_bytes']:,} bytes (shared across all repos indexed with this model)")
    logger.info(f"  Index path:         {stats['index_path']}")


if __name__ == "__main__":
    main()
