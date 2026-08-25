"""Commit-message-based candidate retrieval -- a signal this project's pipeline has never
used before (file content only, via BM25 + embeddings). Real methodology borrowed from
BugSTAiR (Ersahin et al., 2021, Turk J Elec Eng & Comp Sci): retrieve from a source-code index
AND a separate commit-history index, then UNION the two candidate sets before reranking,
rather than relying on content similarity alone.

Cheap and recall-oriented by construction: BM25 over commit subject lines (fast, no new
dependency -- reuses rank_bm25, already a pinned project dependency), unioned with whatever
the main retriever already found. Uses each bug's own already-mirrored git history
(bench4bl_cache/<PROJECT>/gitrepo), no new data needed.
"""

import os
import re
import subprocess

from rank_bm25 import BM25Okapi

from dataset.utils import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench4bl_cache")

_TOKEN_RE = re.compile(r'[a-zA-Z0-9_]+')


def _tokenize(text: str) -> list[str]:
    text = re.sub(r'[/_\-.]', ' ', text)
    text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
    return _TOKEN_RE.findall(text.lower())


def _get_commit_history(gitrepo: str, base_commit: str, max_commits: int) -> list[tuple[str, str, list[str]]]:
    """Returns [(hash, subject, changed_files)] for up to max_commits commits reachable from
    base_commit (the bug's pre-fix snapshot -- history before the fix, matching what a real
    retrieval system would actually have access to at localization time, not the fix commit
    itself)."""
    # %x00 is git's own pretty-format escape (inserts a literal NUL byte in the output) --
    # passed as the 4-character literal string here, not a Python-escaped null, since argv
    # strings can't contain an embedded NUL byte themselves (subprocess rejects that outright).
    try:
        out = subprocess.run(
            ["git", "-C", gitrepo, "log", base_commit, f"-{max_commits}", "--name-only",
             "--format=COMMIT%x00%H%x00%s"],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.warning(f"commit_history_retriever: git log failed for {gitrepo}@{base_commit}: {e.stderr}")
        return []

    commits = []
    current_hash = current_subject = None
    current_files: list[str] = []
    for line in out.stdout.splitlines():
        if line.startswith("COMMIT\x00"):
            if current_hash is not None:
                commits.append((current_hash, current_subject, current_files))
            _, current_hash, current_subject = line.split("\x00")
            current_files = []
        elif line.strip():
            current_files.append(line.strip())
    if current_hash is not None:
        commits.append((current_hash, current_subject, current_files))
    return commits


def rank_files_commit_history(bug, cache_dir: str | None = None, max_commits: int = 3000,
                                top_k_commits: int = 20, java_only: bool = True) -> list[str]:
    """BM25-ranks commit subject lines against the bug report, returns the union of changed
    files from the top `top_k_commits` matches -- meant to be UNIONed with a content-based
    retriever's own candidates (see union_candidates), not used standalone."""
    gitrepo = os.path.join(cache_dir or DEFAULT_CACHE_DIR, bug.repo, "gitrepo")
    if not os.path.isdir(gitrepo):
        return []

    commits = _get_commit_history(gitrepo, bug.base_commit, max_commits)
    if not commits:
        return []

    tokenized_subjects = [_tokenize(subject) for _hash, subject, _files in commits]
    non_empty = [(i, toks) for i, toks in enumerate(tokenized_subjects) if toks]
    if not non_empty:
        return []

    bm25 = BM25Okapi([toks for _i, toks in non_empty])
    query_tokens = _tokenize(bug.bug_report)
    scores = bm25.get_scores(query_tokens)

    ranked_idx = sorted(range(len(non_empty)), key=lambda i: -scores[i])[:top_k_commits]
    candidate_files = []
    seen = set()
    for i in ranked_idx:
        orig_idx, _toks = non_empty[i]
        _hash, _subject, files = commits[orig_idx]
        for f in files:
            if java_only and not f.endswith(".java"):
                continue
            if f not in seen:
                seen.add(f)
                candidate_files.append(f)
    return candidate_files


def rank_files_commit_history_scored(bug, cache_dir: str | None = None, max_commits: int = 3000,
                                       top_k_commits: int = 20, java_only: bool = True) -> list[str]:
    """Like rank_files_commit_history, but returns files ordered by their OWN commit-history
    relevance score instead of first-seen order from iterating top-K commits. The union
    approach (rank_files_bm25_with_history_union) buries every history-only file at the tail
    of the candidate pool regardless of how strong its match was -- a file whose single best
    matching commit scored highest among all 20 gets appended in exactly the same "last"
    position as one that barely made the cut. That's very likely why the union hurt MRR
    (Recall@200 +18.4%, MRR -4% once wired into the full pipeline, 2026-08-20) despite the
    signal itself being real: genuine matches never got a chance to rank near the top.

    This version scores each file by the MAX BM25 score among the commits (of the top
    top_k_commits by subject-line match) that touched it -- "max not mean", matching this
    project's existing convention for per-file aggregation over multiple sub-signals (chunked
    embeddings, embedding-cosine relevance filtering) -- then sorts by that score. Meant to be
    fed into reciprocal_rank_fusion as a genuine third ranking signal alongside BM25 and
    embeddings, not unioned into the pool pre-embedding.
    """
    gitrepo = os.path.join(cache_dir or DEFAULT_CACHE_DIR, bug.repo, "gitrepo")
    if not os.path.isdir(gitrepo):
        return []

    commits = _get_commit_history(gitrepo, bug.base_commit, max_commits)
    if not commits:
        return []

    tokenized_subjects = [_tokenize(subject) for _hash, subject, _files in commits]
    non_empty = [(i, toks) for i, toks in enumerate(tokenized_subjects) if toks]
    if not non_empty:
        return []

    bm25 = BM25Okapi([toks for _i, toks in non_empty])
    query_tokens = _tokenize(bug.bug_report)
    scores = bm25.get_scores(query_tokens)

    ranked_idx = sorted(range(len(non_empty)), key=lambda i: -scores[i])[:top_k_commits]
    best_score_per_file: dict[str, float] = {}
    for i in ranked_idx:
        orig_idx, _toks = non_empty[i]
        _hash, _subject, files = commits[orig_idx]
        commit_score = scores[i]
        for f in files:
            if java_only and not f.endswith(".java"):
                continue
            if commit_score > best_score_per_file.get(f, float("-inf")):
                best_score_per_file[f] = commit_score

    return sorted(best_score_per_file, key=lambda f: -best_score_per_file[f])


def rank_files_bm25_with_history_union(bug, top_k: int | None = 100, cache_dir: str | None = None,
                                         max_commits: int = 3000, top_k_commits: int = 20) -> list[str]:
    """Drop-in BM25-representation-shaped function (bug, top_k) -> ranked paths, matching
    method/bm25_retriever.py's existing _BM25_REPR_FNS convention -- unions skeleton-BM25
    candidates with commit-history candidates, so ALL of them (not just the first top_k) flow
    into whatever downstream embedding/reranking stage consumes this pool. Unlike a plain
    union evaluated standalone (see compare_commit_history_union.py), the extra recall this
    adds is actually reachable here: the embedding stage re-scores every candidate by content
    similarity, not just the first top_k by BM25 rank."""
    from method.bm25_retriever import rank_files_bm25_with_skeleton
    content = rank_files_bm25_with_skeleton(bug, top_k=top_k)
    history = rank_files_commit_history(bug, cache_dir=cache_dir, max_commits=max_commits, top_k_commits=top_k_commits)
    return union_candidates(content, history)


def union_candidates(*candidate_lists: list[str]) -> list[str]:
    """Union multiple candidate lists, preserving first-seen order (first list's ranking
    takes priority, later lists only contribute genuinely new files) -- used to merge a
    content-based retriever's candidates with commit_history's, matching BugSTAiR's
    Mi = Fi u Si merge step."""
    seen = set()
    merged = []
    for candidates in candidate_lists:
        for c in candidates:
            if c not in seen:
                seen.add(c)
                merged.append(c)
    return merged
