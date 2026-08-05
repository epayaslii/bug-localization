import ast
import re
from rank_bm25 import BM25Okapi

from dataset.repo_cache import get_file_contents_batch, is_repo_cached
from dataset.utils import get_logger

logger = get_logger(__name__)


def _tokenize_path(path: str) -> list[str]:
    """Split a file path into lowercase tokens, breaking on separators and camelCase."""
    text = re.sub(r'[/_.\-]', ' ', path)
    text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
    return text.lower().split()


def _tokenize_query(text: str) -> list[str]:
    return re.findall(r'[a-zA-Z0-9_]+', text.lower())


def _extract_skeleton_tokens(content: str) -> list[str]:
    """Extract module docstring + class/function names from Python source (SWE-Fixer-style
    'file skeleton') -- cheap signal beyond the bare path, without sending full file content."""
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return []

    tokens = []
    doc = ast.get_docstring(tree)
    if doc:
        tokens += _tokenize_query(doc[:300])

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # identifiers are snake_case/camelCase, not prose -- split like a path, not a query
            tokens += _tokenize_path(node.name)

    return tokens


def rank_files_bm25(query_text: str, file_paths: list[str], top_k: int | None = 100) -> list[str]:
    """Rank file_paths by relevance to query_text using BM25 over tokenized paths, returning
    the top_k most relevant. Pass top_k=None to get the full ranking (e.g. for screening/
    diagnostics that need every ground-truth file's rank, not just a truncated candidate set).

    Operates on file paths only (no file content), matching how code_files are already
    used elsewhere in this pipeline. Returns file_paths unchanged if there are already
    top_k or fewer (top_k=None always ranks).
    """
    if not file_paths or (top_k is not None and len(file_paths) <= top_k):
        return file_paths

    tokenized_corpus = [_tokenize_path(p) for p in file_paths]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = _tokenize_query(query_text)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(file_paths, scores), key=lambda pair: pair[1], reverse=True)
    return [path for path, _ in ranked[:top_k]]


def rank_files_bm25_with_skeleton(bug, top_k: int = 100) -> list[str]:
    """Like rank_files_bm25, but scores each file using its path plus a lightweight
    'skeleton' (docstring + class/function names) pulled from its actual content via the
    offline repo_cache, when available. Falls back to path-only tokens for any file whose
    content can't be read (including entirely when the repo isn't locally mirrored) --
    never makes a live network call.
    """
    file_paths = bug.code_files
    if not file_paths or len(file_paths) <= top_k:
        return file_paths

    repo_available = is_repo_cached(bug.repo)
    contents = get_file_contents_batch(bug.repo, bug.base_commit, file_paths) if repo_available else {}

    tokenized_corpus = []
    for path in file_paths:
        tokens = _tokenize_path(path)
        content = contents.get(path)
        if content is not None:
            try:
                tokens = tokens + _extract_skeleton_tokens(content)
            except Exception as e:
                logger.debug(f"Skeleton extraction failed for {path}: {e}")
        tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = _tokenize_query(bug.bug_report)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(file_paths, scores), key=lambda pair: pair[1], reverse=True)
    return [path for path, _ in ranked[:top_k]]
