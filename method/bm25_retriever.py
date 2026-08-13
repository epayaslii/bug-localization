import ast
import re
from rank_bm25 import BM25Okapi

from dataset.repo_cache import get_file_contents_batch, is_repo_cached
from dataset.utils import get_logger
from method.java_parsing import extract_java_skeleton_tokens, extract_java_symbol_tokens, is_java_path

logger = get_logger(__name__)


def _tokenize_path(path: str) -> list[str]:
    """Split a file path into lowercase tokens, breaking on separators and camelCase."""
    text = re.sub(r'[/_.\-]', ' ', path)
    text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
    return text.lower().split()


def _tokenize_query(text: str) -> list[str]:
    return re.findall(r'[a-zA-Z0-9_]+', text.lower())


def _extract_skeleton_tokens(content: str, path: str = "") -> list[str]:
    """Extract module docstring + class/function names from Python source (SWE-Fixer-style
    'file skeleton') -- cheap signal beyond the bare path, without sending full file content.
    Dispatches to the Java lexical scanner (method/java_parsing.py) for .java files, since
    ast.parse() only understands Python."""
    if is_java_path(path):
        return extract_java_skeleton_tokens(content)

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


def _extract_symbol_tokens(content: str, path: str = "") -> tuple[list[str], list[str]]:
    """Extract (symbol_tokens, import_tokens) from Python source via AST.

    symbol_tokens are class/function/method names (identifiers, so split like a path
    rather than a query); import_tokens are imported module and name tokens, kept
    separate so callers can ablate them in or out independently (matching the
    path-symbols vs path-symbols-imports comparison in the literature). Returns
    ([], []) on a parse error. Dispatches to the Java lexical scanner
    (method/java_parsing.py) for .java files, since ast.parse() only understands Python.
    """
    if is_java_path(path):
        return extract_java_symbol_tokens(content)

    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return [], []

    symbol_tokens = []
    import_tokens = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbol_tokens += _tokenize_path(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                import_tokens += _tokenize_path(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                import_tokens += _tokenize_path(node.module)
            for alias in node.names:
                import_tokens += _tokenize_path(alias.name)

    return symbol_tokens, import_tokens


def _rank_files_with_content_tokens(bug, extra_tokens_fn, top_k: int | None) -> list[str]:
    """Shared scaffold for BM25 variants that enrich path tokens with content-derived
    tokens: fetches file content via the offline repo_cache when available, applies
    extra_tokens_fn(content, path) -> list[str] per file (falling back to path-only tokens
    on any error or when content isn't available -- including entirely when the repo
    isn't locally mirrored, since this never makes a live network call), then ranks
    by BM25 against the bug report.
    """
    file_paths = bug.code_files
    if not file_paths or (top_k is not None and len(file_paths) <= top_k):
        return file_paths

    repo_available = is_repo_cached(bug.repo)
    contents = get_file_contents_batch(bug.repo, bug.base_commit, file_paths) if repo_available else {}

    tokenized_corpus = []
    for path in file_paths:
        tokens = _tokenize_path(path)
        content = contents.get(path)
        if content is not None:
            try:
                tokens = tokens + extra_tokens_fn(content, path)
            except Exception as e:
                logger.debug(f"Content tokenization failed for {path}: {e}")
        tokenized_corpus.append(tokens)

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = _tokenize_query(bug.bug_report)
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(file_paths, scores), key=lambda pair: pair[1], reverse=True)
    return [path for path, _ in ranked[:top_k]]


def rank_files_bm25_with_skeleton(bug, top_k: int | None = 100) -> list[str]:
    """Like rank_files_bm25, but scores each file using its path plus a lightweight
    'skeleton' (docstring + class/function names) pulled from its actual content via the
    offline repo_cache, when available. Falls back to path-only tokens for any file whose
    content can't be read (including entirely when the repo isn't locally mirrored) --
    never makes a live network call.
    """
    return _rank_files_with_content_tokens(bug, _extract_skeleton_tokens, top_k)


def rank_files_bm25_with_symbols(bug, top_k: int | None = 100, include_imports: bool = True) -> list[str]:
    """Like rank_files_bm25_with_skeleton, but scores each file using its path plus
    extracted class/function/method names, optionally plus imported module/name tokens
    (include_imports=True by default). Unlike the skeleton variant this drops the
    docstring signal and separates symbols from imports, matching the literature's
    path-symbols vs path-symbols-imports representations. Set include_imports=False to
    ablate imports out. Falls back to path-only tokens for any file whose content can't
    be read or parsed -- never makes a live network call.
    """
    def extra_tokens(content: str, path: str) -> list[str]:
        symbol_tokens, import_tokens = _extract_symbol_tokens(content, path)
        return symbol_tokens + (import_tokens if include_imports else [])

    return _rank_files_with_content_tokens(bug, extra_tokens, top_k)


def extract_query_reformulation_terms(paths: list[str], contents: dict[str, str]) -> list[str]:
    """Extract identifier-like terms (class/function/method names) from a set of file
    paths an LLM relevance-feedback step judged relevant, for algorithmic query
    reformulation (BRaIn/IQLoc-style: append expansion terms to the original bug report
    instead of issuing a second LLM call -- see docs/relevance_feedback_scoping.md).
    Reuses the same extraction machinery as rank_files_bm25_with_symbols so the expansion
    vocabulary matches what BM25 already indexes files by. Skips files with no fetched
    content or that fail to parse; import tokens are deliberately excluded here since
    they're module/library names rather than bug-specific vocabulary.
    """
    terms: list[str] = []
    for path in paths:
        content = contents.get(path)
        if content is None:
            continue
        try:
            symbol_tokens, _import_tokens = _extract_symbol_tokens(content, path)
        except Exception as e:
            logger.debug(f"Query reformulation extraction failed for {path}: {e}")
            continue
        terms += symbol_tokens
    return terms
