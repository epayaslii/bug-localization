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


# 2026-08-27: query-side refinements (summary-lite construction + noise filtering + code-aware
# splitting), borrowed from the co-intern's BM25 setup. Everything below only touches the
# QUERY side -- the document-side representations (skeleton/symbols/imports) are untouched,
# so this composes with any existing bm25-repr choice rather than replacing it.

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "i", "we", "you", "he", "she", "it", "they", "my", "our", "your", "his",
    "her", "its", "their", "and", "or", "but", "if", "then", "else", "when", "while", "for",
    "to", "of", "in", "on", "at", "by", "with", "from", "as", "so", "than", "too", "very",
    "can", "could", "would", "should", "will", "shall", "do", "does", "did", "have", "has",
    "had", "not", "no", "yes", "also", "just", "only", "more", "most", "some", "any", "all",
    "each", "other", "such", "what", "which", "who", "whom", "how", "why", "there", "here",
    "get", "getting", "got", "using", "use", "used", "see", "seeing", "seen", "please",
    "thanks", "thank", "hi", "hello",
})

# Common bug-report boilerplate -- generic across nearly every report, so it adds noise to
# BM25 scoring without discriminating between candidate files. Stripped from the free-text
# portion of the query before sentence extraction; identifier-like tokens are pulled from the
# ORIGINAL text separately (see build_summary_lite_query), so nothing discriminative is lost
# even if it happens to fall inside a stripped boilerplate section.
_BOILERPLATE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"steps? to reproduce.*?(?=\n\n|\Z)",
    r"expected (behaviou?r|result)s?.*?(?=\n\n|\Z)",
    r"actual (behaviou?r|result)s?.*?(?=\n\n|\Z)",
    r"environment:.*?(?=\n\n|\Z)",
    r"version:.*?(?=\n\n|\Z)",
    r"i am (using|trying to|getting)[^.\n]*[.\n]",
    r"when i (try to|do)[^.\n]*[.\n]",
    r"could (someone|anyone) (help|please)[^.\n]*[.\n]",
    r"any help (would be|is) (appreciated|great)[^.\n]*[.\n]",
    r"thanks? in advance[^.\n]*[.\n]",
]]


def _strip_boilerplate(text: str) -> str:
    for pat in _BOILERPLATE_PATTERNS:
        text = pat.sub(" ", text)
    return text


def _split_camel_snake(token: str) -> list[str]:
    parts = []
    for chunk in token.split("_"):
        sub = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', chunk).split()
        parts.extend(sub if sub else ([chunk] if chunk else []))
    return [p.lower() for p in parts if p]


def _tokenize_query_code_aware(text: str) -> list[str]:
    """Like _tokenize_query, but also splits camelCase/snake_case identifiers within the
    QUERY text (e.g. "getUserById" -> "get user by id") -- _tokenize_path already does this
    for file paths, but the query side never did, so a bug report mentioning a camelCase
    method name couldn't match against a file's path tokens split the same way."""
    tokens = []
    for raw in re.findall(r'[a-zA-Z0-9_]+', text):
        tokens.extend(_split_camel_snake(raw))
    return tokens


def _looks_like_identifier(token: str) -> bool:
    return bool(re.match(r'^[A-Za-z_][A-Za-z0-9_.]*$', token)) and (
        '_' in token or '.' in token or bool(re.search(r'[a-z][A-Z]', token))
    )


def build_summary_lite_query(bug_report: str, max_sentences: int = 2) -> str:
    """Rule-based ('lite', no LLM) query condensation: strip common bug-report boilerplate,
    keep only the first few sentences of the remaining free text (the core problem statement,
    typically before "steps to reproduce"/environment dumps), and separately re-append every
    code-identifier-looking token (camelCase/snake_case/dotted) pulled verbatim from the
    ORIGINAL (pre-cutoff) text, so a discriminative identifier mentioned later in a long
    report still reaches the query even though the surrounding prose was cut."""
    identifier_tokens = [
        tok for tok in re.findall(r'[A-Za-z_][A-Za-z0-9_.]*', bug_report)
        if _looks_like_identifier(tok)
    ]
    cleaned = _strip_boilerplate(bug_report)
    sentences = re.split(r'(?<=[.!?])\s+', cleaned.strip())
    summary_text = " ".join(s for s in sentences[:max_sentences] if s)
    return (summary_text + " " + " ".join(identifier_tokens)).strip()


def _tokenize_query_refined(text: str) -> list[str]:
    """Full query-side refinement: summary-lite construction -> code-aware tokenization ->
    stopword/noise filtering. This is what "refined BM25" (rank_files_bm25_refined below)
    uses in place of the plain _tokenize_query(bug.bug_report)."""
    summary = build_summary_lite_query(text)
    tokens = _tokenize_query_code_aware(summary)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


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


def _rank_files_with_content_tokens(bug, extra_tokens_fn, top_k: int | None, query_tokenize_fn=None) -> list[str]:
    """Shared scaffold for BM25 variants that enrich path tokens with content-derived
    tokens: fetches file content via the offline repo_cache when available, applies
    extra_tokens_fn(content, path) -> list[str] per file (falling back to path-only tokens
    on any error or when content isn't available -- including entirely when the repo
    isn't locally mirrored, since this never makes a live network call), then ranks
    by BM25 against the bug report.

    query_tokenize_fn(bug_report_text) -> list[str] overrides the plain _tokenize_query
    default -- pass _tokenize_query_refined for the summary-lite/noise-filtered/code-aware
    query construction (see rank_files_bm25_refined). Kept as an opt-in parameter, not the
    new default, so every existing bm25-repr comparison stays exactly reproducible.
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
    query_tokenize_fn = query_tokenize_fn or _tokenize_query
    tokenized_query = query_tokenize_fn(bug.bug_report)
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


def rank_files_bm25_refined(bug, top_k: int | None = 100, include_imports: bool = True) -> list[str]:
    """Like rank_files_bm25_with_symbols, but with the co-intern's query-side refinements
    applied instead of the plain raw-bug-report query: summary-lite construction (strip
    boilerplate, keep the first couple of sentences, re-append identifier tokens verbatim),
    code-aware splitting of camelCase/snake_case identifiers in the query, and stopword/noise
    filtering. Document-side representation (path + symbols (+imports)) is unchanged --
    this isolates the query-side refinement's own effect for direct A/B comparison against
    rank_files_bm25_with_symbols.
    """
    def extra_tokens(content: str, path: str) -> list[str]:
        symbol_tokens, import_tokens = _extract_symbol_tokens(content, path)
        return symbol_tokens + (import_tokens if include_imports else [])

    return _rank_files_with_content_tokens(bug, extra_tokens, top_k, query_tokenize_fn=_tokenize_query_refined)


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
