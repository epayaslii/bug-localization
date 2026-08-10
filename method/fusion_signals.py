"""New hybrid-retrieval fusion signals (Phase 4.1): AST-similarity, dependency-graph, and
commit-history. None of these existed as retrieval signals before this pass -- AST was only
used for chunking/BM25-symbol-tokens, dependency metadata existed in repository_index.py's
sidecar but wasn't wired into retrieval scoring, and commit-history wasn't used anywhere.

Scope decisions (2026-08-10, via AskUserQuestion):
- AST-similarity = symbol-name overlap between bug-report tokens and each file's AST-
  extracted symbols, computed live from repo_cache (reuses the same extraction as
  repository_index.py) rather than requiring a pre-built index for the repo in question.
- Dependency-graph = 1-hop import-neighbor boost against a seed ranking (BM25's own top
  candidates): files that import, or are imported by, an already-strong seed file score
  higher. Import-to-file resolution is a simple dot-to-slash heuristic
  (`myapp.utils` -> `myapp/utils.py`), not a real import resolver -- misses relative
  imports, re-exports, star imports -- but tractable, matching the "simple" scope already
  chosen for repository_index.py's dependency metadata.
- Commit-history = recency: files touched more recently before the bug's base_commit score
  higher, per the standard "recently-changed files are more bug-prone" heuristic.
"""

from dataset.repo_cache import get_file_contents_batch, get_recent_commit_timestamps, is_repo_cached
from dataset.utils import get_logger
from method.bm25_retriever import _tokenize_path
from method.repository_index import extract_symbols_and_imports

logger = get_logger(__name__)


def _fetch_contents(bug, file_paths: list[str]) -> dict:
    if not is_repo_cached(bug.repo):
        return {}
    return get_file_contents_batch(bug.repo, bug.base_commit, file_paths)


def rank_files_ast_similarity(bug, top_k: int | None = None) -> list[str]:
    """Rank bug.code_files by token overlap between the bug report and each file's
    AST-extracted symbol names (classes/functions/methods). Python-only; files that don't
    parse (including all non-Python source) score 0, same fallback as elsewhere.

    Tokenizes the bug report with _tokenize_path (identifier-style: splits camelCase/
    snake_case/separators), not _tokenize_query (plain word regex) -- symbol names are
    identifiers, and a bug report that literally quotes one ("ShapeUtil.compute_area is
    broken") needs the same camelCase/snake_case splitting applied to match a symbol split
    into ["shape", "util"] / ["compute", "area"]; _tokenize_query alone left them as opaque
    single tokens ("shapeutil", "compute_area") that could never overlap with anything.
    """
    file_paths = bug.code_files
    contents = _fetch_contents(bug, file_paths)
    query_tokens = set(_tokenize_path(bug.bug_report))

    scores: dict[str, int] = {}
    for path in file_paths:
        content = contents.get(path)
        if not content:
            scores[path] = 0
            continue
        symbols, _ = extract_symbols_and_imports(content)
        symbol_tokens: set[str] = set()
        for s in symbols:
            symbol_tokens.update(_tokenize_path(s))
        scores[path] = len(query_tokens & symbol_tokens)

    ranked = sorted(file_paths, key=lambda p: scores.get(p, 0), reverse=True)
    return ranked[:top_k] if top_k is not None else ranked


def _resolve_import_to_path(module_name: str, code_files_set: set[str]) -> str | None:
    """Best-effort dot-to-slash heuristic: 'pkg.sub.mod' -> 'pkg/sub/mod.py' or
    'pkg/sub/mod/__init__.py'. Not a real import resolver -- misses relative imports
    (`from . import x`), re-exports, and star imports -- but a tractable approximation."""
    base = module_name.replace(".", "/")
    for candidate in (f"{base}.py", f"{base}/__init__.py"):
        if candidate in code_files_set:
            return candidate
    return None


def rank_files_dependency_graph(bug, seed_ranking: list[str], top_k: int | None = None, seed_size: int = 20) -> list[str]:
    """Rank bug.code_files by 1-hop import-adjacency to the top `seed_size` files of
    `seed_ranking` (e.g. BM25's own ranking): files that import, or are imported by, an
    already-strong seed file score higher. Needs imports for every candidate file (not just
    the seed set), since adjacency is checked in both directions."""
    file_paths = bug.code_files
    contents = _fetch_contents(bug, file_paths)
    code_files_set = set(file_paths)

    file_imports: dict[str, set[str]] = {}
    for path in file_paths:
        content = contents.get(path)
        if not content:
            file_imports[path] = set()
            continue
        _, imports = extract_symbols_and_imports(content)
        resolved = {_resolve_import_to_path(mod, code_files_set) for mod in imports}
        file_imports[path] = {r for r in resolved if r is not None}

    seeds = seed_ranking[:seed_size]
    seed_set = set(seeds)

    scores: dict[str, int] = {}
    for path in file_paths:
        score = 1 if path in seed_set else 0  # a seed is trivially "adjacent" to itself
        score += len(file_imports.get(path, set()) & seed_set)  # this file imports a seed
        score += sum(1 for s in seeds if path in file_imports.get(s, set()))  # a seed imports this file
        scores[path] = score

    ranked = sorted(file_paths, key=lambda p: scores.get(p, 0), reverse=True)
    return ranked[:top_k] if top_k is not None else ranked


def rank_files_commit_recency(bug, top_k: int | None = None, max_commits: int = 2000) -> list[str]:
    """Rank bug.code_files by how recently each was modified before bug.base_commit -- more
    recently touched files score higher. Files never seen in the walked history (or an
    uncached repo) fall back to bug.code_files' own order, not an error."""
    file_paths = bug.code_files
    if not is_repo_cached(bug.repo):
        return file_paths

    timestamps = get_recent_commit_timestamps(bug.repo, bug.base_commit, max_commits=max_commits)
    ranked = sorted(file_paths, key=lambda p: timestamps.get(p, 0), reverse=True)
    return ranked[:top_k] if top_k is not None else ranked
