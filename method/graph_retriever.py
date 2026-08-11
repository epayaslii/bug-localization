"""Graph-guided retrieval (LocAgent-style): instead of a single 1-hop import-adjacency
boost (already tried in an earlier branch and found weak), build a real file-level import
graph and do a bounded BFS traversal outward from BM25-seeded anchor files, ranking
candidates by how close they are to a strong seed.

Scope decisions:
- Seeding is BM25-based (reuses existing, already-validated infra), not LLM entity
  extraction -- free, offline, consistent with how every other retrieval signal in this
  project has been built.
- Graph is file-level, edges from imports only (dot-to-slash heuristic, same tractable
  approximation used elsewhere in this project -- misses relative imports, re-exports,
  star imports). Class/function containment edges were considered but dropped: since
  seeding is already file-level, containment edges wouldn't add any traversal power without
  also doing symbol-level seeding, which is out of scope here.
- Undirected adjacency: "A imports B" and "B imports A" are treated the same for traversal
  purposes -- localization cares about connectivity, not import directionality.
- 2 hops by default: seed -> its direct import-neighbors -> one more hop out. Bounded to
  keep the graph traversal cheap and avoid pulling in the whole repo through a long import
  chain.
"""

import ast

from dataset.repo_cache import get_file_contents_batch, is_repo_cached
from dataset.utils import get_logger

logger = get_logger(__name__)


def _extract_imports(content: str) -> list[str]:
    """Raw imported module names (not tokenized) via AST. Python-only; returns [] on a
    parse error, same fallback as elsewhere in this project."""
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _resolve_import_to_path(module_name: str, code_files_set: set[str]) -> str | None:
    """Best-effort dot-to-slash heuristic: 'pkg.sub.mod' -> 'pkg/sub/mod.py' or
    'pkg/sub/mod/__init__.py'. Not a real import resolver -- misses relative imports
    (`from . import x`), re-exports, and star imports -- but a tractable approximation,
    same one already used for the earlier 1-hop dependency-graph signal."""
    base = module_name.replace(".", "/")
    for candidate in (f"{base}.py", f"{base}/__init__.py"):
        if candidate in code_files_set:
            return candidate
    return None


def build_import_graph(bug) -> dict[str, set[str]]:
    """Undirected adjacency: file -> set of files it imports or is imported by, restricted
    to bug.code_files. Requires the repo to be locally mirrored (repo_cache); returns an
    all-empty graph otherwise -- never makes a live network call."""
    file_paths = bug.code_files
    graph: dict[str, set[str]] = {p: set() for p in file_paths}

    if not is_repo_cached(bug.repo):
        return graph

    contents = get_file_contents_batch(bug.repo, bug.base_commit, file_paths)
    code_files_set = set(file_paths)

    for path in file_paths:
        content = contents.get(path)
        if not content:
            continue
        for module_name in _extract_imports(content):
            target = _resolve_import_to_path(module_name, code_files_set)
            if target and target != path:
                graph[path].add(target)
                graph[target].add(path)

    return graph


def rank_files_graph_traversal(bug, seed_ranking: list[str], top_k: int | None = None,
                                seed_size: int = 10, hops: int = 2) -> list[str]:
    """Rank bug.code_files by BFS distance (up to `hops` steps) from the top `seed_size`
    files of seed_ranking (e.g. BM25's own ranking) along the import-adjacency graph.
    Closer files score higher. Files never reached by traversal keep seed_ranking's own
    relative order (stable sort), rather than being scored arbitrarily -- so a candidate
    the graph has no opinion about doesn't get punished below one it actively pushed down.
    """
    graph = build_import_graph(bug)
    seeds = seed_ranking[:seed_size]

    distance: dict[str, int] = {s: 0 for s in seeds}
    frontier = list(seeds)
    for hop in range(1, hops + 1):
        next_frontier = []
        for node in frontier:
            for neighbor in graph.get(node, set()):
                if neighbor not in distance:
                    distance[neighbor] = hop
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    def score(path):
        return -distance.get(path, hops + 1)

    ranked = sorted(seed_ranking, key=score, reverse=True)

    seed_ranking_set = set(seed_ranking)
    remaining = [p for p in bug.code_files if p not in seed_ranking_set]
    ranked = ranked + remaining

    return ranked[:top_k] if top_k is not None else ranked
