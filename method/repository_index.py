"""Phase 3.2: persistent repository indexing (FAISS + metadata sidecar).

Chunks a repo@commit into files/classes/methods (reusing the same AST chunker as the
hybrid-retrieval work), embeds each chunk once, and persists the vectors + metadata
(path, language, symbols, imports) to disk -- so retrieval reads a saved index instead of
recomputing embeddings on every call. This is the "biggest structural gap" identified in
the official study plan: method/embedding_retriever.py's rank_files_embedding_chunked()
recomputes embeddings fresh on every single call, nothing is persisted.

Scope decisions (2026-08-10, via AskUserQuestion):
- FAISS over Qdrant: embedded/file-based, fits MN5's batch-Slurm/no-persistent-service/
  no-internet constraints; Qdrant's client-server model doesn't (qdrant-client is in
  requirements.txt but was never actually used -- dead dependency from an earlier team).
- Model-agnostic: the embedding model is a parameter (default: UniXCoder, the current
  Phase 3.1 bake-off leader at n=6 -- preliminary, not final), not hard-coded, so 3.2
  doesn't block on 3.1 finishing and re-indexing with a different model is just a
  parameter change, not a redesign.
- Dependency graph kept simple: each chunk's metadata records its file's imported module
  names (a per-file list), not a traversable graph structure (no transitive closure /
  reverse-dependency queries). That's the "dependency graph" scope named in the plan for
  this pass; a real graph is future work if the simple version proves insufficient.
- Python-only, same caching caveat as the rest of this project's AST-based chunking/symbol
  extraction (method/embedding_retriever.py, method/bm25_retriever.py) -- non-Python files
  fall back to a path-token pseudo-chunk, no symbols/imports extracted.
"""

import ast
import json
import os
import tempfile

import faiss
import numpy as np

from dataset.repo_cache import get_file_contents_batch, get_code_files_local, is_repo_cached
from dataset.utils import get_logger
from method.bm25_retriever import _tokenize_path
from method.embedding_retriever import embed_texts, _chunk_file_content

logger = get_logger(__name__)

DEFAULT_INDEX_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "repo_cache", "vector_index"
)
# Current Phase 3.1 bake-off leader (n=6, preliminary) -- swap via the model_name param,
# no code change needed to index with a different model.
DEFAULT_EMBEDDING_MODEL = "microsoft/unixcoder-base"

_LANGUAGE_BY_EXTENSION = {
    ".py": "python", ".java": "java", ".go": "go",
    ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".h": "c", ".hpp": "cpp", ".c": "c",
}


def _language_for_path(path: str) -> str:
    return _LANGUAGE_BY_EXTENSION.get(os.path.splitext(path)[1], "unknown")


def _extract_symbols_and_imports(content: str) -> tuple[list[str], list[str]]:
    """Real (not BM25-tokenized) class/function/method names and imported module names,
    for index metadata rather than retrieval scoring. Python-only; returns ([], []) for
    unparseable content (including all non-Python source, same fallback as elsewhere)."""
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError):
        return [], []

    symbols, imports = [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(node.name)
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return symbols, imports


def _index_paths(repo: str, commit: str, model_name: str, index_root: str | None = None) -> tuple[str, str]:
    root = index_root or DEFAULT_INDEX_ROOT
    base = os.path.join(root, repo.replace("/", "__"), commit, model_name.replace("/", "__"))
    return base + ".faiss", base + ".meta.json"


def _atomic_write_bytes(path: str, write_fn) -> None:
    """write_fn(tmp_path) writes the file; this handles the temp-sibling + os.replace
    dance so a killed/crashed build never leaves a half-written index/metadata file that a
    later load would silently misread as complete."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    os.close(fd)
    try:
        write_fn(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def is_indexed(repo: str, commit: str, model_name: str = DEFAULT_EMBEDDING_MODEL, index_root: str | None = None) -> bool:
    index_path, meta_path = _index_paths(repo, commit, model_name, index_root)
    return os.path.isfile(index_path) and os.path.isfile(meta_path)


def build_repository_index(repo: str, commit: str, file_paths: list[str] | None = None,
                            model_name: str = DEFAULT_EMBEDDING_MODEL,
                            index_root: str | None = None, max_chunk_chars: int = 1500) -> str | None:
    """Chunk every Python file in repo@commit, embed each chunk, and persist to a FAISS
    index + metadata sidecar. Idempotent: always rebuilds from scratch (no incremental
    append) -- call is_indexed() first if you want to skip an already-built index.
    Returns the index path, or None if there was nothing to index.
    """
    if not is_repo_cached(repo):
        raise ValueError(f"{repo} is not in the local repo_cache -- mirror it first (scripts/mirror_repos.py)")

    if file_paths is None:
        file_paths = get_code_files_local(repo, commit, (".py",))

    contents = get_file_contents_batch(repo, commit, file_paths)

    chunk_texts: list[str] = []
    chunk_meta: list[dict] = []
    for path in file_paths:
        content = contents.get(path)
        if content is None:
            continue
        symbols, imports = _extract_symbols_and_imports(content)
        for chunk in _chunk_file_content(content, max_chunk_chars=max_chunk_chars):
            chunk_texts.append(chunk)
            chunk_meta.append({
                "path": path, "language": _language_for_path(path),
                "symbols": symbols, "imports": imports,
            })

    if not chunk_texts:
        logger.warning(f"No indexable Python chunks for {repo}@{commit}")
        return None

    embeddings = embed_texts(chunk_texts, model_name=model_name).numpy().astype("float32")
    faiss.normalize_L2(embeddings)  # cosine similarity via inner product on normalized vectors
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    def _write_meta(tmp_path):
        with open(tmp_path, "w") as f:
            json.dump({"repo": repo, "commit": commit, "model_name": model_name, "chunks": chunk_meta}, f)

    index_path, meta_path = _index_paths(repo, commit, model_name, index_root)
    _atomic_write_bytes(index_path, lambda tmp: faiss.write_index(index, tmp))
    _atomic_write_bytes(meta_path, _write_meta)

    logger.info(f"Indexed {repo}@{commit}: {len(file_paths)} files, {len(chunk_texts)} chunks -> {index_path}")
    return index_path


def load_repository_index(repo: str, commit: str, model_name: str = DEFAULT_EMBEDDING_MODEL,
                           index_root: str | None = None):
    """Returns (faiss_index, chunk_metadata_list), or (None, None) if not yet indexed."""
    index_path, meta_path = _index_paths(repo, commit, model_name, index_root)
    if not os.path.isfile(index_path) or not os.path.isfile(meta_path):
        return None, None
    index = faiss.read_index(index_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return index, meta["chunks"]


def rank_files_from_index(bug, model_name: str = DEFAULT_EMBEDDING_MODEL,
                           index_root: str | None = None, top_k: int | None = None) -> list[str] | None:
    """Like method.embedding_retriever.rank_files_embedding_chunked, but searches a
    PERSISTED index instead of recomputing every file's embedding on every call -- the
    actual point of building the index. Returns None if repo@base_commit isn't indexed yet
    (caller decides whether to build_repository_index() first, since that's a paid-in-
    compute-time step this function deliberately doesn't do implicitly)."""
    index, chunks = load_repository_index(bug.repo, bug.base_commit, model_name, index_root)
    if index is None:
        return None

    query_embedding = embed_texts([bug.bug_report], model_name=model_name).numpy().astype("float32")
    faiss.normalize_L2(query_embedding)

    k = min(index.ntotal, len(chunks))
    scores, idxs = index.search(query_embedding, k)

    code_files = set(bug.code_files)
    file_best_score: dict[str, float] = {}
    for score, idx in zip(scores[0], idxs[0]):
        path = chunks[idx]["path"]
        if path not in code_files:
            continue  # index may cover files outside this bug's current candidate set
        if path not in file_best_score or score > file_best_score[path]:
            file_best_score[path] = float(score)

    ranked = sorted(file_best_score, key=lambda p: file_best_score[p], reverse=True)
    return ranked[:top_k] if top_k is not None else ranked
