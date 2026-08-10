import ast
import os
import time

import requests
import torch
from transformers import AutoTokenizer, AutoModel

from dataset.repo_cache import get_file_contents_batch, is_repo_cached
from dataset.utils import get_logger
from method.bm25_retriever import _extract_skeleton_tokens, _tokenize_path

logger = get_logger(__name__)

_DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

_MODEL_CACHE = {}
_OPENAI_CLIENT = None

# Models served by the OpenAI embeddings API instead of a local HF checkpoint --
# embed_texts() dispatches here instead of loading an AutoModel for these names.
_OPENAI_EMBEDDING_MODELS = {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}

# Same idea for Voyage AI's embeddings API.
_VOYAGE_EMBEDDING_MODELS = {"voyage-code-3", "voyage-code-2"}

# Decoder-based embedding models (BGE-Code-v1, Qwen3-Embedding) use last-token pooling,
# not mean pooling -- the causal attention mask means only the final token's hidden state
# has seen the whole sequence. Confirmed against each model's own card, not assumed:
# BAAI/bge-code-v1 (2B, Qwen2-based) and Qwen/Qwen3-Embedding-* both document this.
_LAST_TOKEN_POOLED_MODELS = {"BAAI/bge-code-v1", "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B"}

# These same models also expect an instruction-prefixed QUERY (not document/chunk text --
# documents are embedded raw). Exact templates per each model's own card.
_QUERY_INSTRUCTION_TEMPLATES = {
    "BAAI/bge-code-v1": "<instruct>Given a bug report, retrieve source code files relevant to localizing the bug.\n<query>{query}",
    "Qwen/Qwen3-Embedding-0.6B": "Instruct: Given a bug report, retrieve source code files relevant to localizing the bug.\nQuery:{query}",
    "Qwen/Qwen3-Embedding-4B": "Instruct: Given a bug report, retrieve source code files relevant to localizing the bug.\nQuery:{query}",
    "Qwen/Qwen3-Embedding-8B": "Instruct: Given a bug report, retrieve source code files relevant to localizing the bug.\nQuery:{query}",
}


def _load_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        logger.info(f"Loading embedding model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(_DEVICE).eval()
        _MODEL_CACHE[model_name] = (tokenizer, model)
    return _MODEL_CACHE[model_name]


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _last_token_pool(last_hidden_state, attention_mask):
    """Right-padding-compatible last-token pooling (our tokenizer calls use HF's default
    right padding, not the left-padding some model cards' example code assumes): the last
    real (non-pad) token's index is attention_mask.sum(dim=1) - 1 for each sequence."""
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_indices, sequence_lengths]


def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        from openai import OpenAI
        # Default max_retries=2 wasn't enough to ride out a sustained 429 burst on a
        # chunk-heavy instance (thousands of chunks -> dozens of requests in quick
        # succession) -- raised well above default, SDK backoff is exponential already.
        _OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=8)
    return _OPENAI_CLIENT


_OPENAI_MAX_INPUT_CHARS = 20000  # conservative vs. the API's 8192-token cap (~4 chars/token for code)


def _embed_texts_openai(texts: list[str], model_name: str, batch_size: int = 100) -> torch.Tensor:
    """Embed via the OpenAI embeddings API (paid). Unlike the local HF path (which silently
    truncates at the tokenizer's max_length), the API hard-rejects oversized input -- the AST
    chunker doesn't cap chunk size for large classes/headers, so this truncates defensively
    rather than crashing mid-run on one pathological chunk. Smaller batches + a short pacing
    delay between requests avoid bursting the rate limit on chunk-heavy instances (thousands
    of chunks -> dozens of back-to-back requests), on top of the client's own retry/backoff."""
    client = _get_openai_client()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = [(t if t.strip() else " ")[:_OPENAI_MAX_INPUT_CHARS] for t in texts[i:i + batch_size]]
        response = client.embeddings.create(model=model_name, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
        if i + batch_size < len(texts):
            time.sleep(0.2)
    return torch.tensor(all_embeddings)


def _embed_texts_voyage(texts: list[str], model_name: str, is_query: bool, batch_size: int = 100,
                         max_retries: int = 8) -> torch.Tensor:
    """Embed via the Voyage AI embeddings API (paid, generous free tier). Plain REST call
    (matches how this codebase already talks to GitHub -- requests, not a vendor SDK) since
    the API is a single simple endpoint. input_type distinguishes query vs. document text
    natively (the API prepends its own retrieval-appropriate instruction), so unlike the
    BGE-Code/Qwen3 path this needs no hand-written instruction template.

    Manual retry/backoff on 429 -- unlike the OpenAI path (whose SDK client retries
    internally), a plain requests.post() has none by default, and this crashed a real
    6-instance run uncaught on its very last model (2026-08-10) after ~55 minutes of
    otherwise-successful work on the model before it -- exactly what backoff exists to
    prevent."""
    api_key = os.getenv("VOYAGE_AI_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_AI_API_KEY not set in the environment")

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = [t if t.strip() else " " for t in texts[i:i + batch_size]]
        for attempt in range(max_retries + 1):
            response = requests.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"input": batch, "model": model_name, "input_type": "query" if is_query else "document"},
            )
            if response.status_code != 429 or attempt == max_retries:
                response.raise_for_status()
                break
            retry_after = float(response.headers.get("Retry-After", 2 ** attempt))
            logger.warning(f"Voyage 429, retrying in {retry_after:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(retry_after)
        data = sorted(response.json()["data"], key=lambda item: item["index"])
        all_embeddings.extend(item["embedding"] for item in data)
        if i + batch_size < len(texts):
            time.sleep(0.2)
    return torch.tensor(all_embeddings)


@torch.no_grad()
def embed_texts(texts: list[str], model_name: str = "microsoft/unixcoder-base", batch_size: int = 32,
                 is_query: bool = False) -> torch.Tensor:
    """Embed a list of texts, returning an (N, hidden_size) tensor. Local HF checkpoints use
    mean pooling by default, or last-token pooling for models in _LAST_TOKEN_POOLED_MODELS
    (decoder-based embedding models); models in _OPENAI_EMBEDDING_MODELS/_VOYAGE_EMBEDDING_MODELS
    are dispatched to their respective APIs instead. Pass is_query=True when embedding the bug
    report/search query (not document/chunk text) -- models with an instruction template (or,
    for Voyage, the API's own native input_type) wrap it accordingly; models without one are
    unaffected."""
    if model_name in _OPENAI_EMBEDDING_MODELS:
        return _embed_texts_openai(texts, model_name)

    if model_name in _VOYAGE_EMBEDDING_MODELS:
        return _embed_texts_voyage(texts, model_name, is_query=is_query)

    if is_query and model_name in _QUERY_INSTRUCTION_TEMPLATES:
        texts = [_QUERY_INSTRUCTION_TEMPLATES[model_name].format(query=t) for t in texts]

    tokenizer, model = _load_model(model_name)
    use_last_token = model_name in _LAST_TOKEN_POOLED_MODELS
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(_DEVICE)
        outputs = model(**inputs)
        pooled = (_last_token_pool if use_last_token else _mean_pool)(outputs.last_hidden_state, inputs["attention_mask"])
        all_embeddings.append(pooled.cpu())

    return torch.cat(all_embeddings, dim=0)


def _file_text_for_embedding(path: str, content: str | None) -> str:
    """Same text basis as the BM25 skeleton retriever (path tokens + docstring/class/function
    names), so the two retrieval methods are compared on equal footing."""
    tokens = _tokenize_path(path)
    if content is not None:
        try:
            tokens = tokens + _extract_skeleton_tokens(content)
        except Exception:
            pass
    return " ".join(tokens)


def _fetch_contents_with_timing(bug, file_paths: list[str]) -> tuple[dict, float]:
    """Fetch file contents via the offline repo_cache (never a live network call), returning
    (contents, elapsed_seconds). Shared by both embedding ranking variants below."""
    t0 = time.time()
    repo_available = is_repo_cached(bug.repo)
    contents = get_file_contents_batch(bug.repo, bug.base_commit, file_paths) if repo_available else {}
    return contents, time.time() - t0


def rank_files_embedding(bug, top_k: int | None = 100, model_name: str = "microsoft/unixcoder-base") -> tuple[list[str], dict]:
    """Rank bug.code_files by cosine similarity of mean-pooled embeddings (path +
    content-skeleton text, same basis as the BM25 skeleton variant) to bug.bug_report,
    returning the top_k most relevant. Pass top_k=None for the full ranking (e.g. for
    screening/diagnostics that need every ground-truth file's rank). Returns
    (ranked_file_paths, timing_info)."""
    file_paths = bug.code_files
    if not file_paths or (top_k is not None and len(file_paths) <= top_k):
        return file_paths, {}

    contents, t_fetch = _fetch_contents_with_timing(bug, file_paths)

    texts = [_file_text_for_embedding(p, contents.get(p)) for p in file_paths]

    t1 = time.time()
    file_embeddings = embed_texts(texts, model_name=model_name)
    query_embedding = embed_texts([bug.bug_report], model_name=model_name, is_query=True)
    t_embed = time.time() - t1

    scores = torch.nn.functional.cosine_similarity(query_embedding, file_embeddings)
    ranked_idx = torch.argsort(scores, descending=True)[:top_k]
    ranked_paths = [file_paths[i] for i in ranked_idx.tolist()]

    timing = {"fetch_s": t_fetch, "embed_s": t_embed, "num_files": len(file_paths)}
    return ranked_paths, timing


def _chunk_file_content(content: str | None, max_chunk_chars: int = 1500, overlap_chars: int = 200) -> list[str]:
    """Split file content into chunks respecting function/class boundaries where possible
    (AST-based: one chunk per top-level function/class, plus a header chunk for imports and
    module docstring), falling back to fixed-size overlapping character windows for content
    that doesn't parse or has no top-level definitions.

    This exists because whole-file embedding is a documented weak strategy: one paper in
    docs/literature_review.md reports whole-file embedding scoring only 3-12% Acc@10 vs.
    33-71% for chunked (code-segment level) embedding on the same task -- a >400% relative
    difference from chunking alone. rank_files_embedding() above does whole-file embedding;
    rank_files_embedding_chunked() below uses this function to test whether that gap holds.
    """
    if not content:
        return []

    try:
        tree = ast.parse(content)
        top_level_defs = [
            n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
    except (SyntaxError, ValueError):
        top_level_defs = None

    if top_level_defs:
        chunks = []
        lines = content.splitlines(keepends=True)
        header = "".join(lines[:top_level_defs[0].lineno - 1]).strip()
        if header:
            chunks.append(header)
        for node in top_level_defs:
            segment = ast.get_source_segment(content, node)
            if segment:
                chunks.append(segment)
        if chunks:
            return chunks

    # Fallback: fixed-size overlapping character windows (no parseable structure to chunk by).
    step = max(max_chunk_chars - overlap_chars, 1)
    return [content[i:i + max_chunk_chars] for i in range(0, len(content), step)]


def rank_files_embedding_chunked(bug, top_k: int | None = 100, model_name: str = "microsoft/unixcoder-base",
                                   max_chunk_chars: int = 1500) -> tuple[list[str], dict]:
    """Like rank_files_embedding, but embeds each file as multiple content chunks
    (_chunk_file_content) instead of one whole-file skeleton text, scoring each file by its
    MAX chunk-to-query cosine similarity (a file is relevant if any one chunk of it is --
    max, not mean, so one relevant function isn't diluted by many irrelevant ones in a large
    file). Falls back to a path-token pseudo-chunk for any file with no fetchable content.
    """
    file_paths = bug.code_files
    if not file_paths or (top_k is not None and len(file_paths) <= top_k):
        return file_paths, {}

    contents, t_fetch = _fetch_contents_with_timing(bug, file_paths)

    chunk_owners = []
    chunk_texts = []
    for path in file_paths:
        file_chunks = _chunk_file_content(contents.get(path), max_chunk_chars=max_chunk_chars)
        if not file_chunks:
            file_chunks = [" ".join(_tokenize_path(path))]
        for chunk in file_chunks:
            chunk_owners.append(path)
            chunk_texts.append(chunk)

    t1 = time.time()
    chunk_embeddings = embed_texts(chunk_texts, model_name=model_name)
    query_embedding = embed_texts([bug.bug_report], model_name=model_name, is_query=True)
    t_embed = time.time() - t1

    chunk_scores = torch.nn.functional.cosine_similarity(query_embedding, chunk_embeddings)

    file_best_score = {}
    for path, score in zip(chunk_owners, chunk_scores.tolist()):
        if path not in file_best_score or score > file_best_score[path]:
            file_best_score[path] = score

    ranked = sorted(file_paths, key=lambda p: file_best_score.get(p, float("-inf")), reverse=True)
    ranked_paths = ranked[:top_k] if top_k is not None else ranked

    timing = {
        "fetch_s": t_fetch, "embed_s": t_embed,
        "num_files": len(file_paths), "num_chunks": len(chunk_texts),
    }
    return ranked_paths, timing
