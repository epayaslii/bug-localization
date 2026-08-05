import time

import torch
from transformers import AutoTokenizer, AutoModel

from dataset.repo_cache import get_file_contents_batch, is_repo_cached
from dataset.utils import get_logger
from method.bm25_retriever import _extract_skeleton_tokens, _tokenize_path

logger = get_logger(__name__)

_DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

_MODEL_CACHE = {}


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


@torch.no_grad()
def embed_texts(texts: list[str], model_name: str = "microsoft/unixcoder-base", batch_size: int = 32) -> torch.Tensor:
    """Embed a list of texts, returning an (N, hidden_size) tensor of mean-pooled embeddings."""
    tokenizer, model = _load_model(model_name)
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt").to(_DEVICE)
        outputs = model(**inputs)
        pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
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


def rank_files_embedding(bug, top_k: int | None = 100, model_name: str = "microsoft/unixcoder-base") -> tuple[list[str], dict]:
    """Rank bug.code_files by cosine similarity of mean-pooled embeddings (path +
    content-skeleton text, same basis as the BM25 skeleton variant) to bug.bug_report,
    returning the top_k most relevant. Pass top_k=None for the full ranking (e.g. for
    screening/diagnostics that need every ground-truth file's rank). Returns
    (ranked_file_paths, timing_info)."""
    file_paths = bug.code_files
    if not file_paths or (top_k is not None and len(file_paths) <= top_k):
        return file_paths, {}

    t0 = time.time()
    repo_available = is_repo_cached(bug.repo)
    contents = get_file_contents_batch(bug.repo, bug.base_commit, file_paths) if repo_available else {}
    t_fetch = time.time() - t0

    texts = [_file_text_for_embedding(p, contents.get(p)) for p in file_paths]

    t1 = time.time()
    file_embeddings = embed_texts(texts, model_name=model_name)
    query_embedding = embed_texts([bug.bug_report], model_name=model_name)
    t_embed = time.time() - t1

    scores = torch.nn.functional.cosine_similarity(query_embedding, file_embeddings)
    ranked_idx = torch.argsort(scores, descending=True)[:top_k]
    ranked_paths = [file_paths[i] for i in ranked_idx.tolist()]

    timing = {"fetch_s": t_fetch, "embed_s": t_embed, "num_files": len(file_paths)}
    return ranked_paths, timing
