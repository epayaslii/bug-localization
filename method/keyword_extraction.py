"""EmbedRank-style keyword extraction (candidate n-grams ranked by cosine similarity to the
document embedding, diversified via Maximal Marginal Relevance) -- the mechanism IQLoc uses
(via KeyBERT) for both its bug-report-side and code-side keyword extraction stages, see
docs/iqloc_replication_scoping.md. IQLoc embeds candidates with a domain-pretrained CodeT5;
this project has no such model, so any embedding model already wired in
method/embedding_retriever.py can be passed instead (an explicit approximation, not a claim
of matching CodeT5's semantics).

No new pip dependency (no KeyBERT/sklearn) -- candidate generation is regex tokenization
(matching method/bm25_retriever.py's existing style) and MMR is a ~15-line numpy loop.
"""

import re

import numpy as np

from method.embedding_retriever import embed_texts

_TOKEN_RE = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*')
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "to", "of", "in", "on", "at", "for", "with", "as", "by",
    "it", "its", "if", "then", "else", "not", "no", "so", "we", "i", "you", "he", "she", "they",
    "from", "into", "when", "while", "than", "there", "here", "which", "who", "what",
}


def _split_identifier(token: str) -> list[str]:
    """camelCase/snake_case/PascalCase -> lowercase word parts, e.g. 'getFileContents' ->
    ['get', 'file', 'contents']. Mirrors method/bm25_retriever.py's _tokenize_path approach."""
    text = re.sub(r'[_\-.]', ' ', token)
    text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', text)
    return [w.lower() for w in text.split() if w]


def _candidate_keywords(text: str, max_candidates: int = 150) -> list[str]:
    """Unigram + bigram candidates from identifier-like tokens, deduplicated, stopwords and
    single-character words dropped. Bounded to max_candidates (by first occurrence) to keep
    the embedding call cost predictable regardless of document length."""
    raw_tokens = _TOKEN_RE.findall(text)
    words = []
    for tok in raw_tokens:
        words.extend(_split_identifier(tok))
    words = [w for w in words if len(w) > 1 and w not in _STOPWORDS]

    seen = set()
    candidates = []
    for w in words:
        if w not in seen:
            seen.add(w)
            candidates.append(w)

    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i + 1]}"
        if bigram not in seen:
            seen.add(bigram)
            candidates.append(bigram)

    return candidates[:max_candidates]


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return a_norm @ b_norm.T


def embedrank_mmr_keywords(
    text: str, model_name: str = "microsoft/unixcoder-base", top_n: int = 15,
    lambda_mult: float = 0.5, max_candidates: int = 150,
) -> list[str]:
    """IQLoc's Algorithm 1: embed each candidate keyword and the whole document, then greedily
    select top_n candidates maximizing `lambda_mult * sim(candidate, doc) -
    (1 - lambda_mult) * max_sim(candidate, already_selected)` -- relevant to the document, but
    not redundant with what's already picked. IQLoc's own N/lambda sweep (N in {5,15,43}) found
    15 the knee of the curve; defaults match that.
    """
    candidates = _candidate_keywords(text, max_candidates=max_candidates)
    if not candidates:
        return []
    if len(candidates) <= top_n:
        return candidates

    doc_embedding = embed_texts([text], model_name=model_name, is_query=False).numpy()
    candidate_embeddings = embed_texts(candidates, model_name=model_name, is_query=False).numpy()

    doc_sim = _cosine_sim_matrix(candidate_embeddings, doc_embedding).squeeze(-1)  # (n_candidates,)
    pairwise_sim = _cosine_sim_matrix(candidate_embeddings, candidate_embeddings)  # (n_candidates, n_candidates)

    selected_idx: list[int] = []
    remaining_idx = list(range(len(candidates)))
    while len(selected_idx) < top_n and remaining_idx:
        if not selected_idx:
            scores = doc_sim[remaining_idx]
        else:
            redundancy = pairwise_sim[np.ix_(remaining_idx, selected_idx)].max(axis=1)
            scores = lambda_mult * doc_sim[remaining_idx] - (1 - lambda_mult) * redundancy
        best_local = int(np.argmax(scores))
        best_idx = remaining_idx[best_local]
        selected_idx.append(best_idx)
        remaining_idx.pop(best_local)

    return [candidates[i] for i in selected_idx]


def reformulate_query_iqloc_style(
    bug_report_keywords: list[str], code_keywords: list[str],
    model_name: str = "microsoft/unixcoder-base", top_matches: int = 15,
) -> list[str]:
    """IQLoc Stage 4c: cosine similarity between bug-report-side and code-side keywords,
    keep the code keywords most similar to (any of) the bug-report keywords -- these are the
    terms the query gets expanded with, not the raw union of both keyword sets."""
    if not bug_report_keywords or not code_keywords:
        return []

    br_embeddings = embed_texts(bug_report_keywords, model_name=model_name, is_query=False).numpy()
    code_embeddings = embed_texts(code_keywords, model_name=model_name, is_query=False).numpy()
    sim = _cosine_sim_matrix(code_embeddings, br_embeddings)  # (n_code, n_br)
    max_sim_per_code = sim.max(axis=1)

    order = np.argsort(-max_sim_per_code)[:top_matches]
    return [code_keywords[i] for i in order]
