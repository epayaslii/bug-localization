# Results

Raw JSON outputs and a human-readable summary of the evaluation runs performed on 2026-08-05, using the evaluation infrastructure in `evaluation/` and the scripts in `scripts/`. All SWE-bench Verified, seed 42.

## Manifest

`manifests/swebench-multi-n30-s42-6757c7d8bb76.json` — 30 instances across 9 repos, `--pool-size 30 --max-per-repo 30` (i.e. no diversity capping), so it's the exact same 30 instances `python main.py --sample-size 30` would sample on its own (both draw from `get_bug_instances(sample_size=30, random_sample=True, random_seed=42)`), chosen deliberately for direct comparability between the free BM25 screening numbers below and the paid end-to-end runs.

## 1. BM25 retrieval-recall ceiling comparison (free, offline)

`bm25_comparison_swebench_30.json` — produced by `scripts/compare_bm25_representations.py`. Measures, for each document representation, how often and how completely the ground-truth file(s) land in the BM25-ranked candidate set, before any LLM touches it.

| Representation | Hit@1 | Hit@5 | Hit@10 | Hit@100 | Hit@200 | Recall@100 | Recall@200 | MRR |
|---|---|---|---|---|---|---|---|---|
| path_only | 10.0% | 20.0% | 23.3% | 60.0% | 73.3% | 58.2% | 71.6% | 0.1496 |
| skeleton | 0.0% | 26.7% | 30.0% | 80.0% | 86.7% | 73.0% | 81.0% | 0.1382 |
| symbols_with_imports | 6.7% | 20.0% | 20.0% | 83.3% | 93.3% | 77.0% | 87.7% | 0.1539 |
| **symbols_no_imports** | 6.7% | 23.3% | 40.0% | **86.7%** | 93.3% | **81.0%** | 87.7% | **0.1769** |

Difficulty bands (best localizable-GT rank; `easy`≤10, `medium`≤100, `hard`≤200, else `outside_top200`):

| Representation | easy | medium | hard | outside_top200 |
|---|---|---|---|---|
| path_only | 7 | 11 | 4 | 8 |
| skeleton | 9 | 15 | 2 | 4 |
| symbols_with_imports | 6 | 19 | 3 | 2 |
| symbols_no_imports | 12 | 14 | 2 | 2 |

`symbols_no_imports` wins on every metric here. Note skeleton's Hit@100=80.0% closely matches an "80% recall ceiling" figure already on record from before this evaluation harness existed — a useful consistency check that the two are measuring the same thing.

## 2. End-to-end evaluation (paid, `openrouter` / `gpt-4o-mini`)

Both runs use `--bm25-top-k 100` on the same 30-instance sample; `--max-files`/BM25 candidate generation differs only by representation. **No-retrieval baseline for comparison: 43.3%** (naive whole-file-list prompt, no BM25 pre-filter, from an earlier 30-sample run — see `docs/literature_review.md`).

| Config | Command | Accuracy | Precision | Recall | F1 | TP/FP/FN |
|---|---|---|---|---|---|---|
| Skeleton | `--bm25-skeleton` | **50.0%** | 50.0% | 37.5% | 42.9% | 15/15/25 |
| Symbols (no imports) | `--bm25-symbols` | **50.0%** | 50.0% | 37.5% | 42.9% | 15/15/25 |

`end_to_end_swebench_30_skeleton.json` / `end_to_end_swebench_30_symbols.json`. Both beat the 43.3% no-retrieval baseline by +6.7pp — **this confirms BM25 pre-filtering genuinely helps end-to-end, not just its retrieval-only ceiling.**

**But the two configs tie exactly at the aggregate level**, despite `symbols_no_imports` having a meaningfully better retrieval ceiling (86.7% vs 80.0% Hit@100). Per-instance, exactly 4 of 30 flip and cancel out:

| Instance | Skeleton | Symbols |
|---|---|---|
| django__django-12125 | miss | **hit** |
| django__django-13551 | miss | **hit** |
| django__django-16100 | hit | **miss** |
| matplotlib__matplotlib-20859 | hit | **miss** |

Two improve, two degrade, net zero change. At n=30 this reads as noise rather than evidence that one representation is truly better end-to-end — a bigger sample is needed to tell whether `symbols_no_imports`' retrieval-ceiling advantage (item 1) actually converts into a real accuracy edge, or whether the LLM reranker's own behavior is simply representation-insensitive once a file is reasonably placed in the candidate set.

(`end_to_end_swebench_30_skeleton.json` was reconstructed from that run's captured log rather than re-run with `--output`, since the flag didn't exist yet at the time — it has full per-bug accuracy/precision/recall/F1 but not the raw candidate-file lists that `end_to_end_swebench_30_symbols.json` includes.)

## 3. Embedding retrieval-recall ceiling (negative result)

`embedding_ceiling_test_swebench_6.json` — produced by `scripts/run_embedding_ceiling_test.py` on a smaller 6-instance manifest. UniXCoder embeddings (whole-file, path+skeleton text, mean-pooled, cosine similarity) vs. BM25 path-only:

| Method | Hit@1 | Hit@5 | Hit@10 | Hit@100 | MRR |
|---|---|---|---|---|---|
| bm25_path_only | 16.7% | 50.0% | 50.0% | 50.0% | 0.2664 |
| embedding (unixcoder-base) | 0.0% | 16.7% | 16.7% | 83.3% | 0.0661 |

Embeddings lose badly at the ranks that matter for a downstream reranker (Hit@1/5/10, MRR), despite broader-but-imprecise Hit@100. Consistent with both the original project team and the co-intern's team deprioritizing embeddings here, and with a specific literature finding (`docs/literature_review.md`) that whole-file embedding is a documented weak strategy vs. chunked embedding. The code for this lives on the `experiment/embedding-ceiling` branch (not merged to `main`, since the result doesn't justify adopting it); only this result JSON is included here on `main` for a complete record.

## 4. Hybrid retrieval: BM25 + chunked embedding (positive, but n=30 revises the n=5 story)

`hybrid_retrieval_swebench_6.json` / [`hybrid_retrieval_report.html`](hybrid_retrieval_report.html) — produced by `scripts/run_hybrid_retrieval_test.py` on a 6-instance manifest, testing whether *chunked* (not whole-file) embedding changes the §3 result, per the literature's own explanation for why whole-file embedding underperforms. Chunking is AST-based (one chunk per top-level function/class, plus a header chunk for imports/docstring); BM25 (symbols representation) narrows the full corpus to a 200-file candidate pool first, only that pool gets chunk-embedded and reranked, fused with BM25's own ranking via Reciprocal Rank Fusion (k=60):

| Config | Hit@1 | Hit@5 | Hit@10 | Hit@100 | MRR | MAP |
|---|---|---|---|---|---|---|
| bm25 (symbols) | 0.0% | 33.3% | 33.3% | 83.3% | 0.178 | 0.178 |
| chunked_embedding | 0.0% | 33.3% | 50.0% | 66.7% | 0.150 | 0.139 |
| **hybrid_rrf** | **16.7%** | 33.3% | 50.0% | 66.7% | **0.282** | **0.264** |

At n=5 this looked like a clean positive result: hybrid RRF fusion beat BM25 alone on MRR (+0.104) and MAP, and landed an actual Hit@1 (`astropy__astropy-14508`: rank 1) that neither BM25 alone (rank 2) nor chunked-embedding alone (rank 7) achieved on its own — evidence, at that sample size, of genuine complementary signal rather than noise.

**Scaled to n=30** (`hybrid_retrieval_swebench_30.json`, diversity-constrained manifest `manifests/swebench-multi-n30-s42-1fb8f4b8d82f.json`, 30 instances / 29 with localizable GT / 11 distinct repos, `--max-per-repo 3` since SWE-bench Verified only spans 12 repos total):

| Config | Hit@1 | Hit@5 | Hit@10 | Hit@100 | MRR | MAP |
|---|---|---|---|---|---|---|
| bm25 (symbols) | 3.3% | 3.3% | 30.0% | 86.7% | 0.087 | 0.088 |
| **chunked_embedding** | **16.7%** | 26.7% | 30.0% | 90.0% | **0.233** | **0.233** |
| hybrid_rrf | 10.0% | 33.3% | 33.3% | 90.0% | 0.206 | 0.206 |

The headline direction **holds and strengthens**: hybrid still more than doubles BM25's MRR (0.087 → 0.206, +0.119). But the n=5 "fusion beats both of its own inputs" story does **not** replicate — at n=30, **chunked_embedding alone now beats the hybrid** on Hit@1 and MRR. Per-instance inspection shows why: in several cases (`django__django-14672`, `scikit-learn__scikit-learn-14629`, `sphinx-doc__sphinx-8459`) chunked embedding alone lands rank 1, but RRF-fusing it with BM25's much weaker ranking drags the fused rank down to 11, 4, and 2 respectively. Only one instance (`scikit-learn__scikit-learn-14710`: bm25=7, embedding=13 → hybrid=1) shows fusion beating both parts, versus this being the *typical* pattern at n=5. Reading: RRF's benefit at n=5 was likely partly a small-sample artifact; at n=30, unweighted RRF against a much-weaker first-stage signal (BM25) net-drags down an already-strong embedding ranking more often than it rescues a weak one. Chunked embedding alone — not the hybrid — looks like the stronger reranking candidate at this scale, worth investigating further (e.g. weighted RRF, or dropping BM25 from the fusion entirely) before treating the hybrid as the answer.

n=30 is still one manifest/one run — directional, not the final word — but larger and more diverse (11 repos vs 5) than the original n=5 test. The code lives on the `experiment/hybrid-retrieval` branch (not merged to `main`, matching this session's branching policy: main only holds validated work); only the result artifacts are included here for a complete record.

## Reproducing

```bash
python scripts/generate_evaluation_manifest.py --dataset swebench --size 30 --pool-size 30 --seed 42 --max-per-repo 30
python scripts/compare_bm25_representations.py --manifest results/manifests/swebench-multi-n30-s42-6757c7d8bb76.json --output results/bm25_comparison_swebench_30.json
python main.py --method openrouter --dataset swebench --model gpt-4o-mini --sample-size 30 --bm25-top-k 100 --bm25-skeleton --output results/end_to_end_swebench_30_skeleton.json
python main.py --method openrouter --dataset swebench --model gpt-4o-mini --sample-size 30 --bm25-top-k 100 --bm25-symbols --output results/end_to_end_swebench_30_symbols.json

# §4 n=30 hybrid retrieval (checkout experiment/hybrid-retrieval first)
python scripts/generate_evaluation_manifest.py --dataset swebench --size 30 --pool-size 500 --seed 42 --max-per-repo 3
python scripts/run_hybrid_retrieval_test.py --manifest results/manifests/swebench-multi-n30-s42-1fb8f4b8d82f.json --candidate-pool-size 200 --output results/hybrid_retrieval_swebench_30.json
```

The two `main.py` runs cost real OpenRouter API usage (`gpt-4o-mini`, paid). §3 and §4 (`run_embedding_ceiling_test.py`, `run_hybrid_retrieval_test.py`) require checking out the `experiment/embedding-ceiling` and `experiment/hybrid-retrieval` branches respectively — those scripts aren't on `main`. The §4 n=30 run is local-compute-only (UniXCoder embeddings, no API cost) but takes ~40 minutes on CPU.
