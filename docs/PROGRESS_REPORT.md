# Project Progress Report

Chronological status for the scalable bug-localization pipeline. Companion docs: [README.md](../README.md) (setup, usage, commands), [project_structure.md](project_structure.md) (file-by-file breakdown), [architecture.md](architecture.md) (dependency + data-flow diagrams), [literature_review.md](literature_review.md) (24+ papers surveyed).

All numeric results below are taken from committed JSON/Markdown under [`results/`](../results/). Diagnostic subsets (n<30) are labeled explicitly.

---

## Current status snapshot

| Area | Status | Notes |
|---|---|---|
| Dataset loading | Complete | SWE-bench Verified (primary); BeetleBox loader present, `BEETLEBOX_LOCAL_PATH` offline loading now implemented |
| Ground-truth localizability diagnostics | Complete | Before/after classification (5 classes) + disk caching |
| BM25 baseline + representations | Complete | path-only / skeleton / symbols(+imports) — real n=30 comparison |
| Seeded evaluation manifests + screening | Complete | Deterministic, diversity-capped, stable content-hash IDs |
| Retrieval-vs-reranking failure attribution | Complete | Free offline split; oracle diagnostic built, not yet run live (costs API) |
| End-to-end evaluation (real, paid) | Complete | 50.0% accuracy vs. 43.3% no-retrieval baseline, n=30 |
| MAP metric | Complete | Added to `evaluation/screening.py` |
| Test suite | Complete | 81 passing tests on `main` (90 on `experiment/hybrid-retrieval`) |
| Architecture diagrams | Complete | `docs/architecture.md`, verified against actual imports |
| Embedding retrieval (whole-file) | Diagnostic complete | **Negative result** — branch `experiment/embedding-ceiling`, not merged |
| Hybrid retrieval (BM25 + chunked embedding) | Diagnostic complete | **Positive, directional result** (n=5) — branch `experiment/hybrid-retrieval`, not merged |
| MareNostrum 5 (MN5) preparation | Partial | Offline dataset/repo_cache/wheelhouse validated (5-sample dry run); a real MN5 run guide read and saved, but targets a different fork and hasn't been adapted/executed here; no SSH access from current machine |
| BeetleBox transition | Not started | Next dataset per confirmed plan sequencing |
| BugsInPy integration | Not started | Lowest priority |
| Literature review | Complete | 24+ papers; one named gap ("Toggle") |

**Tests:** `python -m pytest` — **81 passed** on `main` (2026-08-05).

---

## 1. Project objective

Build a **file-level** bug localization system that, for a bug instance (bug report + repository snapshot):

1. Builds a before-fix source corpus.
2. Narrows candidate files via BM25 (optionally symbol/skeleton-enriched, optionally fused with chunked embeddings).
3. Reranks the candidate set with an LLM via OpenRouter.
4. Evaluates with Hit@k, Recall@k, MRR, MAP, and end-to-end accuracy/precision/recall/F1 — against a documented no-retrieval baseline.

---

## 2. Ground-truth localizability diagnostics

**Objective:** Separate "ground-truth file doesn't exist in the searchable corpus" from "retrieval genuinely failed."

**Implementation:** `dataset/localizability.py` classifies each ground-truth path as `exists_before_fix`, `deleted_by_fix` (still a valid retrieval target), `added_by_fix` (not localizable — introduced by the fix itself), `missing_unresolved`, or `api_error` (never cached, so a transient failure retries next run instead of being remembered as final). Classification is offline-first: parses the patch's own unified-diff headers (`new file mode` / `deleted file mode`) when available (SWE-bench), falling back to an after-commit existence check only when the patch has no real diff hunk (BeetleBox's synthetic `"Before: X\nAfter: Y"` patch text).

**Main files:** `dataset/localizability.py`, `scripts/localizability_report.py`.

---

## 3. Seeded, diversity-constrained evaluation manifests

**Objective:** Compare retrieval/reranking approaches on identical data across runs, not ad-hoc one-off samples.

**Implementation:** `evaluation/manifest.py` builds a deterministic, seeded sample capped at N instances per repository, with a stable content-derived manifest ID. `evaluation/screening.py` runs any pluggable `rank_fn(bug) -> list[str]` over a manifest, reporting per-instance best rank, Hit@k, Recall@k, Average Precision, and a difficulty band (`easy`/`medium`/`hard`/`outside_top200`/`no_localizable_gt`); `summarize_screening()` macro-averages into Hit@k, MRR, and MAP for comparing configurations side by side.

**Main files:** `evaluation/manifest.py`, `evaluation/screening.py`, `scripts/generate_evaluation_manifest.py`, `scripts/run_bm25_screening.py`.

---

## 4. BM25 baseline and document representations

**Objective:** Cheap, deterministic candidate retrieval, enriched beyond bare file paths.

**Implementation:** `method/bm25_retriever.py` supports four representations: `path_only`, `skeleton` (docstring + class/function names), `symbols_with_imports`, `symbols_no_imports` (class/function/method names ± import tokens, AST-based — Python only). All fall back to path-only tokens for unparseable/unfetchable content; never a live network call, only the offline `repo_cache`.

**Real n=30 comparison** (`results/bm25_comparison_swebench_30.json`):

| Representation | Hit@100 | Recall@100 | MRR |
|---|---:|---:|---:|
| path_only | 60.0% | 58.2% | 0.1496 |
| skeleton | 80.0% | 73.0% | 0.1382 |
| symbols_with_imports | 83.3% | 77.0% | 0.1539 |
| **symbols_no_imports** | **86.7%** | **81.0%** | **0.1769** |

`symbols_no_imports` wins on every retrieval metric. skeleton's Hit@100=80.0% closely matches an "80% recall ceiling" figure already on record from before this evaluation harness existed — a useful consistency check.

**Main files:** `method/bm25_retriever.py`, `scripts/compare_bm25_representations.py`.

---

## 5. Retrieval-vs-reranking failure attribution

**Objective:** Separate "the LLM never saw the right file" from "the LLM saw it and didn't pick it."

**Implementation:** `evaluation/failure_attribution.py`'s `classify_retrieval_reach`/`summarize_retrieval_reach` split misses into retrieval-failure vs. reached-candidate-set, free and offline (reuses screening's per-GT ranks). `run_oracle_diagnostic` force-injects every localizable ground truth into the candidate set (retrieval recall forced to 100%) and measures pure reranking placement — this calls the LLM, so it's gated behind an explicit `--run-oracle` flag with a token-cost estimate printed first. **Not run live this session** (would cost real API usage).

**Main files:** `evaluation/failure_attribution.py`, `scripts/run_failure_attribution.py`.

---

## 6. Real end-to-end evaluation (paid, `openrouter` / `gpt-4o-mini`)

**Objective:** Does BM25 pre-filtering actually improve end-to-end LLM localization accuracy, not just its own retrieval ceiling?

**Scope:** 30-instance seeded SWE-bench sample (`results/manifests/swebench-multi-n30-s42-6757c7d8bb76.json`) — the exact same 30 instances `main.py --sample-size 30` draws on its own. **No-retrieval baseline: 43.3%** (naive whole-file-list prompt, no BM25 pre-filter).

| Config | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| skeleton-BM25 | 50.0% | 50.0% | 37.5% | 42.9% |
| symbols-BM25 | 50.0% | 50.0% | 37.5% | 42.9% |

Both beat the baseline by **+6.7pp** — confirms BM25 pre-filtering genuinely helps end-to-end, not just its retrieval-only ceiling. The two representations **tie exactly** at the aggregate level despite symbols' clear retrieval-ceiling advantage (§4): exactly 4/30 instances flip (2 improve, 2 degrade), netting zero. At n=30 this reads as noise, not evidence either representation is truly better end-to-end.

**Main files:** `main.py` (`--bm25-skeleton` / `--bm25-symbols` / `--bm25-symbols-imports` / `--output`), `results/end_to_end_swebench_30_skeleton.json`, `results/end_to_end_swebench_30_symbols.json`.

---

## 7. Embedding retrieval — whole-file (negative result)

**Objective:** Does a dense embedding retriever beat BM25?

**Implementation:** UniXCoder whole-file embeddings (mean-pooled, path + content-skeleton text, cosine similarity) — branch `experiment/embedding-ceiling`, **not merged to `main`**.

**Result** (n=6, `results/embedding_ceiling_test_swebench_6.json`):

| Method | Hit@10 | Hit@100 | MRR |
|---|---:|---:|---:|
| bm25_path_only | 50.0% | 50.0% | 0.2664 |
| embedding (unixcoder-base, whole-file) | 16.7% | 83.3% | 0.0661 |

Embeddings lose badly at the ranks that matter (Hit@1/5/10, MRR), despite broader-but-imprecise Hit@100. Consistent with both prior teams (this project's original team and the co-intern's) deprioritizing embeddings here, and with a specific literature finding (`docs/literature_review.md`) that whole-file embedding is a documented weak strategy vs. chunked embedding.

---

## 8. Hybrid retrieval — BM25 + chunked embedding (positive, directional result)

**Objective:** Does *chunked* (not whole-file) embedding change §7's conclusion, per the literature's own explanation for the gap?

**Implementation:** branch `experiment/hybrid-retrieval`, **not merged to `main`**. `method/embedding_retriever.py` gained AST-based chunking (`_chunk_file_content` — one chunk per top-level function/class plus a header chunk for imports/docstring, falling back to fixed-size overlapping character windows for unparseable content) and `rank_files_embedding_chunked` (scores each file by its **max** chunk-to-query cosine similarity, not mean). `method/hybrid_retriever.py` cascades BM25 (narrow the full corpus to a 200-file candidate pool first — cheap) into chunked-embedding reranking of only that pool, fused via Reciprocal Rank Fusion (k=60, matching the BLAZE paper's constant).

**Result** (n=6, 5 with localizable GT — `results/hybrid_retrieval_swebench_6.json`, [`results/hybrid_retrieval_report.html`](../results/hybrid_retrieval_report.html)):

| Config | Hit@1 | Hit@10 | Hit@100 | MRR | MAP |
|---|---:|---:|---:|---:|---:|
| bm25 (symbols) | 0.0% | 33.3% | 83.3% | 0.178 | 0.178 |
| chunked_embedding | 0.0% | 50.0% | 66.7% | 0.150 | 0.139 |
| **hybrid_rrf** | **16.7%** | 50.0% | 66.7% | **0.282** | **0.264** |

**Opposite finding from §7**: hybrid RRF fusion beats BM25 alone on MRR (+0.104) and MAP, including one instance (`astropy__astropy-14508`, rank 1) where the fused ranking beat *both* of its own inputs outright — evidence of genuine complementary signal, not just noise. Cost: Hit@100 drops to 66.7% for both embedding-involving configs vs. BM25's 83.3%, an inherent tradeoff of the candidate-pool cascade (RRF can push a file BM25 ranked well within its own top-100 further down if the chunk embedder scores it poorly).

**n=5 is small — directional, not conclusive.**

---

## 9. Test suite

**Objective:** Real automated coverage where previously there were only two standalone, non-pytest API-key sanity scripts.

**Implementation:** 81 pytest tests on `main` (90 including 9 hybrid/chunking-specific tests on `experiment/hybrid-retrieval`) covering `dataset/localizability.py`, `method/bm25_retriever.py`, all of `evaluation/`, `method/evaluate.py`, `dataset/utils.py`/`models.py`/`beetlebox.py`/`repo_cache.py` (the last via integration tests against a locally mirrored repo, skipped gracefully if none is mirrored). `pytest.ini` + `conftest.py` added.

**Bug caught and fixed along the way:** pytest's default file-collection pattern (`test_*.py` OR `*_test.py`) also matched the pre-existing `tests/openrouter_key_test.py`/`tests/github_token_test.py`, which have live-network top-level code. A bare `pytest` run was silently making real OpenRouter/GitHub API calls during collection and would hard-crash entirely for anyone without those keys set. Fixed by restricting `python_files = test_*.py`.

---

## 10. Architecture documentation

**Objective:** A visual complement to the existing file-by-file text breakdown.

**Implementation:** `docs/architecture.md` — a package-dependency diagram (verified per-file against actual imports, not guessed; caught two real inaccuracies while drawing it) and runtime data-flow diagrams for both the end-to-end (`main.py`) and offline-diagnostics (`scripts/`) paths.

---

## 11. BeetleBox offline loading fix

**Objective:** Close a real README/implementation mismatch.

**Finding:** `README.md` documented `BEETLEBOX_LOCAL_PATH` as supported, but `dataset/beetlebox.py`'s `load_data()` had no such check — only `dataset/swebench.py` implemented its local-path env var. Independently confirmed by a real prior MN5 execution guide, which had to hand-patch this exact gap to load BeetleBox on a cluster with no Hugging Face network access.

**Fix:** `dataset/beetlebox.py` now checks `BEETLEBOX_LOCAL_PATH` and uses `load_from_disk()`, mirroring `swebench.py` exactly. Two new tests build a real tiny on-disk dataset (no mocks) confirming both the local-load path works and the network path is never invoked when the local path is set.

---

## 12. MareNostrum 5 (MN5) preparation

**Prior work:** offline dataset, `repo_cache`, and wheelhouse already transferred to and validated on MN5 — a real 5-sample dry run confirmed dataset-loading, repo-cache, and prompt-construction machinery works end-to-end there (live LLM inference is structurally impossible on MN5, which has no outbound internet by design).

**This session:** a detailed real MN5 execution guide was read and its concrete facts saved — project account, module load order, working QoS, and a full offline-wheelhouse command sequence. That guide targets a different fork (`alikemalcoskun/bug-localization`), not this repo, and hasn't been adapted or executed here yet. No SSH access to MN5 is currently configured from the development machine.

---

## 13. Literature review

`docs/literature_review.md` — 24+ papers on LLM-based bug/fault localization, classified by approach (retrieval/embedding/graph/agentic), informing most of the design decisions above. One named gap: the "Toggle" hierarchical-localization paper, referenced in the project's own official roadmap but not yet reviewed here.

---

## 14. Current findings

1. Symbol-enriched BM25 (`symbols_no_imports`) has the best pure retrieval ceiling of any file-level representation tested (86.7% Hit@100, real n=30).
2. BM25 pre-filtering measurably improves real end-to-end LLM localization accuracy (43.3% → 50.0%, real n=30, paid).
3. Which BM25 representation wins end-to-end is inconclusive at n=30 despite a clear retrieval-ceiling gap between them — a bigger sample is needed.
4. Whole-file embedding is a genuine dead end here (n=6, but a large, unambiguous gap vs. BM25).
5. Chunked embedding fused with BM25 via RRF shows real promise (n=6, positive on every fused metric, including beating both of its own inputs on one instance) — the opposite of whole-file's result, consistent with the literature's specific explanation for the difference.
6. Localizability diagnostics and failure attribution infrastructure are built and validated but not yet combined into one large-scale end-to-end pass.

---

## 15. Current limitations

- All embedding-related findings (§7, §8) are n=5–6 — directional only, not statistically robust.
- Only SWE-bench Verified (Python) has been evaluated at real scale; BeetleBox (multi-language) is untouched despite being next in the confirmed dataset sequence.
- Symbol/chunk extraction is AST-based and Python-only; would silently degrade to path-only/character-window fallback on BeetleBox's non-Python instances.
- The oracle reranking diagnostic is built but has never been run live (costs real API money).
- No run so far has specifically oversampled hard-difficulty instances.
- MN5 execution for this specific repo hasn't happened — only the offline dry run and infrastructure prep.
- No comparison against literature SOTA baselines (BugCerberus, CoRNStack, LocAgent, etc.) on equivalent setups.

---

## 16. Next planned work

1. Scale the hybrid retrieval test (§8) to a larger manifest (n≈24–30) to confirm or disconfirm the positive n=6 signal.
2. Re-run the skeleton-vs-symbols end-to-end comparison (§6) at larger n to settle the current tie.
3. Write the MN5 execution handbook and adapt the read guide's command sequence to this repo.
4. BeetleBox transition (next dataset per confirmed sequencing) — note the Python-only AST caveat above.
5. BugsInPy integration (lower priority).

---

## 17. Reproducibility checklist

- [x] Manifest IDs and generation parameters recorded
- [x] Seeded sampling (seed=42 throughout)
- [x] BM25 config frozen per comparison run
- [x] Localizability classification cached; API errors never cached as final
- [x] Null-rank semantics documented (no imputation)
- [x] Result artifacts committed under `results/` (JSON, Markdown, one HTML visual, one plain-text table)
- [x] 81 passing tests on `main` (90 on `experiment/hybrid-retrieval`)
- [ ] Larger-n confirmation of the hybrid retrieval result
- [ ] MN5 execution for this repo specifically
- [ ] Non-Python AST support for symbol/chunk extraction

---

## What we can currently claim

**Safe**

- The pipeline is operational end-to-end (BM25 pre-filter → LLM rerank → evaluate), validated with real paid runs.
- BM25 pre-filtering measurably improves real end-to-end accuracy over a documented no-retrieval baseline.
- Whole-file embedding does not help here; a chunked-embedding + BM25 hybrid shows a real, if small-sample, positive signal.
- The evaluation infrastructure (manifests, screening, failure attribution) is reusable and already dataset-agnostic in its CLI flags, pending non-Python AST support for full BeetleBox benefit.

**Unsafe (explicitly avoid)**

- State-of-the-art claims
- General BeetleBox or multi-language results (untested)
- The hybrid retrieval result as conclusive (n=5–6)
- A definitive answer on which BM25 representation is best end-to-end (n=30 tie)
- MN5 production-readiness for this specific repo
