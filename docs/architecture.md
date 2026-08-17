# Architecture

Four views: package dependencies (what imports what), the `main.py` end-to-end runtime path, the retrieval subsystem it calls into (BM25 + embedding + weighted RRF), and the offline evaluation/diagnostics path. Complements the file-by-file breakdown in [project_structure.md](project_structure.md).

**Status (2026-08-17)**: as of the branch merge this session, `main.py` itself wires up the full retrieval stack — dataset selection across three benchmarks (Bench4BL primary, SWE-bench and BeetleBox secondary), optional BM25-only narrowing, and optional hybrid BM25+embedding narrowing via weighted RRF — not just the standalone `scripts/run_hybrid_*` tools Figure 3 used to describe as a separate unmerged pipeline.

## Figure 1a — Dataset & retrieval methods

```mermaid
graph TB
    subgraph dataset["dataset/ (self-contained)"]
        base[base.py]
        swebench[swebench.py]
        beetlebox[beetlebox.py]
        bench4bl["bench4bl.py\n(primary dataset)"]
        models[models.py]
        repo_cache["repo_cache.py\n(bare-clone + Bench4BL gitrepo fallback)"]
        utils_d[utils.py]
        localizability[localizability.py]
    end

    subgraph method["method/ (depends on dataset only)"]
        bm25["bm25_retriever.py"]
        java["java_parsing.py\n(lexical scanner, non-Python content)"]
        embedding["embedding_retriever.py\n(6 models incl. Qwen3-Embedding-0.6B)"]
        hybrid["hybrid_retriever.py\n(weighted RRF fusion)"]
        repoindex["repository_index.py\n(FAISS persistent index)"]
        fusion["fusion_signals.py\n(AST / dep-graph / commit-recency)"]
        prompt[prompt.py]
        evaluate[evaluate.py]
        llm[llm.py]
        utils_m[utils.py]
    end

    swebench --> base
    beetlebox --> base
    bench4bl --> base
    swebench --> models
    beetlebox --> models
    bench4bl --> models
    swebench --> utils_d
    beetlebox --> utils_d
    bench4bl --> utils_d
    utils_d --> repo_cache
    localizability --> utils_d

    bm25 --> repo_cache
    bm25 --> java
    embedding --> bm25
    embedding --> repo_cache
    embedding --> java
    hybrid --> bm25
    hybrid --> embedding
    repoindex --> repo_cache
    repoindex --> bm25
    repoindex --> embedding
    fusion --> repo_cache
    fusion --> bm25
    fusion --> repoindex
    utils_m --> utils_d
    method --> utils_d

    style embedding fill:#2f6f4f,stroke:#1a3f2c,color:#fff
    style hybrid fill:#2f6f4f,stroke:#1a3f2c,color:#fff
    style repoindex fill:#2f6f4f,stroke:#1a3f2c,color:#fff
    style bench4bl fill:#2f6f4f,stroke:#1a3f2c,color:#fff
    style java fill:#2f6f4f,stroke:#1a3f2c,color:#fff
    style fusion fill:#8a5a2b,stroke:#5c3b1a,color:#fff
```

**Legend**: **Green = the current recommended path** — Bench4BL (primary dataset), its Java-aware lexical scanner, chunked-embedding retrieval, weighted RRF fusion, and the FAISS persistent index all merged and validated. **Amber = tried, evaluated, closed as an informative negative result** (AST-similarity/dependency-graph/commit-recency fusion signals underperformed embedding-alone) — kept in the tree for reference, not adopted into the recommended path. `graph_retriever.py` (2-hop import-graph traversal) from Figure 1's earlier version has been removed here — that branch (`research/graph-traversal`) was never merged and the file doesn't exist on `main`; see the branch table below Figure 3 for its (also negative) result.

## Figure 1b — Localizers, evaluation & entry points

```mermaid
graph TB
    subgraph method_l["method/ (localizers)"]
        base_m[base.py]
        openrouter["openrouter_localizer.py\n(+ optional reasoning step)"]
        openai_l[openai_localizer.py]
        openai_free[openai_free_localizer.py]
        opensource["opensource_localizer.py\n(disabled)"]
        rag["rag_localizer.py\n(dead code)"]
    end

    subgraph evaluation["evaluation/ (depends on dataset + method.bm25_retriever)"]
        manifest[manifest.py]
        screening[screening.py]
        failure_attribution[failure_attribution.py]
    end

    subgraph entry["Entry points (depend on everything above)"]
        main_py["main.py\n(dataset -> optional BM25/hybrid retrieval\n-> localizer -> evaluator, see Figure 2)"]
        scripts["scripts/*.py"]
    end

    openrouter --> base_m
    openrouter --> prompt_ref[prompt.py]
    openrouter --> rag
    openai_l --> base_m
    openai_l --> prompt_ref
    openai_l --> llm_ref[llm.py]
    openai_free --> base_m
    openai_free --> prompt_ref
    openai_free --> llm_ref

    screening --> bm25_ref[bm25_retriever.py]
    screening --> localizability_ref[localizability.py]
    failure_attribution --> localizability_ref
    failure_attribution -.->|prepare_oracle_candidate_set only| bm25_ref
    manifest --> utils_ref[dataset/utils.py]

    main_py --> openrouter
    main_py --> openai_l
    main_py --> openai_free
    main_py --> bm25_ref
    main_py --> hybrid_ref["hybrid_retriever.py\n(--retrieval-top-k hybrid-rrf)"]
    main_py --> embedding_ref["embedding_retriever.py\n(--retrieval-top-k embedding)"]
    main_py --> evaluate_ref[evaluate.py]

    scripts --> dataset_ref[dataset/]
    scripts --> evaluation
    scripts -.->|run_failure_attribution --run-oracle only| openrouter

    style opensource fill:#666,stroke:#333,color:#ccc
    style rag fill:#666,stroke:#333,color:#ccc
```

**Layering is still strict and one-directional**: `dataset` never imports `method` or `evaluation`; `method` never imports `evaluation`. `evaluation` is the only package that reaches into `method` (just `bm25_retriever`, to screen candidate rankings). `main.py` and `evaluation` remain siblings — `main.py` never imports `evaluation`.

## Figure 2 — Runtime data flow: `main.py` end-to-end path

```mermaid
flowchart LR
    A["Bench4BL / SWEBench / BeetleBox\n.get_bug_instances()"] --> B["BugInstance\n(bug_report, repo, base_commit,\ncode_files, ground_truths)"]
    B --> C{"--retrieval-top-k set?"}
    C -->|yes| D{"--retrieval-mode"}
    D -->|hybrid-rrf, default| D1["hybrid_retriever.rank_files_hybrid\nBM25 pre-filter -> chunked embedding\n-> weighted RRF (default 1:5)"]
    D -->|embedding| D2["embedding_retriever.rank_files_embedding_chunked\nBM25 pre-filter -> embedding rank only"]
    C -->|no| E2{"--bm25-top-k set?"}
    E2 -->|yes| D3["bm25_retriever\npath_only / skeleton / symbols(+imports)"]
    E2 -->|no| E
    D1 --> E["Localizer.localize(bug)\nOpenRouter / OpenAI / OpenAI-free"]
    D2 --> E
    D3 --> E
    E --> F["OpenAILocalizerResponse\n(candidate_files)"]
    F --> G["Evaluator.evaluate()\naccuracy / precision / recall / F1"]
    G --> H["--output -> JSON report\n(config + per-bug + overall)"]

    style D1 fill:#2f6f4f,stroke:#1a3f2c,color:#fff
```

Candidate file content, for any BM25/embedding step, comes from `dataset/repo_cache.py`'s offline cache (`get_file_contents_batch`) — the bare-clone mirror for SWE-bench/BeetleBox, or Bench4BL's own working-tree `gitrepo/` as a fallback — never a live network call. `--retrieval-top-k` and `--bm25-top-k` are mutually exclusive; `--retrieval-top-k` takes precedence if both are passed. See Figure 3 for what happens inside the `hybrid_retriever`/`embedding_retriever` boxes.

## Figure 3 — Retrieval subsystem detail: BM25 + embedding + weighted RRF

```mermaid
flowchart LR
    A[BugInstance] --> B["bm25_retriever\nsymbols_no_imports\n(narrows full corpus to a\ncandidate pool, e.g. top-200)"]
    B --> C["embedding_retriever\nrank_files_embedding_chunked\n(Java-aware or AST-chunked,\nmax cosine sim per file)"]
    C --> D["hybrid_retriever\nweighted RRF (default 1:5\nBM25:embedding)"]
    D --> G[Final file ranking]

    style D fill:#2f6f4f,stroke:#1a3f2c,color:#fff
```

**Confirmed at n=30 with Qwen3-Embedding-0.6B on two benchmarks** — weighted RRF beats embedding-alone on both, peaking at the same 1:5 ratio:

| Benchmark | BM25 alone | Embedding alone | RRF 1:5 (best) |
|---|---:|---:|---:|
| Bench4BL (primary) | MRR 0.143 | MRR 0.688 | **MRR 0.714** |
| SWE-bench Verified | MRR 0.086 | MRR 0.316 | **MRR 0.422** |

Bench4BL's 0.714 MRR is the strongest confirmed result in the project. (An earlier n=6 Bench4BL sanity check had shown unweighted RRF 1:1 winning instead — that ordering didn't survive n=30, the same small-sample pattern already seen once with SWE-bench's own n=6→n=30 transition. See `docs/bench4bl_result.md` / `docs/qwen3_rrf_result.md` for full detail, including the void pre-fix numbers this replaced.) BeetleBox has a confirmed BM25 representation comparison (`symbols_with_imports` best, MRR 0.500 at n=15) but no hybrid RRF run yet.

A separate FAISS-backed persistent index (`method/repository_index.py`, merged) implements the same chunked-embedding step with on-disk caching/dedup instead of recomputing embeddings per run, verified end-to-end on a real 144-file repo (81s build, 0.16s/search after). Real per-instance Java-aware chunking on Bench4BL is slow enough (~900s/instance) that n=30-scale runs go through Slurm array-job sharding (`scripts/run_hybrid_rrf_weighting_shard.py` + `scripts/aggregate_rrf_shards.py`) rather than one long serial job — see Figure 4.

### Branches tried against this pipeline, and their verdict

| Branch | Idea | Result |
|---|---|---|
| `research/hybrid-fusion-signals` (merged) | Add AST-similarity, 1-hop dependency-graph, commit-recency as extra RRF signals | **Negative** — each signal individually weak (0.02–0.06 MRR); naive fusion drags the strong embedding signal down |
| `research/graph-traversal` (not merged, kept separate) | Real 2-hop BFS over the import graph, seeded from BM25 | **Negative** — MRR roughly flat (0.163→0.158) but Hit@100 regresses 0.867→0.733; import connectivity is a weak relevance proxy in this codebase style |
| `research/reasoning-rerank` (not merged, kept separate) | LLM writes bug-specific reasoning per candidate before ranking (RGFL-style) | **Negative** — worse than direct-pick baseline both tries (36.7%→30.0%→23.3%); final ranking wasn't reliably grounded in the reasoning text in this single-call design |
| `experiment/embedding-ceiling` (not merged, kept separate) | Whole-file embedding vs. BM25, no chunking | **Negative** — a clean, documented dead end |

## Figure 4 — Offline evaluation / diagnostics path (`scripts/`)

```mermaid
flowchart LR
    A["generate_evaluation_manifest.py\n(--dataset bench4bl/swebench/beetlebox)"] --> M["manifest.json\n(seeded, max-N-per-repo,\nstable content-hash ID)"]
    M --> B["run_bm25_screening.py\ncompare_bm25_representations.py"]
    M --> EMB["compare_embedding_models.py\nrun_hybrid_rrf_weighting_test.py\n(serial, small n)"]
    M --> SHARD["run_hybrid_rrf_weighting_shard.py\n(Slurm array job, one shard per task)"]
    SHARD --> AGG["aggregate_rrf_shards.py\n(merges shard JSONs -> one report)"]
    B --> S["screening.py\nHit@k / Recall@k / MRR / MAP\ndifficulty band per instance"]
    EMB --> S
    AGG --> S
    S --> R1["results/*.json"]
    M --> F["run_failure_attribution.py"]
    F --> FA["failure_attribution.py\nretrieval-failure vs\nreached-candidate-set split (free)"]
    FA -.->|"--run-oracle (costs LLM calls)"| O["oracle diagnostic\nforce-inject GT, measure\nOpenRouterLocalizer placement"]
    FA --> R1
    O --> R1
```

`run_hybrid_rrf_weighting_shard.py`/`aggregate_rrf_shards.py` exist specifically because real per-instance Java-aware chunking on Bench4BL is too slow for one long serial MN5 job (~900s/instance would mean hours for n=30, with no output until the very end if it's killed partway) — the array-job path splits the manifest into shards run in parallel, each writing its own JSON atomically. `scripts/mirror_bench4bl.py` (not shown above, upstream of the manifest step) mirrors Bench4BL's per-project SourceForge archives into `bench4bl_cache/`.
