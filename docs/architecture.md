# Architecture

Two views: package dependencies (what imports what) and runtime data flow (how a bug instance actually moves through the system). Complements the file-by-file breakdown in [project_structure.md](project_structure.md).

## Package dependencies

```mermaid
graph TB
    subgraph dataset["dataset/ (self-contained)"]
        base[base.py]
        swebench[swebench.py]
        beetlebox[beetlebox.py]
        models[models.py]
        repo_cache[repo_cache.py]
        utils_d[utils.py]
        localizability[localizability.py]
    end

    subgraph method["method/ (depends on dataset only)"]
        base_m[base.py]
        openrouter[openrouter_localizer.py]
        openai_l[openai_localizer.py]
        openai_free[openai_free_localizer.py]
        opensource["opensource_localizer.py (disabled)"]
        rag["rag_localizer.py (dead code)"]
        bm25["bm25_retriever.py"]
        embedding["embedding_retriever.py (experiment branch)"]
        prompt[prompt.py]
        evaluate[evaluate.py]
        llm[llm.py]
        utils_m[utils.py]

    end

    subgraph evaluation["evaluation/ (depends on dataset + method.bm25_retriever)"]
        manifest[manifest.py]
        screening[screening.py]
        failure_attribution[failure_attribution.py]
    end

    subgraph entry["Entry points (depend on everything below them)"]
        main_py[main.py]
        scripts["scripts/*.py"]
    end

    swebench --> base
    beetlebox --> base
    swebench --> models
    beetlebox --> models
    swebench --> utils_d
    beetlebox --> utils_d
    utils_d --> repo_cache
    localizability --> utils_d

    openrouter --> base_m
    openrouter --> prompt
    openrouter --> rag
    openrouter --> utils_m
    openai_l --> base_m
    openai_l --> prompt
    openai_l --> llm
    openai_free --> base_m
    openai_free --> prompt
    openai_free --> llm
    bm25 --> repo_cache
    embedding --> bm25
    embedding --> repo_cache
    utils_m --> utils_d
    method --> utils_d

    screening --> bm25
    screening --> localizability
    failure_attribution --> localizability
    failure_attribution -.->|prepare_oracle_candidate_set only| bm25
    manifest --> utils_d

    main_py --> swebench
    main_py --> beetlebox
    main_py --> openrouter
    main_py --> openai_l
    main_py --> openai_free
    main_py --> bm25
    main_py --> evaluate

    scripts --> dataset
    scripts --> evaluation
    scripts -.->|run_failure_attribution --run-oracle only| openrouter

    style opensource fill:#666,stroke:#333,color:#ccc
    style rag fill:#666,stroke:#333,color:#ccc
    style embedding fill:#457,stroke:#333,color:#fff
```

Grey = disabled/dead code. Blue = lives only on the `experiment/embedding-ceiling` branch, not `main`.

**Layering is strict and one-directional**: `dataset` never imports `method` or `evaluation`; `method` never imports `evaluation`. `evaluation` is the only package that reaches into `method` (just `bm25_retriever`, to screen candidate rankings) — it never touches the LLM localizers directly except via the opt-in oracle diagnostic in `scripts/run_failure_attribution.py`, which imports `OpenRouterLocalizer` only when `--run-oracle` is passed. `main.py` and `evaluation` are siblings, not a hierarchy: `main.py` never imports `evaluation`, so the end-to-end pipeline and the offline diagnostics tooling can be developed and tested independently.

## Runtime data flow

### End-to-end path (`main.py`)

```mermaid
flowchart LR
    A["SWEBench / BeetleBox\n.get_bug_instances()"] --> B["BugInstance\n(bug_report, repo, base_commit,\ncode_files, ground_truths)"]
    B --> C{"--bm25-top-k set?"}
    C -->|no| E
    C -->|yes| D["bm25_retriever\npath_only / skeleton / symbols(+imports)"]
    D --> E["Localizer.localize(bug)\nOpenRouter / OpenAI / OpenAI-free"]
    E --> F["OpenAILocalizerResponse\n(candidate_files)"]
    F --> G["Evaluator.evaluate()\naccuracy / precision / recall / F1"]
    G --> H["--output → JSON report\n(config + per-bug + overall)"]
```

Candidate file content, when a BM25 content-based variant is used, comes from `dataset/repo_cache.py`'s offline bare-clone cache (`get_file_contents_batch`) if the repo is locally mirrored, falling back to path-only tokens rather than a live network call.

### Offline evaluation / diagnostics path (`scripts/`)

```mermaid
flowchart LR
    A["generate_evaluation_manifest.py"] --> M["manifest.json\n(seeded, max-N-per-repo,\nstable content-hash ID)"]
    M --> B["run_bm25_screening.py\ncompare_bm25_representations.py"]
    B --> S["screening.py\nHit@k / Recall@k / MRR / MAP\ndifficulty band per instance"]
    S --> R1["results/*.json"]
    M --> F["run_failure_attribution.py"]
    F --> FA["failure_attribution.py\nretrieval-failure vs\nreached-candidate-set split (free)"]
    FA -.->|"--run-oracle (costs LLM calls)"| O["oracle diagnostic\nforce-inject GT, measure\nOpenRouterLocalizer placement"]
    FA --> R1
    O --> R1
```

`screening.py` and `failure_attribution.py` both depend on `dataset/localizability.py` to exclude ground-truth files that were introduced by the fix itself (`added_by_fix`) from being counted as retrieval misses — this classification is shared infrastructure across every diagnostic script, not duplicated per-script logic.
