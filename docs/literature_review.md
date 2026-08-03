# Literature review: LLM-based bug/fault localization (2024-2026)

Scope: papers on using LLMs (with or without retrieval) to predict which source file(s)/function(s) need to change to fix a bug, given a bug report/issue. Grounds the four things asked for in the project brief: compare algorithms/datasets/metrics, unify benchmark handling, design offline code representation + vector indexing, and evaluate hybrid retrieval/reranking/AST/semantic-similarity methods for Top-1 accuracy.

**Baselines named in the original handover doc:** BLAZE, BugCerberus, FlexFL, MarsCode. Status: BLAZE, BugCerberus, and FlexFL have been located below. **MarsCode has not yet been found** in either research pass — needs a targeted follow-up search.

## Priority tiers

### Tier 1 — core (must include)

| Paper | Why it matters |
|---|---|
| BugCerberus — *Bridging Bug Localization and Issue Fixing: A Hierarchical Localization Framework Leveraging LLMs* | Directly studies hierarchical retrieval (file → function → statement) on SWE-bench Lite; measures Top-k localization and repair improvement. Named in the original handover doc. |
| BLAZE — *Cross-Language and Cross-Project Bug Localization* | Introduces dynamic chunking, the BeetleBox benchmark, evaluates on SWE-bench with Top-1/MAP/MRR. Directly relevant to chunking design. Named in the original handover doc. |
| Kept — *A Knowledge Enhanced Large Language Model for Bug Localization* | Uses knowledge graphs to enhance LLM retrieval; compares against GraphCodeBERT, CodeT5, OpenAI embeddings. |
| Toggle — *A Deep Dive into Large Language Models for Automated Bug Localization and Repair* | Strong baseline paper; token-level vs. line-level localization and prompt engineering. |

### Tier 2 — supporting

| Paper | Contribution |
|---|---|
| LinuxFLBench — *Benchmarking and Enhancing LLM Agents in Localizing Linux Kernel Bugs* | Evaluates Agentless, SWE-Agent, AutoCodeRover on a much larger codebase; proposes LinuxFL+. Relevant for scalability discussion. |
| AutoFL — *A Quantitative and Qualitative Evaluation of LLM-Based Explainable Fault Localization* | Repository navigation via function calls instead of vector retrieval; adds explainability. |

### Tier 3 — likely out of scope

| Paper | Reason |
|---|---|
| SPICED | Analog circuit Trojan detection via LLM prompting — hardware security, not software bug localization. Excluded from the retrieval-architecture comparison. |

### Additional high-priority papers (starred)

| Paper | Method Category | Why it matters | Priority |
|---|---|---|---|
| FaR-Loc — *Enhancing LLM-based Fault Localization with a Functionality-Aware Retrieval-Augmented Generation Framework* | Dense retrieval + LLM reranking (RAG) | Closest paper to this project's direction: complete retrieve → rerank pipeline with functionality extraction + semantic retrieval + LLM reranking. Reports SOTA on Defects4J. | ⭐⭐⭐⭐⭐ |
| FlexFL — *Flexible and Effective Fault Localization with Open-Source LLMs* | Hybrid retrieval + LLM agent | Two-stage: classical FL narrows search space, then LLM refines. Discusses contamination via unseen-bug evaluation. Named in the original handover doc. | ⭐⭐⭐⭐⭐ |
| Hierarchical Knowledge Injection for Improving LLM-based Program Repair | Hierarchical context retrieval / knowledge injection | Shows progressively adding bug-, repo-, and project-level context improves repair rate. | ⭐⭐⭐⭐☆ |
| MemFL — *Improving LLM-Based Fault Localization with External Memory and Project Context* | External memory retrieval | Static + dynamic project memory instead of standard vector retrieval; improves localization while cutting runtime/cost. | ⭐⭐⭐⭐⭐ |
| DEVLoRe — *Integrating Various Software Artifacts for Better LLM-based Bug Localization and Program Repair* | Multi-source retrieval | Combining issue descriptions + stack traces + debugging info beats any single artifact. | ⭐⭐⭐⭐⭐ |
| IQLoc — *Improving IR-based Bug Localization with Semantics-Driven Query Reduction* | Hybrid IR + LLM reranking | Modernizes BM25 with transformer-based semantic reranking + query reformulation. | ⭐⭐⭐⭐☆ |

## Master comparison table

Columns filled in only where an actual per-paper review has extracted the data. `TBD` means the paper is in scope but hasn't had a full review pass yet (use the template below).

| Paper | Method Category | Code Granularity | Embedding Model / Vector DB | Benchmark(s) | Metric(s) | Best Top-1 / Pass@1 Result | Notes |
|---|---|---|---|---|---|---|---|
| ReCode (Zhao et al., arXiv 2025) | Dense retrieval + LLM (RAG) | Function/snippet level | OASIS-code-1.3B (code encoder), BGE-M3 (text encoder); vector DB not specified | RACodeBench (new), competitive programming datasets | Test Pass Rate, Strict Accuracy | Not directly reported as Top-1 | Fine-grained algorithm-aware retrieval; introduces RACodeBench |
| RAGFix (Mansur et al., IEEE BigData 2024) | RAG | Function level | Chroma; Stack Overflow embeddings via Llama 3 | HumanEvalFix | Repair accuracy | Not reported | Python-only; needs docstrings; small benchmark |
| SWE-Fixer (Xie et al., Findings ACL 2025) | Hybrid retrieval (BM25 + LLM retriever + LLM repair) | File level | BM25 + fine-tuned Qwen2.5 retriever; no vector DB | SWE-bench Lite, SWE-bench Verified | Pass@1, Best@1 (P2P filtering) | 30.2% Pass@1 on Verified (32.8% Best@1) | Coarse-to-fine: BM25 → neural retrieval → repair |
| ExpeRepair (Mu et al., IJCAI 2025 / arXiv 2025) | Retrieval-based memory system | Repository level | Embedding similarity or BM25 memory; vector DB unspecified | SWE-bench Lite, SWE-bench Verified | Pass@1 | **74.6% on Verified** (Claude 3.7 Sonnet) — highest surveyed | Episodic + semantic memory of previous repairs |
| KGCompass (Antoniades et al., arXiv 2025) | Knowledge graph / structural retrieval | Function + file level | No vector DB; knowledge graph traversal | SWE-bench Lite | Pass@1, localization accuracy | 58.3% on Lite | Repo knowledge graph instead of dense retrieval |
| CoRNStack (Zhu et al., arXiv 2024) | Retrieve-then-rerank | Function level | E5, Arctic Embed, CodeSage-style embeddings; vector DB not specified | GitHub issue localization datasets | Recall, MRR, localization accuracy | Not evaluated on SWE-bench | Trains both retriever and reranker |
| Bug Fixing with Broader Context (Ehsani et al., arXiv 2025) | Layered context injection | Repository + project level | None | BugsInPy (314 bugs) | Fix Rate | 79% (250/314) | Repo/project-level context boosts repair |
| Fact Selection Problem in LLM-Based Program Repair (Parasaram et al., ICSE 2025) | Pure LLM prompting | Function level | None | BugsInPy | Repair Rate | ~56% (88/157) | Prompt fact selection, no retrieval; introduces Maniple |
| BugCerberus | Hierarchical retrieval (file → function → statement) | Hierarchical | TBD | SWE-bench Lite | Top-k localization, repair improvement | TBD | Needs full paper review |
| BLAZE | Dynamic chunking, cross-language/cross-project | Dynamic (semantic boundaries) | TBD | SWE-bench, BeetleBox (introduces) | Top-1, MAP, MRR | TBD | Needs full paper review |
| Kept | Knowledge graph enhanced retrieval | TBD | Compares GraphCodeBERT, CodeT5, OpenAI embeddings | TBD | TBD | TBD | Needs full paper review |
| Toggle | Pure LLM prompting w/ prompt engineering | Token-level vs. line-level | None (no retrieval) | TBD | TBD | TBD | Needs full paper review |
| LinuxFLBench | Agent-based (Agentless, SWE-Agent, AutoCodeRover evaluated) + proposes LinuxFL+ | TBD | TBD | Linux kernel (own benchmark) | TBD | TBD | Needs full paper review |
| AutoFL | Repository navigation via function calls (no vector retrieval) | Function level (navigated, not embedded) | None | TBD | TBD | TBD | Adds explainability |
| FaR-Loc | Dense retrieval + LLM reranking (RAG) | TBD | TBD | Defects4J | TBD | SOTA on Defects4J (exact number TBD) | Needs full paper review |
| FlexFL | Hybrid retrieval + LLM agent | TBD | Classical FL techniques + LLM refinement | TBD (tests on unseen bugs re: contamination) | TBD | TBD | Named in original handover doc |
| Hierarchical Knowledge Injection | Hierarchical context retrieval / knowledge injection | Bug / repo / project levels | TBD | TBD | Repair rate | TBD | Needs full paper review |
| MemFL | External memory retrieval | TBD | Static + dynamic project memory (not standard vector retrieval) | TBD | TBD | Improves localization, cuts runtime/cost | Needs full paper review |
| DEVLoRe | Multi-source retrieval | TBD | TBD | TBD | TBD | TBD | Combines issue text + stack trace + debug info |
| IQLoc | Hybrid IR + LLM reranking | TBD | BM25 + transformer reranker | TBD | TBD | TBD | Modernized BM25 pipeline |

## Key trends across both research passes

1. **No retrieval is the losing strategy.** Every paper surveyed that uses any retrieval step (dense, BM25, hybrid, knowledge-graph, or memory-based) outperforms pure-prompting baselines. This project's current pipeline (dump the full file-path list into one prompt) falls in the "pure prompting" category — the weakest one in the literature.
2. **Hierarchical retrieval (file → function → statement) is the emerging dominant pattern** (BugCerberus, KGCompass, Kept), not flat file-level or symbol-level alone.
3. **Dynamic/semantic chunking is replacing fixed-size chunking** — BLAZE segments at class/method/interface boundaries rather than arbitrary token windows.
4. **Hybrid retrieval (BM25 + dense + rerank) remains highly competitive** and is cheaper to build than full knowledge-graph or memory systems (SWE-Fixer, IQLoc, FlexFL).
5. **Project/repository-level context and memory of past repairs are the biggest recent lever** for the very top results (ExpeRepair 74.6% Pass@1, MemFL, Hierarchical Knowledge Injection) — but these are also the most complex to build.
6. **Localization and repair are increasingly separated as explicit stages** (Toggle) — validates keeping this project scoped to localization only, rather than also attempting patch generation.
7. **Agent-based repository navigation (AutoFL, LinuxFLBench's evaluated agents) is a no-vector-DB alternative** worth knowing about, though the team's own prior "multi-agent workflow" experiments were reportedly abandoned as not clearly helping — a discrepancy worth resolving by reading AutoFL/LinuxFLBench specifically for *why* their agent approach differs from what was tried before.
8. **Metric mismatch risk:** several top numbers (ExpeRepair, SWE-Fixer, Ehsani et al.) report Pass@1 or Fix Rate — which require a working generated *patch*, not just correct file identification. This project's `evaluate.py` currently measures pure localization Top-1 (hit@k on file paths only). Don't directly compare this project's accuracy number against Pass@1 figures without noting the metric is different.

## Per-paper review template

Copy this block for each `TBD` paper above once it's been read (via ChatGPT deep research or directly), then fold the results back into the master table.

```markdown
### <Paper Title>

**1. Bibliographic info**
- Authors:
- Venue / Year:
- Link (DOI/arXiv/OpenReview):

**2. Method category** (a) pure LLM prompting  (b) dense/vector retrieval  (c) hybrid BM25+dense  (d) AST/structural/knowledge graph  (e) retrieve→rerank  (f) other
- Category:
- Why:

**3. Code representation granularity**
(repository / whole-file / class / method-function / symbol / statement / dynamic chunking / hierarchical)

**4. Embedding model / vector database**
- Embedding model:
- Vector DB:
(state "Not specified" if the paper doesn't say)

**5. Benchmark dataset(s)**

**6. Evaluation metric(s)** (include exact k for Top-k)

**7. Best reported result** (metric, value, dataset, split, model, setting)

**8. Limitations / failure modes** (hallucination, retrieval failure, context window limits, scalability, contamination/SWE-bench leakage, doc dependence, missing project history, compute cost, cross-project generalization)
```
