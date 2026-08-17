# Failure case analysis

Splits every miss in the real n=30 end-to-end runs (`results/end_to_end_swebench_30_skeleton.json`,
`results/end_to_end_swebench_30_symbols.json`) into two causes that call for opposite fixes,
using the existing `evaluation/failure_attribution.py` machinery — **free and offline**, no
new runs, no LLM calls:

- **Retrieval failure** — the ground-truth file never made it into the top-100 BM25
  candidates the LLM reranker was shown. No amount of prompt/model improvement could have
  fixed this; only better retrieval can.
- **Reranking failure** — the ground-truth file *was* in the candidate set, but `gpt-4o-mini`
  didn't pick it. Retrieval did its job; this calls for reranker changes, not retrieval changes.

## Headline result

| Config | Hits | Retrieval failures | Reranking failures |
|---|---:|---:|---:|
| symbols_no_imports (authoritative — this run's own recorded candidate list) | 15/30 | 6 | 9 |
| skeleton (best-effort — see caveat below) | 15/30 | 6 | 9 |

**Both configs land on the identical 6/9 split.** This directly answers the open question
from `results/README.md` §6: the two representations tied exactly at n=30 (50.0% each)
despite `symbols_no_imports` having a clearly better *retrieval-only* ceiling (86.7% vs.
80.0% Hit@100, §4 of the same doc). The failure breakdown shows why the ceiling gap doesn't
show up in the end-to-end number: **retrieval isn't this pipeline's current bottleneck at
n=30 — reranking is.** Of the 15 total misses per config, 9 (60%) are cases where the
correct file was right there in front of the LLM and it still didn't pick it.

## Consistently hard instances (retrieval failure under both representations)

```
astropy__astropy-7166
django__django-13401
pylint-dev__pylint-7080
scikit-learn__scikit-learn-25973
```

These four fail to reach the top-100 candidate set regardless of which BM25 representation
is used — the bottleneck for these specific bugs is BM25 itself (or the 100-file cutoff),
not the choice between path/skeleton/symbol tokens. Worth a closer look if retrieval work
resumes: is it a vocabulary mismatch BM25 fundamentally can't bridge (a case for embeddings,
which is exactly what the hybrid-retrieval work already targets), or genuinely deep in a
large repo regardless of method?

Two instances flip depending on representation — `django__django-12125` and
`pytest-dev__pytest-10356` are retrieval failures under skeleton but reachable under symbols
(consistent with symbols' better ceiling); `django__django-11292` and `django__django-13158`
go the other way. Small numbers, but the direction matches the aggregate ceiling difference.

## Caveat — a real, small data-consistency issue found and worked around

`results/bm25_comparison_swebench_30.json`'s `symbols_no_imports` screening report and
`end_to_end_swebench_30_symbols.json`'s actual recorded `candidate_files` **disagree on 1 of
30 instances** (`django__django-12125`: the separately-computed BM25 screening run classifies
it as unreachable, but the real paid end-to-end run's own candidate list shows the ground
truth was in fact present). Likely cause: BM25 tie-breaking order isn't guaranteed stable
across separate Python process invocations when many candidates score identically. The
symbols numbers above use the end-to-end run's own recorded `candidate_files` (the
authoritative record of what the LLM actually saw), not the separately-computed screening
report, specifically to avoid this. **The skeleton numbers can't be corrected the same way**
— that run predates the `--output` flag and only has accuracy/precision/recall/F1 per bug,
not the raw candidate list — so its retrieval/reranking split is a best-effort reconstruction
from the separate BM25 screening run, with the same ~3% (1/30) mismatch risk just
demonstrated on the symbols side, unverified.

## What this suggests, not yet acted on

The oracle diagnostic (`evaluation/failure_attribution.py`'s `run_oracle_diagnostic`,
force-injects every ground truth into the candidate set to isolate pure reranking ability)
would directly test the "reranking is the bottleneck" reading above — but it calls the LLM
and costs real API usage, so it hasn't been run this session. Given 9/15 misses per config
are reranking failures, this is now a better-motivated next spend than further retrieval
representation tuning, which is the reverse of where n=30 effort has gone so far (BM25
representations, hybrid embedding fusion) — worth a real conversation before committing to it.

## Reproducing

```python
import json
from evaluation.failure_attribution import classify_retrieval_reach, RETRIEVAL_FAILURE

bm25 = json.load(open("results/bm25_comparison_swebench_30.json"))
e2e = json.load(open("results/end_to_end_swebench_30_symbols.json"))

for instance_id, bug in e2e["per_bug"].items():
    if not bug["ground_truths"]:
        continue
    reached = any(gt in bug["candidate_files"] for gt in bug["ground_truths"])
    cause = "hit" if bug["accuracy"] == 1 else ("reranking_failure" if reached else "retrieval_failure")
    print(instance_id, cause)
```
