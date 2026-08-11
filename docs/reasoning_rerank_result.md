# Reasoning-augmented reranking result

Tests whether having the LLM reranker write bug-specific reasoning per candidate file
*before* producing its final ranking (RGFL-style: "for each candidate, why does/doesn't
this file relate to the bug, then rank based on that reasoning") beats the existing
direct-pick rerank. **Negative result** — worse than the baseline in both variants tried,
with a diagnosed, reproducible cause rather than noise.

## What was built

- `method/models.py`: `FileReasoning` (file + reasoning text) and
  `ReasoningLocalizerResponse` (list of those + final `candidate_files`).
- `method/prompt.py`: `generate_reasoning_prompt()` — asks the LLM to reason about every
  candidate file, one sentence each, then produce a final ranked list from that reasoning.
  Single API call (reasoning and ranking are one continuous generation), same call budget
  as the direct-pick baseline.
- `method/openrouter_localizer.py`: `localize_with_reasoning()` wiring it together.
- `main.py`: `--reasoning-rerank` flag.
- Fixed a real bug along the way in `method/utils.py`'s `generate_json_schema()`: it only
  set `additionalProperties: false` on the top-level schema, not on nested `$defs` (e.g.
  `FileReasoning`) — OpenAI-style strict structured output requires it at every level, so
  any nested-model response schema would have silently failed the moment one was used.

## Result

30 instances (SWE-bench Verified, seed 42, same set for every run below), `gpt-4o-mini`,
`--bm25-top-k 15 --bm25-symbols` for both arms (a narrower shortlist than the published
100-file pool, to keep reasoning-per-file cheap and within the response token budget —
**these numbers are not directly comparable to the project's published 51.7%/50.0%
end-to-end figures**, which use `--bm25-top-k 100`; they're only meaningful relative to
each other, since both arms share the identical BM25 stage).

| Config | Accuracy |
|---|---:|
| **baseline (direct rerank)** | **36.7% (11/30)** |
| reasoning-rerank v1 | 30.0% (9/30) |
| reasoning-rerank v2 (test-file-bias prompt fix applied) | 23.3% (7/30) |

Zero instances were *gained* by reasoning-rerank in either variant — every flip across all
three runs was a regression relative to baseline.

## Diagnosed cause: the final ranking isn't reliably grounded in the reasoning text

**v1** had one clear, consistent bias: when both a source file and its own test file were
candidates, the model routinely ranked the test file above the source file in the final
list — even when the source file's own reasoning correctly named it as central to the bug
(`astropy/io/fits/diff.py`: "contains the implementation for FITSDiff... directly manages
the comparison logic" — ranked below `test_diff.py` anyway).

**v2** added an explicit instruction to prefer implementation files over their test files
unless there's specific evidence otherwise. This fixed exactly the case it targeted
(`astropy__astropy-14539` flipped from miss to hit) but broke three *new* instances the
same way, just with a different kind of file winning instead of a test file (e.g.
`django/contrib/auth/tokens.py`: "generating tokens for password resets... where the token
generation logic can be found" — GT, correct reasoning — ranked below `base_user.py`, a
sibling file with no test-file involved).

**Reading**: this isn't one fixable bias, it's a structural gap. Reasoning and the final
ranked list are produced in one continuous generation, so nothing forces the ranking step
to actually be *computed from* the reasoning rather than drifting alongside it. Patching
the symptom that showed up first (test-file preference) just exposed the next one
(sibling-file preference) in a different subset of instances — the underlying problem,
weak grounding between reasoning and ranking, was untouched.

## What would be worth trying if this direction is resumed

Split into two separate API calls instead of one: generate reasoning for every candidate
first, then feed that reasoning text back as *input* to a distinct ranking call. That
forces the ranking step to genuinely depend on the reasoning (it's the only thing in that
call's context) rather than letting one autoregressive generation wander between the two.
This is a real architectural change (2x calls, new code path), not attempted here — closed
out as a documented negative result instead, per the project's own scoping decision.

## Reproducing

```bash
# baseline
python main.py --method openrouter --dataset swebench --model gpt-4o-mini --sample-size 30 \
  --bm25-top-k 15 --bm25-symbols --output results/reasoning_rerank_baseline_30.json

# reasoning-rerank (same instances, same seed)
python main.py --method openrouter --dataset swebench --model gpt-4o-mini --sample-size 30 \
  --bm25-top-k 15 --bm25-symbols --reasoning-rerank --output results/reasoning_rerank_treatment_30_v2.json
```
