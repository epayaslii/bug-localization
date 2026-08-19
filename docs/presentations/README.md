# Presentations

Live Artifact decks (redeployed in place as results land, not committed as static
HTML/pptx — see the note at the bottom). Same convention as the co-intern's
`docs/presentations/` on `github.com/adisenaa/Bug-Report-Localization`.

- [Bench4BL Progress & IQLoc Scoping](https://claude.ai/code/artifact/cfe81973-3797-44d8-a87b-da5f545e8b6c) —
  current, updated 2026-08-19. Hybrid-retrieval results on the diverse 46-project mirror
  (BM25 4-representation comparison, hybrid-RRF weight sweeps, both at n=30 and full
  population), the local-LLM (Ollama) query-reformulation pipeline and its confirmed
  negative result, a stage-by-stage comparison against IQLoc's actual published pipeline,
  the `iqloc-publication-replication` branch's progress (including a real embedding-quality
  bug found and fixed in the shared Java chunker), and next steps.
- [Bug Localization — Progress Review](https://claude.ai/code/artifact/4b93c69d-c238-4025-a442-f2aec44f9749) —
  the longer-running general progress-review deck, updated as major results land (last:
  2026-08-17, Bench4BL n=30 confirmed, MN5 GPU fix, hybrid-RRF end-to-end 76.7%).
- [Bench4BL deep-dive](https://claude.ai/code/artifact/1121036e-b173-482a-a4af-b9c7f5f4bf07) —
  2026-08-13 session, superseded by the confirmed n=30 numbers in the decks above.

No static HTML/pptx copies are committed here by design — a prior session removed
`docs/presentation.html` at explicit user request in favor of links-only, since these decks
are redeployed to the same URL rather than versioned as files.
