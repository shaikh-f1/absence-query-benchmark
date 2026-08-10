# Local-model extraction findings (Ollama)

Evaluation of **local, zero-cost, fully-private** LLM extraction on real contract
prose, using the `ollama` provider added to this rig. The question this answers:
*is a local model good enough to be the extractor behind an absence/completeness
query system, and does a bigger local model help?*

All numbers are on **CUAD** (real commercial contracts), 30-document slice, 41
clause types, 1,230 clause slots, **72.2% absent**. Absence-F1 is gameable on this
imbalance (an always-"absent" predictor scores 0.839), so the metrics that matter
are **MCC** and **presence-recall** — the two the degenerate baseline cannot
inflate.

## TL;DR

- **The architecture is exact.** Under the oracle detector the derivation store
  scores **MCC 1.00** — every error below is *extraction* error, not a flaw in the
  rollup/provenance/merge design.
- **A local 7B is the sweet spot.** Qwen2.5-7B: **MCC 0.61, presence-recall 0.60,
  0 extraction failures, $0, ~40 s/doc** on a consumer laptop GPU (RTX 3060).
- **A bigger local model buys nothing.** Qwen2.5-14B is statistically
  indistinguishable on accuracy (MCC 0.608, and *worse* on presence-recall) while
  running **~6× slower**. On this hardware/task, 14B is strictly dominated by 7B.
- **The gap is clause-specific, not capacity-bound.** Both models are strong on
  well-defined clauses and *worse than chance* on subtle liability/IP/rights
  clauses. You cannot close that gap by scaling the local model — it needs **tier
  escalation** to a materially stronger (hosted) model for the hard clause types.

## Setup

| | |
|---|---|
| Hardware | Intel i5-10500H, 31 GB RAM, **NVIDIA RTX 3060 Mobile (6 GB VRAM)** |
| Runtime | Ollama 0.32.6, native `/api/chat` endpoint (only path that honors `num_ctx`) |
| Context | `num_ctx=16384` — exceeds the ~13k-token contracts so results are not confounded by truncation |
| Decoding | temperature 0, **schema-constrained** (`format` = JSON Schema forcing exact clause keys as booleans) |

## Results

| Metric | Oracle | always-absent | **Qwen2.5-7B** | Qwen2.5-14B |
|---|---|---|---|---|
| **MCC** | 1.00 | 0.00 | **0.609** | 0.608 |
| **Presence recall** | 1.00 | 0.00 | **0.599** | 0.559 |
| Absence F1 (gameable) | 1.00 | 0.839 | 0.902 | 0.903 |
| Balanced accuracy | 1.00 | 0.500 | 0.774 | 0.761 |
| Extraction failures | 0/30 | — | 0/30 | 0/30 |
| Speed (GPU) | — | — | **~40 s/doc** | ~254 s/doc |

*14B partial-offloads (61% CPU / 39% GPU) because its 12 GB footprint exceeds the
6 GB card — hence the ~6× slowdown.*

### Per-clause pattern (stable across both models)

**Reliable (MCC ≥ 0.80):** `governing_law` (0.88), `cap_on_liability` (0.86),
`irrevocable_or_perpetual_license` (0.85), `license_grant` (0.81),
`renewal_term` (0.81), `competitive_restriction_exception` (0.80).

**Worse than chance (MCC < 0):** `uncapped_liability`, `liquidated_damages`,
`covenant_not_to_sue`, `joint_ip_ownership`, `rofr_rofo_rofn`.

The model reliably reads clauses that are stated plainly and fails on the subtle,
negation-heavy, cross-referenced ones — exactly the clauses where "the topic is
discussed but no obligation is established" is hardest to tell apart.

## Why schema-constrained decoding matters

Without it, the 7B failed outright on **3/30 docs** (MCC 0.574 with those docs
excluded) — not from truncation, but by **inventing its own JSON key scheme**
(`"1","2","3"`, `"A","B","C"`, `"Section 3.1"`) and collapsing to "answer true to
everything." Passing a JSON Schema in Ollama's `format` field forces the exact
clause keys as booleans, making those failure modes structurally impossible:
**0/30 failures** and MCC 0.574 → 0.609. This is both a cleaner benchmark and the
way you'd actually deploy a local extractor.

## Implications

- **Local extraction is viable for triage / human-in-the-loop**, not yet for
  *unattended* absence assertion: missing ~40% of present clauses means ~40% of
  "this contract has no X" answers on hard clauses would be confidently wrong.
- **Default to the 7B locally** — it is free, private, reliable (with schema
  decoding), and fast enough on a consumer GPU. Do not reach for a bigger local
  model to fix accuracy; it doesn't.
- **Close the hard-clause gap with tier escalation:** run the local model on the
  ~80% of clause types it handles well and escalate only the worse-than-chance
  ones to a hosted model. That is the cost-vs-accuracy architecture worth
  proving next (a hosted-model run on the same 30 docs would quantify the gain).

## Reproduce

```bash
# architecture check — expect MCC 1.0 (no key, no GPU needed)
python3 run_cuad.py --detector oracle --limit 120

# local 7B (needs Ollama + `ollama pull qwen2.5:7b-instruct`)
python3 run_cuad.py --detector llm --provider ollama \
  --model qwen2.5:7b-instruct --limit 30 --workers 1 \
  --out results/ollama_qwen7b_limit30_schema.json

# local 14B comparison
python3 run_cuad.py --detector llm --provider ollama \
  --model qwen2.5:14b-instruct --limit 30 --workers 1 \
  --out results/ollama_qwen14b_limit30_schema.json
```

Results cache per-document to `cache/llm/`, so re-runs and resumes cost nothing.

## Limitations

- 30 documents / 1,230 slots is a modest sample; scale to 100+ before treating
  the point estimates as final.
- One model family (Qwen2.5) at two sizes; other families (Llama 3.1, Mistral-Nemo)
  are untested here.
- No hosted-model number yet, so the *size* of the tier-escalation gain is
  estimated, not measured.
