# Absence Query Benchmark

Standalone test rig for a **derivation store** vs **global top-k RAG** vs **per-document scan** on contract-style **absence queries**, using synthetic data with ground truth. No production dependencies; no API key required for the default run.

```bash
pip install -r requirements.txt
python3 experiments.py --docs 300 --detector negation_aware --events 100
python3 experiments.py --docs 300 --detector oracle          # isolate architecture error
python3 experiments.py --docs 300 --detector keyword         # weak extractor failure mode
python3 experiments.py --docs 2000 --events 200              # scaling
python3 experiments.py --json                                # machine-readable
```

Requires `scikit-learn` (TF-IDF baseline only). If it is not installed, `baselines.py` uses a minimal pure-Python TF-IDF fallback. `--detector llm` uses **OpenRouter** by default when `OPENROUTER_API_KEY` is set in `.env.local` (Anthropic still works via `ANTHROPIC_API_KEY`); everything else runs offline.

## Files

| file | what it does |
|---|---|
| `corpus.py` | synthetic contracts + ground truth, with difficulty tiers and adversarial distractors |
| `datasets.py` | CUAD real-contract loader (auto-download) |
| `store.py` | content-addressed nodes, provenance DAG, support counting, ACL closure |
| `operators.py` | Class M extractors (keyword / negation-aware / oracle / LLM), Class A mergeable rollup |
| `llm_extractor.py` | cached LLM extraction (OpenRouter default / Anthropic), live pricing |
| `env.py` | loads `.env.local` at import |
| `list_models.py` | list OpenRouter models by price |
| `engine.py` | rollup tree, semantic no-op filter, propagation halting, path-to-root maintenance |
| `baselines.py` | global top-k RAG (arm A) and per-document scan (arm B) |
| `experiments.py` | E0, extraction quality, E1, E2, ACL test |
| `run_cuad.py` | real-corpus runner with imbalance-aware metrics |
| `server.py` | inspector API |
| `web/index.html` | inspector UI |
| `tests/` | pytest suite (12 invariant checks) |

## The design decision that matters most

The corpus is **adversarial on purpose**. Clauses are realised at three difficulty tiers
(literal heading / paraphrase / oblique cross-reference), and ~39% of absences carry a
**distractor**: text that discusses the clause topic while establishing no obligation
("the parties elected not to include any ceiling on damages").

Without this, a templated corpus plus a keyword extractor scores ~100% and the experiment
proves nothing. The distractors are what separate a real extractor from a pattern matcher.

Similarly, **arm B (per-document scan) exists to keep the comparison honest.** The claim is
not "RAG cannot answer absence questions." Arm B is RAG and it answers them perfectly. The
claim is that the only way to answer them is a full pass over the corpus, so you should pay
for that once at write time rather than on every query.

## Findings from the first run (300 docs, 100 change events)

**1. The architecture is exact; all error is extraction error.** Under the oracle detector
the derivation store scores P=1.00 / R=1.00 / F1=1.00 on absence queries. The rollup tree,
provenance DAG, and merge algebra introduce zero error. That means every accuracy number
below is a statement about the extractor, not about the design.

**2. Extraction recall is the binding constraint.** With the negation-aware heuristic:

| | recall |
|---|---|
| tier 0 (literal heading) | 1.00 |
| tier 1 (paraphrase) | 0.71 |
| tier 2 (oblique / cross-reference) | 0.71 |

Specificity against distractors is 0.91 for the negation-aware detector and **0.00** for the
keyword detector — the keyword extractor fires on every single distractor. A missed clause
becomes a **false absence**, which is a confidently wrong answer delivered by a system that
looks authoritative. This is the real risk in the headline claim and it is not fixed by any
amount of architecture.

**3. Global top-k RAG fails structurally, and worsens with scale.**

| corpus | derivation store F1 | global top-k RAG F1 |
|---|---|---|
| 300 docs | 0.824 | 0.019 |
| 2,000 docs | 0.838 | 0.000 |

It cannot assert absence for documents it never retrieved, and the fraction it retrieves
shrinks as the corpus grows. This is not a tuning problem.

**4. The derivation store's win is cost, not accuracy.** It ties per-document scan exactly
(F1 0.824 vs 0.824) — same extractor, same answer. The difference is query cost:

| arm | extraction calls per query |
|---|---|
| derivation store | 0 |
| global top-k RAG | 20 (and wrong) |
| per-document scan | 300 (2,000 at scale) |

**5. Incremental maintenance clears the ≥50× gate by a wide margin.** Over 100 change events
on 300 documents: 54 filtered as semantic no-ops, 6 halted after extraction showed no
propositional change, 40 propagated. Total 46 extraction calls and 166 merges, versus
30,000 / 34,400 for naive full rebuild — **652× on extraction calls.** At 2,000 documents it
is 4,124×, because the saving scales with corpus size while incremental cost stays flat.

Note that the two cost gates did most of the work: 60% of change events never reached an
extraction call at all.

**6. ACL closure behaves as predicted, and the implication is awkward.** One restricted
document makes the global rollup restricted-only, because the visible-user set is the
*intersection* of inputs. Confirmed by the test. Production therefore needs per-band
rollups, which multiplies derivation cost by band count — so band count has to be small,
and that is an organisational negotiation, not an engineering one.

## Real corpus

[CUAD](https://www.atticusprojectai.org/cuad/) (Contract Understanding Atticus Dataset):
**510** real commercial contracts, **41** clause categories, **20,910** clause slots,
**67.9% absent**, expert-annotated under **CC BY 4.0**. `datasets.py` auto-downloads it
on first use.

```bash
cp .env.local.example .env.local   # then paste your key from https://openrouter.ai/keys
python run_cuad.py --estimate --limit 50                    # cost preflight, spends nothing
python list_models.py claude                                # live OpenRouter prices
python run_cuad.py --detector oracle --limit 120            # architecture check (expect MCC 1.0)
python run_cuad.py --detector llm --limit 50                # OpenRouter default model
python run_cuad.py --detector llm --model anthropic/claude-sonnet-4.5 --limit 50
```

OpenRouter is the default provider when `OPENROUTER_API_KEY` is present. Pass `--provider anthropic` (and set `ANTHROPIC_API_KEY`) to use Anthropic directly.

## Metrics

Absence-F1 is **gameable** on this corpus. A predictor that always answers "absent" scores
**0.807** absence-F1 because absence is the majority class. Report **MCC** and
**presence-recall** — those are the metrics the degenerate baseline cannot inflate.
`run_cuad.py` always prints the always-absent column beside the model.

## Inspector UI

```bash
python server.py                              # synthetic, instant
python server.py --corpus cuad --limit 60     # real contracts
# open http://127.0.0.1:8000
```

The centre of the screen is a presence matrix: established clauses are solid squares;
absences are literal holes outlined in red. That is the thesis as an interface.

## What this rig does not yet test

The verifier and its calibration (E4) — testable in isolation with prompts and labelled
examples, no graph infrastructure needed. Class N drift (E7), since the only operator here is
associative. Cartridges. The planner and tier escalation. Real linguistic variation: synthetic
tier-2 clauses are harder than tier 0 but still far tamer than genuine contract prose.

## Next step

Run `run_cuad.py --detector llm` and read MCC / presence-recall against the always-absent
baseline. Everything else is already validated; extraction quality on real contract prose is
the number that decides whether this is production-viable.
