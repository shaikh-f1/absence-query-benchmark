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

Requires `scikit-learn` (TF-IDF baseline only). If it is not installed, `baselines.py` uses a minimal pure-Python TF-IDF fallback. `--detector llm` uses the Anthropic API if `ANTHROPIC_API_KEY` is set; everything else runs offline.

## Files

| file | what it does |
|---|---|
| `corpus.py` | synthetic contracts + ground truth, with difficulty tiers and adversarial distractors |
| `store.py` | content-addressed nodes, provenance DAG, support counting, ACL closure |
| `operators.py` | Class M extractors (keyword / negation-aware / oracle / LLM), Class A mergeable rollup |
| `engine.py` | rollup tree, semantic no-op filter, propagation halting, path-to-root maintenance |
| `baselines.py` | global top-k RAG (arm A) and per-document scan (arm B) |
| `experiments.py` | E0, extraction quality, E1, E2, ACL test |

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

## What this rig does not yet test

The verifier and its calibration (E4) — testable in isolation with prompts and labelled
examples, no graph infrastructure needed. Class N drift (E7), since the only operator here is
associative. Cartridges. The planner and tier escalation. Real linguistic variation: synthetic
tier-2 clauses are harder than tier 0 but still far tamer than genuine contract prose.

## Next step

Swap in `--detector llm` and re-measure extraction recall by tier. Everything else is
already validated; extraction quality on tier 1 and tier 2 is the number that decides whether
this is production-viable, and it is the only number a heuristic cannot tell you.
