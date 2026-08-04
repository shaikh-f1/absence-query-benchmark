# Setup — repo integration, testing, and the inspector UI

## 1. Where it goes in your repo

Keep it as a **self-contained subpackage with no imports into your existing code**. That
was the point of building it standalone, and it stays true until the numbers justify
integration.

```
your-repo/
├── ...your existing project...
└── derivation-store/          ← drop the whole folder here
    ├── corpus.py              synthetic adversarial corpus + ground truth
    ├── datasets.py            CUAD loader (real contracts)
    ├── store.py               content-addressed nodes, provenance DAG, ACL closure
    ├── operators.py           Class M extractors, Class A mergeable rollup
    ├── engine.py              rollup tree, no-op filter, incremental maintenance
    ├── baselines.py           global top-k RAG + per-document scan
    ├── llm_extractor.py       real LLM extraction (OpenRouter / Anthropic), cached, costed
    ├── env.py                 loads .env.local
    ├── list_models.py         OpenRouter model price listing
    ├── experiments.py         synthetic run (E0/E1/E2/ACL)
    ├── run_cuad.py            real-corpus run, imbalance-aware metrics
    ├── server.py              inspector API
    ├── web/index.html         inspector UI
    ├── tests/                 pytest suite
    ├── .env.local.example     template for API keys
    └── data/, cache/          gitignored
```

Add to `.gitignore`:

```
derivation-store/data/
derivation-store/cache/
derivation-store/.env.local
derivation-store/.env
```

`data/` holds CUAD (~100MB unpacked) and `cache/` holds LLM responses keyed by content
hash. Both regenerate; neither belongs in git. **Do not commit `cache/`** — it contains
verbatim contract text in the cache keys' source documents. **Do not commit `.env.local`.**

Dependencies (add to a separate `requirements-derivstore.txt`, not your main one):

```
scikit-learn>=1.3
fastapi>=0.110
uvicorn>=0.27
pytest>=8.0
python-dotenv>=1.0
```

### API keys

OpenRouter is the default provider. Copy the example and paste a key from
https://openrouter.ai/keys:

```bash
cp .env.local.example .env.local
# edit .env.local → OPENROUTER_API_KEY=sk-or-...
```

`env.py` loads `.env.local` automatically when you run `run_cuad.py` or `server.py`.
Anthropic still works if you set `ANTHROPIC_API_KEY` and pass `--provider anthropic`.

## 2. Testing

```bash
cd derivation-store
pip install -r requirements-derivstore.txt
pytest -q                       # 12 tests, ~2s, no API key needed
```

The suite tests invariants that would **silently corrupt results**, not happy paths:

| test | claim it protects |
|---|---|
| `oracle_extraction_gives_lossless_absence` | the architecture itself is exact |
| `merge_is_associative_and_commutative` | Class A maintenance is mathematically valid |
| `content_addressing_is_deterministic` | cache keys are stable, so the cache can hit |
| `cosmetic_change_costs_nothing` | cost gate 1 works |
| `semantic_change_recomputes_log_n_not_n` | blast radius stays on the path to root |
| `maintenance_beats_rebuild_by_50x` | the E2 gate holds |
| `acl_closure_uses_intersection_not_union` | **a restricted doc restricts the aggregate** |
| `support_counting_retires_only_unsupported` | deletion semantics are correct |
| `global_rag_fails_structurally_on_absence` | the baseline really is the weak one |
| `store_ties_per_document_scan` | accuracy parity — the win is cost, not quality |
| `distractors_defeat_the_keyword_extractor` | the corpus is genuinely adversarial |

If that last one ever passes trivially, your corpus has stopped being a real test.

Note on the 50× gate: incremental cost is 1 extraction per change, so **the maximum
achievable speedup is the corpus size**. The gate is unreachable below ~50 documents —
that test builds its own 300-doc corpus rather than reusing the small fixture.

### Experiment runs

```bash
python experiments.py --docs 300 --detector negation_aware --events 100
python experiments.py --docs 300 --detector oracle          # isolate architecture error
python run_cuad.py --detector oracle --limit 120            # real contracts, no API key
python run_cuad.py --estimate --limit 50                    # cost preflight (OpenRouter prices)
python list_models.py claude                                # live model list + $/MTok
python run_cuad.py --detector llm --limit 50                # OpenRouter default model
python run_cuad.py --detector llm --model anthropic/claude-sonnet-4.5 --limit 50
```

## 3. The inspector UI

```bash
python server.py                              # synthetic, instant
python server.py --corpus cuad --limit 60     # real contracts
# open http://127.0.0.1:8000
```

**What it shows.** The centre of the screen is a presence matrix — rows are documents,
columns are clause types. A filled square means the clause is established. **An absence is
a literal hole in the card**, outlined in red. That is the entire thesis rendered as an
interface: you are looking at the thing retrieval cannot return.

Four panels underneath:

- **Selected clause** — click any clause in the left register to see which documents lack
  it, how many of those are *false* absences the extractor invented, and the LLM-call count
  for this store (0) versus a full scan (N).
- **Accuracy vs the degenerate baseline** — every metric printed beside what a predictor
  that always answers "absent" would score. MCC and presence-recall are highlighted because
  they are the two it cannot game.
- **Write-time cost** — what the build cost once, and the rollup tree shape.
- **Invalidation ledger** — press *Change meaning* or *Reformat only* and watch the blast
  radius. Cosmetic edits log as `noop` with zero calls; semantic edits log as `propagated`
  with 1 extraction and ~4 merges against a 120-extraction rebuild.

That ledger is the demo to run in front of your team. Press *Reformat only* three times,
then *Change meaning* once, and the cost asymmetry explains itself without a slide.

## 4. What is deliberately not wired up

No verifier, no planner tiers, no cartridges, no distilled models. Those are Phase 3+ and
building them now would be committing before the extraction number is known.

The one measurement that still matters: `run_cuad.py --detector llm`. Everything else is
already validated.
