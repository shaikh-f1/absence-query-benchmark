"""
Real-corpus runner for CUAD, with imbalance-aware metrics.

Absence is ~68% of clause slots here. Absence-F1 is therefore gameable: a
predictor that always answers "absent" scores ~0.81. This module always reports
that degenerate baseline beside the model, and elevates MCC and presence-recall
— the two numbers it cannot inflate by refusing to see any clause.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Optional, Sequence

import env  # noqa: F401 — load .env.local before any provider lookup
import datasets as ds
from engine import RollupEngine
from llm_extractor import LLMClauseExtractor, estimate_cost


# ---------------------------------------------------------------------------
# Detectors local to the real corpus (text-keyed oracle; no CONTRACT_N regex)
# ---------------------------------------------------------------------------

class OracleDetector:
    """
    Perfect extraction from CUAD ground truth. Maps full document text → labels
    via a text-keyed dict, so it works without the synthetic CONTRACT_N marker.
    Chunks that are not exact document texts return all-False (absent) — the
    oracle is only meaningful on whole documents.
    """
    name = "oracle"
    version = "1"

    def __init__(self, truth, clause_types, docs):
        self.truth = truth
        self.clause_types = list(clause_types)
        self.by_text = {docs[d].text: d for d in docs}

    def __call__(self, text: str) -> Dict[str, bool]:
        doc_id = self.by_text.get(text)
        if doc_id is None:
            return {c: False for c in self.clause_types}
        gt = self.truth[doc_id]
        return {c: (c in gt.present) for c in self.clause_types}


class AlwaysAbsentDetector:
    """Degenerate baseline: every clause is absent. Scores high absence-F1 here."""
    name = "always_absent"
    version = "1"

    def __init__(self, clause_types):
        self.clause_types = list(clause_types)

    def __call__(self, text: str) -> Dict[str, bool]:
        return {c: False for c in self.clause_types}


# ---------------------------------------------------------------------------
# Metrics — "predicted absent" is the positive class
# ---------------------------------------------------------------------------

def _cell(pred_absent: bool, gold_absent: bool):
    if pred_absent and gold_absent:
        return "tp"
    if pred_absent and not gold_absent:
        return "fp"
    if (not pred_absent) and gold_absent:
        return "fn"
    return "tn"


def _scores(tp: int, fp: int, fn: int, tn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # presence_recall = specificity when absent is the positive class
    presence_rec = tn / (tn + fp) if (tn + fp) else 0.0
    bal = 0.5 * (rec + presence_rec)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    return {
        "absence_precision": round(prec, 4),
        "absence_recall": round(rec, 4),
        "absence_f1": round(f1, 4),
        "presence_recall": round(presence_rec, 4),
        "balanced_accuracy": round(bal, 4),
        "mcc": round(mcc, 4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def _accumulate(predictions, truth, clause_types, force_absent: bool = False):
    """Micro counts over all (doc, clause) slots; also per-clause tallies."""
    micro = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per: Dict[str, Dict[str, int]] = {
        c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in clause_types
    }
    for doc_id, gt in truth.items():
        pred = predictions.get(doc_id, {})
        for c in clause_types:
            gold_absent = c in gt.absent
            if force_absent:
                pred_absent = True
            else:
                # predictions[doc][c] True means present
                pred_absent = not bool(pred.get(c, False))
            bucket = _cell(pred_absent, gold_absent)
            micro[bucket] += 1
            per[c][bucket] += 1
    return micro, per


def evaluate(predictions, truth, clause_types) -> dict:
    """
    Imbalance-aware evaluation. Always includes the always-absent baseline
    scored on identical metrics so a gamed absence-F1 cannot hide.
    """
    micro_c, per_c = _accumulate(predictions, truth, clause_types, False)
    base_c, _ = _accumulate(predictions, truth, clause_types, True)

    per_scores = {c: _scores(**per_c[c]) for c in clause_types}
    macro = {
        k: round(sum(per_scores[c][k] for c in clause_types) / len(clause_types), 4)
        for k in ("absence_precision", "absence_recall", "absence_f1",
                  "presence_recall", "balanced_accuracy", "mcc")
    }

    by_mcc = sorted(clause_types, key=lambda c: per_scores[c]["mcc"])
    return {
        "micro": _scores(**micro_c),
        "macro_over_clause_types": macro,
        "always_absent_baseline": _scores(**base_c),
        "per_clause": per_scores,
        "worst_5_by_mcc": [
            {"clause": c, **{k: per_scores[c][k] for k in
              ("mcc", "absence_f1", "presence_recall")}}
            for c in by_mcc[:5]
        ],
        "best_5_by_mcc": [
            {"clause": c, **{k: per_scores[c][k] for k in
              ("mcc", "absence_f1", "presence_recall")}}
            for c in by_mcc[-5:][::-1]
        ],
    }


def _predictions_from_engine(engine, docs, clause_types) -> Dict[str, Dict[str, bool]]:
    out = {}
    for doc_id in docs:
        key = engine.store.by_logical_id.get(f"extract:{doc_id}")
        present = set(engine.store.nodes[key].output["present"]) if key else set()
        out[doc_id] = {c: (c in present) for c in clause_types}
    return out


def _print_table(metrics: dict, detector_label: str) -> None:
    m, b = metrics["micro"], metrics["always_absent_baseline"]
    rows = [
        ("absence_precision", "Absence precision"),
        ("absence_recall", "Absence recall"),
        ("absence_f1", "Absence F1"),
        ("presence_recall", "Presence recall"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("mcc", "MCC"),
    ]
    print(f"\n{'Metric':<22}{detector_label:<14}{'always_absent':<14}")
    print("-" * 50)
    for k, label in rows:
        print(f"{label:<22}{m[k]:<14}{b[k]:<14}")
    print(f"\n{metrics.get('note', '')}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="CUAD real-corpus absence benchmark")
    ap.add_argument("--detector", choices=["oracle", "always_absent", "llm"],
                    default="oracle")
    ap.add_argument("--model", default=None,
                    help="provider model id; default is provider-appropriate")
    ap.add_argument("--provider", choices=["openrouter", "anthropic"], default=None,
                    help="force provider; default auto-detects from env keys")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-chars", type=int, default=None)
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent LLM calls; keep low on free-tier rate limits")
    ap.add_argument("--estimate", action="store_true",
                    help="print cost estimates and exit without spending")
    ap.add_argument("--out", default=None, help="write full JSON metrics here")
    args = ap.parse_args(argv)

    docs, truth, clause_types = ds.load_cuad(
        limit=args.limit, max_chars=args.max_chars, categories=args.categories,
    )
    prof = ds.profile(docs, truth, clause_types)
    print(f"CUAD  docs={prof['n_docs']}  clauses={prof['n_clause_types']}  "
          f"slots={prof['clause_slots']}  "
          f"absence_rate={prof['overall_absence_rate']:.1%}")
    print(prof["degenerate_baseline_warning"])

    if args.estimate:
        provider = args.provider or "openrouter"
        if args.model:
            models = [args.model]
        else:
            # Default pair: quality default + a cheap alternative.
            models = ["anthropic/claude-sonnet-4.5", "openai/gpt-4o-mini"]
        for model in models:
            est = estimate_cost(docs, model, provider)
            print(f"  [{est['provider']}] {model}: ~${est['est_cost_usd']:.2f}  "
                  f"(~{est['est_input_tokens']:,} in + "
                  f"{est['est_output_tokens']:,} out tokens, "
                  f"${est['price_in_per_mtok']}/{est['price_out_per_mtok']} per MTok)")
        return

    if args.detector == "oracle":
        detector = OracleDetector(truth, clause_types, docs)
    elif args.detector == "always_absent":
        detector = AlwaysAbsentDetector(clause_types)
    else:
        detector = LLMClauseExtractor(
            clause_types, model=args.model, provider=args.provider,
        )
        est = estimate_cost(docs, detector.model, detector.provider)
        print(f"preflight [{detector.provider}] {detector.model}: "
              f"~${est['est_cost_usd']:.2f}")
        print("prewarming LLM cache…")
        detector.prewarm(docs, workers=args.workers, progress=True)
        print("usage:", detector.usage.summary(detector.model, detector.provider))

    failed_docs = getattr(detector, "failed_docs", set())
    n_total = len(docs)
    n_excluded = len(failed_docs)
    if n_excluded:
        docs = {k: v for k, v in docs.items() if k not in failed_docs}
        truth = {k: v for k, v in truth.items() if k not in failed_docs}

    engine = RollupEngine(detector)
    engine.build(docs)
    preds = _predictions_from_engine(engine, docs, clause_types)

    metrics = evaluate(preds, truth, clause_types)
    metrics["profile"] = prof
    metrics["detector"] = f"{detector.name}:{detector.version}"
    metrics["n_excluded"] = n_excluded
    metrics["note"] = (
        "Read MCC and presence-recall. Absence-F1 is inflated by class imbalance."
    )

    _print_table(metrics, detector.name)
    if n_excluded:
        print(f"\nn_excluded (failed extraction): {n_excluded} / {n_total}")
    print(f"\nworst-5 MCC: "
          + ", ".join(f"{x['clause']}={x['mcc']}" for x in metrics["worst_5_by_mcc"]))
    print(f"best-5 MCC:  "
          + ", ".join(f"{x['clause']}={x['mcc']}" for x in metrics["best_5_by_mcc"]))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
