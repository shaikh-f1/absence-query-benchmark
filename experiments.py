"""
Experiment harness for the derivation-store test rig.

Run:  python experiments.py [--docs 300] [--detector negation_aware|keyword|oracle|llm]
"""

import argparse
import json
import random
from typing import Dict, List, Set

import corpus
from baselines import GlobalTopKRAG, PerDocumentScanRAG
from engine import RollupEngine
from operators import (KeywordDetector, LLMDetector, NegationAwareDetector,
                       OracleDetector)


def prf(pred: Set[str], gold: Set[str]) -> Dict[str, float]:
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": len(pred - gold), "fn": len(gold - pred)}


def extraction_quality(engine: RollupEngine, docs, truth) -> dict:
    """
    Measured SEPARATELY from architecture. A false absence -- extraction missing a
    clause that is present -- is a confidently wrong answer, and it is the real
    risk in the headline claim. Broken out by difficulty tier and by distractor.
    """
    by_tier = {0: [0, 0], 1: [0, 0], 2: [0, 0]}   # [correct, total]
    distractor = [0, 0]
    plain_absent = [0, 0]

    for doc_id in docs:
        key = engine.store.by_logical_id[f"extract:{doc_id}"]
        pred = set(engine.store.nodes[key].output["present"])
        gt = truth[doc_id]
        for clause, tier in gt.present.items():
            by_tier[tier][1] += 1
            if clause in pred:
                by_tier[tier][0] += 1
        for clause in gt.absent:
            bucket = distractor if clause in gt.distractor_only else plain_absent
            bucket[1] += 1
            if clause not in pred:
                bucket[0] += 1

    return {
        "recall_by_tier": {
            f"tier{t}": round(c / n, 4) if n else None for t, (c, n) in by_tier.items()
        },
        "tier_counts": {f"tier{t}": n for t, (c, n) in by_tier.items()},
        "specificity_plain_absent": round(plain_absent[0] / plain_absent[1], 4),
        "specificity_with_distractor": round(distractor[0] / distractor[1], 4),
        "distractor_count": distractor[1],
    }


def e1_absence(engine, docs, truth, detector) -> dict:
    """Absence query: which docs lack clause X. Three arms, same corpus, same detector."""
    n = len(docs)
    global_rag = GlobalTopKRAG(docs, k=20)
    perdoc_rag = PerDocumentScanRAG(docs)

    rows = []
    for clause in corpus.CLAUSE_TYPES:
        gold = {d for d in docs if clause in truth[d].absent}
        rows.append({
            "clause": clause,
            "gold_absent": len(gold),
            "derivation_store": prf(set(engine.docs_missing(clause)), gold),
            "global_topk_rag": prf(set(global_rag.docs_missing(clause, detector)), gold),
            "per_doc_scan_rag": prf(set(perdoc_rag.docs_missing(clause, detector)), gold),
        })

    def mean(arm, metric):
        return round(sum(r[arm][metric] for r in rows) / len(rows), 4)

    return {
        "per_clause": rows,
        "macro": {
            arm: {m: mean(arm, m) for m in ("precision", "recall", "f1")}
            for arm in ("derivation_store", "global_topk_rag", "per_doc_scan_rag")
        },
        "query_cost_extraction_calls": {
            "derivation_store": 0,
            "global_topk_rag": 20,
            "per_doc_scan_rag": n,
        },
    }


def e2_maintenance(engine, docs, truth, n_events: int = 100, seed: int = 11,
                   p_cosmetic: float = 0.55) -> dict:
    """
    Replay a change stream. Measure incremental cost vs full rebuild.
    p_cosmetic reflects that most real document churn is formatting noise.
    """
    rng = random.Random(seed)
    engine.store.counters.reset()
    paths = {"noop": 0, "halted": 0, "propagated": 0}
    ext, mrg = 0, 0
    ids = list(docs)

    for _ in range(n_events):
        doc_id = rng.choice(ids)
        d = docs[doc_id]
        if rng.random() < p_cosmetic:
            new_text = d.text.replace("\n\n", "\n\n ").upper().lower() \
                if rng.random() < 0.5 else d.text + "  "
            new_text = d.text.replace(". ", ".  ")
        else:
            clause = rng.choice(corpus.CLAUSE_TYPES)
            if clause in truth[doc_id].present:
                target = [l for l in d.text.split("\n\n")
                          if any(k in l.lower() for k in
                                 __import__("operators").KEYWORDS[clause])]
                new_text = d.text.replace(target[0], "") if target else d.text + "\n\nAmended."
                truth[doc_id].present.pop(clause, None)
                truth[doc_id].absent.add(clause)
            else:
                add = __import__("corpus").REALISATIONS[clause][0][0]
                new_text = d.text + "\n\n" + add
                truth[doc_id].absent.discard(clause)
                truth[doc_id].present[clause] = 0

        res = engine.apply_change(doc_id, new_text, d.acl_band)
        docs[doc_id] = corpus.Document(doc_id, d.revision + 1, new_text, d.acl_band)
        paths[res["path"]] += 1
        ext += res["extractions"]
        mrg += res["merges"]

    fb_ext, fb_mrg = engine.full_rebuild_cost(len(docs))
    naive_ext = fb_ext * n_events
    naive_mrg = fb_mrg * n_events

    return {
        "events": n_events,
        "paths": paths,
        "incremental": {"extraction_calls": ext, "merge_calls": mrg},
        "naive_full_rebuild_per_event": {"extraction_calls": naive_ext,
                                        "merge_calls": naive_mrg},
        "speedup_extraction": round(naive_ext / ext, 1) if ext else None,
        "speedup_merge": round(naive_mrg / mrg, 1) if mrg else None,
        "gate_50x_extraction": (naive_ext / ext) >= 50 if ext else None,
    }


def acl_test(engine, docs) -> dict:
    """
    ACL closure: a rollup over mixed bands must not be visible below the most
    restrictive input band. Verifies the intersection rule, not the union rule.
    """
    root_key = engine.tree[-1][0]
    root_band = engine.store.nodes[root_key].acl_band
    bands_present = sorted({d.acl_band for d in docs.values()})
    return {
        "input_bands_present": bands_present,
        "root_rollup_band": root_band,
        "visible_to_public_user": engine.store.visible_to(root_key, "public"),
        "visible_to_restricted_user": engine.store.visible_to(root_key, "restricted"),
        "intersection_rule_holds": root_band == max(
            bands_present, key=lambda b: {"public": 0, "internal": 1, "restricted": 2}[b]
        ),
        "implication": ("a single restricted document makes the global rollup "
                        "restricted-only; production needs per-band rollups"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=300)
    ap.add_argument("--detector", default="negation_aware",
                    choices=["keyword", "negation_aware", "oracle", "llm"])
    ap.add_argument("--events", type=int, default=100)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    docs, truth = corpus.generate(n_docs=args.docs)

    detector = {
        "keyword": KeywordDetector(),
        "negation_aware": NegationAwareDetector(),
        "oracle": lambda: OracleDetector(truth),
        "llm": LLMDetector,
    }[args.detector]
    if callable(detector) and not hasattr(detector, "name"):
        detector = detector()

    engine = RollupEngine(detector)
    engine.build(docs)

    report = {
        "E0_corpus": corpus.summarise(docs, truth),
        "store": engine.store.stats(),
        "detector": f"{detector.name}:{detector.version}",
        "extraction_quality": extraction_quality(engine, docs, truth),
        "E1_absence": e1_absence(engine, docs, truth, detector),
        "ACL_closure": acl_test(engine, docs),
        "E2_maintenance": e2_maintenance(engine, docs, truth, n_events=args.events),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    r = report
    print("=" * 74)
    print(f"DERIVATION STORE TEST RIG   detector={r['detector']}")
    print("=" * 74)
    print("\n-- E0 corpus --")
    for k, v in r["E0_corpus"].items():
        print(f"   {k:32} {v}")
    print(f"   store nodes                      {r['store']}")

    print("\n-- Extraction quality (isolated from architecture) --")
    eq = r["extraction_quality"]
    for k, v in eq["recall_by_tier"].items():
        print(f"   recall {k:26} {v}   (n={eq['tier_counts'][k]})")
    print(f"   specificity, plain absent        {eq['specificity_plain_absent']}")
    print(f"   specificity, distractor present  {eq['specificity_with_distractor']}"
          f"   (n={eq['distractor_count']})")

    print("\n-- E1 absence query, macro over 10 clause types --")
    print(f"   {'arm':22} {'P':>8} {'R':>8} {'F1':>8}  {'query cost':>12}")
    costs = r["E1_absence"]["query_cost_extraction_calls"]
    for arm, m in r["E1_absence"]["macro"].items():
        print(f"   {arm:22} {m['precision']:>8} {m['recall']:>8} {m['f1']:>8}"
              f"  {costs[arm]:>12}")

    print("\n-- ACL closure --")
    for k, v in r["ACL_closure"].items():
        print(f"   {k:32} {v}")

    print("\n-- E2 incremental maintenance --")
    e2 = r["E2_maintenance"]
    print(f"   events                           {e2['events']}  paths={e2['paths']}")
    print(f"   incremental                      {e2['incremental']}")
    print(f"   naive rebuild per event          {e2['naive_full_rebuild_per_event']}")
    print(f"   speedup, extraction calls        {e2['speedup_extraction']}x"
          f"   (>=50x gate: {e2['gate_50x_extraction']})")
    print(f"   speedup, merge calls             {e2['speedup_merge']}x")
    print("=" * 74)


if __name__ == "__main__":
    main()
