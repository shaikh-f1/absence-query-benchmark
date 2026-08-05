"""
Inspection server for the derivation store.

  pip install fastapi uvicorn scikit-learn
  python server.py                      # synthetic corpus, instant, no API key
  python server.py --corpus cuad --limit 60   # real contracts

Then open http://127.0.0.1:8000
"""

import argparse
import random
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import datasets as ds
import env  # noqa: F401 — load .env.local before any provider lookup
import corpus as synth
from engine import RollupEngine
from operators import NegationAwareDetector
from run_cuad import AlwaysAbsentDetector, OracleDetector, evaluate

app = FastAPI(title="Derivation Store Inspector")
STATE: Dict = {}


# ---------------------------------------------------------------- bootstrap

def build_state(corpus_name: str = "synthetic", limit: int = 200,
                detector_name: str = "heuristic") -> None:
    if corpus_name == "cuad":
        docs, truth, clause_types = ds.load_cuad(limit=limit)
        detector = OracleDetector(truth, clause_types, docs) \
            if detector_name != "always_absent" else AlwaysAbsentDetector(clause_types)
        label = "CUAD (real contracts)"
    else:
        docs, truth = synth.generate(n_docs=limit)
        clause_types = synth.CLAUSE_TYPES
        detector = NegationAwareDetector()
        label = "Synthetic (adversarial)"

    engine = RollupEngine(detector)
    engine.build(docs)

    STATE.update({
        "corpus_label": label,
        "corpus_name": corpus_name,
        "docs": docs,
        "truth": truth,
        "clause_types": clause_types,
        "engine": engine,
        "detector": f"{detector.name}:{detector.version}",
        "log": [],
        "build_cost": {
            "extraction_calls": engine.store.counters.extraction_calls,
            "merge_calls": engine.store.counters.merge_calls,
        },
    })


def predictions() -> Dict[str, Dict[str, bool]]:
    e, cts = STATE["engine"], STATE["clause_types"]
    out = {}
    for doc_id in STATE["docs"]:
        key = e.store.by_logical_id.get(f"extract:{doc_id}")
        present = set(e.store.nodes[key].output["present"]) if key else set()
        out[doc_id] = {c: (c in present) for c in cts}
    return out


# ---------------------------------------------------------------- api

@app.get("/api/state")
def api_state():
    e = STATE["engine"]
    n = len(STATE["docs"])
    fb_e, fb_m = e.full_rebuild_cost(n)
    return {
        "corpus": STATE["corpus_label"],
        "corpus_name": STATE["corpus_name"],
        "detector": STATE["detector"],
        "n_docs": n,
        "clause_types": STATE["clause_types"],
        "tree_levels": [len(l) for l in e.tree],
        "branching": e.branching,
        "build_cost": STATE["build_cost"],
        "full_rebuild_cost": {"extraction_calls": fb_e, "merge_calls": fb_m},
        "store": e.store.stats(),
        "log": STATE["log"][-40:],
    }


@app.get("/api/matrix")
def api_matrix(limit: int = 64):
    doc_ids = sorted(STATE["docs"])[:limit]
    pred, truth, cts = predictions(), STATE["truth"], STATE["clause_types"]
    rows = []
    for d in doc_ids:
        gt = truth[d]
        rows.append({
            "doc": d,
            "cells": [
                {
                    "pred_present": pred[d][c],
                    "gold_present": c in gt.present,
                }
                for c in cts
            ],
        })
    return {"clause_types": cts, "rows": rows}


@app.get("/api/absence")
def api_absence(clause: str):
    e, truth = STATE["engine"], STATE["truth"]
    predicted = e.docs_missing(clause)
    gold = sorted([d for d, gt in truth.items() if clause in gt.absent])
    ps, gs = set(predicted), set(gold)
    tp = len(ps & gs)
    prec = tp / len(ps) if ps else 0.0
    rec = tp / len(gs) if gs else 0.0
    return {
        "clause": clause,
        "predicted_missing": predicted[:200],
        "n_predicted": len(predicted),
        "n_gold": len(gold),
        "false_absences": sorted(ps - gs)[:50],   # said missing, actually present
        "missed_absences": sorted(gs - ps)[:50],
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
        "query_llm_calls": 0,
        "scan_llm_calls": len(STATE["docs"]),
    }


@app.get("/api/metrics")
def api_metrics():
    return evaluate(predictions(), STATE["truth"], STATE["clause_types"])


class ChangeRequest(BaseModel):
    doc_id: Optional[str] = None
    kind: str = "semantic"     # "cosmetic" | "semantic"


@app.post("/api/change")
def api_change(req: ChangeRequest):
    """Apply one edit and report the blast radius against a full rebuild."""
    docs, truth = STATE["docs"], STATE["truth"]
    e, cts = STATE["engine"], STATE["clause_types"]
    doc_id = req.doc_id or random.choice(sorted(docs))
    d = docs[doc_id]

    if req.kind == "cosmetic":
        new_text = d.text.replace(". ", ".  ")
        detail = "whitespace only"
    else:
        gt = truth[doc_id]
        present = sorted(gt.present)
        if present:
            clause = random.choice(present)
            lines = [l for l in d.text.split("\n\n")
                     if clause.split("_")[0].lower() in l.lower()]
            new_text = d.text.replace(lines[0], "") if lines else d.text + "\n\nAmended."
            gt.present.pop(clause, None)
            gt.absent.add(clause)
            detail = f"removed {clause.replace('_', ' ')}"
        else:
            clause = random.choice(cts)
            new_text = d.text + f"\n\n{clause.upper()}. The parties so agree."
            gt.absent.discard(clause)
            gt.present[clause] = 0
            detail = f"added {clause.replace('_', ' ')}"

    res = e.apply_change(doc_id, new_text, d.acl_band)
    docs[doc_id] = type(d)(doc_id, d.revision + 1, new_text, d.acl_band)

    fb_e, fb_m = e.full_rebuild_cost(len(docs))
    entry = {
        "doc": doc_id,
        "kind": req.kind,
        "detail": detail,
        "path": res["path"],
        "extraction_calls": res["extractions"],
        "merge_calls": res["merges"],
        "rebuild_extraction_calls": fb_e,
        "saving": round(fb_e / res["extractions"], 1) if res["extractions"] else None,
    }
    STATE["log"].append(entry)
    return entry


@app.post("/api/reset")
def api_reset():
    build_state(STATE["corpus_name"], len(STATE["docs"]))
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse("web/index.html")


app.mount("/web", StaticFiles(directory="web"), name="web")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="synthetic", choices=["synthetic", "cuad"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    build_state(args.corpus, args.limit)
    print(f"corpus={STATE['corpus_label']}  docs={len(STATE['docs'])}  "
          f"detector={STATE['detector']}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)
