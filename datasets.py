"""
CUAD loader — the real-corpus half of this rig.

Contract Understanding Atticus Dataset (CUAD): 510 commercial contracts, 41
clause categories, 20,910 clause slots, lawyer-annotated under CC BY 4.0.
Absence is the majority class (~67.9%), and the labels are expert-verified, so
this is a genuine absence benchmark rather than a synthetic proxy.

***************************************************************************
WARNING — severe class imbalance.
Some categories are absent in ~97% of contracts. A predictor that always
answers "absent" scores ~0.81 absence-F1 on this corpus. Any evaluation MUST
report that degenerate baseline alongside the model, or the number is
meaningless. Prefer MCC and presence-recall; they cannot be gamed that way.
***************************************************************************
"""

from __future__ import annotations

import io
import json
import random
import re
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Nested: repo zip → data.zip → CUADv1.json. TheAtticusProject redirects; keep
# both spellings so offline mirrors and renamed orgs both work.
_CUAD_REPO_URLS = (
    "https://codeload.github.com/TheAtticusProject/cuad/zip/refs/heads/main",
    "https://codeload.github.com/The-Atticus-Project/cuad/zip/refs/heads/main",
)


@dataclass
class GroundTruth:
    doc_id: str
    present: Dict[str, int] = field(default_factory=dict)   # clause -> tier (0)
    absent: Set[str] = field(default_factory=set)
    distractor_only: Set[str] = field(default_factory=set)
    contradictions: List[str] = field(default_factory=list)
    spans: Dict[str, List[str]] = field(default_factory=dict)  # gold answer text


@dataclass
class Document:
    doc_id: str
    revision: int
    text: str
    acl_band: str


def _normalise_category(qid: str) -> str:
    """
    `"X__Governing Law_0"` → `"governing_law"`.
    Takes the part after `__`, strips a trailing `_\\d+`, lowercases, and
    replaces non-alphanumerics with `_`.
    """
    part = qid.split("__", 1)[1] if "__" in qid else qid
    part = re.sub(r"_\d+$", "", part)
    part = part.lower()
    part = re.sub(r"[^a-z0-9]+", "_", part).strip("_")
    return part


def ensure_cuad(root: str = "data/cuad") -> str:
    """
    Download and unpack CUAD if `CUADv1.json` is absent. Returns its path.
    Handles the nested zip-inside-zip layout of the Atticus GitHub repo.
    """
    root_p = Path(root)
    target = root_p / "CUADv1.json"
    if target.exists():
        return str(target)

    root_p.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    repo_bytes: Optional[bytes] = None
    for url in _CUAD_REPO_URLS:
        try:
            with urllib.request.urlopen(url, timeout=120) as resp:
                repo_bytes = resp.read()
            break
        except Exception as e:  # noqa: BLE001 — try next mirror
            last_err = e
    if repo_bytes is None:
        raise RuntimeError(f"failed to download CUAD repo zip: {last_err}")

    data_zip_bytes: Optional[bytes] = None
    with zipfile.ZipFile(io.BytesIO(repo_bytes)) as repo_zf:
        # Prefer an entry literally named data.zip (possibly under a top folder).
        candidates = [n for n in repo_zf.namelist() if n.endswith("data.zip")]
        if not candidates:
            # Some mirrors ship CUADv1.json at the repo root already.
            json_hits = [n for n in repo_zf.namelist() if n.endswith("CUADv1.json")]
            if json_hits:
                target.write_bytes(repo_zf.read(json_hits[0]))
                return str(target)
            raise RuntimeError("CUAD archive contains neither data.zip nor CUADv1.json")
        data_zip_bytes = repo_zf.read(candidates[0])

    with zipfile.ZipFile(io.BytesIO(data_zip_bytes)) as data_zf:
        json_hits = [n for n in data_zf.namelist() if n.endswith("CUADv1.json")]
        if not json_hits:
            raise RuntimeError("data.zip does not contain CUADv1.json")
        target.write_bytes(data_zf.read(json_hits[0]))

    return str(target)


def load_cuad(
    path: Optional[str] = None,
    limit: Optional[int] = None,
    max_chars: Optional[int] = None,
    categories: Optional[Sequence[str]] = None,
    acl_bands: Sequence[str] = ("public", "internal", "restricted"),
    seed: int = 7,
) -> Tuple[Dict[str, Document], Dict[str, GroundTruth], List[str]]:
    """
    Load CUAD into the same (docs, truth) shape the synthetic corpus uses, plus
    the ordered list of clause type ids.

    max_chars truncates document text. Default is off: truncation invalidates
    ground truth (gold spans may sit past the cut), so only use it for cost
    estimation / smoke tests, never for reported accuracy.
    """
    json_path = path or ensure_cuad()
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    contracts = raw["data"]
    if limit is not None:
        contracts = contracts[:limit]

    # Discover clause types in stable first-seen order across the (possibly
    # limited) slice, then optionally filter.
    discovered: List[str] = []
    seen: Set[str] = set()
    for contract in contracts:
        for qa in contract["paragraphs"][0]["qas"]:
            cat = _normalise_category(qa["id"])
            if cat not in seen:
                seen.add(cat)
                discovered.append(cat)

    if categories is not None:
        wanted = {_normalise_category(c) if "__" in c else
                  re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")
                  for c in categories}
        clause_types = [c for c in discovered if c in wanted]
    else:
        clause_types = discovered

    rng = random.Random(seed)
    docs: Dict[str, Document] = {}
    truth: Dict[str, GroundTruth] = {}

    for i, contract in enumerate(contracts):
        doc_id = contract["title"]
        para = contract["paragraphs"][0]
        text = para["context"]
        if max_chars is not None:
            # Truncation invalidates ground truth for spans past the cut.
            text = text[:max_chars]

        present: Dict[str, int] = {}
        absent: Set[str] = set()
        spans: Dict[str, List[str]] = {}

        # One qa per category in CUADv1.json; still union defensively.
        by_cat: Dict[str, List[dict]] = {}
        for qa in para["qas"]:
            cat = _normalise_category(qa["id"])
            if cat not in clause_types:
                continue
            by_cat.setdefault(cat, []).append(qa)

        for cat in clause_types:
            qas = by_cat.get(cat, [])
            if not qas:
                absent.add(cat)
                continue
            # Present if any qa for this category is answerable.
            if any(not q.get("is_impossible", False) for q in qas):
                present[cat] = 0
                texts: List[str] = []
                for q in qas:
                    for a in q.get("answers") or []:
                        t = (a.get("text") or "").strip()
                        if t:
                            texts.append(t)
                if texts:
                    spans[cat] = texts
            else:
                absent.add(cat)

        band = rng.choice(list(acl_bands))
        docs[doc_id] = Document(doc_id=doc_id, revision=1, text=text, acl_band=band)
        truth[doc_id] = GroundTruth(
            doc_id=doc_id, present=present, absent=absent, spans=spans,
        )

    return docs, truth, clause_types


def profile(docs, truth, clause_types) -> dict:
    """Corpus shape + the degenerate-baseline warning every consumer must see."""
    n_docs = len(docs)
    n_cts = len(clause_types)
    slots = n_docs * n_cts
    absent_n = sum(1 for gt in truth.values() for c in clause_types if c in gt.absent)
    present_n = slots - absent_n

    per_cat_absent = Counter()
    for gt in truth.values():
        for c in clause_types:
            if c in gt.absent:
                per_cat_absent[c] += 1

    chars = [len(d.text) for d in docs.values()]
    avg_chars = sum(chars) // max(n_docs, 1)
    max_chars = max(chars) if chars else 0
    # Rough: ~3.8 chars/token for legal English; one call per doc covering all cats.
    est_tokens = int(sum(chars) / 3.8) + n_docs * 350

    ranked = sorted(
        ((c, per_cat_absent[c] / max(n_docs, 1)) for c in clause_types),
        key=lambda x: x[1],
        reverse=True,
    )
    rate = absent_n / slots if slots else 0.0
    return {
        "n_docs": n_docs,
        "n_clause_types": n_cts,
        "clause_slots": slots,
        "present_slots": present_n,
        "absent_slots": absent_n,
        "overall_absence_rate": round(rate, 4),
        "avg_chars": avg_chars,
        "max_chars": max_chars,
        "estimated_tokens_full_pass": est_tokens,
        "most_absent": [
            {"clause": c, "absence_rate": round(r, 4)} for c, r in ranked[:5]
        ],
        "least_absent": [
            {"clause": c, "absence_rate": round(r, 4)} for c, r in ranked[-5:][::-1]
        ],
        "degenerate_baseline_warning": (
            f"A predictor that always answers 'absent' scores roughly "
            f"{2 * rate / (1 + rate):.3f} absence-F1 on this slice "
            f"(absence rate {rate:.1%}). Report MCC and presence-recall."
        ),
    }
