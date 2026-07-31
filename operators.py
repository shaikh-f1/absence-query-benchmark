"""
Operators.

Class M (per-source extraction): pluggable detectors of deliberately different
quality, so extraction error can be measured SEPARATELY from architecture error.
This separation is the whole point -- an absence answer is only as good as
extraction recall, and a false absence is a confidently wrong answer.

Class A (associative aggregate): the missing-clause rollup. Designed to be
mergeable so it maintains in O(log n) rather than O(n).
"""

import re
from typing import Dict, List, Set

from corpus import CLAUSE_TYPES

# ---------------------------------------------------------------------------
# Detectors (Class M)
# ---------------------------------------------------------------------------

# Keyword surface forms -- fires on topic mention, not on obligation.
KEYWORDS = {
    "liability_cap": ["liability", "limitation of liability", "exceed", "ceiling"],
    "termination_notice": ["terminat", "notice period", "bring this arrangement to an end"],
    "indemnification": ["indemnif", "hold harmless", "defend"],
    "payment_terms": ["payment", "invoice", "due", "settlement"],
    "confidentiality": ["confidential", "non-public", "nda"],
    "governing_law": ["governing law", "governed by the laws", "law applies", "jurisdiction"],
    "force_majeure": ["force majeure", "acts of god", "beyond its reasonable control"],
    "data_protection": ["data protection", "personal data", "personal information", "processing"],
    "assignment": ["assign", "transfer of this contract", "change of contracting party"],
    "warranty": ["warrant", "assurance", "quality undertaking"],
}

# Negation / exclusion cues that mean the topic is mentioned but NOT established.
NEGATION_CUES = [
    "not to include", "no such cap", "no notice mechanism", "expressly excluded",
    "not addressed", "arise outside", "remains open", "no force majeure relief",
    "not yet executed", "deliberately left the position unstated", "are disclaimed",
    "nothing herein", "nothing in this agreement", "shall apply",
    "was the subject of extensive negotiation", "reserve their positions",
    "settled separately", "elected not",
]


class KeywordDetector:
    """Weak baseline. Fires on topic mention; blind to negation. Fast, no LLM."""
    name = "keyword"
    version = "1"

    def __call__(self, text: str) -> Dict[str, bool]:
        low = text.lower()
        return {c: any(k in low for k in KEYWORDS[c]) for c in CLAUSE_TYPES}


class NegationAwareDetector:
    """
    Stronger heuristic: finds the sentence(s) mentioning the topic and rejects
    the clause if the local context carries an exclusion cue. Stands in for a
    competent small-model extractor without needing an API key.
    """
    name = "negation_aware"
    version = "2"

    def __call__(self, text: str) -> Dict[str, bool]:
        sentences = [s.strip().lower() for s in re.split(r"\n\n|\. ", text) if s.strip()]
        out = {}
        for c in CLAUSE_TYPES:
            hit = False
            for s in sentences:
                if any(k in s for k in KEYWORDS[c]):
                    if not any(n in s for n in NEGATION_CUES):
                        hit = True
                        break
            out[c] = hit
        return out


class OracleDetector:
    """
    Perfect extraction, from ground truth. Isolates ARCHITECTURE error alone --
    if the derivation store is not exact under an oracle, the bug is structural.

    Only meaningful on whole documents. When handed a chunk (as the RAG baseline
    does) it falls back to the heuristic, since ground truth is per document.
    """
    name = "oracle"
    version = "1"

    def __init__(self, truth):
        self.truth = truth
        self.fallback = NegationAwareDetector()

    def __call__(self, text: str) -> Dict[str, bool]:
        m = re.search(r"CONTRACT_(\d+)", text)
        key = f"contract_{m.group(1)}" if m else None
        gt = self.truth.get(key) if key else None
        if gt is None:
            return self.fallback(text)
        return {c: (c in gt.present) for c in CLAUSE_TYPES}


class LLMDetector:
    """
    Real extraction via the Anthropic API. Requires ANTHROPIC_API_KEY.
    Included so the rig can be run with a genuine small-model extractor; the
    heuristics above let you run everything else offline first.
    """
    name = "llm"
    version = "1"

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.model = model
        self.version = f"1:{model}"

    def __call__(self, text: str) -> Dict[str, bool]:
        import json as _json
        import os
        import urllib.request

        prompt = (
            "For each clause type, answer true only if this contract actually "
            "ESTABLISHES that obligation. Answer false if the topic is merely "
            "mentioned, discussed, disclaimed, excluded, or deferred elsewhere.\n\n"
            f"Clause types: {', '.join(CLAUSE_TYPES)}\n\n"
            f"Contract:\n{text}\n\n"
            "Reply with JSON only: an object mapping each clause type to true or false."
        )
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=_json.dumps({
                "model": self.model,
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            }).encode(),
            headers={
                "content-type": "application/json",
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req) as r:
            body = _json.loads(r.read())
        raw = "".join(b.get("text", "") for b in body["content"])
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        parsed = _json.loads(raw)
        return {c: bool(parsed.get(c, False)) for c in CLAUSE_TYPES}


# ---------------------------------------------------------------------------
# Class M operator: per-document extraction -> leaf summary
# ---------------------------------------------------------------------------

def make_extraction_op(detector):
    def op(inputs: List[str]):
        text = inputs[0]
        found = detector(text)
        return {
            "present": sorted([c for c, v in found.items() if v]),
            "absent": sorted([c for c, v in found.items() if not v]),
        }
    return op


# ---------------------------------------------------------------------------
# Class A operator: associative merge of clause-presence counts
# ---------------------------------------------------------------------------

def leaf_to_partial(doc_id: str, extraction: dict) -> dict:
    """Convert one extraction into a mergeable partial aggregate."""
    return {
        "n_docs": 1,
        "present_counts": {c: 1 for c in extraction["present"]},
        "absent_docs": {c: [doc_id] for c in extraction["absent"]},
    }


def merge_partials(partials: List[dict]) -> dict:
    """
    Associative, commutative merge. This property is what buys O(log n)
    maintenance -- a non-mergeable formulation would force full rebuild.
    """
    out = {"n_docs": 0, "present_counts": {}, "absent_docs": {}}
    for p in partials:
        out["n_docs"] += p["n_docs"]
        for c, v in p["present_counts"].items():
            out["present_counts"][c] = out["present_counts"].get(c, 0) + v
        for c, docs in p["absent_docs"].items():
            out["absent_docs"].setdefault(c, []).extend(docs)
    for c in out["absent_docs"]:
        out["absent_docs"][c] = sorted(set(out["absent_docs"][c]))
    return out


def rollup_merge_op(inputs: List[dict]) -> dict:
    return merge_partials(inputs)
