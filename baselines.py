"""
Baselines for the absence query.

Two arms, deliberately:

  A. GLOBAL TOP-K RAG -- what production RAG actually does. Retrieve k chunks for
     the query, reason over them. Structurally cannot enumerate absences across a
     corpus: it never sees the documents that fail to match.

  B. PER-DOCUMENT SCAN -- ask the extraction question of every document at query
     time. This DOES work. Including it is what makes the comparison honest:
     the derivation store's claim is not "RAG cannot answer absence questions",
     it is "the only way to answer them is write-time work, so pay for it once
     instead of on every query". Arm B pays N calls per query; the store pays
     N once and ~1 per change.
"""

import math
import re
from typing import Dict, List, Set, Tuple

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # offline rig: minimal TF-IDF for global top-k baseline only

    _EN_STOP = frozenset(
        "a about above after again against all am an and any are aren't as at be because been "
        "before being below between both but by can't cannot could couldn't did didn't do does "
        "doesn't doing don't down during each few for from further had hadn't has hasn't have "
        "haven't having he he'd he'll he's her here here's hers herself him himself his how "
        "how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most "
        "mustn't my myself no nor not of off on once only or other ought our ours ourselves out "
        "over own same shan't she she'd she'll she's should shouldn't so some such than that "
        "that's the their theirs them themselves then there there's these they they'd they'll "
        "they're they've this those through to too under until up very was wasn't we we'd we'll "
        "we're we've were weren't what what's when when's where where's which while who who's "
        "whom why why's with won't would wouldn't you you'd you'll you're you've your yours "
        "yourself yourselves".split()
    )

    def _ngrams(text: str, ngram_range: Tuple[int, int], stop: frozenset) -> List[str]:
        tokens = [t for t in re.findall(r"\b[a-z]+\b", text.lower()) if t not in stop]
        out: List[str] = []
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(max(0, len(tokens) - n + 1)):
                out.append(" ".join(tokens[i : i + n]))
        return out

    class TfidfVectorizer:  # noqa: N801 — sklearn-compatible name
        def __init__(self, stop_words="english", ngram_range=(1, 2)):
            self.stop_words = _EN_STOP if stop_words == "english" else frozenset(stop_words or [])
            self.ngram_range = ngram_range
            self._idf: Dict[str, float] = {}

        def _fit_idf(self, docs: List[str]) -> None:
            df: Dict[str, int] = {}
            for doc in docs:
                for term in set(_ngrams(doc, self.ngram_range, self.stop_words)):
                    df[term] = df.get(term, 0) + 1
            n = len(docs)
            self._idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}

        def _vec(self, doc: str) -> Dict[str, float]:
            counts: Dict[str, int] = {}
            for term in _ngrams(doc, self.ngram_range, self.stop_words):
                counts[term] = counts.get(term, 0) + 1
            total = sum(counts.values()) or 1
            return {t: (c / total) * self._idf.get(t, 0.0) for t, c in counts.items()}

        def fit_transform(self, raw_docs: List[str]) -> List[Dict[str, float]]:
            self._fit_idf(raw_docs)
            return [self._vec(d) for d in raw_docs]

        def transform(self, raw_docs: List[str]) -> List[Dict[str, float]]:
            return [self._vec(d) for d in raw_docs]

    def cosine_similarity(q_rows: List[Dict[str, float]], doc_rows: List[Dict[str, float]]):
        sims = []
        for q in q_rows:
            q_norm = math.sqrt(sum(v * v for v in q.values())) or 1.0
            row_sims = []
            for d in doc_rows:
                dot = sum(q.get(k, 0.0) * v for k, v in d.items())
                d_norm = math.sqrt(sum(v * v for v in d.values())) or 1.0
                row_sims.append(dot / (q_norm * d_norm))
            sims.append(row_sims)
        return sims


def chunk(text: str, size: int = 400) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) < size:
            cur = f"{cur}\n{p}".strip()
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out


class GlobalTopKRAG:
    """Arm A. Cost: 1 retrieval + 1 reasoning call per query."""

    def __init__(self, docs, k: int = 20):
        self.k = k
        self.chunks, self.owner = [], []
        for doc_id, d in docs.items():
            for c in chunk(d.text):
                self.chunks.append(c)
                self.owner.append(doc_id)
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform(self.chunks)
        self.all_docs = set(docs)

    def docs_missing(self, clause: str, detector) -> List[str]:
        query = clause.replace("_", " ")
        qv = self.vec.transform([query])
        sims_raw = cosine_similarity(qv, self.matrix)
        sims = sims_raw[0] if isinstance(sims_raw, list) else sims_raw.ravel()
        if hasattr(sims, "argsort"):
            top = sims.argsort()[::-1][: self.k]
        else:
            top = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[: self.k]

        seen_docs: Set[str] = set()
        has_clause: Set[str] = set()
        for i in top:
            doc_id = self.owner[i]
            seen_docs.add(doc_id)
            if detector(self.chunks[i]).get(clause):
                has_clause.add(doc_id)

        # A RAG system can only speak about what it retrieved. Documents it never
        # saw are simply invisible -- it cannot assert absence for them.
        return sorted(seen_docs - has_clause)

    def cost_per_query(self, n_docs: int) -> Dict[str, int]:
        return {"extraction_calls": 1, "note": "reasons only over k retrieved chunks"}


class PerDocumentScanRAG:
    """Arm B. Correct, but pays a full extraction pass on every single query."""

    def __init__(self, docs):
        self.docs = docs

    def docs_missing(self, clause: str, detector) -> List[str]:
        out = []
        for doc_id, d in self.docs.items():
            if not detector(d.text).get(clause):
                out.append(doc_id)
        return sorted(out)

    def cost_per_query(self, n_docs: int) -> Dict[str, int]:
        return {"extraction_calls": n_docs, "note": "write-time work paid at query time"}
