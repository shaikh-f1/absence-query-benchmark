"""
Test suite. Run:  pytest -q

These test the invariants that would silently corrupt results if broken --
not the happy path. Each maps to a claim in the architecture doc.
"""

import pytest

import corpus
from baselines import GlobalTopKRAG, PerDocumentScanRAG
from engine import RollupEngine, _is_semantic_noop
from operators import (KeywordDetector, NegationAwareDetector, OracleDetector,
                       merge_partials)
from store import BAND_ORDER, DerivationStore


@pytest.fixture(scope="module")
def small():
    docs, truth = corpus.generate(n_docs=48, seed=3)
    return docs, truth


# ---------------------------------------------------------------- architecture

def test_oracle_extraction_gives_lossless_absence(small):
    """The load-bearing claim: with perfect extraction the store is exact."""
    docs, truth = small
    engine = RollupEngine(OracleDetector(truth))
    engine.build(docs)
    for clause in corpus.CLAUSE_TYPES:
        gold = {d for d in docs if clause in truth[d].absent}
        assert set(engine.docs_missing(clause)) == gold, clause


def test_merge_is_associative_and_commutative():
    """Class A maintenance is only valid if the merge really is associative."""
    a = {"n_docs": 1, "present_counts": {"x": 1}, "absent_docs": {"y": ["d1"]}}
    b = {"n_docs": 1, "present_counts": {"x": 1}, "absent_docs": {"y": ["d2"]}}
    c = {"n_docs": 1, "present_counts": {"z": 1}, "absent_docs": {}}
    assert merge_partials([merge_partials([a, b]), c]) == \
           merge_partials([a, merge_partials([b, c])])
    assert merge_partials([a, b, c]) == merge_partials([c, b, a])


def test_content_addressing_is_deterministic(small):
    """Same corpus, same keys -- otherwise the cache never hits."""
    docs, truth = small
    k1 = RollupEngine(OracleDetector(truth)); k1.build(docs)
    k2 = RollupEngine(OracleDetector(truth)); k2.build(docs)
    assert k1.tree[-1][0] == k2.tree[-1][0]


# ---------------------------------------------------------------- maintenance

def test_cosmetic_change_costs_nothing(small):
    docs, truth = small
    engine = RollupEngine(NegationAwareDetector())
    engine.build(docs)
    d = docs["contract_0000"]
    res = engine.apply_change(d.doc_id, d.text.replace(". ", ".  "), d.acl_band)
    assert res["path"] == "noop"
    assert res["extractions"] == 0 and res["merges"] == 0


def test_semantic_change_recomputes_log_n_not_n(small):
    """Blast radius must be the path to root, not the whole corpus."""
    docs, truth = small
    engine = RollupEngine(NegationAwareDetector())
    engine.build(docs)
    depth = len(engine.tree) - 1
    d = docs["contract_0000"]
    res = engine.apply_change(
        d.doc_id, d.text + "\n\nLIMITATION OF LIABILITY. Liability shall not "
                           "exceed the fees paid.", d.acl_band)
    assert res["extractions"] == 1
    assert res["merges"] <= depth + 1, "blast radius exceeded path-to-root"


def test_maintenance_beats_rebuild_by_50x():
    """
    Note the ceiling: incremental cost is 1 extraction per change, so the maximum
    achievable speedup IS the corpus size. A >=50x gate is unreachable below ~50
    documents -- this test therefore builds its own larger corpus rather than
    using the shared small fixture.
    """
    docs, truth = corpus.generate(n_docs=300, seed=5)
    engine = RollupEngine(NegationAwareDetector())
    engine.build(docs)
    fb_ext, _ = engine.full_rebuild_cost(len(docs))
    total = 0
    for i in range(20):
        did = f"contract_{i:04d}"
        d = docs[did]
        r = engine.apply_change(did, d.text + f"\n\nAMENDMENT {i}. Liability "
                                              f"shall not exceed fees paid.", d.acl_band)
        total += r["extractions"]
    speedup = (fb_ext * 20) / max(total, 1)
    assert speedup >= 50, f"speedup {speedup:.0f}x below gate"


def test_noop_detector_ignores_formatting_but_not_meaning():
    assert _is_semantic_noop("Hello,  World.", "hello world")
    assert not _is_semantic_noop("Liability is capped.", "Liability is not capped.")


# ---------------------------------------------------------------- ACL

def test_acl_closure_uses_intersection_not_union():
    """One restricted input must restrict the derived aggregate."""
    s = DerivationStore()
    a = s.put_source("a", "public text", "public")
    b = s.put_source("b", "secret text", "restricted")
    node = s.get_or_compute("agg", "merge", "1", [a.key, b.key],
                            lambda ins: {"n": len(ins)})
    assert node.acl_band == "restricted"
    assert not s.visible_to(node.key, "public")
    assert s.visible_to(node.key, "restricted")


def test_support_counting_retires_only_unsupported_derivations():
    s = DerivationStore()
    a = s.put_source("a", "x", "public")
    b = s.put_source("b", "y", "public")
    node = s.get_or_compute("agg", "merge", "1", [a.key, b.key], lambda i: len(i))
    s.retire_source("a")
    assert node.key in s.nodes, "derivation retired while still supported"
    s.retire_source("b")
    assert node.key not in s.nodes, "derivation survived with zero support"


# ---------------------------------------------------------------- baselines

def test_global_rag_fails_structurally_on_absence(small):
    docs, truth = small
    rag = GlobalTopKRAG(docs, k=20)
    det = NegationAwareDetector()
    gold = {d for d in docs if "liability_cap" in truth[d].absent}
    pred = set(rag.docs_missing("liability_cap", det))
    recall = len(pred & gold) / len(gold)
    assert recall < 0.35, "baseline unexpectedly strong -- check corpus size"


def test_store_ties_per_document_scan(small):
    """Accuracy parity is expected. The win is cost, not quality."""
    docs, truth = small
    det = NegationAwareDetector()
    engine = RollupEngine(det)
    engine.build(docs)
    scan = PerDocumentScanRAG(docs)
    for clause in corpus.CLAUSE_TYPES:
        assert set(engine.docs_missing(clause)) == set(scan.docs_missing(clause, det))


# ---------------------------------------------------------------- corpus

def test_distractors_defeat_the_keyword_extractor(small):
    """If this passes trivially, the corpus is not adversarial enough."""
    docs, truth = small
    kw = KeywordDetector()
    fired = total = 0
    for doc_id, gt in truth.items():
        pred = kw(docs[doc_id].text)
        for clause in gt.distractor_only:
            total += 1
            if pred[clause]:
                fired += 1
    assert total > 0, "corpus generated no distractors"
    assert fired / total > 0.8, "keyword extractor should be fooled by distractors"
