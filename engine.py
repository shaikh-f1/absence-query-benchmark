"""
Rollup tree + incremental maintenance.

Full build:        N extraction calls + ~N/(b-1) merges
Incremental (1 doc): 1 extraction call + ceil(log_b N) merges  (if it propagates at all)

Two cost gates from the architecture doc, both implemented and both measured:
  1. semantic no-op detection  -- change event never reaches extraction
  2. propagation halting       -- extraction ran but output unchanged, so no merges
"""

import math
import re
from typing import Dict, List, Tuple

from operators import leaf_to_partial, make_extraction_op, rollup_merge_op
from store import DerivationStore

BRANCHING = 8


def _is_semantic_noop(old: str, new: str) -> bool:
    """
    Cheap classifier stand-in. Normalises whitespace, case and punctuation-only
    differences. Real systems use a small model here; the point of measuring it
    separately is that missed meaning-changes silently serve stale answers.
    """
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return norm(old) == norm(new)


class RollupEngine:
    def __init__(self, detector, branching: int = BRANCHING):
        self.store = DerivationStore()
        self.detector = detector
        self.extract_op = make_extraction_op(detector)
        self.branching = branching
        self.leaf_partial_key: Dict[str, str] = {}   # doc_id -> partial node key
        self.tree: List[List[str]] = []              # levels of node keys, [0] = leaves
        self.doc_order: List[str] = []

    # ---------------- build ----------------

    def _extract_and_partial(self, doc_id: str, src_key: str) -> str:
        ex = self.store.get_or_compute(
            logical_id=f"extract:{doc_id}",
            operator=f"clause_extract:{self.detector.name}",
            operator_version=self.detector.version,
            input_keys=[src_key],
            compute=self.extract_op,
            cost="extraction",
        )
        part = self.store.get_or_compute(
            logical_id=f"partial:{doc_id}",
            operator="leaf_partial",
            operator_version="1",
            input_keys=[ex.key],
            compute=lambda ins: leaf_to_partial(doc_id, ins[0]),
            cost="merge",
        )
        self.leaf_partial_key[doc_id] = part.key
        return part.key

    def build(self, docs) -> dict:
        self.doc_order = sorted(docs)
        leaves = []
        for doc_id in self.doc_order:
            d = docs[doc_id]
            src = self.store.put_source(doc_id, d.text, d.acl_band)
            leaves.append(self._extract_and_partial(doc_id, src.key))

        self.tree = [leaves]
        level, idx = leaves, 0
        while len(level) > 1:
            nxt = []
            for i in range(0, len(level), self.branching):
                group = level[i:i + self.branching]
                node = self.store.get_or_compute(
                    logical_id=f"rollup:L{idx+1}:{i//self.branching}",
                    operator="missing_clause_rollup",
                    operator_version="1",
                    input_keys=group,
                    compute=rollup_merge_op,
                    cost="merge",
                )
                nxt.append(node.key)
            self.tree.append(nxt)
            level = nxt
            idx += 1
        return self.root()

    def root(self) -> dict:
        return self.store.nodes[self.tree[-1][0]].output

    # ---------------- maintenance ----------------

    def apply_change(self, doc_id: str, new_text: str, acl_band: str) -> dict:
        """
        Apply one change event and return what it cost.
        """
        old_key = self.store.by_logical_id[f"src:{doc_id}"]
        old_text = self.store.nodes[old_key].output

        if _is_semantic_noop(old_text, new_text):
            self.store.counters.noop_filtered += 1
            return {"path": "noop", "extractions": 0, "merges": 0}

        before = self.store.counters.as_dict()

        old_partial = self.leaf_partial_key[doc_id]
        old_extract_out = self.store.nodes[
            self.store.by_logical_id[f"extract:{doc_id}"]
        ].output

        src = self.store.put_source(doc_id, new_text, acl_band)
        new_partial = self._extract_and_partial(doc_id, src.key)

        new_extract_out = self.store.nodes[
            self.store.by_logical_id[f"extract:{doc_id}"]
        ].output

        if new_extract_out == old_extract_out:
            self.store.counters.propagation_halted += 1
            after = self.store.counters.as_dict()
            return {
                "path": "halted",
                "extractions": after["extraction_calls"] - before["extraction_calls"],
                "merges": after["merge_calls"] - before["merge_calls"],
            }

        # rebuild only the path from this leaf to the root
        pos = self.doc_order.index(doc_id)
        self.tree[0][pos] = new_partial
        level_idx, i = 0, pos
        while level_idx + 1 < len(self.tree):
            group_i = i // self.branching
            start = group_i * self.branching
            group = self.tree[level_idx][start:start + self.branching]
            node = self.store.get_or_compute(
                logical_id=f"rollup:L{level_idx+1}:{group_i}",
                operator="missing_clause_rollup",
                operator_version="1",
                input_keys=group,
                compute=rollup_merge_op,
                cost="merge",
            )
            self.tree[level_idx + 1][group_i] = node.key
            level_idx += 1
            i = group_i

        after = self.store.counters.as_dict()
        return {
            "path": "propagated",
            "extractions": after["extraction_calls"] - before["extraction_calls"],
            "merges": after["merge_calls"] - before["merge_calls"],
        }

    def full_rebuild_cost(self, n_docs: int) -> Tuple[int, int]:
        """What a naive re-crawl would cost: every extraction + every merge."""
        merges = n_docs  # leaf partials
        level = n_docs
        while level > 1:
            level = math.ceil(level / self.branching)
            merges += level
        return n_docs, merges

    # ---------------- absence query ----------------

    def docs_missing(self, clause: str, user_band: str = "restricted") -> List[str]:
        """The headline query: answered by a filter over precomputed state."""
        root_key = self.tree[-1][0]
        if not self.store.visible_to(root_key, user_band):
            return []
        return self.root()["absent_docs"].get(clause, [])
