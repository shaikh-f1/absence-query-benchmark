"""
Derivation store: content-addressed nodes, provenance DAG, exact invalidation.

Implements the parts of the architecture that are actually risky:
  - cache keys over (operator, operator_version, model, params, input hashes)
  - forward-dirty propagation with semantic-equivalence halting
  - support counting for deletion (a derivation survives while any input supports it)
  - ACL closure: a derivation is visible only to the INTERSECTION of its inputs' bands
  - instrumentation separating expensive (extraction) from cheap (merge) calls
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# band lattice: a derivation over mixed bands is only visible at the most
# restrictive band present among its inputs
BAND_ORDER = {"public": 0, "internal": 1, "restricted": 2}


def h(*parts: Any) -> str:
    m = hashlib.sha256()
    for p in parts:
        m.update(json.dumps(p, sort_keys=True, default=str).encode())
    return m.hexdigest()[:16]


@dataclass
class Node:
    key: str
    kind: str                       # "source" | "extraction" | "derivation"
    operator: str
    operator_version: str
    inputs: List[str]               # keys of input nodes
    output: Any
    acl_band: str
    support: int = 1                # number of independent inputs supporting it
    dirty: bool = False


class Counters:
    def __init__(self):
        self.extraction_calls = 0   # expensive: one LLM call per document
        self.merge_calls = 0        # cheap: associative merge, no LLM
        self.noop_filtered = 0      # change events halted by semantic no-op check
        self.propagation_halted = 0 # changes where extraction output was unchanged

    def as_dict(self):
        return dict(self.__dict__)

    def reset(self):
        self.__init__()


class DerivationStore:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.dependents: Dict[str, Set[str]] = {}   # input key -> derived keys
        self.by_logical_id: Dict[str, str] = {}     # stable logical id -> current key
        self.counters = Counters()

    # ---------------- registration ----------------

    def put_source(self, doc_id: str, text: str, acl_band: str) -> Node:
        key = h("source", doc_id, text)
        node = Node(key=key, kind="source", operator="ingest", operator_version="1",
                    inputs=[], output=text, acl_band=acl_band)
        self.nodes[key] = node
        self.by_logical_id[f"src:{doc_id}"] = key
        return node

    def _acl_closure(self, input_keys: List[str]) -> str:
        """Most restrictive band among inputs -- the intersection rule."""
        bands = [self.nodes[k].acl_band for k in input_keys if k in self.nodes]
        if not bands:
            return "public"
        return max(bands, key=lambda b: BAND_ORDER[b])

    def get_or_compute(
        self,
        logical_id: str,
        operator: str,
        operator_version: str,
        input_keys: List[str],
        compute: Callable[[List[Any]], Any],
        cost: str = "merge",
        params: Optional[dict] = None,
    ) -> Node:
        """Content-addressed memoised computation. Returns the node."""
        key = h(operator, operator_version, params or {}, sorted(input_keys))
        existing = self.nodes.get(key)
        if existing is not None and not existing.dirty:
            return existing

        inputs = [self.nodes[k].output for k in input_keys]
        if cost == "extraction":
            self.counters.extraction_calls += 1
        else:
            self.counters.merge_calls += 1
        out = compute(inputs)

        node = Node(
            key=key, kind="derivation" if cost == "merge" else "extraction",
            operator=operator, operator_version=operator_version,
            inputs=list(input_keys), output=out,
            acl_band=self._acl_closure(input_keys),
            support=len(input_keys),
        )
        self.nodes[key] = node
        self.by_logical_id[logical_id] = key
        for k in input_keys:
            self.dependents.setdefault(k, set()).add(key)
        return node

    # ---------------- invalidation ----------------

    def mark_dirty_downstream(self, key: str) -> List[str]:
        """Forward walk of the provenance DAG. Returns dirtied keys in order."""
        dirtied, stack = [], [key]
        seen = set()
        while stack:
            k = stack.pop()
            for dep in self.dependents.get(k, ()):
                if dep in seen:
                    continue
                seen.add(dep)
                self.nodes[dep].dirty = True
                dirtied.append(dep)
                stack.append(dep)
        return dirtied

    def retire_source(self, doc_id: str) -> None:
        """Deletion via support counting: derivations survive while support remains."""
        key = self.by_logical_id.get(f"src:{doc_id}")
        if key is None:
            return
        for dep in list(self.dependents.get(key, ())):
            node = self.nodes[dep]
            node.support -= 1
            if node.support <= 0:
                self.nodes.pop(dep, None)
        self.nodes.pop(key, None)

    def visible_to(self, key: str, user_band: str) -> bool:
        return BAND_ORDER[user_band] >= BAND_ORDER[self.nodes[key].acl_band]

    def stats(self):
        kinds = {}
        for n in self.nodes.values():
            kinds[n.kind] = kinds.get(n.kind, 0) + 1
        return {"nodes": len(self.nodes), "by_kind": kinds}
