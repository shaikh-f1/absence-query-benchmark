"""
Synthetic contract corpus with GROUND TRUTH and controlled linguistic difficulty.

The critical design decision here: clauses are realised at three difficulty tiers,
and the corpus contains DISTRACTORS -- text that mentions a clause topic without
actually establishing the clause. Without both of these, an extraction operator
scores ~100% and the whole experiment proves nothing.

Ground truth records, per document:
  - which clause types are actually PRESENT (and at what difficulty tier)
  - which clause types are ABSENT
  - which clause types appear only as a DISTRACTOR (topic mentioned, no obligation)
  - planted contradictions
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Set

CLAUSE_TYPES = [
    "liability_cap",
    "termination_notice",
    "indemnification",
    "payment_terms",
    "confidentiality",
    "governing_law",
    "force_majeure",
    "data_protection",
    "assignment",
    "warranty",
]

# ---------------------------------------------------------------------------
# Clause realisations by difficulty tier.
#   tier 0 (literal)   : canonical heading + standard boilerplate
#   tier 1 (paraphrase): no heading, reworded, same legal effect
#   tier 2 (oblique)   : indirect, cross-referenced, or buried in another clause
# ---------------------------------------------------------------------------

REALISATIONS: Dict[str, Dict[int, List[str]]] = {
    "liability_cap": {
        0: [
            "LIMITATION OF LIABILITY. In no event shall the aggregate liability of "
            "either party exceed the total fees paid under this Agreement in the twelve "
            "(12) months preceding the claim.",
            "LIMITATION OF LIABILITY. Each party's total liability arising out of this "
            "Agreement shall not exceed USD 500,000.",
        ],
        1: [
            "Neither party will be answerable to the other for amounts greater than the "
            "sums actually invoiced and settled during the preceding annual period.",
            "The parties agree that recovery under this contract is bounded by the "
            "consideration exchanged hereunder and shall go no further.",
        ],
        2: [
            "Financial exposure of the parties is constrained in the manner described in "
            "Schedule C, paragraph 4, which the parties acknowledge as binding.",
            "Save for the ceiling recorded in the Commercial Terms annexure, the parties "
            "waive claims beyond that recorded figure.",
        ],
    },
    "termination_notice": {
        0: [
            "TERMINATION. Either party may terminate this Agreement upon thirty (30) "
            "days prior written notice to the other party.",
        ],
        1: [
            "Should either side wish to bring this arrangement to an end, it must inform "
            "the other in writing a full month in advance.",
        ],
        2: [
            "Exit from this engagement follows the notice period stipulated in the "
            "Master Services Framework referenced in the preamble.",
        ],
    },
    "indemnification": {
        0: [
            "INDEMNIFICATION. Supplier shall indemnify, defend and hold harmless "
            "Customer from any third-party claims arising from Supplier's negligence.",
        ],
        1: [
            "The Supplier will cover and defend the Customer against outside claims "
            "brought as a result of the Supplier's failure to take due care.",
        ],
        2: [
            "Protective obligations in respect of third-party actions rest with the "
            "party whose conduct gave rise to them, as further detailed in Annex 2.",
        ],
    },
    "payment_terms": {
        0: [
            "PAYMENT. Customer shall pay all undisputed invoices within thirty (30) days "
            "of receipt.",
        ],
        1: [
            "Invoices not in dispute fall due one month from the date they reach the "
            "Customer's accounts department.",
        ],
        2: [
            "Settlement timing is governed by the schedule appended as Exhibit A.",
        ],
    },
    "confidentiality": {
        0: [
            "CONFIDENTIALITY. Each party shall keep confidential all non-public "
            "information disclosed by the other party for a period of five (5) years.",
        ],
        1: [
            "Information shared between the parties that is not already public must be "
            "kept private for half a decade following disclosure.",
        ],
        2: [
            "Non-public material exchanged hereunder is subject to the protective regime "
            "described in the parties' existing mutual NDA, incorporated by reference.",
        ],
    },
    "governing_law": {
        0: [
            "GOVERNING LAW. This Agreement shall be governed by the laws of the State of "
            "Delaware.",
        ],
        1: [
            "Delaware law applies to the interpretation and enforcement of this contract.",
        ],
        2: [
            "Questions of construction fall to be decided under the legal system of the "
            "jurisdiction in which the Customer maintains its registered office.",
        ],
    },
    "force_majeure": {
        0: [
            "FORCE MAJEURE. Neither party shall be liable for delays caused by events "
            "beyond its reasonable control, including acts of God, war, or pandemic.",
        ],
        1: [
            "Delays brought about by circumstances neither side could reasonably manage "
            "-- natural disaster, armed conflict, widespread illness -- excuse performance.",
        ],
        2: [
            "Performance is suspended during the categories of disruption enumerated at "
            "clause 14.3 of the framework agreement.",
        ],
    },
    "data_protection": {
        0: [
            "DATA PROTECTION. Supplier shall process personal data only as instructed by "
            "Customer and in compliance with applicable data protection legislation.",
        ],
        1: [
            "Any handling of individuals' personal information by the Supplier must follow "
            "the Customer's directions and remain within the bounds of relevant privacy law.",
        ],
        2: [
            "Processing activities are constrained by the Data Processing Addendum executed "
            "concurrently with this Agreement.",
        ],
    },
    "assignment": {
        0: [
            "ASSIGNMENT. Neither party may assign this Agreement without the prior "
            "written consent of the other party.",
        ],
        1: [
            "Transfer of this contract to another entity requires the written blessing of "
            "the non-transferring side.",
        ],
        2: [
            "Change of contracting party is permitted only on the conditions set out in "
            "the Corporate Transactions annex.",
        ],
    },
    "warranty": {
        0: [
            "WARRANTY. Supplier warrants that the Services will be performed in a "
            "professional and workmanlike manner.",
        ],
        1: [
            "The Supplier gives its assurance that the work will be carried out to a "
            "competent professional standard.",
        ],
        2: [
            "Quality undertakings are those, and only those, recorded in the Statement of "
            "Work incorporated herein.",
        ],
    },
}

# ---------------------------------------------------------------------------
# DISTRACTORS: text that mentions the clause topic but establishes NO obligation.
# These are the adversarial cases. A keyword or embedding-similarity extractor
# will fire on these; a correct extractor must not.
# ---------------------------------------------------------------------------

DISTRACTORS: Dict[str, List[str]] = {
    "liability_cap": [
        "The parties have discussed the question of liability limits and have elected not "
        "to include any ceiling on damages in this Agreement.",
        "WHEREAS the Customer sought a limitation of liability during negotiations, the "
        "final commercial position reflects no such cap.",
    ],
    "termination_notice": [
        "The parties note that termination was the subject of extensive negotiation; no "
        "notice mechanism was ultimately agreed.",
    ],
    "indemnification": [
        "Indemnification was raised in preliminary discussions but is expressly excluded "
        "from the scope of this Agreement.",
    ],
    "payment_terms": [
        "Payment matters, including timing, are to be settled separately and are not "
        "addressed by this instrument.",
    ],
    "confidentiality": [
        "The parties acknowledge that confidentiality obligations, if any, arise outside "
        "this Agreement and are not created by it.",
    ],
    "governing_law": [
        "Choice of governing law remains open and the parties reserve their positions.",
    ],
    "force_majeure": [
        "No force majeure relief is provided under this Agreement notwithstanding the "
        "parties' discussion of such events.",
    ],
    "data_protection": [
        "Data protection compliance is the subject of a separate instrument not yet "
        "executed; nothing herein imposes processing obligations.",
    ],
    "assignment": [
        "Assignment rights were considered and the parties have deliberately left the "
        "position unstated.",
    ],
    "warranty": [
        "ALL WARRANTIES, EXPRESS OR IMPLIED, ARE DISCLAIMED. Nothing in this Agreement "
        "constitutes a warranty as to the Services.",
    ],
}

FILLER = [
    "The parties have entered into this Agreement as of the date last signed below.",
    "Headings are for convenience only and do not affect interpretation.",
    "This Agreement constitutes the entire understanding between the parties.",
    "Any amendment must be in writing and signed by authorised representatives.",
    "Notices shall be sent to the addresses recorded in the signature block.",
    "The parties are independent contractors and nothing herein creates a partnership.",
    "If any provision is held unenforceable, the remainder shall continue in effect.",
    "Counterparts may be executed electronically and shall be deemed originals.",
]


@dataclass
class GroundTruth:
    doc_id: str
    present: Dict[str, int] = field(default_factory=dict)   # clause -> tier
    absent: Set[str] = field(default_factory=set)
    distractor_only: Set[str] = field(default_factory=set)  # subset of absent
    contradictions: List[str] = field(default_factory=list)


@dataclass
class Document:
    doc_id: str
    revision: int
    text: str
    acl_band: str


def _tier_for(rng: random.Random, tier_mix) -> int:
    r = rng.random()
    c = 0.0
    for tier, p in enumerate(tier_mix):
        c += p
        if r <= c:
            return tier
    return len(tier_mix) - 1


def generate(
    n_docs: int = 300,
    seed: int = 7,
    tier_mix=(0.5, 0.3, 0.2),
    p_absent: float = 0.25,
    p_distractor_given_absent: float = 0.4,
    p_contradiction: float = 0.08,
    acl_bands=("public", "internal", "restricted"),
):
    """Returns (documents, ground_truth) both keyed by doc_id."""
    rng = random.Random(seed)
    docs: Dict[str, Document] = {}
    truth: Dict[str, GroundTruth] = {}

    for i in range(n_docs):
        doc_id = f"contract_{i:04d}"
        gt = GroundTruth(doc_id=doc_id)
        body: List[str] = [
            f"VENDOR SERVICES AGREEMENT -- {doc_id.upper()}",
            f"This Agreement is made between Customer Holdings Inc. and Supplier "
            f"Entity {i % 37 + 1} Ltd.",
        ]

        for clause in CLAUSE_TYPES:
            if rng.random() < p_absent:
                gt.absent.add(clause)
                if rng.random() < p_distractor_given_absent:
                    gt.distractor_only.add(clause)
                    body.append(rng.choice(DISTRACTORS[clause]))
                continue

            tier = _tier_for(rng, tier_mix)
            pool = REALISATIONS[clause].get(tier) or REALISATIONS[clause][0]
            body.append(rng.choice(pool))
            gt.present[clause] = tier

            # planted contradiction: present clause plus a conflicting statement
            if rng.random() < p_contradiction:
                body.append(
                    f"Notwithstanding the foregoing, the parties agree that no "
                    f"{clause.replace('_', ' ')} obligation shall apply."
                )
                gt.contradictions.append(clause)

        for _ in range(rng.randint(3, 7)):
            body.append(rng.choice(FILLER))
        rng.shuffle(body[2:])

        docs[doc_id] = Document(
            doc_id=doc_id,
            revision=1,
            text="\n\n".join(body),
            acl_band=rng.choice(acl_bands),
        )
        truth[doc_id] = gt

    return docs, truth


def summarise(docs, truth):
    from collections import Counter
    tiers = Counter()
    absent = Counter()
    distract = Counter()
    for gt in truth.values():
        for c, t in gt.present.items():
            tiers[t] += 1
        for c in gt.absent:
            absent[c] += 1
        for c in gt.distractor_only:
            distract[c] += 1
    return {
        "n_docs": len(docs),
        "avg_chars": sum(len(d.text) for d in docs.values()) // len(docs),
        "clause_instances_by_tier": dict(sorted(tiers.items())),
        "total_absences": sum(absent.values()),
        "absences_with_distractor_text": sum(distract.values()),
        "docs_with_contradiction": sum(1 for gt in truth.values() if gt.contradictions),
    }
