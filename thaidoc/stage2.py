"""Stage 2 — text-based refinement to the exact physical_type within a family.

Given the predicted visual family and the reader's output, resolve the exact
canonical type. Two paths:

  * type_hypotheses present (VLM-as-classifier): map the hypothesis to the
    nearest canonical label in the family.
  * text present (OCR/mock): score each candidate label in the family by text
    similarity to the reader text. The best score and the MARGIN to the runner-up
    drive confidence: a small margin (e.g. the subtype marker was unreadable)
    yields low confidence -> human review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from . import labels
from .readers.base import ReaderResult

# Below this best-similarity the read carries no usable signal -> abstain.
SCORE_FLOOR = 0.20


@dataclass
class Stage2Result:
    physical_type: Optional[str]
    score: float          # best similarity in [0,1]
    margin: float         # gap between best and 2nd-best similarity
    candidates_considered: int


def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def refine(family: str, reader: ReaderResult) -> Stage2Result:
    candidates = labels.types_by_family(family)
    if not candidates:
        return Stage2Result(None, 0.0, 0.0, 0)

    # VLM hypothesis path: snap to nearest canonical label in the family.
    if reader.type_hypotheses:
        hyp = reader.type_hypotheses[0][0]
        scored = sorted(
            ((c.physical_type, _sim(hyp, c.display_name)) for c in candidates),
            key=lambda t: t[1], reverse=True)
    elif reader.text is not None:
        text = reader.text

        def score_candidate(c) -> float:
            s = _sim(text, c.display_name)
            # Decisive bonus when the distinguishing subtype marker is actually
            # READABLE in the text (e.g. 'CODE B' or '"B"'). On adversarial
            # samples the marker was stripped, so no bonus -> small margin ->
            # human review. This is the crux: reading the code resolves subtypes.
            m = c.subtype_marker
            if m:
                em = re.escape(m)
                if re.search(rf'(?:\bCODE\s+{em}\b)|(?:"{em}")', text):
                    s += 0.45
            return min(s, 1.0)

        scored = sorted(((c.physical_type, score_candidate(c)) for c in candidates),
                        key=lambda t: t[1], reverse=True)
    else:
        return Stage2Result(None, 0.0, 0.0, len(candidates))

    best_type, best_score = scored[0]
    second = scored[1][1] if len(scored) > 1 else 0.0
    margin = best_score - second

    # Explicit abstention at the layer that knows the scores: if the best match
    # is too weak (empty/out-of-distribution read), return no type rather than an
    # arbitrary sort-order winner. Downstream routing then reports UNKNOWN.
    if best_score < SCORE_FLOOR:
        return Stage2Result(None, float(best_score), 0.0, len(candidates))
    return Stage2Result(best_type, float(best_score), float(margin), len(candidates))
