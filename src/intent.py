"""Query-intent analysis: safety hazards, compound questions, comparisons.

Port addition. Three behaviours the original pipeline could not express, each
one observed as a real failure against the NOVAHAUS corpus:

* **Safety.** ``My smart switch is sparking and there is a burning smell`` was
  refused with the generic no-match string. The corpus dedicates a section
  (``NOVAHAUS_FAQ.md`` §13) to exactly this report type, instructing support to
  escalate. Hazard detection must therefore run *before* the similarity gate,
  because "we could not find a product page for your burning smell" is a bad
  answer to a safety question.
* **Compound questions.** ``what are the shipping costs and how long does
  delivery take?`` only ever answered half of itself, because the answering
  engine selects a single context item and cannot merge two.
* **Comparisons.** ``compare the Glow Bulb A19 and the Glow Strip 2M`` returned
  one product's socket specification and never mentioned the second product.

This module only *classifies* a query — it never answers one. Everything it
detects is answered by :mod:`src.answerer` strictly from retrieved catalog text.
"""

from __future__ import annotations

import re
from typing import List

from .lexical import meaningful_words

# --------------------------------------------------------------------------- #
# Safety hazards
# --------------------------------------------------------------------------- #

#: Unambiguous electrical-safety signals. Any one of these is enough on its own,
#: mirroring the escalation trigger list in ``NOVAHAUS_FAQ.md`` §13
#: ("electrical shock, burning smell, smoke, melting, exposed wiring, or
#: repeated circuit trips").
_STRONG_HAZARD_RE = re.compile(
    r"\b(?:"
    r"spark(?:s|ing|ed)?|"
    r"smok(?:e|es|ing|y)|"
    r"melt(?:s|ing|ed)?|"
    r"scorch(?:es|ed|ing)?|"
    r"charred|"
    r"burning\s+(?:smell|odou?r|scent)|smell(?:s|ing)?\s+burning|"
    r"electric(?:al)?\s+shock|shock(?:ed|ing)?\s+me|"
    r"electrocut\w*|"
    r"exposed\s+wir\w*|"
    r"circuit\s+trip\w*|"
    r"trip(?:s|ping)?\s+the\s+(?:breaker|circuit)|"
    r"burnt\s+smell"
    r")\b",
    re.IGNORECASE,
)

#: Weak signals that only count as a hazard alongside a device/electrical word,
#: so "the app is hot garbage" or "fire up the hub" stay ordinary questions.
_WEAK_HAZARD_RE = re.compile(
    r"\b(?:burn\w*|hot|overheat\w*|fire|buzz\w*|pop\w*|crackl\w*|flame\w*)\b",
    re.IGNORECASE,
)

_DEVICE_RE = re.compile(
    r"\b(?:switch|plug|bulb|light\w*|strip|ceiling|camera|cam|doorbell|sensor|"
    r"hub|device|charger|adapter|socket|outlet|wir\w*|cable|mains|power|"
    r"electric\w*)\b",
    re.IGNORECASE,
)


def hazard_intent(query: str) -> str:
    """Return the hazard term found in ``query``, or an empty string.

    A strong signal matches alone; a weak one must appear next to a
    device/electrical word so that ordinary questions are not escalated.
    """
    strong = _STRONG_HAZARD_RE.search(query)
    if strong:
        return strong.group(0)
    weak = _WEAK_HAZARD_RE.search(query)
    if weak and _DEVICE_RE.search(query):
        return weak.group(0)
    return ""


# --------------------------------------------------------------------------- #
# Compound questions and comparisons
# --------------------------------------------------------------------------- #

#: A lead-in that announces a comparison instead of carrying meaning itself.
_COMPARE_LEAD_RE = re.compile(
    r"^\s*(?:please\s+|can you\s+|could you\s+)?(?:"
    r"compare|contrast|"
    r"what(?:'s| is) the difference between|difference(?:s)? between|"
    r"which is better[:,]?"
    r")\s+",
    re.IGNORECASE,
)

_VERSUS_RE = re.compile(r"\s+(?:vs\.?|versus)\s+", re.IGNORECASE)
_CONJUNCTION_RE = re.compile(r"\s+(?:and|as well as|plus|&)\s+", re.IGNORECASE)
#: A conjunction that merely opens a follow-up ("and what about X?").
_FOLLOWUP_LEAD_RE = re.compile(
    r"^\s*(?:and|also|then|what about|how about|what of)\s+", re.IGNORECASE
)

#: Each side of a split must carry at least this many content words.
MIN_PART_WORDS = 2
#: Upper bound on stitched sub-answers; longer chains read worse than one refusal.
MAX_COMPOUND_PARTS = 4


def is_comparison(query: str) -> bool:
    """Whether ``query`` explicitly asks for a comparison."""
    return bool(_COMPARE_LEAD_RE.search(query) or _VERSUS_RE.search(query))


def split_compound(query: str) -> List[str]:
    """Split a compound question into independent sub-questions.

    Returns an empty list when ``query`` is atomic — which is the safe default,
    since every returned part is answered separately.

    Splitting is deliberately conservative: it happens only on an explicit
    comparison lead, ``vs``/``versus``, or a coordinating conjunction, and every
    resulting part must still carry at least :data:`MIN_PART_WORDS` content
    words. Anything less and the query is treated as atomic.
    """
    body = _COMPARE_LEAD_RE.sub("", str(query).strip())
    if _VERSUS_RE.search(body):
        raw_parts = _VERSUS_RE.split(body)
    elif _CONJUNCTION_RE.search(body):
        raw_parts = _CONJUNCTION_RE.split(body)
    else:
        return []

    parts: List[str] = []
    for part in raw_parts:
        cleaned = _FOLLOWUP_LEAD_RE.sub("", part.strip()).strip(" \t?.,!")
        if len(meaningful_words(cleaned)) < MIN_PART_WORDS:
            # One unusable side means the whole split is unsafe.
            return []
        parts.append(cleaned)

    if len(parts) < 2:
        return []
    return parts[:MAX_COMPOUND_PARTS]
