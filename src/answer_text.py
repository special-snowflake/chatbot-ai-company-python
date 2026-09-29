"""Answer rendering and node-quality guards.

Port addition. The JS original funnelled every retrieved chunk straight into the
generative provider and returned whatever came back. Measured against the
NOVAHAUS corpus, that pipeline had two defects:

1. **Heading-only nodes get selected.** :mod:`src.source_loader` splits a
   markdown document at *every* heading, so a file whose first line is a title
   produces a node whose entire body is that title. Such a node embeds close to
   any question about the document, which is how ``what is the company about?``
   came back answered with the literal string
   ``# NOVAHAUS Company Information & Corporate Knowledge Base``.
2. **Raw chunks read badly in a chat reply.** A node carries loader scaffolding
   (``**Q: ...**``, the document title) that is noise once it is an answer.

:func:`is_contentless` is the guard for (1); :func:`render_answer` handles (2).
Neither invents a single word — every string they produce is copied verbatim out
of the source document.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #

#: Any ATX markdown heading line (``#`` through ``######``).
_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t].*$", re.MULTILINE)
#: Fenced code blocks — never an answer in this corpus.
_FENCE_RE = re.compile(r"^[ \t]*```.*?^[ \t]*```[ \t]*$", re.MULTILINE | re.DOTALL)
#: Alphabetic words of two or more letters: the prose signal.
_WORD_RE = re.compile(r"[A-Za-z]{2,}")

#: A node must carry at least this many body words to be answerable. One, not
#: three: the guard exists to reject nodes whose entire body *is* their heading —
#: which is how the company document produced a first node that was nothing but
#: its title — and a deliberately terse node such as
#: "### Glow Bulb A19\n- Socket: E27" is a real answer that must survive.
MIN_BODY_WORDS = 1

# Q/A scaffolding, in both shapes the corpus and the JSON loader produce:
#   ``**Q: How long is shipping?**`` / ``**A: Three days.**``      (markdown FAQ)
#   ``Question: How long is shipping?`` / ``Answer: Three days.``  (JSON entries)
_QUESTION_RE = re.compile(
    r"^[ \t]*(?:\*\*)?(?:Q|Question)[ \t]*:[ \t]*(?P<text>.+?)(?:\*\*)?[ \t]*$",
    re.MULTILINE,
)
_ANSWER_RE = re.compile(
    r"^[ \t]*(?:\*\*)?(?:A|Answer)[ \t]*:[ \t]*(?P<text>.+?)(?:\*\*)?[ \t]*$",
    re.MULTILINE,
)
_BLANK_RUN_RE = re.compile(r"\n{3,}")

#: The refusal string, kept identical to the JS service.
REFUSAL = "Sorry, I cannot help with that based on the available catalog."


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #

def body_text(text: str) -> str:
    """Return ``text`` with heading lines and code fences removed."""
    return _FENCE_RE.sub("", _HEADING_RE.sub("", text))


def is_contentless(text: str) -> bool:
    """Whether ``text`` is a heading/label shell with no answerable prose.

    A node counts as contentless when, after removing markdown headings and code
    fences, fewer than :data:`MIN_BODY_WORDS` alphabetic words remain. Table rows
    and list bullets are deliberately *kept*: the company and product sections
    are tables and bullet lists, and those are legitimate answers.
    """
    return len(_WORD_RE.findall(body_text(text))) < MIN_BODY_WORDS


def probe_text(entry: Dict[str, Any]) -> str:
    """Return the text a node-quality guard should inspect.

    The loader prepends the document title to every entry's searchable ``text``
    (``"NOVAHAUS_FAQ\\n**Q: ...**"``). Guarding on that string would never fire,
    because the injected title alone clears the word threshold — so the raw
    ``content`` is preferred when present.
    """
    content = entry.get("content")
    if isinstance(content, str) and content.strip():
        return content
    return str(entry.get("text", ""))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def extract_qa(text: str) -> Optional[Tuple[str, str]]:
    """Return ``(question, answer)`` scaffolding found in ``text``, else ``None``."""
    answer = _ANSWER_RE.search(text)
    if answer is None:
        return None
    question = _QUESTION_RE.search(text)
    return (
        question.group("text").strip() if question else "",
        answer.group("text").strip(),
    )


def clean_markdown(text: str) -> str:
    """Tidy whitespace and line endings without altering a single word."""
    cleaned = text.replace("\r\n", "\n").strip()
    return _BLANK_RUN_RE.sub("\n\n", cleaned)


_PRODUCT_HEADING_RE = re.compile(r"^[ \t]*#{3,6}[ \t]+(?P<name>.+?)[ \t]*$", re.MULTILINE)


def product_name(entry: Dict[str, Any]) -> str:
    """Return a catalog product's display name, or ``""`` when it is not one.

    Only the ``### Name`` detail sections of ``NOVAHAUS_Product_Catalog.md``
    count. The distinction matters for budget queries: several FAQ nodes mention
    a price in passing, and listing those alongside real products would be
    wrong.
    """
    match = _PRODUCT_HEADING_RE.search(probe_text(entry))
    return match.group("name").strip() if match else ""


def node_label(entry: Dict[str, Any]) -> str:
    """Return a short human label for a node: its question or first heading.

    Used when several node answers are stitched into one reply, so the reader can
    tell which answer belongs to which part of their question. Returns an empty
    string when the node offers no natural label.
    """
    question = entry.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip().rstrip("?")

    content = entry.get("content")
    if isinstance(content, str) and content.strip():
        heading = _HEADING_RE.search(content)
        if heading:
            return heading.group(0).lstrip("# \t").strip()
        qa = extract_qa(content)
        if qa is not None and qa[0]:
            return qa[0].rstrip("?")
    return ""


def render_answer(entry: Dict[str, Any], *, include_question: bool = False) -> str:
    """Render ``entry`` as a chat answer, verbatim from the source.

    Preference order mirrors how the loader stores content: an explicit
    ``answer`` field first, then markdown Q/A inside ``content``, then the raw
    ``content``/``text`` block (catalog specs, company sections).

    Args:
        entry: An indexed catalog entry.
        include_question: Prefix the answer with its question. Used when several
            answers are stitched together, where the label keeps them readable.
    """
    answer_field = entry.get("answer")
    if isinstance(answer_field, str) and answer_field.strip():
        answer = answer_field.strip()
        question = entry.get("question")
        if include_question and isinstance(question, str) and question.strip():
            return f"{question.strip()}\n{answer}"
        return answer

    content = entry.get("content")
    if not isinstance(content, str) or not content.strip():
        content = entry.get("text")
    if not isinstance(content, str) or not content.strip():
        return ""

    qa = extract_qa(content)
    if qa is not None:
        question, answer = qa
        if include_question and question:
            return f"**{question}**\n{answer}"
        return answer
    return clean_markdown(content)
