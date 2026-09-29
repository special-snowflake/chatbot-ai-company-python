"""Catalog source loader.

Python port of ``src/source-loader.js``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

# Matches ``**Q: ...`` FAQ blocks (port of the JS split regex).
_FAQ_SPLIT_RE = re.compile(r"(?=^\*\*Q:\s)", re.MULTILINE)
_FAQ_TEST_RE = re.compile(r"^\*\*Q:\s", re.MULTILINE)
# Matches markdown headings level 1-3.
_HEADING_SPLIT_RE = re.compile(r"(?=^#{1,3}\s)", re.MULTILINE)


def markdown_entries(filename: str, content: str) -> List[Dict[str, Any]]:
    """Split a markdown document into FAQ blocks or heading sections."""
    base_id = os.path.splitext(os.path.basename(filename))[0]

    faq_blocks = [block for block in _FAQ_SPLIT_RE.split(content) if _FAQ_TEST_RE.search(block)]
    if faq_blocks:
        return [
            {"id": f"{base_id}-{i + 1}", "title": base_id, "content": block.strip()}
            for i, block in enumerate(faq_blocks)
        ]

    sections = [section.strip() for section in _HEADING_SPLIT_RE.split(content) if section.strip()]
    return [
        {"id": f"{base_id}-{i + 1}", "title": base_id, "content": section}
        for i, section in enumerate(sections)
    ]


def load_source_entries(source_path: str) -> List[Dict[str, Any]]:
    """Load entries from a JSON file or a directory of ``.md``/``.txt`` files."""
    if os.path.isfile(source_path):
        with open(source_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    filenames = sorted(
        name
        for name in os.listdir(source_path)
        if os.path.isfile(os.path.join(source_path, name))
        and os.path.splitext(name)[1].lower() in (".md", ".txt")
    )

    documents: List[Dict[str, Any]] = []
    for filename in filenames:
        with open(os.path.join(source_path, filename), "r", encoding="utf-8") as handle:
            documents.extend(markdown_entries(filename, handle.read()))
    return documents
