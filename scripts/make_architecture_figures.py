"""Generate the before/after architecture figures for the answering rework.

Port addition. Two figures are produced as self-contained HTML files with inline
SVG, then rendered to PNG with headless Chrome:

    docs/answering-before-after.html  -- the pipeline as ported, then as reworked
    docs/answering-flow.html          -- one request's journey, with the branch
                                         counts measured by scripts/run_eval.py

Every number in the figures comes from results/eval-*.json; nothing is
illustrative. Re-run after changing the strategy:

    python scripts/make_architecture_figures.py
"""

from __future__ import annotations

import html
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# --------------------------------------------------------------------------- #
# Palette (tailwind-ish, matching the architecture-diagram house style)
# --------------------------------------------------------------------------- #

BG = "#020617"
GRID = "#1e293b"
PANEL = "#0f172a"

CYAN = "#22d3ee"
EMERALD = "#34d399"
VIOLET = "#a78bfa"
AMBER = "#fbbf24"
ROSE = "#fb7185"
SLATE = "#94a3b8"
MUTED = "#64748b"

FILL_CYAN = "rgba(8, 51, 68, 0.45)"
FILL_EMERALD = "rgba(6, 78, 59, 0.45)"
FILL_VIOLET = "rgba(76, 29, 149, 0.45)"
FILL_AMBER = "rgba(120, 53, 15, 0.35)"
FILL_ROSE = "rgba(136, 19, 55, 0.45)"
FILL_SLATE = "rgba(30, 41, 59, 0.55)"

# --------------------------------------------------------------------------- #
# Measured data (results/eval-*.json). Keyed by config name.
# --------------------------------------------------------------------------- #

LABELS = {
    "original-058": "as ported — gate 0.58",
    "threshold-045": "gate relaxed — 0.45",
    "hybrid-045": "strategy — gate 0.45",
}

MEASURED = {
    "original-058": {
        "passed": 28, "cases": 54, "p50": "3 368 ms", "p95": "8 200 ms",
        "rss": "2 581 MB", "calls": 33, "refused": 27, "model": 24,
    },
    "threshold-045": {
        "passed": 29, "cases": 54, "p50": "3 368 ms", "p95": "8 200 ms",
        "rss": "2 578 MB", "calls": 33, "refused": 18, "model": 33,
    },
    "hybrid-045": {
        "passed": 52, "cases": 54, "p50": "22 ms", "p95": "68 ms",
        "rss": "898 MB", "calls": 0, "refused": 9, "model": 0,
    },
}

#: Branch counts from the hybrid run (`summary.paths`), highest count first.
PATHS_HYBRID: List[Tuple[str, int, str, str]] = [
    ("deterministic", 17, "the winning node's own text", EMERALD),
    ("compound", 10, "each half answered, then stitched", CYAN),
    ("refused", 9, "7 correct out-of-scope + 2 known gaps", MUTED),
    ("tie-expansion", 6, "every plausible variant returned", CYAN),
    ("budget-filter", 5, "all products at or under the ceiling", AMBER),
    ("hazard-escalation", 4, "the catalog's own safety policy", ROSE),
    ("conversational", 3, "canned table, before retrieval", SLATE),
    ("layap", 0, "the model — never reached", VIOLET),
]

#: The eight strategy steps, in evaluation order.
STEPS: List[Tuple[str, str, str, str]] = [
    ("1", "Hazard report?", "quote the catalog's safety policy", ROSE),
    ("2", "Budget parsed?", "list every product at or under the ceiling", AMBER),
    ("3", "Score below the gate?", "refuse — unchanged behaviour", MUTED),
    ("4", "Only scaffolding retrieved?", "refuse — a heading is not an answer", MUTED),
    ("5", "Compound question?", "split it, carry the subject, stitch", CYAN),
    ("6", "Near-tie or variant family?", "return every plausible node", CYAN),
    ("7", "A node renders to text", "answer verbatim — NO model call", EMERALD),
    ("8", "Nothing renderable", "optional synthesizer, else LAYA", VIOLET),
]


def esc(text: str) -> str:
    """Escape text for inline SVG."""
    return html.escape(text, quote=False)


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str,
         rx: int = 8, dash: Optional[str] = None) -> str:
    """A rounded component rectangle."""
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{extra}/>'
    )


def text(x: float, y: float, body: str, size: float = 11, fill: str = "#ffffff",
         weight: str = "600", anchor: str = "middle") -> str:
    """A text run."""
    return (
        f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{esc(body)}</text>'
    )


def arrow(x1: float, y1: float, x2: float, y2: float, color: str = MUTED,
          dash: Optional[str] = None, width: float = 1.5) -> str:
    """A connector with the shared arrowhead marker."""
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{width}" marker-end="url(#arrowhead)"{extra}/>'
    )


# --------------------------------------------------------------------------- #
# Shared HTML shell
# --------------------------------------------------------------------------- #

FILLS = {
    CYAN: FILL_CYAN, EMERALD: FILL_EMERALD, VIOLET: FILL_VIOLET,
    AMBER: FILL_AMBER, ROSE: FILL_ROSE, MUTED: FILL_SLATE, SLATE: FILL_SLATE,
}

CSS = f"""
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'JetBrains Mono', 'DejaVu Sans Mono', monospace;
    background: {BG}; color: #fff; padding: 26px;
  }}
  .header-row {{ display: flex; align-items: center; gap: 10px; }}
  .pulse-dot {{ width: 11px; height: 11px; background: {CYAN}; border-radius: 50%; }}
  h1 {{ font-size: 19px; font-weight: 700; letter-spacing: -0.02em; }}
  .subtitle {{ color: {SLATE}; font-size: 12px; margin: 5px 0 0 21px; }}
  .panel {{
    background: rgba(15, 23, 42, 0.5); border: 1px solid {GRID};
    border-radius: 14px; padding: 14px; margin-top: 16px;
  }}
  svg {{ display: block; }}
  .footer {{ text-align: center; margin-top: 12px; color: {MUTED}; font-size: 10px; }}
"""

DEFS = f"""
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="{MUTED}" />
    </marker>
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="{GRID}" stroke-width="0.5" />
    </pattern>
  </defs>
"""


def html_shell(title: str, subtitle: str, svg: str, footer: str, width: int) -> str:
    """Wrap an SVG body in the shared dark card layout."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{esc(title)}</title>
<style>{CSS}</style>
</head>
<body style="width:{width + 52}px">
  <div style="width:{width}px">
    <div class="header-row"><div class="pulse-dot"></div><h1>{esc(title)}</h1></div>
    <p class="subtitle">{esc(subtitle)}</p>
    <div class="panel">{svg}</div>
    <p class="footer">{esc(footer)}</p>
  </div>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Figure 1 — the pipeline before and after
# --------------------------------------------------------------------------- #

def _pipeline_row(y: float) -> str:
    """The six shared front-end components, drawn identically in both panels."""
    parts: List[str] = []
    boxes = [
        ("Client", "POST /catalog/query", SLATE),
        ("Conversational", "canned table", EMERALD),
        ("Embed + score", "MiniLM + lexical", EMERALD),
        ("Gate", "reject below", ROSE),
        ("top-k = 3", "context items", EMERALD),
        ("Answerer", "see below", CYAN),
    ]
    for index, (name, sub, color) in enumerate(boxes):
        x = 20 + index * 170
        parts.append(rect(x, y, 150, 54, FILLS[color], color))
        centre = x + 75
        parts.append(text(centre, y + 24, name, 11))
        parts.append(text(centre, y + 40, sub, 8.5, SLATE, "400"))
        if index < len(boxes) - 1:
            parts.append(arrow(x + 152, y + 27, x + 168, y + 27))
    return "".join(parts)


def figure_before_after() -> str:
    """Render the ported pipeline and the reworked one, stacked."""
    out: List[str] = ['<svg viewBox="0 0 1040 880" width="1040">', DEFS]
    out.append('<rect width="100%" height="100%" fill="url(#grid)" />')

    # ---------------------------------------------------------------- BEFORE #
    out.append(text(22, 26, "BEFORE  ·  one answering path", 13, ROSE, "700", "start"))
    out.append(_pipeline_row(46))
    # the gate refuses, LAYA echoes one item
    out.append(arrow(605, 100, 605, 128))
    out.append(rect(360, 130, 310, 54, FILL_ROSE, ROSE))
    out.append(text(515, 154, "REFUSED", 12, ROSE))
    out.append(text(515, 170, "27 of 54 outcomes", 9, SLATE, "400"))
    out.append(arrow(945, 100, 945, 128))
    out.append(rect(690, 130, 330, 54, FILL_VIOLET, VIOLET))
    out.append(text(855, 154, "ONE CONTEXT ITEM", 11.5, VIOLET))
    out.append(text(855, 170, "echoed verbatim — never merged", 9, SLATE, "400"))
    out.append(rect(20, 200, 1000, 40, FILL_SLATE, GRID, 8))
    out.append(text(520, 225,
                    "p50 3 368 ms  ·  p95 8 200 ms  ·  2 581 MB RSS  ·  two cores pinned  "
                    "·  24 of 54 calls reached the model",
                    10.5, SLATE, "500"))
    out.append(f'<line x1="20" y1="262" x2="1020" y2="262" stroke="{GRID}" stroke-width="1" '
               'stroke-dasharray="6,6" />')

    # ----------------------------------------------------------------- AFTER #
    out.append(text(22, 296, "AFTER  ·  eight-step strategy, model last", 13, EMERALD, "700", "start"))
    out.append(_pipeline_row(316))

    notes = {
        "1": "never gated on score",
        "6": "the three delivery regions",
        "7": "most common path — 17 of 54",
        "8": "rare — 0 of 54",
    }
    for index, (num, question, outcome, color) in enumerate(STEPS):
        y = 400 + index * 50
        highlighted = num == "7"
        if highlighted:
            out.append(rect(14, y - 5, 1012, 52, "rgba(6, 78, 59, 0.28)", EMERALD, 10, "4,3"))
        dash = "5,4" if num == "8" else None
        out.append(rect(20, y, 470, 42, FILLS[color], color, 8, dash))
        out.append(text(34, y + 26, f"{num}.  {question}", 10.5, "#ffffff", "600", "start"))
        out.append(arrow(492, y + 21, 506, y + 21))
        out.append(rect(508, y, 262, 42, FILLS[color], color, 8, dash))
        out.append(text(639, y + 26, outcome, 9.5, "#ffffff", "500"))
        note = notes.get(num)
        if note:
            out.append(text(782, y + 26, note, 9, color, "500", "start"))

    out.append(rect(20, 812, 1000, 40, FILL_SLATE, GRID, 8))
    out.append(text(520, 837,
                    "p50 22 ms  ·  p95 68 ms  ·  898 MB RSS  ·  no core saturation  "
                    "·  0 of 54 calls reached the model",
                    10.5, EMERALD, "500"))
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Figure 2 — one request's journey through the strategy
# --------------------------------------------------------------------------- #

#: Where the 54 eval cases landed is PATHS_HYBRID above; figure 2 draws it.


def figure_flow() -> str:
    """Render the request journey beside where the 54 eval cases actually landed."""
    out: List[str] = ['<svg viewBox="0 0 1040 470" width="1040">', DEFS]
    out.append('<rect width="100%" height="100%" fill="url(#grid)" />')

    # ------------------------------------------------- left: the journey ---- #
    out.append(text(30, 44, "ONE REQUEST", 11, SLATE, "700", "start"))
    stages = [
        ("POST /catalog/query", "the only endpoint", SLATE),
        ("Conversational match?", "canned table, answered before retrieval", EMERALD),
        ("Embed + score", "384-dim MiniLM over 124 chunks", EMERALD),
        ("Eight-step strategy", "decides WITHOUT calling a model", CYAN),
        ("Response", "verbatim from the catalog, or a refusal", SLATE),
    ]
    for index, (name, sub, color) in enumerate(stages):
        y = 62 + index * 74
        if index:
            out.append(arrow(200, y - 22, 200, y - 4))
        out.append(rect(30, y, 340, 52, FILLS[color], color))
        out.append(text(200, y + 22, name, 10.5))
        out.append(text(200, y + 38, sub, 8.5, SLATE, "400"))

    # ----------------------------------- right: where the cases landed ----- #
    out.append(text(430, 44, "WHERE THE 54 EVAL CASES LANDED", 11, SLATE, "700", "start"))
    for index, (path, count, blurb, color) in enumerate(PATHS_HYBRID):
        y = 62 + index * 48
        # Bars are sized by count, but the count and the description sit in fixed
        # columns so they line up as a table instead of a ragged staircase.
        width = 70 + count * 14
        out.append(rect(430, y, width, 38, FILLS[color], color, 6,
                        "5,4" if count == 0 else None))
        out.append(text(442, y + 24, path, 9.5, "#ffffff", "600", "start"))
        out.append(text(790, y + 24, str(count), 11, color, "700", "end"))
        out.append(text(806, y + 24, blurb, 8, SLATE, "400", "start"))
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

FIGURES = [
    ("answering-before-after", figure_before_after, (1092, 1050),
     "Catalog assistant — answering architecture",
     "NOVAHAUS catalog port · the ported pipeline, then the reworked one",
     "54 labeled cases · scripts/run_eval.py · scripts/make_architecture_figures.py"),
    ("answering-flow", figure_flow, (1092, 640),
     "Catalog assistant — request flow",
     "One POST /catalog/query through the eight-step strategy",
     "branch counts from results/eval-hybrid-045.json · model calls: 0 of 54"),
]


def render_png(html_path: Path, png_path: Path, size: Tuple[int, int]) -> bool:
    """Screenshot an HTML file with headless Chrome. Returns success."""
    command = [
        "google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
        "--disable-dev-shm-usage", "--hide-scrollbars",
        "--force-device-scale-factor=2",
        f"--window-size={size[0]},{size[1]}",
        f"--screenshot={png_path}",
        html_path.as_uri(),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as error:
        print(f"  chrome failed: {error}", file=sys.stderr)
        return False
    if not png_path.exists():
        print(f"  chrome produced no file (exit {result.returncode})", file=sys.stderr)
        return False
    return True


def main() -> int:
    """Write both HTML figures, then render each to PNG."""
    DOCS.mkdir(parents=True, exist_ok=True)
    for name, builder, size, title, subtitle, footer in FIGURES:
        html_path = DOCS / f"{name}.html"
        png_path = DOCS / f"{name}.png"
        svg = builder()
        html_path.write_text(html_shell(title, subtitle, svg, footer, 1040),
                             encoding="utf-8")
        ok = render_png(html_path, png_path, size)
        status = f"{png_path.stat().st_size // 1024} KB" if ok else "FAILED"
        print(f"{name}: html {html_path.stat().st_size // 1024} KB, png {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
