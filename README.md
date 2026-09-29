# chatbot-ai-company — Python port with LAYA

A Python (FastAPI) rewrite of [`special-snowflake/chatbot-ai-company`](https://github.com/special-snowflake/chatbot-ai-company).

The original is a Node.js/Hapi knowledge-base chatbot that answered catalog questions with a
local `node-llama-cpp` model ("Jev AI"). This port keeps the application flow, the routing
contract, and the retrieval maths **identical**, and swaps the generative LLM for
[**LAYA**](https://github.com/NandhaKishorM/laya) — a non-autoregressive System 1 decision
engine. Jev AI is removed completely.

---

## 1. Stack mapping

| Concern | Original (Node.js) | This port (Python) |
| :--- | :--- | :--- |
| HTTP framework | Hapi.js | FastAPI |
| Schema validation | Joi | Pydantic v2 |
| Answer engine | `node-llama-cpp` + local `.gguf` ("Jev AI") | **LAYA** (`from laya.router import Router`) |
| Embeddings | `@xenova/transformers` `Xenova/all-MiniLM-L6-v2` | `sentence-transformers/all-MiniLM-L6-v2`, deterministic hashed fallback |
| Logging | pino | stdlib `logging` (same request/response events) |
| Config | `process.env` + `dotenv` | `os.environ` + `python-dotenv` (same names, same defaults) |
| Test runner | `node --test` | `pytest` |

API paths are preserved **1:1**. Every request and response shape is byte-compatible.

---

## 2. API

### `POST /catalog/ingest`

Replaces the in-memory index wholesale.

```json
{
  "documents": [
    { "id": "faq-1", "question": "How long is the return window?", "answer": "7 calendar days." },
    { "id": "doc-2", "title": "Company", "content": "NOVAHAUS is based in Jakarta." },
    { "id": "raw-3", "text": "Free-form text is also accepted." }
  ]
}
```

Response: `{ "count": 3 }`

Each entry needs an `id` plus one of `question`+`answer`, `title`+`content`, or `text`.

### `POST /catalog/query`

```json
{ "query": "How long does delivery take in Jabodetabek?" }
```

Response:

```json
{ "matched": true, "answer": "Typical delivery is 1-3 business days.", "score": 0.8123 }
```

Before anything is ingested the endpoint returns **409**. Invalid payloads return **400** in the
original Hapi/Joi error shape.

### Docs

Interactive schema is served at **`/documentation`** (the original's route name), with the raw
document at `/openapi.json`.

---

## 3. Retrieval and answering

The retrieval pipeline is unchanged from the JS build. The answering step on top of it is new — see
§3.2.

1. **Conversational short-circuit** — greetings, thanks and goodbyes are answered from a canned
   table, with a mixed-intent path that strips a prefix (`"Hi, what are your hours?"`) and
   prepends the acknowledgement to the factual answer.
2. **Embed** the (rewritten) query.
3. **Score every entry** with the original hybrid formula:

   ```
   score = max(cosine, 0.7 * cosine + 0.3 * lexical)
   ```

   `lexical` is the fraction of query words (stop-words removed, aliases normalised) present in
   the entry.
4. **Budget filter** — a query like `"smart plug under 200k"` is parsed with the original regex
   and restricts candidates to entries whose extracted `Rp` price is at or below the budget.
   Every product under the ceiling is listed; if none qualifies, the reply names the actual
   cheapest product rather than answering from an unrelated node.
5. **Threshold rejection** — below `SIMILARITY_THRESHOLD` (default `0.45`) with no budget match,
   the API returns the `NO_MATCH` refusal rather than guessing.
6. **Answering** — see §3.2.

### 3.2 Answering strategy (`src/answerer.py`)

The original called LAYA on every match and returned whichever context item LAYA picked. LAYA is a
**selector, not a generator**: it returns an index pointing at one context item, which the API then
echoes verbatim. Measured against this corpus that produced three defects — compound questions
answered with half their intent, an ambiguous question answered with one arbitrary variant, and
correct in-corpus questions refused outright. The strategy therefore answers from the retrieved
nodes first and only reaches for a model when no node answer can be produced:

| Order | Step | `path` label |
| :--- | :--- | :--- |
| 1 | Electrical-safety report → the catalog's own escalation policy, quoted. Never gated on score. | `hazard-escalation` |
| 2 | Budget query → every product at or below the ceiling | `budget-filter` |
| 3 | Below the gate → refusal (unchanged from the JS build) | `refused` |
| 4 | Everything retrieved was scaffolding → refusal | `refused` |
| 5 | Compound question → each half answered separately, then stitched | `compound` |
| 6 | Near-tie or variant family → every plausible node | `tie-expansion` |
| 7 | Clear winner → that node's own text, no model call | `deterministic` |
| 8 | Otherwise → optional synthesizer, else LAYA | `synthesizer` / `layap` |

Two guard rails make this behave:

* a node whose entire body **is** its heading is never an answer (`answer_text.is_contentless`) —
  the guard for the "what is the company about?" failure, where the winning node was the document
  title; and
* a question that matched **strongly** (`score >= 0.70`) is answered from its single best node, so
  asking about one delivery region does not return all of them.

The decision path is logged per request (`layap answered`, `tie expanded`, …) and returned by
`CatalogService.query_with_diagnostics` for the tests and the evaluation harness.

---

## 4. Layout

```
.
├── src/
│   ├── config.py           # port of config.js — same env names, same defaults
│   ├── lexical.py          # stop words, aliases, price/budget parsing, entry text
│   ├── similarity.py       # cosine similarity
│   ├── embeddings.py       # MiniLM provider + deterministic hashed fallback
│   ├── laya_engine.py      # LAYA decision engine (replaces llm.js / Jev AI)
│   ├── answer_text.py      # node rendering + the contentless-node guard
│   ├── intent.py           # hazard, compound and comparison detection
│   ├── answerer.py         # answering strategy (see §3.2)
│   ├── synthesizer.py      # optional OpenAI-compatible synthesizer
│   ├── cache.py            # normalised-query response cache
│   ├── catalog_service.py  # ingest, scoring pipeline, conversational intent
│   ├── source_loader.py    # port of source-loader.js (markdown → entries)
│   ├── routes.py           # Pydantic schemas + the two endpoints
│   └── server.py           # app factory, middleware, CORS, auto-ingest on boot
├── scripts/
│   ├── ingest.py           # port of scripts/ingest.js — CLI indexer
│   ├── questions.json      # grounded paraphrase question set
│   ├── run_test_suite.py   # multi-checkpoint harness (which LAYA model answers)
│   ├── eval_questions.py   # 54 labeled cases with cited expectations
│   └── run_eval.py         # before/after harness → EVAL_REPORT.md
├── test/
│   ├── test_api.py         # pytest suite (API + scoring parity)
│   └── test_answering.py   # pytest suite (strategy, guards, cache)
├── source/                 # NOVAHAUS knowledge base (.md)
├── data/                   # generated catalog index
├── EVAL_REPORT.md          # generated answer-quality report
├── TEST_REPORT.md          # generated checkpoint-comparison report
└── requirements.txt
```

---

## 5. Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # optional, defaults match the JS build
python -m src.server
```

The server ingests `./source` on boot and listens on `PORT` (default `3000`).

Index the catalog ahead of time instead:

```bash
python scripts/ingest.py --source ./source --out ./data/catalog-index.json
```

### Configuration

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `PORT` | `3000` | HTTP port |
| `SIMILARITY_THRESHOLD` | `0.45` | minimum score to answer (the JS build used `0.58`) |
| `TOP_K` | `3` | contexts passed to the model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | embedding provider |
| `LAYA_MODEL` | `english` | LAYA checkpoint family (`english`, `multilingual`, `typed-decisions`) |
| `LLM_TIMEOUT_MS` | `30000` | decision-engine timeout |
| `CATALOG_FILE` | `./source` | JSON file or directory of `.md`/`.txt` |
| `CATALOG_INDEX_FILE` | `./data/catalog-index.json` | persisted index path |

Port additions (see §3.2):

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `ANSWER_MODE` | `hybrid` | `hybrid` = deterministic first; `legacy` = original pipeline |
| `TIE_EPSILON` | `0.05` | score window treated as a tie; every node in it is returned |
| `MAX_TIE_NODES` | `3` | cap on nodes stitched into one reply |
| `DROP_CONTENTLESS_NODES` | `true` | never answer with a heading-only node |
| `ENABLE_COMPOUND_SPLIT` | `true` | split `A and B` / `compare A and B` and answer both halves |
| `ENABLE_HAZARD_ESCALATION` | `true` | answer safety reports with the catalog's escalation policy |
| `ENABLE_BUDGET_LISTING` | `true` | list every product at or below a stated ceiling |
| `QUERY_CACHE_SIZE` | `256` | normalised-query response cache; `0` disables |
| `WARM_LAYAP` | `false` | load the LAYA checkpoint at startup (~1.6 GB) |
| `SYNTHESIZER_BASE_URL` | _(empty)_ | optional OpenAI-compatible endpoint for open-ended asks |
| `SYNTHESIZER_MODEL` | _(empty)_ | model name for the above |
| `SYNTHESIZER_API_KEY` | _(empty)_ | key for the above; empty means stay local |
| `SYNTHESIZER_TIMEOUT_MS` | `20000` | synthesizer timeout |

---

## 6. Tests

```bash
pytest test/ -v                                   # API + scoring-parity suite
python scripts/run_test_suite.py english multilingual   # end-to-end eval + report
```

The evaluation harness boots the API once per LAYA checkpoint, replays
`scripts/questions.json` (paraphrased variants of the same intent), and records the exact
request body, response body, wall-clock timestamp, and latency for every call. It emits
`TEST_REPORT.md` plus raw JSON under `results/`.

### 6.1 Measured results

`scripts/run_eval.py` replays the 54 labeled cases in `scripts/eval_questions.py` against three
configurations, each in its own process so the memory figures are not polluted by a previously
loaded model. Full output and every failure is in `EVAL_REPORT.md`.

| configuration | accuracy | p50 | p95 | max | model calls | RSS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `original-058` — as ported | 28/54 (51.9%) | 47 ms | 7 450 ms | 10 221 ms | 24 | 2 581 MB |
| `threshold-045` — gate relaxed only | 29/54 (53.7%) | 3 368 ms | 8 200 ms | 10 641 ms | 33 | 2 578 MB |
| `hybrid-045` — gate + strategy | **52/54 (96.3%)** | **22 ms** | **68 ms** | **70 ms** | **0** | **898 MB** |

| category | original-058 | threshold-045 | hybrid-045 |
| :--- | :--- | :--- | :--- |
| single-intent | 9/12 | 9/12 | 12/12 |
| paraphrase | 4/8 | 5/8 | 6/8 |
| tie / ambiguous | 0/3 | 0/3 | 3/3 |
| compound | 1/8 | 1/8 | 8/8 |
| comparison | 3/3 | 3/3 | 3/3 |
| safety | 0/4 | 0/4 | 4/4 |
| budget | 0/5 | 0/5 | 5/5 |
| out-of-scope (must refuse) | 7/7 | 7/7 | 7/7 |
| conversational | 4/4 | 4/4 | 4/4 |

Two things the numbers settle directly:

* **Relaxing the gate was never the fix.** 0.58 → 0.45 bought one case out of 54 (51.9% → 53.7%);
  the strategy moved it to 96.3%. It also made latency *worse* (p50 47 ms → 3 368 ms), because more
  queries cleared the gate and reached LAYA. Both changes are kept — the gate relaxation is still
  worth having — but the ordering of their contributions is not what it looked like.
* **Out-of-scope precision did not regress.** All three configurations refuse all seven
  out-of-scope questions, so the accuracy gains are not bought with invented answers.

The two remaining failures are both **refusals rather than wrong answers**: `para-send-back`
(score 0.409) and `para-cheapest` (score 0.427) sit below the 0.45 gate while the out-of-scope
ceiling is 0.441. The distributions overlap, so lowering the gate further would start admitting
unrelated questions.

---

## 7. Notes

### Answering

* **LAYAP is a selector, not a generator.** It returns an index pointing at one of the context items
  it was given, and the API echoes that item verbatim. It cannot merge two nodes, so a compound
  question ("what do the bulbs cost *and* what warranty do they carry") is structurally unanswerable
  through it — no amount of prompt tuning changes that, which is why `src/answerer.py` answers each
  half itself.
* **The similarity gate cannot separate this corpus.** Measured over the eval set, the lowest score
  of a genuinely answerable question is `0.409` and the highest score of an out-of-scope question is
  `0.441`: the two distributions overlap. Any threshold therefore trades false refusals against
  false answers, which is why the strategy decides on *what was retrieved* (is it prose? is it
  priced? is it a hazard?) rather than on the score alone. See `EVAL_REPORT.md`.
* **The LAYA checkpoint is not warmed at boot by default.** `WARM_LAYAP=false` because the
  deterministic path means most installs never call it; when it *is* called it costs roughly 1.6 GB
  of resident memory and 3–6.5 s per question on a dual-core CPU. Set it to `true` if you route a
  lot of open-ended traffic its way.

### LAYA

* `from laya import Router` does not work on every LAYA release — import the submodule:
  `from laya.router import Router`.
* **LAYAP ignores the checkpoint you ask for, twice over.** Both failures are silent and both
  produce byte-identical output, so a "compare two models" run can easily be a comparison of one
  model with itself. `LayaEngine` works around both:
  1. `Router()` has no `model=` argument. Routing precedence is explicit `model` > `task` >
     workflow > `lang` > `lang_guess` > **detected language** > `default` — and detected English
     returns `english` *before* `default` is consulted. So `Router(default="multilingual")` still
     runs the English checkpoint on English input. The checkpoint must be pinned per call with
     `predict(model=...)`, which is what this port does.
  2. `DEFAULT_MODELS` maps `multilingual` / `typed-decisions` to revisions of the *same* repo that
     do not exist on the Hub (`GET /api/models/convaiinnovations/laya/revision/multilingual` →
     `404 Invalid rev id`), so LAYA falls back to `main`. The genuinely distinct checkpoints are
     standalone repos — `convaiinnovations/laya-multilingual`, `convaiinnovations/laya-typed-decisions`
     — selected with `standalone_repos=True`.
* LAYA ships some checkpoints with out-of-range temperature values and warns about uncalibrated
  confidence. The warning is expected and does not affect answer selection.
* The raw embedding provider is optional by design: without `sentence-transformers` the app still
  runs using deterministic hashed embeddings, which keeps the demo reproducible on small machines.
* The knowledge base in `source/` is a **fictional** synthetic dataset about "NOVAHAUS", inherited
  from the upstream repository.
