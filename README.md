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

## 3. Retrieval pipeline (unchanged from the JS build)

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
5. **Threshold rejection** — below `SIMILARITY_THRESHOLD` (default `0.58`) with no budget match,
   the API returns the `NO_MATCH` refusal rather than guessing.
6. **Answer synthesis** — the top-`k` contexts (default `3`) are handed to LAYA, which selects the
   answer. If LAYA returns nothing usable the request degrades to returning the best raw match.

LAYA additionally has a *relevance* gate: even when the score clears the threshold it may rule
that the retrieved context does not answer the question, and the API then returns the refusal.

---

## 4. Layout

```
.
├── src/
│   ├── config.py           # port of config.js — same env names, same defaults
│   ├── similarity.py       # cosine similarity
│   ├── embeddings.py       # MiniLM provider + deterministic hashed fallback
│   ├── laya_engine.py      # LAYA decision engine (replaces llm.js / Jev AI)
│   ├── catalog_service.py  # ingest, scoring pipeline, conversational intent
│   ├── source_loader.py    # port of source-loader.js (markdown → entries)
│   ├── routes.py           # Pydantic schemas + the two endpoints
│   └── server.py           # app factory, middleware, CORS, auto-ingest on boot
├── scripts/
│   ├── ingest.py           # port of scripts/ingest.js — CLI indexer
│   ├── questions.json      # grounded paraphrase question set
│   └── run_test_suite.py   # multi-model evaluation harness
├── test/test_api.py        # pytest suite (API + scoring parity)
├── source/                 # NOVAHAUS knowledge base (.md)
├── data/                   # generated catalog index
├── TEST_REPORT.md          # generated evaluation report
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
| `SIMILARITY_THRESHOLD` | `0.58` | minimum score to answer |
| `TOP_K` | `3` | contexts passed to LAYA |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | embedding provider |
| `LAYA_MODEL` | `english` | LAYA checkpoint family (`english`, `multilingual`, `typed-decisions`) |
| `LLM_TIMEOUT_MS` | `30000` | decision-engine timeout |
| `CATALOG_FILE` | `./source` | JSON file or directory of `.md`/`.txt` |
| `CATALOG_INDEX_FILE` | `./data/catalog-index.json` | persisted index path |

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

---

## 7. Notes

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
