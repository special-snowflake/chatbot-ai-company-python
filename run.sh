#!/usr/bin/env bash
# One command to run the NOVAHAUS catalog chatbot: API + chat UI.
#
#   ./run.sh              # start on 127.0.0.1:3000
#   PORT=8080 ./run.sh    # start on a different port
#
# The chat UI is served by the same FastAPI process at "/" and the
# interactive API console at "/documentation", so a single process is all
# that is needed. Creates the virtualenv and installs requirements on first
# run (uv if available, otherwise python3 -m venv).
set -euo pipefail

cd "$(dirname "$0")"

VENV="${VENV:-/tmp/chatbot-ai-env}"
PORT="${PORT:-3000}"
HOST="${HOST:-127.0.0.1}"

# --- virtualenv ---------------------------------------------------------- #
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "==> creating virtualenv at $VENV"
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV" --python 3.12
  else
    python3 -m venv "$VENV"
  fi
  echo "==> installing requirements"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$VENV/bin/python" -r requirements.txt
  else
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -r requirements.txt
  fi
fi

# --- launch -------------------------------------------------------------- #
echo
echo "  Chat UI    http://$HOST:$PORT/"
echo "  API docs   http://$HOST:$PORT/documentation"
echo "  Query API  POST http://$HOST:$PORT/catalog/query"
echo
exec "$VENV/bin/python" -m uvicorn src.server:app --host "$HOST" --port "$PORT"
