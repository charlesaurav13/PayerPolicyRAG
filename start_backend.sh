#!/bin/bash
# Start the PayerPolicy RAG backend server
set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
export PATH="/opt/homebrew/opt/openjdk@21/bin:$PATH"
export JAVA_HOME="/opt/homebrew/opt/openjdk@21"

echo "Starting PayerPolicy RAG backend on http://localhost:8001"
cd "$REPO/backend"
"$REPO/venv/bin/uvicorn" main:app --host 0.0.0.0 --port 8001 --reload
