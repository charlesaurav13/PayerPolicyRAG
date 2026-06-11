#!/bin/bash
# Start the PayerPolicy RAG frontend
set -e

cd "$(dirname "$0")/frontend"
echo "Starting Next.js frontend on http://localhost:3000"
npm run dev
