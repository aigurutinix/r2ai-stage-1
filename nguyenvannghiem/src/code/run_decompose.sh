#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Query Decomposition Pipeline ==="

echo "$(date): Step 1 - Decompose queries..."
PYTHONUNBUFFERED=1 python3 query_decomposition.py decompose --workers 8

echo ""
echo "$(date): Step 2 - Retrieve with sub-queries (Dense)..."
PYTHONUNBUFFERED=1 python3 query_decomposition.py retrieve --source dense --top-k 100

echo ""
echo "$(date): Step 3 - LLM rerank..."
PYTHONUNBUFFERED=1 python3 query_decomposition.py rerank --workers 8 --max-candidates 40

echo ""
echo "$(date): DONE"
echo "Submit: submission_3_1_decompose.json"
