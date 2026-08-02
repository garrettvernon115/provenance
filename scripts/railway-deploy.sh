#!/usr/bin/env bash
# One-command Railway deploy for Provenance.
#
# `railway up` obeys .gitignore, which excludes the serving artifacts (the ~91 MB
# trained re-ranker ONNX + tokenizer, and data/eval/results.json). This script
# temporarily un-ignores exactly those three, uploads, then restores .gitignore —
# so the built image gets the model without committing a 91 MB blob to git.
#
# Usage:  bash scripts/railway-deploy.sh
# Requires: railway login already done; the app service is named provenance-app.
set -euo pipefail
cd "$(dirname "$0")/.."

SERVICE="provenance-app"
BAK="$(mktemp)"
cp .gitignore "$BAK"
restore() { cp "$BAK" .gitignore; rm -f "$BAK"; echo "restored .gitignore"; }
trap restore EXIT

# models/ is excluded as a whole dir; switch to per-child so children can be re-included
sed -i 's#^models/$#models/*#' .gitignore
# Re-include just the runtime artifacts (last-match-wins, so append at the end).
cat >> .gitignore <<'EOF'

# --- temp (railway-deploy.sh): include serving artifacts for `railway up` ---
!models/reranker.onnx
!models/reranker-tokenizer/
!data/eval/
data/eval/*
!data/eval/results.json
EOF

echo "Deploying to Railway (uploading incl. the trained re-ranker)…"
railway up --detach --service "$SERVICE"
echo "Upload complete — build is running on Railway."
echo "Watch:  railway logs --build --service $SERVICE"
