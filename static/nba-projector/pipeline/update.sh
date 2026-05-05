#!/usr/bin/env bash
# Refresh data + retrain models + regenerate the static page bundle.
# Run weekly during the season, monthly off-season.  See README.md.
set -euo pipefail

cd "$(dirname "$0")"

echo "[1/3] Fetching latest team-game totals..."
python3 fetch_data.py --totals

echo "[2/3] Refitting models (this can take ~1-2 minutes)..."
python3 build_models.py "$@"

echo "[3/3] Done. New artifacts in:"
echo "      ../models.js"
echo "      ./data/models.json"
echo "      ./data/team_priors.csv"
echo "      ./data/training_snapshots.csv"
echo "      ./data/model_results.csv"

# Optional: auto-commit + push so Vercel rebuilds the page automatically.
# Uncomment if you want that behavior.
#
# git add ../models.js ./data/*
# git diff --cached --quiet || {
#   git commit -m "nba-projector: refresh models ($(date -u +%Y-%m-%d))"
#   git push
# }
