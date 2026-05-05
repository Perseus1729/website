# NBA Score Projector — Data & Model Pipeline

Self-contained refresh + retrain pipeline for the projector page at
`/nba-projector/`. All inputs and outputs live under
`static/nba-projector/`, so future updates are local to this folder.

## Files

```
static/nba-projector/
├── index.html             ← the static page (loads ./models.js)
├── models.js              ← bundled model coefficients + team priors (auto-generated)
├── data/
│   ├── nba_team_game_totals.csv     raw team-game finals (latest fetch)
│   ├── team_priors.csv              per-team PPG with 3 history windows
│   ├── training_snapshots.csv       96k (minute, current_score, ppg) -> final
│   ├── models.json                  every model's coefficients
│   └── model_results.csv            held-out MAE / RMSE leaderboard
└── pipeline/
    ├── fetch_data.py        pull latest CSV (and optional play-by-play / live scores)
    ├── build_models.py      compute priors, simulate, fit 17 models, regenerate models.js
    ├── update.sh            wrapper: fetch_data.py + build_models.py
    └── README.md            this file
```

## One-time setup

```bash
# From the repo root
cd static/nba-projector/pipeline
python3 -m pip install -r <(echo -e "pandas\nnumpy\nscikit-learn")
chmod +x update.sh
```

## Manual refresh

```bash
./update.sh                 # fetch + retrain (~2 min)
./update.sh --no-nn         # skip neural net (~30s)
python3 build_models.py     # refit only, reuse cached CSV
python3 fetch_data.py --pbp --seasons 2024 2025   # pull play-by-play (large)
```

After running, commit `models.js` and the `data/` directory; Vercel will
rebuild the site on push.

## Background pipeline (recommended)

### Option A — GitHub Actions (zero local setup)

Add `.github/workflows/nba-refresh.yml` to the repo:

```yaml
name: Refresh NBA models
on:
  schedule:
    - cron: "0 7 * * 1"        # every Monday 07:00 UTC
  workflow_dispatch:           # manual trigger button

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { token: ${{ secrets.GITHUB_TOKEN }} }
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install pandas numpy scikit-learn
      - run: cd static/nba-projector/pipeline && bash update.sh
      - name: Commit if changed
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add static/nba-projector/models.js static/nba-projector/data
          git diff --cached --quiet || (
            git commit -m "nba-projector: refresh models" && git push
          )
```

### Option B — Local cron

```cron
# Refresh models every Monday at 02:00 local time
0 2 * * 1 cd /Users/hiteshkumar/Desktop/Extras/Code/website/static/nba-projector/pipeline && /usr/bin/env bash update.sh >> ~/.nba-projector.log 2>&1
```

### Option C — Cowork scheduled task

If you're running this in Cowork mode, ask Claude to:

> "Schedule the NBA projector pipeline to run every Monday at 7am."

Claude will register the task via `mcp__scheduled-tasks__create_scheduled_task`
and trigger `update.sh` for you.

## Live data sources

`fetch_data.py` knows about three sources:

| Source                                 | Type              | Rate-limit | Notes                                       |
| -------------------------------------- | ----------------- | ---------- | ------------------------------------------- |
| `github.com/NocturneBear/NBA-Data-...` | Team-game totals  | Free       | 33k+ rows, 2010-present (default)           |
| `github.com/shufinskiy/nba_data`       | Play-by-play CSVs | Free       | Large (~2 GB); fetch with `--pbp --seasons` |
| `site.api.espn.com/.../scoreboard`     | Live game state   | Generous   | May be blocked by sandboxed networks        |
| `cdn.nba.com/static/.../scoreboard`    | Live game state   | Generous   | Backup live source                          |

The HTML page also accepts a manual paste of play-by-play text (pull from
ESPN GameCast or NBA.com) so you can use rich within-game dynamics without
the live API.

## Adding a new feature to all models

1. Add the column to the snapshot generator in `build_models.py:simulate_snapshots`
2. Append it to the `FEATURES` list at the top
3. Run `python3 build_models.py` — every regression / piecewise / NN model
   automatically picks it up
4. Update `index.html` to surface a UI control + pass the new value into the
   prediction call

## Known limitations / TODO

- Snapshots are simulated (Poisson + quarter-effect) because raw play-by-play
  gives ~50× more rows than needed for the current model size — switch to
  real PBP-derived snapshots once disk budget allows.
- Home-court advantage is currently a +2.5 pt post-hoc shift. Once we have
  PBP data parsed, fold `is_home` directly into the feature set and refit.
- Opponent strength / pace adjustment is not yet a feature; PBP fetch unlocks it.
- Neural net is a tiny MLP (16-8 ReLU). Bumping to a sequence model over the
  per-minute scoring vector would likely beat the polynomials on late-game
  prediction.
