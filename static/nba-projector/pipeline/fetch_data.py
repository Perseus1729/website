"""
Fetch the latest NBA team-game totals CSV from a configured source.

Default source is github.com/NocturneBear/NBA-Data-2010-2024 (regular season
team-level box-score totals, 2010-present).  Add or swap sources by editing
`SOURCES` below.  Each source is `(name, fetch_fn, validator_fn)` and
`fetch_fn(out_path)` should write a CSV with at minimum the columns:

    SEASON_YEAR, TEAM_ABBREVIATION, TEAM_NAME, GAME_ID, MATCHUP, PTS

Optional extras the rest of the pipeline will use if present:
    GAME_DATE  – enables per-game date sorting / recency weighting
    OPP_PTS    – enables opponent-aware features
    HOME_AWAY  – 'H' / 'A' for home-court features
"""
from __future__ import annotations
import os, shutil, subprocess, tempfile, sys
from pathlib import Path

REPO_URL_NOCT = "https://github.com/NocturneBear/NBA-Data-2010-2024.git"

# Optional add-ons – any URL on github.com works because raw files come down
# via `git clone --depth 1` (avoids needing raw.githubusercontent.com).
REPO_URL_SHUF = "https://github.com/shufinskiy/nba_data.git"  # play-by-play
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
NBA_LIVE_SCOREBOARD = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"


def _run(cmd: list[str]) -> None:
    print("    $", " ".join(cmd))
    subprocess.run(cmd, check=True)


def fetch_team_game_totals(out_path: Path) -> Path:
    """Clone the NocturneBear repo, copy regular-season + playoff totals,
    concatenate, and write to out_path."""
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _run(["git", "clone", "--depth", "1", REPO_URL_NOCT, str(tmp / "noct")])
        src1 = tmp / "noct" / "regular_season_totals_2010_2024.csv"
        src2 = tmp / "noct" / "play_off_totals_2010_2024.csv"

        import pandas as pd
        frames = []
        if src1.exists(): frames.append(pd.read_csv(src1))
        if src2.exists(): frames.append(pd.read_csv(src2))
        if not frames:
            raise FileNotFoundError("no totals CSVs found in NocturneBear repo")
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(out_path, index=False)
        print(f"    Wrote {len(df):,} rows -> {out_path}")
    return out_path


def fetch_play_by_play(out_dir: Path, seasons: list[int] | None = None) -> Path:
    """Pull play-by-play parquet/csv files from the shufinskiy repo into
    `out_dir/pbp/`.  Note this repo is large (~2 GB); pass `seasons` to limit."""
    out_dir = Path(out_dir); (out_dir / "pbp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _run(["git", "clone", "--depth", "1", "--filter=blob:none",
              "--sparse", REPO_URL_SHUF, str(tmp / "shuf")])
        if seasons:
            patterns = [f"{s}_pbp.csv" for s in seasons]
            _run(["git", "-C", str(tmp / "shuf"), "sparse-checkout", "set"] + patterns)
        for f in (tmp / "shuf").rglob("*pbp*.csv"):
            dest = out_dir / "pbp" / f.name
            shutil.copy(f, dest)
            print(f"    Copied {f.name} -> {dest}")
    return out_dir / "pbp"


def fetch_live_scoreboard() -> dict | None:
    """Fetch today's live NBA scoreboard.  Returns None if the host network
    blocks ESPN/NBA.com (commonly the case in sandboxed envs).  In CI,
    set up a proxy / allowlist to enable live updates."""
    try:
        import urllib.request, json
        req = urllib.request.Request(
            ESPN_SCOREBOARD, headers={"User-Agent": "Mozilla/5.0 nba-projector"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"    [warn] live ESPN fetch failed: {e}")
    try:
        import urllib.request, json
        req = urllib.request.Request(
            NBA_LIVE_SCOREBOARD, headers={"User-Agent": "Mozilla/5.0 nba-projector"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"    [warn] live NBA.com fetch failed: {e}")
    return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="NBA data fetcher")
    ap.add_argument("--totals", action="store_true", help="Refresh team-game totals CSV")
    ap.add_argument("--pbp",    action="store_true", help="Pull play-by-play files")
    ap.add_argument("--live",   action="store_true", help="Print today's scoreboard")
    ap.add_argument("--seasons", nargs="*", type=int, default=None)
    args = ap.parse_args()

    HERE = Path(__file__).resolve().parent
    DATA = HERE.parent / "data"
    if args.totals or not any([args.pbp, args.live]):
        fetch_team_game_totals(DATA / "nba_team_game_totals.csv")
    if args.pbp:
        fetch_play_by_play(DATA, seasons=args.seasons)
    if args.live:
        sb = fetch_live_scoreboard()
        print(sb if sb else "[live fetch unavailable in this network context]")
