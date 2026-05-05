"""
NBA Final Score Projector — model fitting pipeline.

Reads the latest team-game CSV (default: data/nba_team_game_totals_<years>.csv,
auto-fetched by fetch_data.py from github.com/NocturneBear/NBA-Data-2010-2024
or another configured source), computes per-team priors with three history
windows (recent-5, weighted, full), simulates minute-level scoring trajectories
that match the observed PPG distribution + quarter-effects, and fits 17 models:

    Naive linear, Linear regression, Polynomial deg 2/3/4 + ridge,
    Piecewise linear (2/3/4 segments uniform; 2/3/4 segments data-driven knots),
    Piecewise polynomial deg 2/3 over quarters, deg-2 with data-driven knots,
    Random Forest, Gradient Boosting, Bayesian shrinkage (k=12),
    Neural network MLP (compact 3-layer, weights exported for browser use).

Outputs:
    data/team_priors.csv         per-team PPG priors (3 windows)
    data/training_snapshots.csv  96k (minute, current_score, team_ppg) rows
    data/models.json             coefficients for every model
    data/model_results.csv       held-out MAE/RMSE leaderboard
    ../models.js                 bundle for the static page (overwrites)

Usage:
    python build_models.py             # uses data/ as-is
    python build_models.py --refetch   # pull latest CSV first
    python build_models.py --no-nn     # skip the neural network fit (fast)
"""
from __future__ import annotations
import argparse, json, os, sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

HERE      = Path(__file__).resolve().parent
PROJ_DIR  = HERE.parent                       # static/nba-projector/
DATA_DIR  = PROJ_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV   = DATA_DIR / "nba_team_game_totals.csv"
GAME_MIN  = 48
RNG       = np.random.default_rng(42)
FEATURES  = ["minute", "current_score", "team_ppg"]


def load_raw(refetch: bool = False) -> pd.DataFrame:
    if refetch or not RAW_CSV.exists():
        from fetch_data import fetch_team_game_totals
        fetch_team_game_totals(RAW_CSV)
    df = pd.read_csv(RAW_CSV, usecols=[
        "SEASON_YEAR", "TEAM_ABBREVIATION", "TEAM_NAME",
        "GAME_ID", "MATCHUP", "PTS",
    ]).dropna(subset=["PTS"])
    df["season_start"] = df["SEASON_YEAR"].str[:4].astype(int)
    return df


def compute_priors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = df["season_start"] - df["season_start"].min() + 1
    priors = (
        df.groupby(["TEAM_ABBREVIATION", "TEAM_NAME"], group_keys=False)
          .apply(lambda g: pd.Series({
              "ppg_weighted": np.average(g["PTS"], weights=g["weight"]),
              "ppg_recent5":  g[g["season_start"] >= g["season_start"].max() - 4]["PTS"].mean(),
              "ppg_overall":  g["PTS"].mean(),
              "ppg_std":      g["PTS"].std(),
              "n_games":      len(g),
          }))
          .reset_index()
          .sort_values("ppg_weighted", ascending=False)
    )
    priors.to_csv(DATA_DIR / "team_priors.csv", index=False)
    return priors


# ---------------------------------------------------------------------------
# Simulation (used to generate minute-level training snapshots)
# ---------------------------------------------------------------------------
def quarter_rate(minute: int, base: float) -> float:
    q = minute // 12
    return base * (0.97 if q == 0 else 1.00 if q == 1 else 1.03 if q == 2 else 1.00)


def simulate_snapshots(priors: pd.DataFrame, n_games: int = 4000) -> pd.DataFrame:
    team_ppgs = priors["ppg_weighted"].values
    rows = []
    for _ in range(n_games):
        base_ppg     = float(RNG.choice(team_ppgs)) + RNG.normal(0, 4)
        base_per_min = base_ppg / GAME_MIN
        cum, hist    = 0.0, [0.0]
        for m in range(GAME_MIN):
            scored = max(0.0, RNG.poisson(quarter_rate(m, base_per_min)) + RNG.normal(0, 0.6))
            cum   += scored
            hist.append(cum)
        final = hist[-1]
        for m in range(1, GAME_MIN, 2):
            rows.append((m, hist[m], base_ppg, final))
    snap = pd.DataFrame(rows, columns=FEATURES + ["final_score"])
    snap.to_csv(DATA_DIR / "training_snapshots.csv", index=False)
    return snap


# ---------------------------------------------------------------------------
# Model fitting helpers
# ---------------------------------------------------------------------------
def split_train_test(snap: pd.DataFrame, test_frac: float = 0.2):
    n = len(snap)
    idx = RNG.permutation(n); cut = int(n * (1 - test_frac))
    train, test = snap.iloc[idx[:cut]], snap.iloc[idx[cut:]]
    return train, test


def record(name: str, y_true, y_pred, results: list):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    results.append({"model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3)})
    print(f"    {name:<48s}  MAE={mae:5.2f}  RMSE={rmse:5.2f}")


def fit_piecewise(snap: pd.DataFrame, knots: list[float], degree: int):
    segments = []
    for lo, hi in zip(knots[:-1], knots[1:]):
        mask = (snap["minute"] >= lo) & ((snap["minute"] < hi) if hi < GAME_MIN else (snap["minute"] <= hi))
        seg = snap[mask]
        if len(seg) < 20:
            segments.append(None); continue
        Xs, ys = seg[FEATURES].values, seg["final_score"].values
        pipe = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), Ridge(alpha=1.0))
        pipe.fit(Xs, ys)
        poly  = pipe.named_steps["polynomialfeatures"]
        ridge = pipe.named_steps["ridge"]
        segments.append({
            "minute_lo": float(lo), "minute_hi": float(hi), "degree": degree,
            "powers":    poly.powers_.tolist(),
            "coef":      [float(c) for c in ridge.coef_],
            "intercept": float(ridge.intercept_),
        })
    return {"knots": [float(k) for k in knots], "degree": degree, "segments": segments}


def predict_piecewise(mod: dict, X: np.ndarray) -> np.ndarray:
    knots = mod["knots"]; out = np.zeros(len(X))
    inner = np.array(knots[1:-1]) if len(knots) > 2 else np.array([])
    for i, row in enumerate(X):
        m = row[0]
        si = min(len(knots) - 2, max(0, int(np.searchsorted(inner, m, side="right"))))
        seg = mod["segments"][si]
        y = seg["intercept"]
        for pw, c in zip(seg["powers"], seg["coef"]):
            term = c
            for j, p in enumerate(pw):
                if p: term *= row[j] ** p
            y += term
        out[i] = y
    return out


def optimal_knots(snap: pd.DataFrame, X_train, y_train, K: int, candidates, degree: int):
    sub_idx = RNG.choice(len(X_train), size=min(8000, len(X_train)), replace=False)
    Xs, ys  = X_train[sub_idx], y_train[sub_idx]
    best = None
    for combo in combinations(candidates, K - 1):
        knots = [0.0] + [float(c) for c in combo] + [float(GAME_MIN)]
        try:
            mod  = fit_piecewise(snap, knots, degree)
            pred = predict_piecewise(mod, Xs)
            mae  = mean_absolute_error(ys, pred)
            if best is None or mae < best[0]:
                best = (mae, knots, mod)
        except Exception:
            continue
    return best


# ---------------------------------------------------------------------------
# Main fit
# ---------------------------------------------------------------------------
def fit_all(snap: pd.DataFrame, fit_nn: bool = True) -> dict:
    train, test = split_train_test(snap)
    X_train, y_train = train[FEATURES].values, train["final_score"].values
    X_test,  y_test  = test[FEATURES].values,  test["final_score"].values
    results = []

    pred_naive = np.where(X_test[:,0] > 0,
                          X_test[:,1] * GAME_MIN / np.maximum(X_test[:,0], 1e-9), X_test[:,2])
    record("Naive linear (current * 48/min)", y_test, pred_naive, results)

    m_lin = LinearRegression().fit(X_train, y_train)
    record("Linear regression", y_test, m_lin.predict(X_test), results)

    poly_models = {}
    for deg in (2, 3, 4):
        pipe = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False), Ridge(alpha=1.0))
        pipe.fit(X_train, y_train)
        poly_models[deg] = pipe
        record(f"Polynomial deg {deg} + ridge (global)", y_test, pipe.predict(X_test), results)

    pw = {}
    pw["linear_2seg_uniform"] = fit_piecewise(snap, [0,24,48], 1);     record("Piecewise LINEAR — 2 segments (halves)",   y_test, predict_piecewise(pw["linear_2seg_uniform"], X_test), results)
    pw["linear_3seg_uniform"] = fit_piecewise(snap, [0,16,32,48], 1);   record("Piecewise LINEAR — 3 segments (thirds)",   y_test, predict_piecewise(pw["linear_3seg_uniform"], X_test), results)
    pw["linear_4seg_uniform"] = fit_piecewise(snap, [0,12,24,36,48],1); record("Piecewise LINEAR — 4 segments (quarters)", y_test, predict_piecewise(pw["linear_4seg_uniform"], X_test), results)
    pw["poly2_4seg_uniform"]  = fit_piecewise(snap, [0,12,24,36,48],2); record("Piecewise POLY-2 — 4 segments (quarters)", y_test, predict_piecewise(pw["poly2_4seg_uniform"],  X_test), results)
    pw["poly3_4seg_uniform"]  = fit_piecewise(snap, [0,12,24,36,48],3); record("Piecewise POLY-3 — 4 segments (quarters)", y_test, predict_piecewise(pw["poly3_4seg_uniform"],  X_test), results)

    candidates = [4,8,12,16,20,24,28,32,36,40,44]
    for K in (2,3,4):
        _, knots, mod = optimal_knots(snap, X_train, y_train, K, candidates, degree=1)
        pw[f"linear_{K}seg_optimal"] = mod
        record(f"Piecewise LIN — {K} segs (data-driven knots {knots[1:-1]})",
               y_test, predict_piecewise(mod, X_test), results)
    _, knots, mod = optimal_knots(snap, X_train, y_train, 4, candidates, degree=2)
    pw["poly2_4seg_optimal"] = mod
    record(f"Piecewise POLY-2 — 4 segs (data-driven knots {knots[1:-1]})",
           y_test, predict_piecewise(mod, X_test), results)

    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1).fit(X_train, y_train)
    record("Random Forest (100 trees)", y_test, rf.predict(X_test), results)
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=42).fit(X_train, y_train)
    record("Gradient Boosting (150)", y_test, gb.predict(X_test), results)

    def bayes_pred(X, k=12.0):
        m,s,p = X[:,0], X[:,1], X[:,2]
        prior = p / GAME_MIN
        obs   = np.where(m > 0, s / np.maximum(m, 1e-9), prior)
        a     = m / (m + k)
        rate  = a*obs + (1-a)*prior
        return s + rate * (GAME_MIN - m)
    record("Bayesian shrinkage (k=12)", y_test, bayes_pred(X_test), results)

    nn_dump = None
    if fit_nn:
        scaler = StandardScaler().fit(X_train)
        nn = MLPRegressor(hidden_layer_sizes=(16, 8), activation="relu",
                          max_iter=400, random_state=42, learning_rate_init=0.005).fit(
                              scaler.transform(X_train), y_train)
        record("Neural Net MLP (16-8 ReLU)", y_test, nn.predict(scaler.transform(X_test)), results)
        nn_dump = {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "layers": [
                {"W": l.tolist(), "b": b.tolist(), "activation": "relu"}
                for l, b in zip(nn.coefs_, nn.intercepts_)
            ],
        }
        # final layer is identity in MLPRegressor
        nn_dump["layers"][-1]["activation"] = "identity"

    def poly_dump(pipe, deg):
        poly  = pipe.named_steps["polynomialfeatures"]
        ridge = pipe.named_steps["ridge"]
        return {"degree": deg, "powers": poly.powers_.tolist(),
                "coef": [float(c) for c in ridge.coef_],
                "intercept": float(ridge.intercept_),
                "feature_order": FEATURES}
    return {
        "linear":     {"intercept": float(m_lin.intercept_),
                       "coef": [float(c) for c in m_lin.coef_],
                       "feature_order": FEATURES},
        "polynomial": {str(d): poly_dump(p, d) for d, p in poly_models.items()},
        "piecewise":  pw,
        "neural":     nn_dump,
        "results":    results,
    }


def build_teams_payload(priors: pd.DataFrame, league_avg: float) -> list[dict]:
    """Pick the 30 active teams (most games) and emit name/abbr + 3 priors."""
    active = priors.sort_values("n_games", ascending=False).head(30).sort_values("TEAM_NAME")
    teams = [{"name": "League average", "abbr": "AVG",
              "ppg": round(league_avg, 2),
              "ppg_recent5":  round(league_avg, 2),
              "ppg_overall":  round(league_avg, 2),
              "ppg_weighted": round(league_avg, 2)}]
    for _, r in active.iterrows():
        teams.append({
            "name": r["TEAM_NAME"], "abbr": r["TEAM_ABBREVIATION"],
            "ppg":          round(r["ppg_recent5"],  2),
            "ppg_recent5":  round(r["ppg_recent5"],  2),
            "ppg_overall":  round(r["ppg_overall"],  2),
            "ppg_weighted": round(r["ppg_weighted"], 2),
        })
    return teams


def write_outputs(priors: pd.DataFrame, models: dict, league_avg: float,
                  league_std: float, n_real: int, n_sim: int, n_snap: int):
    payload = {
        "metadata": {
            "n_simulated_games":    n_sim,
            "n_training_snapshots": n_snap,
            "n_real_team_games":    n_real,
            "league_avg_ppg":       round(float(league_avg), 3),
            "league_std_ppg":       round(float(league_std), 3),
            "seed":                 42,
            "shrink_k":             12.0,
            "home_advantage_pts":   2.5,
            "last_updated":         pd.Timestamp.utcnow().date().isoformat(),
            "data_source":          "github.com/NocturneBear/NBA-Data-2010-2024",
        },
        **models,
        "teams": build_teams_payload(priors, league_avg),
    }
    (DATA_DIR / "models.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(models["results"]).to_csv(DATA_DIR / "model_results.csv", index=False)

    # Emit models.js for the static page
    js_payload = json.dumps(payload, indent=2)
    js_text    = ("/* Generated by pipeline/build_models.py — do not edit by hand. */\n"
                  "window.MODELS = " + js_payload + ";\n")
    (PROJ_DIR / "models.js").write_text(js_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true", help="Re-pull raw NBA CSV before fitting")
    ap.add_argument("--no-nn",   action="store_true", help="Skip neural network (fast path)")
    ap.add_argument("--n-games", type=int, default=4000, help="Simulated games for training snapshots")
    args = ap.parse_args()

    df = load_raw(refetch=args.refetch)
    print(f">>> Loaded {len(df):,} team-game rows across {df['SEASON_YEAR'].nunique()} seasons")
    league_avg, league_std = float(df["PTS"].mean()), float(df["PTS"].std())
    priors  = compute_priors(df); print(f"    {len(priors)} team priors written")
    snap    = simulate_snapshots(priors, n_games=args.n_games)
    print(f"    {len(snap):,} training snapshots from {args.n_games} simulated games")
    models  = fit_all(snap, fit_nn=not args.no_nn)
    write_outputs(priors, models, league_avg, league_std,
                  n_real=len(df), n_sim=args.n_games, n_snap=len(snap))
    print("\n>>> Wrote models.js, data/models.json, data/model_results.csv, data/team_priors.csv")
    best = min(models["results"], key=lambda r: r["MAE"])
    print(f"    Best by MAE: {best['model']} ({best['MAE']})")


if __name__ == "__main__":
    main()
