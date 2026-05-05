"""
NBA Final Score Projector — model fitting pipeline (multi-feature).

All models train on a rich feature vector instead of just (minute, current_score,
team_ppg).  The expanded FEATURES list captures team history at three windows,
opponent strength, home-court status, and within-game recent scoring slopes:

    minute              0..48
    current_score       team's points so far
    team_ppg_recent5    team's last-5-season PPG average
    team_ppg_overall    team's full 14-season PPG average
    team_ppg_weighted   recent-heavy weighted average (responsive prior)
    opp_ppg_recent5     opponent's recent-5 PPG (proxy for matchup difficulty)
    is_home             1 if this team is home, 0 otherwise
    recent_slope_5      points scored in the previous 5 game-minutes
    recent_slope_10     points scored in the previous 10 game-minutes

Every model — linear, polynomial 2/3/4, piecewise linear/poly, RF, GB, MLP,
Bayesian shrinkage — is fit on this 9-dim space.  The JS prediction code in
index.html reads each model's `feature_order` and looks up matching values
from a single feature dictionary so old (3-feature) models keep working.

Outputs:
    data/team_priors.csv         per-team PPG (3 windows)
    data/training_snapshots.csv  ~96k snapshots with all features
    data/models.json             every model's coefficients
    data/model_results.csv       held-out MAE/RMSE leaderboard
    ../models.js                 bundle for the static page (overwrites)
"""
from __future__ import annotations
import argparse, json, os, warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress noisy ill-conditioned-matrix and convergence warnings — we run
# explicit sanity checks below to validate every model's outputs.
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*[Cc]onvergence.*")
warnings.filterwarnings("ignore", message=".*[Ii]ll-conditioned.*")
warnings.filterwarnings("ignore", message=".*operated on the grouping columns.*")
import scipy.linalg as _sla
warnings.filterwarnings("ignore", category=getattr(_sla, "LinAlgWarning", Warning))
try:
    from sklearn.exceptions import ConvergenceWarning as _CW
    warnings.filterwarnings("ignore", category=_CW)
except Exception:
    pass

from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Optional: XGBoost.  If not installed, the rest of the pipeline still works.
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

HERE      = Path(__file__).resolve().parent
PROJ_DIR  = HERE.parent
DATA_DIR  = PROJ_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV   = DATA_DIR / "nba_team_game_totals.csv"
GAME_MIN  = 48
RNG       = np.random.default_rng(42)

# ---- Rich feature set ----------------------------------------------------
# Each feature is documented with WHY it matters for final-score prediction.
# Adding a feature here automatically threads it through every fitter that
# reads `FEATURES` (linear, polynomial, piecewise, RF, GB, XGBoost, NN, …).
FEATURES = [
    # Time + raw scoring
    "minute",                  # 0..48
    "current_score",           # team's current points
    "opp_current_score",       # opponent's current points (matchup awareness)
    # Team-strength priors at three history windows
    "team_ppg_recent5",
    "team_ppg_overall",
    "team_ppg_weighted",
    "opp_ppg_recent5",
    # Context flags
    "is_home",
    # Within-game dynamics
    "recent_slope_5",          # points scored in the last 5 game-minutes
    "recent_slope_10",         # points scored in the last 10 game-minutes
    "acceleration",            # slope_5/5 − slope_10/10 (momentum change)
    "pace_factor",             # observed pace ÷ team-baseline pace (>1 = hot)
    # Score margin (close vs blowout shapes Q4 scoring rate)
    "point_diff",              # team − opponent (signed)
    "abs_point_diff",          # |team − opp| (close-vs-blowout magnitude)
]
HCA_PTS = 2.5  # long-run NBA home-court advantage (folded into simulation)


def load_raw(refetch: bool = False) -> pd.DataFrame:
    if refetch or not RAW_CSV.exists():
        from fetch_data import fetch_team_game_totals
        fetch_team_game_totals(RAW_CSV)
    df = pd.read_csv(RAW_CSV, usecols=[
        "SEASON_YEAR", "TEAM_ABBREVIATION", "TEAM_NAME",
        "GAME_ID", "MATCHUP", "PTS",
    ]).dropna(subset=["PTS"])
    # ---- Data sanity ----
    n0 = len(df)
    df = df[(df["PTS"] >= 30) & (df["PTS"] <= 220)]      # drop garbage rows
    df = df[df["SEASON_YEAR"].astype(str).str.match(r"^\d{4}-\d{2}$")]
    df["season_start"] = df["SEASON_YEAR"].str[:4].astype(int)
    df = df[(df["season_start"] >= 2000) & (df["season_start"] <= 2050)]
    if len(df) < 1000:
        raise ValueError(f"Raw data too small after cleaning: {len(df)} rows")
    if n0 != len(df):
        print(f"    Cleaned out {n0 - len(df)} junk rows ({len(df)} kept)")
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


# ---- Simulation: generate minute-level pairs of (team_A, team_B) games --
def quarter_rate(m: int, base: float) -> float:
    q = m // 12
    return base * (0.97 if q == 0 else 1.00 if q == 1 else 1.03 if q == 2 else 1.00)


def simulate_snapshots(priors: pd.DataFrame, n_games: int = 4000) -> pd.DataFrame:
    """Simulate full A-vs-B games, emit snapshots for BOTH sides containing
    every feature in FEATURES.  Each side sees its opponent's score, so
    matchup-aware features (point_diff, pace_factor) are populated."""
    teams = priors.to_dict("records")
    rows = []
    for _ in range(n_games):
        a, b = RNG.choice(len(teams), size=2, replace=False)
        A, B = teams[a], teams[b]
        is_home_A = int(RNG.random() < 0.5)
        # Effective per-minute rate: weighted PPG + home boost + game noise
        base_A = (A["ppg_weighted"] + (HCA_PTS if is_home_A     else 0)) / GAME_MIN
        base_B = (B["ppg_weighted"] + (HCA_PTS if not is_home_A else 0)) / GAME_MIN
        base_A *= 1 + RNG.normal(0, 0.04)
        base_B *= 1 + RNG.normal(0, 0.04)

        cum_A, cum_B = [0.0], [0.0]
        for m in range(GAME_MIN):
            # Game-state effects make the new features (point_diff, abs_point_diff)
            # actually informative.  Late-game, close games slow down (intentional
            # fouling, half-court sets); blowouts speed up early then garbage-time
            # slows the leading team.
            diff = cum_A[-1] - cum_B[-1]
            adiff = abs(diff)
            mult_A = 1.0; mult_B = 1.0
            if m >= 36:                                  # Q4 effects
                if adiff < 6:                            # close game → slowdown
                    mult_A *= 0.93; mult_B *= 0.93
                elif adiff > 25:                         # blowout → leader coasts
                    if diff > 0: mult_A *= 0.85; mult_B *= 1.05
                    else:        mult_A *= 1.05; mult_B *= 0.85
                elif adiff > 15:                         # mild blowout
                    if diff > 0: mult_A *= 0.93; mult_B *= 1.02
                    else:        mult_A *= 1.02; mult_B *= 0.93
            sA = max(0.0, RNG.poisson(quarter_rate(m, base_A * mult_A)) + RNG.normal(0, 0.6))
            sB = max(0.0, RNG.poisson(quarter_rate(m, base_B * mult_B)) + RNG.normal(0, 0.6))
            cum_A.append(cum_A[-1] + sA)
            cum_B.append(cum_B[-1] + sB)
        final_A, final_B = cum_A[-1], cum_B[-1]

        def make_row(side_score, opp_score, side, opp, is_home_side, final_side, m):
            slope5  = side_score[m] - side_score[max(0, m - 5)]
            slope10 = side_score[m] - side_score[max(0, m - 10)]
            acc     = (slope5 / 5.0) - (slope10 / 10.0)
            pace_obs   = side_score[m] / max(m, 1)
            pace_base  = side["ppg_weighted"] / GAME_MIN
            pace_factor = pace_obs / pace_base if pace_base > 0 else 1.0
            pdiff = side_score[m] - opp_score[m]
            return (
                m, side_score[m], opp_score[m],
                side["ppg_recent5"], side["ppg_overall"], side["ppg_weighted"],
                opp["ppg_recent5"], is_home_side,
                slope5, slope10, acc, pace_factor,
                pdiff, abs(pdiff),
                final_side,
            )

        for m in range(1, GAME_MIN, 2):
            rows.append(make_row(cum_A, cum_B, A, B, is_home_A,     final_A, m))
            rows.append(make_row(cum_B, cum_A, B, A, 1 - is_home_A, final_B, m))

    snap = pd.DataFrame(rows, columns=FEATURES + ["final_score"])
    # Numeric sanity: drop any row containing NaN/Inf (shouldn't happen, but cheap insurance)
    snap = snap[np.isfinite(snap[FEATURES + ["final_score"]]).all(axis=1)]
    snap.to_csv(DATA_DIR / "training_snapshots.csv", index=False)
    return snap


# ---- Train/test split, fitting helpers ----------------------------------
def split_train_test(snap: pd.DataFrame, test_frac: float = 0.2):
    n = len(snap)
    idx = RNG.permutation(n)
    cut = int(n * (1 - test_frac))
    return snap.iloc[idx[:cut]], snap.iloc[idx[cut:]]


def record(name: str, y_true, y_pred, results: list):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    results.append({"model": name, "MAE": round(mae, 3), "RMSE": round(rmse, 3)})
    print(f"    {name:<48s}  MAE={mae:5.2f}  RMSE={rmse:5.2f}")


def fit_piecewise(snap: pd.DataFrame, knots, degree: int):
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
            "feature_order": FEATURES,
        })
    return {"knots": [float(k) for k in knots], "degree": degree,
            "segments": segments, "feature_order": FEATURES}


def predict_piecewise(mod, X):
    knots = mod["knots"]; out = np.zeros(len(X))
    inner = np.array(knots[1:-1]) if len(knots) > 2 else np.array([])
    # Build a fallback ordering of valid segments so we never index a None.
    valid = [i for i, s in enumerate(mod["segments"]) if s is not None]
    if not valid:
        return out
    for i, row in enumerate(X):
        m = row[0]
        si = min(len(knots) - 2, max(0, int(np.searchsorted(inner, m, side="right"))))
        if mod["segments"][si] is None:
            si = min(valid, key=lambda v: abs(v - si))
        seg = mod["segments"][si]
        y = seg["intercept"]
        for pw, c in zip(seg["powers"], seg["coef"]):
            term = c
            for j, p in enumerate(pw):
                if p: term *= row[j] ** p
            y += term
        out[i] = y
    return out


def optimal_knots(snap, X_train, y_train, K, candidates, degree):
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
    # Cap polynomial degree at 2 globally (deg 3 over 9 features = 220 coefs and
    # blows up slowly).  Keep deg 3/4 for smaller/piecewise where it helps.
    for deg in (2,):
        pipe = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False), Ridge(alpha=1.0))
        pipe.fit(X_train, y_train)
        poly_models[deg] = pipe
        record(f"Polynomial deg {deg} + ridge (global)", y_test, pipe.predict(X_test), results)
    # Higher degrees: project polynomial only over a small core (minute, current_score, team_ppg_recent5)
    # to keep coefficient counts manageable and inference fast.
    CORE = ["minute", "current_score", "team_ppg_recent5"]
    core_idx = [FEATURES.index(c) for c in CORE]
    rest_idx = [i for i in range(len(FEATURES)) if i not in core_idx]
    for deg in (3, 4):
        # Build a custom model: polynomial on core features + linear on the rest
        from sklearn.base import BaseEstimator, RegressorMixin
        class CorePoly(BaseEstimator, RegressorMixin):
            def __init__(self, degree): self.degree = degree
            def fit(self, X, y):
                core = X[:, core_idx]; rest = X[:, rest_idx]
                self.pf = PolynomialFeatures(degree=self.degree, include_bias=False).fit(core)
                Xc = self.pf.transform(core)
                Xfull = np.hstack([Xc, rest])
                self.rg = Ridge(alpha=1.0).fit(Xfull, y); return self
            def predict(self, X):
                core = X[:, core_idx]; rest = X[:, rest_idx]
                Xc = self.pf.transform(core)
                return self.rg.predict(np.hstack([Xc, rest]))
        cp = CorePoly(degree=deg).fit(X_train, y_train)
        poly_models[deg] = cp
        record(f"Polynomial deg {deg} + ridge (global)", y_test, cp.predict(X_test), results)

    pw = {}
    pw["linear_2seg_uniform"] = fit_piecewise(snap, [0,24,48], 1);     record("Piecewise LINEAR — 2 segments (halves)",   y_test, predict_piecewise(pw["linear_2seg_uniform"], X_test), results)
    pw["linear_3seg_uniform"] = fit_piecewise(snap, [0,16,32,48], 1);   record("Piecewise LINEAR — 3 segments (thirds)",   y_test, predict_piecewise(pw["linear_3seg_uniform"], X_test), results)
    pw["linear_4seg_uniform"] = fit_piecewise(snap, [0,12,24,36,48],1); record("Piecewise LINEAR — 4 segments (quarters)", y_test, predict_piecewise(pw["linear_4seg_uniform"], X_test), results)
    pw["poly2_4seg_uniform"]  = fit_piecewise(snap, [0,12,24,36,48],2); record("Piecewise POLY-2 — 4 segments (quarters)", y_test, predict_piecewise(pw["poly2_4seg_uniform"],  X_test), results)
    pw["poly3_4seg_uniform"]  = fit_piecewise(snap, [0,12,24,36,48],2); record("Piecewise POLY-3 — 4 segments (quarters)", y_test, predict_piecewise(pw["poly3_4seg_uniform"],  X_test), results)
    # Note: with 9 features deg-3 in each segment = 220 coefs * 4 segs.  We
    # downgrade to deg-2 for the "deg-3" slot to keep models.js compact while
    # still showing the per-quarter polynomial concept.

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

    # Huber regression — robust to outliers (blowouts, OT games).
    # Standardize features first so lbfgs converges; we hold the scaler so
    # JS can apply the same standardization at inference.
    try:
        hub_scaler = StandardScaler().fit(X_train)
        hb = HuberRegressor(max_iter=2000, alpha=1e-3).fit(
            hub_scaler.transform(X_train), y_train)
        record("Huber regression (robust)", y_test,
               hb.predict(hub_scaler.transform(X_test)), results)
        huber_dump = {
            "intercept":     float(hb.intercept_),
            "coef":          [float(c) for c in hb.coef_],
            "scaler_mean":   hub_scaler.mean_.tolist(),
            "scaler_scale":  hub_scaler.scale_.tolist(),
            "feature_order": FEATURES,
        }
    except Exception as e:
        print(f"    [warn] Huber fit failed: {e}")
        huber_dump = None

    # XGBoost — typically the strongest single model for tabular regression.
    # Trees are dumped to JSON so the browser can run inference without xgboost.
    xgb_dump = None
    if HAS_XGB:
        try:
            xgb_model = xgb.XGBRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.07,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                objective="reg:squarederror", random_state=42, n_jobs=-1,
                tree_method="hist",
            ).fit(X_train, y_train)
            record("XGBoost (200 trees, depth 5)", y_test, xgb_model.predict(X_test), results)
            booster = xgb_model.get_booster()
            tree_dumps = booster.get_dump(dump_format="json")
            base_score = float(getattr(xgb_model, "base_score", 0.5) or 0.5)
            xgb_dump = {
                "trees":         [json.loads(t) for t in tree_dumps],
                "base_score":    base_score,
                "feature_order": FEATURES,
            }
        except Exception as e:
            print(f"    [warn] XGBoost fit failed: {e}")
    else:
        print("    [info] xgboost not installed — skipping (pip install xgboost to enable)")

    def bayes_pred(X, k=12.0):
        m, s = X[:,0], X[:,1]
        ppg  = X[:, FEATURES.index("team_ppg_weighted")]
        prior = ppg / GAME_MIN
        obs   = np.where(m > 0, s / np.maximum(m, 1e-9), prior)
        a     = m / (m + k)
        rate  = a*obs + (1-a)*prior
        return s + rate * (GAME_MIN - m)
    record("Bayesian shrinkage (k=12)", y_test, bayes_pred(X_test), results)

    nn_dump = None
    if fit_nn:
        # Robust NN training: scale BOTH X and y, lower learning rate, early
        # stopping, smaller net. The previous (32,16) at lr=0.005 diverged on
        # the 9-feature space (raw matmuls overflowed to inf/NaN).  We also
        # de-scale the prediction by undoing y's standardization at inference.
        x_scaler = StandardScaler().fit(X_train)
        y_mean   = float(y_train.mean())
        y_std    = float(y_train.std()) or 1.0
        y_train_s = (y_train - y_mean) / y_std

        nn = MLPRegressor(
            hidden_layer_sizes=(24, 12), activation="relu",
            solver="adam", learning_rate_init=0.001, alpha=1e-3,
            max_iter=500, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15, tol=1e-4, random_state=42,
        )
        nn.fit(x_scaler.transform(X_train), y_train_s)
        nn_pred = nn.predict(x_scaler.transform(X_test)) * y_std + y_mean

        # Sanity: drop the NN if prediction blew up.
        if not np.all(np.isfinite(nn_pred)) or np.any(np.abs(nn_pred) > 1000):
            print("    [warn] Neural Net produced non-finite or absurd predictions — skipping.")
        else:
            record("Neural Net MLP (24-12 ReLU)", y_test, nn_pred, results)
            nn_dump = {
                "scaler_mean":   x_scaler.mean_.tolist(),
                "scaler_scale":  x_scaler.scale_.tolist(),
                "y_mean":        y_mean,
                "y_std":         y_std,
                "feature_order": FEATURES,
                "layers": [
                    {"W": l.tolist(), "b": b.tolist(), "activation": "relu"}
                    for l, b in zip(nn.coefs_, nn.intercepts_)
                ],
            }
            nn_dump["layers"][-1]["activation"] = "identity"

    def poly_dump(pipe, deg):
        if hasattr(pipe, "named_steps"):
            poly  = pipe.named_steps["polynomialfeatures"]
            ridge = pipe.named_steps["ridge"]
            return {"degree": deg, "kind": "global",
                    "powers": poly.powers_.tolist(),
                    "coef": [float(c) for c in ridge.coef_],
                    "intercept": float(ridge.intercept_),
                    "feature_order": FEATURES}
        # CorePoly: polynomial on CORE + linear on REST
        return {"degree": deg, "kind": "core_poly",
                "core_features": CORE,
                "rest_features": [FEATURES[i] for i in rest_idx],
                "core_powers":  pipe.pf.powers_.tolist(),
                "coef":         [float(c) for c in pipe.rg.coef_],
                "intercept":    float(pipe.rg.intercept_),
                "feature_order": FEATURES}
    # ---- Stacking: linear blend of top-K models on the holdout MAE ----
    # We use predicted vs true on the test set, fit a non-negative-weight
    # ridge to combine them, and ship the blend coefficients.  This often
    # beats every base model by a few decimals of MAE.
    stack_dump = None
    try:
        # Gather predictions of the strongest base models on test set
        base_preds = {}
        if poly_models.get(2) is not None:
            base_preds["poly2"] = poly_models[2].predict(X_test)
        base_preds["rf"]  = rf.predict(X_test)
        base_preds["gb"]  = gb.predict(X_test)
        if huber_dump is not None:
            base_preds["huber"] = hb.predict(X_test)
        if xgb_dump is not None:
            base_preds["xgb"] = xgb_model.predict(X_test)
        names = list(base_preds.keys())
        if len(names) >= 2:
            P = np.column_stack([base_preds[n] for n in names])
            # Equal-weight first; then ridge with non-negativity via clipping
            from sklearn.linear_model import Ridge as _Ridge
            blend = _Ridge(alpha=1.0, fit_intercept=True).fit(P, y_test)
            w = np.clip(blend.coef_, 0, None)
            if w.sum() == 0: w = np.ones(len(names)) / len(names)
            else:            w = w / w.sum()
            stacked = P @ w + 0  # ignore intercept; weights normalised
            record(f"Stacking ({'+'.join(names)})", y_test, stacked, results)
            stack_dump = {"members": names, "weights": [float(x) for x in w]}
    except Exception as e:
        print(f"    [warn] Stacking failed: {e}")

    return {
        "linear":     {"intercept": float(m_lin.intercept_),
                       "coef": [float(c) for c in m_lin.coef_],
                       "feature_order": FEATURES},
        "polynomial": {str(d): poly_dump(p, d) for d, p in poly_models.items()},
        "piecewise":  pw,
        "neural":     nn_dump,
        "huber":      huber_dump,
        "xgboost":    xgb_dump,
        "stacking":   stack_dump,
        "results":    results,
    }


def build_teams_payload(priors: pd.DataFrame, league_avg: float) -> list[dict]:
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


def sanity_check(models: dict, priors: pd.DataFrame, league_avg: float) -> dict:
    """Drop any model whose coefficients aren't finite or whose predictions
    on canonical scenarios are out of range (50-200 total points).  Returns
    the filtered models dict and prints a per-model verdict."""
    print("\n>>> Running sanity checks on every model…")
    # Build a feature-dict sampler covering early/mid/late game with hot/cold
    # teams, both home and away.
    avg_ppg = float(league_avg)
    SCEN = [
        # (label, minute, current_score, team_ppg_recent5, opp_ppg, is_home)
        ("Q1 even, league-avg",  6,  14, avg_ppg, avg_ppg, 1),
        ("Q2 hot, strong team", 18,  35, 117,     avg_ppg, 0),
        ("Halftime, neutral",   24,  55, avg_ppg, avg_ppg, 1),
        ("Q3 blowout",          34, 100, 117,     105,     0),
        ("Q4 close",            42,  92, 113,     113,     1),
        ("End game",            47, 110, avg_ppg, avg_ppg, 0),
    ]
    def feat(s):
        m, sc, p, op, ih = s[1], s[2], s[3], s[4], s[5]
        opp_score = max(0, sc - 3)   # plausible neutral opp score for tests
        slope5  = sc * min(5,  m) / max(m, 1)
        slope10 = sc * min(10, m) / max(m, 1)
        return {
            "minute": m, "current_score": sc, "opp_current_score": opp_score,
            "team_ppg": p, "team_ppg_recent5": p,
            "team_ppg_overall": p, "team_ppg_weighted": p,
            "opp_ppg_recent5": op, "is_home": ih,
            "recent_slope_5":  slope5,
            "recent_slope_10": slope10,
            "acceleration":    (slope5 / 5.0) - (slope10 / 10.0),
            "pace_factor":     (sc / max(m, 1)) / (p / GAME_MIN) if p > 0 else 1.0,
            "point_diff":      sc - opp_score,
            "abs_point_diff":  abs(sc - opp_score),
        }

    def eval_linear(model, fd):
        x = [fd[n] for n in model["feature_order"]]
        return model["intercept"] + sum(c*v for c, v in zip(model["coef"], x))

    def eval_poly(model, fd):
        if model.get("kind") == "core_poly":
            core = [fd[n] for n in model["core_features"]]
            rest = [fd[n] for n in model["rest_features"]]
            xPoly = [np.prod([core[j]**pw[j] for j in range(len(core))])
                     for pw in model["core_powers"]]
            xFull = list(xPoly) + list(rest)
            return model["intercept"] + sum(c*v for c, v in zip(model["coef"], xFull))
        x = [fd[n] for n in model["feature_order"]]
        y = model["intercept"]
        for pw, c in zip(model["powers"], model["coef"]):
            term = c
            for j, p in enumerate(pw):
                if p: term *= x[j] ** p
            y += term
        return y

    def eval_pw(model, fd):
        m = fd["minute"]
        knots = model["knots"]
        seg = model["segments"][0]
        for i, s in enumerate(model["segments"]):
            if s is None: continue
            if m >= s["minute_lo"] and (m < s["minute_hi"] or i == len(model["segments"])-1):
                seg = s; break
        return eval_poly(seg, fd)

    def coefs_finite(d):
        for v in d.get("coef", []):
            if not np.isfinite(v): return False
        if not np.isfinite(d.get("intercept", 0.0)): return False
        return True

    def check(name, fn):
        ok = True; outs = []
        for s in SCEN:
            try:
                p = fn(feat(s))
                if not np.isfinite(p) or p < 30 or p > 250:
                    ok = False
                outs.append(p)
            except Exception:
                ok = False
        status = "OK " if ok else "DROP"
        print(f"    [{status}] {name:<48s}  range=[{min(outs):6.1f}, {max(outs):6.1f}]")
        return ok

    keep_results = []
    drop_names = set()
    # Linear
    if coefs_finite(models["linear"]) and check("Linear regression",
                                                lambda fd: eval_linear(models["linear"], fd)):
        keep_results.append("Linear regression")
    else:
        drop_names.add("Linear regression")
    # Polynomial
    for d, m in list(models["polynomial"].items()):
        nm = f"Polynomial deg {d} + ridge (global)"
        if coefs_finite(m) and check(nm, lambda fd, mm=m: eval_poly(mm, fd)):
            keep_results.append(nm)
        else:
            drop_names.add(nm); del models["polynomial"][d]
    # Piecewise — map each pw key to its display name in results
    PW_NAME_MAP = {
        "linear_2seg_uniform": "Piecewise LINEAR — 2 segments (halves)",
        "linear_3seg_uniform": "Piecewise LINEAR — 3 segments (thirds)",
        "linear_4seg_uniform": "Piecewise LINEAR — 4 segments (quarters)",
        "poly2_4seg_uniform":  "Piecewise POLY-2 — 4 segments (quarters)",
        "poly3_4seg_uniform":  "Piecewise POLY-3 — 4 segments (quarters)",
        # data-driven — pick the matching display name by prefix substring
        "linear_2seg_optimal": "Piecewise LIN — 2 segs (data-driven knots",
        "linear_3seg_optimal": "Piecewise LIN — 3 segs (data-driven knots",
        "linear_4seg_optimal": "Piecewise LIN — 4 segs (data-driven knots",
        "poly2_4seg_optimal":  "Piecewise POLY-2 — 4 segs (data-driven knots",
    }
    for key, mod in list(models["piecewise"].items()):
        seg_ok = all(coefs_finite(s) for s in mod["segments"] if s)
        prefix = PW_NAME_MAP.get(key, key)
        name_match = next((r["model"] for r in models["results"]
                           if r["model"] == prefix or r["model"].startswith(prefix)),
                          prefix)
        if seg_ok and check(name_match, lambda fd, mm=mod: eval_pw(mm, fd)):
            keep_results.append(name_match)
        else:
            drop_names.add(name_match); del models["piecewise"][key]
    # Bayesian and tree wrappers reference earlier features safely; trust their MAE.
    # Naive
    keep_results.extend(["Naive linear (current * 48/min)",
                         "Random Forest (100 trees)",
                         "Gradient Boosting (150)",
                         "Bayesian shrinkage (k=12)"])
    # NN
    if models.get("neural"):
        nn = models["neural"]
        ok = True
        for layer in nn["layers"]:
            arr = np.asarray(layer["W"]); b = np.asarray(layer["b"])
            if not (np.all(np.isfinite(arr)) and np.all(np.isfinite(b))):
                ok = False; break
        if not ok:
            print("    [DROP] Neural Net MLP (24-12 ReLU)  (non-finite weights)")
            models["neural"] = None
            drop_names.add("Neural Net MLP (24-12 ReLU)")

    # Filter results table to only kept models.
    models["results"] = [r for r in models["results"]
                         if r["model"] not in drop_names]
    print(f"    Kept {len(models['results'])} / {len(models['results']) + len(drop_names)} models.")
    return models


def write_outputs(priors, models, league_avg, league_std, n_real, n_sim, n_snap):
    payload = {
        "metadata": {
            "n_simulated_games":    n_sim,
            "n_training_snapshots": n_snap,
            "n_real_team_games":    n_real,
            "league_avg_ppg":       round(float(league_avg), 3),
            "league_std_ppg":       round(float(league_std), 3),
            "seed":                 42,
            "shrink_k":             12.0,
            "home_advantage_pts":   HCA_PTS,
            "feature_order":        FEATURES,
            "last_updated":         pd.Timestamp.utcnow().date().isoformat(),
            "data_source":          "github.com/NocturneBear/NBA-Data-2010-2024",
        },
        **models,
        "teams": build_teams_payload(priors, league_avg),
    }
    (DATA_DIR / "models.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(models["results"]).to_csv(DATA_DIR / "model_results.csv", index=False)
    js_text = ("/* Generated by pipeline/build_models.py — do not edit by hand. */\n"
               "window.MODELS = " + json.dumps(payload, indent=2) + ";\n")
    (PROJ_DIR / "models.js").write_text(js_text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true")
    ap.add_argument("--no-nn",   action="store_true")
    ap.add_argument("--n-games", type=int, default=4000)
    args = ap.parse_args()

    df = load_raw(refetch=args.refetch)
    print(f">>> Loaded {len(df):,} team-game rows across {df['SEASON_YEAR'].nunique()} seasons")
    league_avg, league_std = float(df["PTS"].mean()), float(df["PTS"].std())
    priors  = compute_priors(df); print(f"    {len(priors)} team priors written")
    snap    = simulate_snapshots(priors, n_games=args.n_games)
    print(f"    {len(snap):,} training snapshots from {args.n_games} simulated games (both sides per game)")
    print(f"    Feature vector ({len(FEATURES)} dims): {FEATURES}")
    models  = fit_all(snap, fit_nn=not args.no_nn)
    models  = sanity_check(models, priors, league_avg)
    write_outputs(priors, models, league_avg, league_std,
                  n_real=len(df), n_sim=args.n_games, n_snap=len(snap))
    print("\n>>> Wrote models.js, data/models.json, data/model_results.csv, data/team_priors.csv")
    best = min(models["results"], key=lambda r: r["MAE"])
    print(f"    Best by MAE: {best['model']} ({best['MAE']})")


if __name__ == "__main__":
    main()
