# momentum_train.py
# momentum_train.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple
from scipy.special import expit, logit
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Feature engineering (match-level)
# -----------------------------
_SCORE_MAP_NORMAL = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, 0: 0, 15: 1, 30: 2, 40: 3}


def _parse_point_score(x) -> float:
    if pd.isna(x):
        return np.nan
    if x in _SCORE_MAP_NORMAL:
        return float(_SCORE_MAP_NORMAL[x])
    try:
        return float(int(x))  # tie-break numeric like "7"
    except Exception:
        return np.nan


def _is_game_point(p_score_num: float, o_score_num: float) -> bool:
    if np.isnan(p_score_num) or np.isnan(o_score_num):
        return False
    return (p_score_num >= 3) and ((p_score_num - o_score_num) >= 1)


def _infer_is_tie_break(game_no: int) -> bool:
    # Wimbledon dataset uses game_no==13 to denote tie-break game within a set (approx)
    return int(game_no) == 13


def _infer_is_set_point_row(r: pd.Series) -> bool:
    # For tie-break: set point if >=6 and lead >=1 (approx)
    if _infer_is_tie_break(r["game_no"]):
        p1s = _parse_point_score(r["p1_score"])
        p2s = _parse_point_score(r["p2_score"])
        if np.isnan(p1s) or np.isnan(p2s):
            return False
        return (p1s >= 6 and (p1s - p2s) >= 1) or (p2s >= 6 and (p2s - p1s) >= 1)

    p1_games = int(r["p1_games"])
    p2_games = int(r["p2_games"])
    p1_score = _parse_point_score(r["p1_score"])
    p2_score = _parse_point_score(r["p2_score"])
    p1_gp = _is_game_point(p1_score, p2_score)
    p2_gp = _is_game_point(p2_score, p1_score)

    # approximate: set can be clinched at 6-0..6-4 on game point while leading 5-x
    p1_clinch_6 = (p1_games == 5 and p2_games <= 4 and p1_gp)
    p2_clinch_6 = (p2_games == 5 and p1_games <= 4 and p2_gp)

    # 7-5 clinch: at 6-5 on game point
    p1_clinch_75 = (p1_games == 6 and p2_games == 5 and p1_gp)
    p2_clinch_75 = (p2_games == 6 and p1_games == 5 and p2_gp)
    return bool(p1_clinch_6 or p2_clinch_6 or p1_clinch_75 or p2_clinch_75)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "match_id",
        "set_no",
        "game_no",
        "point_no",
        "server",
        "serve_no",
        "point_victor",
        "game_victor",
        "p1_sets",
        "p2_sets",
        "p1_games",
        "p2_games",
        "p1_score",
        "p2_score",
        "p1_break_pt",
        "p2_break_pt",
        "player1",
        "player2",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    # y: 1 if player1 wins point
    out["y"] = (out["point_victor"] == 1).astype(int)

    # Base features
    out["is_server_p1"] = (out["server"] == 1).astype(int)
    out["is_second_serve"] = (out["serve_no"] == 2).astype(int)

    # Pressure flags
    out["is_break_point"] = ((out["p1_break_pt"] == 1) | (out["p2_break_pt"] == 1)).astype(bool)
    out["is_tie_break"] = out["game_no"].apply(_infer_is_tie_break).astype(bool)
    out["is_set_point"] = out.apply(_infer_is_set_point_row, axis=1).astype(bool)

    # Score gaps (player1 - player2)
    out["score_gap_sets"] = (out["p1_sets"].astype(int) - out["p2_sets"].astype(int)).astype(int)
    out["score_gap_games"] = (out["p1_games"].astype(int) - out["p2_games"].astype(int)).astype(int)

    # Ensure correct order
    out = out.sort_values(["match_id", "set_no", "game_no", "point_no"], kind="mergesort").reset_index(drop=True)
    return out


def add_future_targets(df: pd.DataFrame, K: int = 5) -> pd.DataFrame:
    """Add point-level future targets for training.

    - future_winrate_K: mean of next K points' outcomes (player1 win = 1)
    - future_wins_K: count of next K points won by player1

    Notes:
      * Targets are computed within each match (no cross-match leakage)
      * For the last K points in a match, targets are NaN.
    """
    if K <= 0:
        raise ValueError("K must be positive")

    out = df.copy()
    out["future_wins_K"] = np.nan
    out["future_winrate_K"] = np.nan

    for _, d in out.groupby("match_id", sort=False):
        y = d["y"].to_numpy(float)
        n = len(d)
        wins = np.full(n, np.nan, float)
        rate = np.full(n, np.nan, float)
        if n > K:
            for t in range(0, n - K):
                c = float(np.sum(y[t + 1 : t + K + 1]))
                wins[t] = c
                rate[t] = c / float(K)
        out.loc[d.index, "future_wins_K"] = wins
        out.loc[d.index, "future_winrate_K"] = rate

    return out


# -----------------------------
# Base model (logistic regression)
# -----------------------------
def fit_base_model(df: pd.DataFrame) -> Tuple[LogisticRegression, pd.DataFrame]:
    X = df[["is_server_p1", "is_second_serve"]].values
    y = df["y"].values
    model = LogisticRegression(solver="lbfgs", max_iter=500)
    model.fit(X, y)

    prob = model.predict_proba(X)[:, 1]
    prob = np.clip(prob, 1e-5, 1 - 1e-5)

    out = df.copy()
    out["base_prob"] = prob
    out["base_logit"] = logit(prob)
    return model, out


# -----------------------------
# Momentum model
# -----------------------------
@dataclass
class MomentumParams:
    alpha: float
    k: float
    gamma_break: float
    gamma_tie: float
    gamma_set: float
    w_s: float
    w_g: float


def _gamma_t(is_set: bool, is_break: bool, is_tie: bool, p: MomentumParams) -> float:
    if is_set:
        return p.gamma_set
    if is_break:
        return p.gamma_break
    if is_tie:
        return p.gamma_tie
    return 1.0


def calculate_nll_next_point(x: np.ndarray, df: pd.DataFrame, fixed_alpha: float) -> float:
    """Original objective: next-point negative log-likelihood."""
    # x = [k, gamma_break, gamma_tie, gamma_set, w_s, w_g]
    p = MomentumParams(
        alpha=float(fixed_alpha),
        k=float(x[0]),
        gamma_break=float(x[1]),
        gamma_tie=float(x[2]),
        gamma_set=float(x[3]),
        w_s=float(x[4]),
        w_g=float(x[5]),
    )

    eps = 1e-12
    nll = 0.0

    for _, d in df.groupby("match_id", sort=False):
        y = d["y"].to_numpy(float)
        base_prob = d["base_prob"].to_numpy(float)
        base_logit = d["base_logit"].to_numpy(float)
        gap_sets = d["score_gap_sets"].to_numpy(float)
        gap_games = d["score_gap_games"].to_numpy(float)

        is_break = d["is_break_point"].to_numpy(bool)
        is_tie = d["is_tie_break"].to_numpy(bool)
        is_set = d["is_set_point"].to_numpy(bool)

        n = len(d)
        if n < 2:
            continue

        v_prev = 0.0
        for t in range(n - 1):
            gamma = _gamma_t(bool(is_set[t]), bool(is_break[t]), bool(is_tie[t]), p)

            Omega = gamma * (y[t] - base_prob[t])
            v_t = p.alpha * Omega + (1.0 - p.alpha) * v_prev
            v_prev = v_t

            inertia = p.w_s * gap_sets[t] + p.w_g * gap_games[t]
            M_t = np.tanh(p.k * (v_t + inertia))

            # predict next point
            p_hat = expit(base_logit[t + 1] + M_t)
            p_hat = float(np.clip(p_hat, 1e-5, 1 - 1e-5))
            y_next = y[t + 1]

            nll -= y_next * np.log(p_hat + eps) + (1 - y_next) * np.log(1 - p_hat + eps)

    # mild regularization (optional)
    nll += 0.01 * (p.w_s**2 + p.w_g**2)
    return float(nll)


def calculate_loss_future_winrateK(
    x: np.ndarray,
    df: pd.DataFrame,
    fixed_alpha: float,
    K: int = 5,
    weight_per_point: bool = True,
) -> float:
    """Train objective: predict *future K points win-rate* from current momentum state.

    For each point t, define target:
        r_t = mean(y_{t+1}..y_{t+K})

    We predict r_hat_t by rolling out K steps in **expectation** (no sampling):
        p_{t+s} = sigmoid(base_logit_{t+s} + M_{t+s-1})
        then update v/M using expected outcome E[y_{t+s}] = p_{t+s}.

    Loss uses Bernoulli cross-entropy with fractional label r_t.
    """
    if K <= 0:
        raise ValueError("K must be positive")

    p = MomentumParams(
        alpha=float(fixed_alpha),
        k=float(x[0]),
        gamma_break=float(x[1]),
        gamma_tie=float(x[2]),
        gamma_set=float(x[3]),
        w_s=float(x[4]),
        w_g=float(x[5]),
    )

    eps = 1e-12
    loss = 0.0

    for _, d in df.groupby("match_id", sort=False):
        y = d["y"].to_numpy(float)
        base_prob = d["base_prob"].to_numpy(float)
        base_logit = d["base_logit"].to_numpy(float)
        gap_sets = d["score_gap_sets"].to_numpy(float)
        gap_games = d["score_gap_games"].to_numpy(float)
        is_break = d["is_break_point"].to_numpy(bool)
        is_tie = d["is_tie_break"].to_numpy(bool)
        is_set = d["is_set_point"].to_numpy(bool)

        n = len(d)
        if n <= K:
            continue

        v_prev = 0.0
        for t in range(n):
            # --- update momentum state using *observed* point t ---
            gamma = _gamma_t(bool(is_set[t]), bool(is_break[t]), bool(is_tie[t]), p)
            Omega = gamma * (y[t] - base_prob[t])
            v_t = p.alpha * Omega + (1.0 - p.alpha) * v_prev
            v_prev = v_t

            inertia = p.w_s * gap_sets[t] + p.w_g * gap_games[t]
            M_t = np.tanh(p.k * (v_t + inertia))

            # --- can we form a K-step target from t? (needs points t+1..t+K) ---
            if t + K >= n:
                break

            r_true = float(np.mean(y[t + 1 : t + K + 1]))

            # --- roll out K steps in expectation to get r_hat ---
            v_roll = float(v_t)
            M_roll = float(M_t)
            p_sum = 0.0
            for s in range(1, K + 1):
                idx = t + s

                p_s = float(expit(base_logit[idx] + M_roll))
                p_s = float(np.clip(p_s, 1e-5, 1 - 1e-5))
                p_sum += p_s

                # update state using expected y = p_s at THIS predicted point
                gamma_s = _gamma_t(bool(is_set[idx]), bool(is_break[idx]), bool(is_tie[idx]), p)
                Omega_hat = gamma_s * (p_s - base_prob[idx])
                v_roll = p.alpha * Omega_hat + (1.0 - p.alpha) * v_roll
                inertia_s = p.w_s * gap_sets[idx] + p.w_g * gap_games[idx]
                M_roll = float(np.tanh(p.k * (v_roll + inertia_s)))

            r_hat = float(np.clip(p_sum / float(K), 1e-5, 1 - 1e-5))

            ce = -(r_true * np.log(r_hat + eps) + (1.0 - r_true) * np.log(1.0 - r_hat + eps))
            loss += ce * (K if weight_per_point else 1.0)

    # mild regularization (optional)
    loss += 0.01 * (p.w_s**2 + p.w_g**2)
    return float(loss)


def compute_momentum_series(df: pd.DataFrame, p: MomentumParams) -> pd.Series:
    M = np.zeros(len(df), float)

    start = 0
    for _, d in df.groupby("match_id", sort=False):
        y = d["y"].to_numpy(float)
        base_prob = d["base_prob"].to_numpy(float)
        gap_sets = d["score_gap_sets"].to_numpy(float)
        gap_games = d["score_gap_games"].to_numpy(float)

        is_break = d["is_break_point"].to_numpy(bool)
        is_tie = d["is_tie_break"].to_numpy(bool)
        is_set = d["is_set_point"].to_numpy(bool)

        v_prev = 0.0
        n = len(d)
        for t in range(n):
            gamma = _gamma_t(bool(is_set[t]), bool(is_break[t]), bool(is_tie[t]), p)

            Omega = gamma * (y[t] - base_prob[t])
            v_t = p.alpha * Omega + (1.0 - p.alpha) * v_prev
            v_prev = v_t

            inertia = p.w_s * gap_sets[t] + p.w_g * gap_games[t]
            M[start + t] = np.tanh(p.k * (v_t + inertia))

        start += n

    return pd.Series(M, index=df.index, name="momentum")


# -----------------------------
# Training + LRT
# -----------------------------
def train_momentum_model(
    df_raw: pd.DataFrame,
    objective: str = "future_winrate",
    K: int = 5,
    weight_per_point: bool = True,
) -> Tuple[Dict[str, float], float, pd.DataFrame]:
    """Train momentum params.

    Parameters
    ----------
    objective:
        - "future_winrate": minimize loss on K-step future win-rate (recommended)
        - "next_point": original next-point negative log-likelihood
    K:
        horizon for objective="future_winrate".
    weight_per_point:
        if True, multiply each window loss by K (so each point contributes ~equally).
    """
    if objective not in {"future_winrate", "next_point"}:
        raise ValueError("objective must be 'future_winrate' or 'next_point'")

    df = prepare_features(df_raw)
    _, df = fit_base_model(df)

    if objective == "future_winrate":
        df = add_future_targets(df, K=K)

    alpha_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    bounds = [
        (0.1, 10.0),  # k
        (1.0, 3.0),  # gamma_break
        (1.0, 3.0),  # gamma_tie
        (1.0, 4.0),  # gamma_set
        (0.0, 1.0),  # w_s
        (0.0, 0.5),  # w_g
    ]
    x0 = np.array([1.0, 1.5, 1.5, 2.0, 0.10, 0.05], float)

    best = None  # (loss, alpha, xopt)
    for alpha in alpha_grid:
        if objective == "future_winrate":
            fun = lambda x: calculate_loss_future_winrateK(
                x, df, fixed_alpha=alpha, K=K, weight_per_point=weight_per_point
            )
        else:
            fun = lambda x: calculate_nll_next_point(x, df, fixed_alpha=alpha)

        res = minimize(fun, x0=x0, bounds=bounds, method="L-BFGS-B")
        loss = float(res.fun)
        if best is None or loss < best[0]:
            best = (loss, alpha, res.x.copy())

    best_loss, best_alpha, best_x = best
    p = MomentumParams(
        alpha=float(best_alpha),
        k=float(best_x[0]),
        gamma_break=float(best_x[1]),
        gamma_tie=float(best_x[2]),
        gamma_set=float(best_x[3]),
        w_s=float(best_x[4]),
        w_g=float(best_x[5]),
    )

    df_out = df.copy()
    df_out["momentum"] = compute_momentum_series(df_out, p)

    # ---- LRT ----
    # Null log-likelihood: base only (still next-point likelihood for significance testing)
    def ll_null(df_: pd.DataFrame) -> float:
        eps = 1e-12
        ll = 0.0
        for _, d in df_.groupby("match_id", sort=False):
            y = d["y"].to_numpy(float)
            base_logit = d["base_logit"].to_numpy(float)
            n = len(d)
            if n < 2:
                continue
            for t in range(n - 1):
                ph = expit(base_logit[t + 1])
                ph = float(np.clip(ph, 1e-5, 1 - 1e-5))
                yn = y[t + 1]
                ll += yn * np.log(ph + eps) + (1 - yn) * np.log(1 - ph + eps)
        return float(ll)

    # Model log-likelihood: base + momentum (same recursion; next-point)
    def ll_model(df_: pd.DataFrame, p_: MomentumParams) -> float:
        eps = 1e-12
        ll = 0.0
        for _, d in df_.groupby("match_id", sort=False):
            y = d["y"].to_numpy(float)
            base_prob = d["base_prob"].to_numpy(float)
            base_logit = d["base_logit"].to_numpy(float)
            gap_sets = d["score_gap_sets"].to_numpy(float)
            gap_games = d["score_gap_games"].to_numpy(float)

            is_break = d["is_break_point"].to_numpy(bool)
            is_tie = d["is_tie_break"].to_numpy(bool)
            is_set = d["is_set_point"].to_numpy(bool)

            n = len(d)
            if n < 2:
                continue

            v_prev = 0.0
            for t in range(n - 1):
                gamma = _gamma_t(bool(is_set[t]), bool(is_break[t]), bool(is_tie[t]), p_)

                Omega = gamma * (y[t] - base_prob[t])
                v_t = p_.alpha * Omega + (1.0 - p_.alpha) * v_prev
                v_prev = v_t

                inertia = p_.w_s * gap_sets[t] + p_.w_g * gap_games[t]
                M_t = np.tanh(p_.k * (v_t + inertia))

                ph = expit(base_logit[t + 1] + M_t)
                ph = float(np.clip(ph, 1e-5, 1 - 1e-5))
                yn = y[t + 1]
                ll += yn * np.log(ph + eps) + (1 - yn) * np.log(1 - ph + eps)

        return float(ll)

    ll0 = ll_null(df_out)
    ll1 = ll_model(df_out, p)

    lrt_stat = -2.0 * (ll0 - ll1)

    # df: 6 params optimized + alpha chosen by grid => 7 (你也可改成6)
    df_lrt = 7
    p_value = float(chi2.sf(lrt_stat, df=df_lrt))

    best_params = {
        "alpha": p.alpha,
        "k": p.k,
        "gamma_break": p.gamma_break,
        "gamma_tie": p.gamma_tie,
        "gamma_set": p.gamma_set,
        "w_s": p.w_s,
        "w_g": p.w_g,
        "objective": objective,
        "K": int(K),
        "train_loss": best_loss,
        "ll_null": ll0,
        "ll_model": ll1,
        "lrt_stat": float(lrt_stat),
        "lrt_df": int(df_lrt),
    }
    return best_params, p_value, df_out
