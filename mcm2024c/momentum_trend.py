#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm


KEY_COLS = [
    "match_id","player1","player2","elapsed_time",
    "set_no","game_no","point_no",
    "p1_sets","p2_sets","p1_games","p2_games","p1_score","p2_score",
    "server","serve_no","point_victor","game_victor","set_victor",
    "is_tie_break","is_break_point","is_set_point",
    "future_winrate_K","momentum"
]

def ensure_bool(df, col):
    if col in df.columns and df[col].dtype != bool:
        # 兼容 True/False 字符串、0/1
        df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})
        if df[col].isna().any():
            df[col] = df[col].fillna(0).astype(int).astype(bool)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 基本检查
    if "momentum" not in df.columns:
        raise ValueError("CSV 中找不到 momentum 列")
    if "match_id" not in df.columns:
        raise ValueError("CSV 中找不到 match_id 列")

    # 统一 bool 列
    for c in ["is_tie_break", "is_break_point", "is_set_point"]:
        ensure_bool(df, c)

    # 构造每场的顺序索引（保证绘图/rolling 正确）
    sort_cols = [c for c in ["match_id","set_no","game_no","point_no"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df

def print_extremes(df: pd.DataFrame):
    imax = df["momentum"].idxmax()
    imin = df["momentum"].idxmin()

    cols = [c for c in KEY_COLS if c in df.columns]
    print("\n=== momentum MAX ===")
    print(df.loc[imax, cols].to_string())

    print("\n=== momentum MIN ===")
    print(df.loc[imin, cols].to_string())

def add_trend_columns(d: pd.DataFrame, span=25, win=31) -> pd.DataFrame:
    """给每场比赛加 EWMA 和 rolling mean，用于趋势观察"""
    d = d.copy()
    d["m_ewm"] = (
        d.groupby("match_id")["momentum"]
        .apply(lambda s: s.ewm(span=span, adjust=False).mean())
        .reset_index(level=0, drop=True)
    )
    d["m_roll"] = (
        d.groupby("match_id")["momentum"]
        .apply(lambda s: s.rolling(win, min_periods=max(5, win//4)).mean())
        .reset_index(level=0, drop=True)
    )
    return d

def detect_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    关键节点事件：
    - tiebreak_end: 抢七结束点（本分导致 game_victor>0 且 is_tie_break=True）
    - break_end: 破发局结束点（本分导致 game_victor>0 且 game_victor != server 且 非抢七）
    - hold_end: 保发局结束点（本分导致 game_victor>0 且 game_victor == server 且 非抢七）
    注意：这些是“局结束点”，用于研究事件前后 momentum 跳变。
    """
    d = df.copy()
    for c in ["game_victor","server","is_tie_break"]:
        if c not in d.columns:
            d[c] = np.nan

    d["tiebreak_end"] = (d.get("is_tie_break", False) == True) & (d["game_victor"].fillna(0).astype(int) > 0)
    d["break_end"] = (d.get("is_tie_break", False) == False) & (d["game_victor"].fillna(0).astype(int) > 0) & (d["game_victor"] != d["server"])
    d["hold_end"]  = (d.get("is_tie_break", False) == False) & (d["game_victor"].fillna(0).astype(int) > 0) & (d["game_victor"] == d["server"])

    return d

def event_study(df: pd.DataFrame, event_col: str, k_before=10, k_after=10) -> pd.DataFrame:
    """
    事件研究：对每个事件点，取事件前后 [-k_before, +k_after] 的 momentum 序列，汇总平均。
    返回：relative_t, mean_momentum, count
    """
    if event_col not in df.columns:
        raise ValueError(f"找不到事件列 {event_col}")

    rows = []
    for mid, g in df.groupby("match_id"):
        g = g.reset_index(drop=True)
        idxs = np.where(g[event_col].values.astype(bool))[0]
        for idx in idxs:
            for dt in range(-k_before, k_after + 1):
                j = idx + dt
                if 0 <= j < len(g):
                    rows.append((dt, g.loc[j, "momentum"]))
    if not rows:
        return pd.DataFrame(columns=["relative_t","mean_momentum","count"])

    tmp = pd.DataFrame(rows, columns=["relative_t","momentum"])
    out = tmp.groupby("relative_t")["momentum"].agg(["mean","count"]).reset_index()
    out = out.rename(columns={"mean":"mean_momentum"})
    return out

def plot_match(df: pd.DataFrame, match_id: str, outdir: str):
    g = df[df["match_id"] == match_id].copy()
    if g.empty:
        print(f"[warn] match_id={match_id} 不存在")
        return

    g = g.reset_index(drop=True)
    x = np.arange(len(g))

    fig, ax = plt.subplots(figsize=(14, 5))

    # ====== 1) 按“盘-局”分段给 momentum 上色 ======
    # 组合成唯一的局ID（同一盘同一局=同一段颜色）
    if "set_no" in g.columns and "game_no" in g.columns:
        game_key = list(zip(g["set_no"].astype(int), g["game_no"].astype(int)))
    else:
        # 如果缺列，就退化成每 10 个点一段（兜底）
        game_key = (x // 10).tolist()

    # 把每个 game_key 映射到 0..(n_games-1)
    uniq_games = pd.Series(game_key).astype(str).unique().tolist()
    game_to_idx = {k: i for i, k in enumerate(uniq_games)}
    game_idx = np.array([game_to_idx[str(k)] for k in game_key], dtype=int)

    y = g["momentum"].values

    # 构造线段 (x_i,y_i) -> (x_{i+1},y_{i+1})
    pts = np.column_stack([x, y])
    segs = np.stack([pts[:-1], pts[1:]], axis=1)

    # 每段颜色取“该段起点所在局”
    seg_game = game_idx[:-1]
    n_games = max(1, len(uniq_games))

    # 用 colormap 自动生成颜色（不手工指定具体颜色）
    cmap = cm.get_cmap("tab20", n_games if n_games <= 20 else 20)
    colors = [cmap(int(i % cmap.N)) for i in seg_game]

    lc = LineCollection(segs, colors=colors, linewidths=1.5, alpha=0.95)
    ax.add_collection(lc)

    # 设定坐标范围（LineCollection 不会自动撑开坐标轴）
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min() - 0.02, y.max() + 0.02)

    # ====== 2) 趋势线（EWMA / RollingMean）保持单色，方便读趋势 ======
    if "m_ewm" in g.columns:
        ax.plot(x, g["m_ewm"].values, linewidth=2.5, label="EWMA")
    if "m_roll" in g.columns:
        ax.plot(x, g["m_roll"].values, linewidth=2.5, label="RollingMean")

    # ====== 3) 分割线：每局 / 每盘 ======
    # 每局边界：game_no变化处
    if "game_no" in g.columns:
        game_change = np.where(g["game_no"].astype(int).diff().fillna(0).values != 0)[0]
        for idx in game_change:
            ax.axvline(idx, linewidth=0.8, alpha=0.15)  # 淡灰分割局

    # 每盘边界：set_no变化处（更醒目）
    if "set_no" in g.columns:
        set_change = np.where(g["set_no"].astype(int).diff().fillna(0).values != 0)[0]
        for idx in set_change:
            ax.axvline(idx, linestyle="--", linewidth=1.3, alpha=0.8)  # 分割盘

    # ====== 4) 关键点标注（保持你原来逻辑） ======
    for col, mk in [("break_end", "x"), ("tiebreak_end", "o")]:
        if col in g.columns:
            idxs = np.where(g[col].values.astype(bool))[0]
            if len(idxs) > 0:
                ax.scatter(idxs, g.loc[idxs, "momentum"], marker=mk, s=45, label=col)

    # ====== 5) 标题坐标轴 ======
    p1 = g["player1"].iloc[0] if "player1" in g.columns else "p1"
    p2 = g["player2"].iloc[0] if "player2" in g.columns else "p2"
    ax.set_title(f"{match_id}: {p1} vs {p2} momentum trend (colored by game)")
    ax.set_xlabel("point index within match")
    ax.set_ylabel("momentum")

    # momentum 分段上色后不适合做“每局颜色图例”（太多），所以这里不加 momentum 的 legend
    ax.legend()
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{match_id}_momentum_colored_by_game.png")
    fig.savefig(outpath, dpi=160)
    plt.close(fig)
    print(f"[saved] {outpath}")


def plot_event_study(es: pd.DataFrame, title: str, outpath: str):
    if es.empty:
        print(f"[warn] no events for {title}")
        return
    plt.figure(figsize=(8, 4))
    plt.plot(es["relative_t"], es["mean_momentum"], linewidth=2)
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("relative point index (0 = event point)")
    plt.ylabel("mean momentum")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()
    print(f"[saved] {outpath}")

def summarize_set_game(df: pd.DataFrame, match_id: str, outdir: str):
    """输出：该场比赛按盘、按局的 momentum 统计表（CSV）"""
    g = df[df["match_id"] == match_id].copy()
    if g.empty:
        print(f"[warn] match_id={match_id} not found for summary")
        return

    os.makedirs(outdir, exist_ok=True)

    # ---------- 按盘统计 ----------
    grp_set = g.groupby(["match_id", "set_no"])["momentum"]
    set_summary = grp_set.agg(
        n="count",
        mean="mean",
        std="std",
        mean_abs=lambda s: float(np.mean(np.abs(s))),
        min="min",
        max="max",
    ).reset_index()

    set_path = os.path.join(outdir, f"{match_id}_set_momentum_summary.csv")
    set_summary.to_csv(set_path, index=False)
    print(f"[saved] {set_path}")

    # ---------- 按盘-局统计 ----------
    grp_game = g.groupby(["match_id", "set_no", "game_no"])

    # momentum 统计
    game_summary = grp_game["momentum"].agg(
        n="count",
        mean="mean",
        std="std",
        mean_abs=lambda s: float(np.mean(np.abs(s))),
        min="min",
        max="max",
    ).reset_index()

    # 可选：加一些“这局的元信息”（不影响你原模型）
    # 该局发球方（取该局第一分的 server）
    if "server" in g.columns:
        first_server = grp_game["server"].first().reset_index(name="server_first")
        game_summary = game_summary.merge(first_server, on=["match_id", "set_no", "game_no"], how="left")

    # 该局是否出现 break_end / tiebreak_end（如果你前面 detect_events() 已经生成了这两列）
    for ev in ["break_end", "tiebreak_end", "hold_end"]:
        if ev in g.columns:
            ev_flag = grp_game[ev].any().reset_index(name=f"has_{ev}")
            game_summary = game_summary.merge(ev_flag, on=["match_id", "set_no", "game_no"], how="left")

    game_path = os.path.join(outdir, f"{match_id}_game_momentum_summary.csv")
    game_summary.to_csv(game_path, index=False)
    print(f"[saved] {game_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=r"D:\download\D__download_mcm2024c_with_momentum.csv",
                help="path to csv with momentum")
    ap.add_argument("--outdir", default="out_momentum", help="output directory for plots")
    ap.add_argument("--match", default="", help="optional: plot one match_id only")
    args = ap.parse_args()

    df = load_data(args.csv)
    print(f"loaded: {df.shape}, matches={df['match_id'].nunique()}")

    # 1) 极值点
    print_extremes(df)

    # 2) 趋势列 + 事件列
    df = add_trend_columns(df, span=25, win=31)
    df = detect_events(df)

    os.makedirs(args.outdir, exist_ok=True)

    # 3) 事件研究：破发结束点、抢七结束点（看事件前后平均走势）
    es_break = event_study(df, "break_end", k_before=12, k_after=12)
    plot_event_study(es_break, "Event Study: break_end", os.path.join(args.outdir, "event_break_end.png"))

    es_tb = event_study(df, "tiebreak_end", k_before=12, k_after=12)
    plot_event_study(es_tb, "Event Study: tiebreak_end", os.path.join(args.outdir, "event_tiebreak_end.png"))

    # 4) 每场比赛趋势图
        # 4) 每场比赛趋势图
    # --- 修改开始：默认只画最后一场（优先1701） ---
    if args.match:
        target_match = args.match
    else:
        all_mids = list(df["match_id"].unique())

        # 优先找 2023-wimbledon-1701
        prefer = [m for m in all_mids if str(m).endswith("-1701") or str(m) == "2023-wimbledon-1701"]
        if prefer:
            target_match = prefer[0]
        else:
            # 否则取按 match_id 排序后的最后一个
            target_match = sorted(all_mids)[-1]

    print(f"[info] plotting match_id = {target_match}")
    plot_match(df, target_match, args.outdir)
    summarize_set_game(df, target_match, args.outdir)
    # --- 修改结束 ---


    # 5) 输出一个汇总表（每场的波动强度）
    summary = (
        df.groupby("match_id")["momentum"]
        .agg(mean="mean", std="std", mean_abs=lambda s: np.mean(np.abs(s)), min="min", max="max", n="count")
        .reset_index()
        .sort_values("mean_abs", ascending=False)
    )
    summary_path = os.path.join(args.outdir, "match_momentum_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

if __name__ == "__main__":
    main()
