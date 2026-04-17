"""
Trial-to-trial ΔT Analysis — Hamilos 2025 Framework
=====================================================
Computes ΔT(n+1) = T(n+1) - T(n) stratified by trial outcome,
corrected for regression-to-the-median (R2M), with bootstrap CIs.

Data source: trials_training_analyzed.csv
  (produced by session_processing_helper.py pipeline)

Timing anchor — change TIMING_ANCHOR to compare different references:
  'cue_on'        → time_waited_since_cue_on
  'cue_off'       → time_waited  (default: cue offset = wait period start)
  'last_reward'   → time_waited_since_last_reward
  'last_lick'     → time_waited_since_last_lick
  'last_lick_bout'→ time_waited_since_last_lick_bout

Outcome stratification:
  'rewarded'     — reward > 0
  'not_rewarded' — reward == 0 and miss_trial == False
  'miss'         — miss_trial == True  (excluded from stratified analysis)
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — avoids macOS backend version conflicts

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# TIMING ANCHOR — change this single variable to compare anchors
# ─────────────────────────────────────────────────────────────────────────────
TIMING_ANCHOR = 'cue_off'   # options: 'cue_on' | 'cue_off' | 'last_reward' | 'last_lick' | 'last_lick_bout'

TIMING_REFS = {
    'cue_on':         'time_waited_since_cue_on',
    'cue_off':        'time_waited',
    'last_reward':    'time_waited_since_last_reward',
    'last_lick':      'time_waited_since_last_lick',
    'last_lick_bout': 'time_waited_since_last_lick_bout',
}

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN MAP — real column names from trials_training_analyzed.csv
# wait_time is set dynamically from TIMING_ANCHOR at runtime
# ─────────────────────────────────────────────────────────────────────────────
COLUMN_MAP = {
    "wait_time":  TIMING_REFS[TIMING_ANCHOR],
    "outcome":    "outcome",    # derived column added by derive_outcome()
    "session_id": "dir",
    "mouse_id":   "mouse",
}

# ─────────────────────────────────────────────────────────────────────────────
# OUTCOME LABELS
# ─────────────────────────────────────────────────────────────────────────────
REWARDED_LABEL     = "rewarded"
NOT_REWARDED_LABEL = "not_rewarded"

OUTCOME_COLORS = {
    REWARDED_LABEL:     "#2196F3",   # blue
    NOT_REWARDED_LABEL: "#F44336",   # red
}

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
N_BOOTSTRAP   = 10_000
N_DECILES     = 10    # for R2M methods A & B
MOVING_WINDOW = 100   # trials, for R2M method C (centered)

# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def derive_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'outcome' column derived from reward and miss_trial columns.

    reward > 0          → 'rewarded'
    reward == 0 and not miss_trial → 'not_rewarded'
    miss_trial == True  → 'miss'  (excluded from stratified ΔT analysis)
    """
    df = df.copy()
    df["outcome"] = np.where(
        df["miss_trial"] == True, "miss",
        np.where(df["reward"] > 0, REWARDED_LABEL, NOT_REWARDED_LABEL)
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_delta_t(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ΔT column to a single-session DataFrame sorted by trial order.
    ΔT(n) = T(n+1) - T(n)  →  positive = later lick next trial.
    Last trial has NaN delta_t.
    """
    df = df.copy().reset_index(drop=True)
    wait = COLUMN_MAP["wait_time"]
    df["delta_t"] = df[wait].shift(-1) - df[wait]
    return df


def r2m_method_A(df: pd.DataFrame) -> pd.Series:
    """
    R2M Method A: decile medians.
    For each trial, R2M = median(T in that decile) - T(n).
    """
    wait = COLUMN_MAP["wait_time"]
    n = len(df)
    decile_idx = pd.qcut(np.arange(n), N_DECILES, labels=False)
    decile_medians = df.groupby(decile_idx)[wait].transform("median")
    return decile_medians - df[wait].values


def r2m_method_B(df: pd.DataFrame, n_boot: int = 1000) -> pd.Series:
    """
    R2M Method B: bootstrapped decile medians.
    Same as A but uses 1000-bootstrap estimate of decile median.
    """
    wait = COLUMN_MAP["wait_time"]
    n = len(df)
    decile_idx = pd.qcut(np.arange(n), N_DECILES, labels=False)
    r2m = np.zeros(n)
    for dec in range(N_DECILES):
        mask = decile_idx == dec
        vals = df.loc[mask, wait].values
        boot_medians = np.median(
            np.random.choice(vals, size=(n_boot, len(vals)), replace=True), axis=1
        )
        r2m[mask] = np.mean(boot_medians) - df.loc[mask, wait].values
    return pd.Series(r2m, index=df.index)


def r2m_method_C(df: pd.DataFrame) -> pd.Series:
    """
    R2M Method C: local moving median (100-trial centered window).
    """
    wait = COLUMN_MAP["wait_time"]
    moving_med = (
        df[wait]
        .rolling(window=MOVING_WINDOW, center=True, min_periods=20)
        .median()
    )
    return moving_med - df[wait].values


def compute_r2m(df: pd.DataFrame, method: str = "A") -> pd.Series:
    if method == "A":
        return r2m_method_A(df)
    elif method == "B":
        return r2m_method_B(df)
    elif method == "C":
        return r2m_method_C(df)
    else:
        raise ValueError("method must be 'A', 'B', or 'C'")


def process_all_sessions(df: pd.DataFrame, r2m_method: str = "A",
                         r2m_scope: str = "block") -> pd.DataFrame:
    """
    Process all sessions:
      - ΔT computed within each session (consecutive trial order, including cross-block transitions)
      - R2M computed within each group defined by r2m_scope:
          'block'   → (session, block_num)  — distribution reference is block-specific (default)
          'session' → session only           — distribution reference spans the full session
    """
    sid = COLUMN_MAP["session_id"]

    # Step 1: ΔT within session
    df_dt = (
        df.groupby(sid, group_keys=False)
          .apply(lambda g: compute_delta_t(
              g.sort_values("session_trial_num").reset_index(drop=True)
          ))
    )

    # Step 2: R2M within the chosen scope
    def _add_r2m(g):
        g = g.copy()
        g["r2m"] = compute_r2m(g, method=r2m_method)
        g["excess_delta_t"] = g["delta_t"] - g["r2m"]
        return g

    if r2m_scope == "block":
        return df_dt.groupby([sid, "block_num"], group_keys=False).apply(_add_r2m)
    elif r2m_scope == "session":
        return df_dt.groupby(sid, group_keys=False).apply(_add_r2m)
    else:
        raise ValueError("r2m_scope must be 'block' or 'session'")


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, ci: float = 0.95,
                      clt_threshold: int = 1000) -> tuple:
    """
    Return (mean, lower_ci, upper_ci).

    For n >= clt_threshold: uses CLT (mean ± z*SEM) — exact for large n, no memory blowup.
    For n <  clt_threshold: uses bootstrap (safe at small n).
    """
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = values.mean()
    if n >= clt_threshold:
        z = stats.norm.ppf((1 + ci) / 2)
        sem = values.std(ddof=1) / np.sqrt(n)
        return mean, mean - z * sem, mean + z * sem
    boot = np.random.choice(values, size=(n_boot, n), replace=True)
    boot_means = boot.mean(axis=1)
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return mean, lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def run_delta_t_analysis(df_raw: pd.DataFrame, r2m_method: str = "A"):
    """
    Full pipeline. Returns (results_dict, processed_df).

    results_dict keys: 'rewarded' / 'not_rewarded'  →  dict with:
      raw_mean, raw_lo, raw_hi         (raw ΔT stats, ms)
      excess_mean, excess_lo, excess_hi (ΔT - R2M stats, ms)
      n_trials, wilcoxon_p
    """
    print(f"Running ΔT analysis (timing anchor: {TIMING_ANCHOR!r}, R2M method {r2m_method})")
    print(f"  Wait time column: '{COLUMN_MAP['wait_time']}'")
    print(f"  Total trials: {len(df_raw)}")

    processed = process_all_sessions(df_raw, r2m_method)

    outcome_col = COLUMN_MAP["outcome"]
    results = {}

    for label in [REWARDED_LABEL, NOT_REWARDED_LABEL]:
        mask = (processed[outcome_col] == label) & processed["delta_t"].notna()
        subset = processed.loc[mask]

        raw_vals    = subset["delta_t"].values
        excess_vals = subset["excess_delta_t"].values

        raw_mean, raw_lo, raw_hi   = bootstrap_mean_ci(raw_vals)
        exc_mean, exc_lo, exc_hi   = bootstrap_mean_ci(excess_vals)

        clean = excess_vals[~np.isnan(excess_vals)]
        if len(clean) > 10:
            stat, pval = stats.wilcoxon(clean)
        else:
            stat, pval = np.nan, np.nan

        results[label] = {
            "raw_mean": raw_mean * 1000,    "raw_lo": raw_lo * 1000,
            "raw_hi":   raw_hi * 1000,
            "excess_mean": exc_mean * 1000, "excess_lo": exc_lo * 1000,
            "excess_hi":   exc_hi * 1000,
            "n_trials": int(mask.sum()),
            "wilcoxon_p": pval,
        }

        print(f"\n  Outcome: {label} (n={int(mask.sum())} trials)")
        print(f"    Raw ΔT:    {raw_mean*1000:+.1f} ms  [{raw_lo*1000:+.1f}, {raw_hi*1000:+.1f}]")
        p_str = f"  (Wilcoxon p={pval:.3e})" if not np.isnan(pval) else ""
        print(f"    Excess ΔT: {exc_mean*1000:+.1f} ms  [{exc_lo*1000:+.1f}, {exc_hi*1000:+.1f}]{p_str}")

    return results, processed


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

GROUP_COL    = "group"
GROUP_LABELS = {"s": "Short", "l": "Long"}   # display names for group values


def _get_groups(processed: pd.DataFrame) -> list:
    """Return sorted unique group values present in the data."""
    return sorted(processed[GROUP_COL].dropna().unique())


def _compute_summary_stats(df: pd.DataFrame) -> dict:
    """
    Compute mean/CI/p stats for each outcome label from a processed DataFrame slice.
    Returns the same dict structure as run_delta_t_analysis results.
    """
    outcome_col = COLUMN_MAP["outcome"]
    result = {}
    for label in [REWARDED_LABEL, NOT_REWARDED_LABEL]:
        mask        = (df[outcome_col] == label) & df["delta_t"].notna()
        subset      = df.loc[mask]
        raw_vals    = subset["delta_t"].values
        excess_vals = subset["excess_delta_t"].values

        raw_mean, raw_lo, raw_hi = bootstrap_mean_ci(raw_vals)
        exc_mean, exc_lo, exc_hi = bootstrap_mean_ci(excess_vals)

        clean = excess_vals[~np.isnan(excess_vals)]
        _, pval = stats.wilcoxon(clean) if len(clean) > 10 else (np.nan, np.nan)

        result[label] = {
            "raw_mean": raw_mean * 1000,    "raw_lo": raw_lo * 1000,    "raw_hi": raw_hi * 1000,
            "excess_mean": exc_mean * 1000, "excess_lo": exc_lo * 1000, "excess_hi": exc_hi * 1000,
            "n_trials": int(mask.sum()),
            "wilcoxon_p": pval,
        }
    return result


def _draw_bar_panel(ax, group_results: dict, key: str, title: str):
    """Draw a single bar panel (raw or excess ΔT) for one group."""
    lo_key = key.replace("mean", "lo")
    hi_key = key.replace("mean", "hi")
    labels_plot = [REWARDED_LABEL, NOT_REWARDED_LABEL]
    valid  = [l for l in labels_plot if l in group_results]
    means  = [group_results[l][key]    for l in valid]
    los    = [group_results[l][lo_key] for l in valid]
    his    = [group_results[l][hi_key] for l in valid]
    cols   = [OUTCOME_COLORS[l]        for l in valid]

    ax.bar(range(len(valid)), means, color=cols, alpha=0.8, edgecolor="k", linewidth=0.8, width=0.55)
    for i, (lo, hi) in enumerate(zip(los, his)):
        ax.plot([i, i], [lo, hi], color="k", linewidth=2)
        ax.plot([i-0.1, i+0.1], [lo, lo], color="k", linewidth=2)
        ax.plot([i-0.1, i+0.1], [hi, hi], color="k", linewidth=2)

    ax.axhline(0, color="k", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(valid)))
    # n= embedded in tick labels to avoid overlap with annotations
    ax.set_xticklabels(
        [f"{v.replace('_', ' ').capitalize()}\n(n={group_results[v]['n_trials']:,})" for v in valid],
        fontsize=9,
    )
    ax.set_ylabel("ΔT (ms)", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # p= placed just above each CI whisker
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i, l in enumerate(valid):
        p = group_results[l]["wilcoxon_p"]
        if not np.isnan(p):
            whisker_top = his[i]
            ax.text(i, whisker_top + y_range * 0.03, f"p={p:.2e}",
                    ha="center", va="bottom", fontsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_delta_t_summary(processed: pd.DataFrame, save_path: str = None):
    """
    Figure 1 — Bar chart of excess ΔT per outcome, one row per group (short / long).
    Stats are computed from processed directly so each row reflects only that group's data.
    """
    groups = _get_groups(processed)
    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 2, figsize=(9, 4.5 * n_groups), squeeze=False)
    fig.suptitle(f"Trial-to-trial timing update (ΔT)\nAnchor: {TIMING_ANCHOR!r}")

    for row, grp in enumerate(groups):
        grp_df      = processed[processed[GROUP_COL] == grp]
        grp_results = _compute_summary_stats(grp_df)
        grp_label   = GROUP_LABELS.get(grp, grp)

        for col, (key, title) in enumerate([
            ("raw_mean",    "Raw ΔT (ms)"),
            ("excess_mean", "Excess ΔT — R2M corrected (ms)"),
        ]):
            ax = axes[row, col]
            _draw_bar_panel(ax, grp_results, key, title)
            if col == 0:
                ax.set_ylabel(f"{grp_label}\nΔT (ms)", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_delta_t_by_t_n(processed: pd.DataFrame, save_path: str = None):
    """
    Figure 2 — Excess ΔT(n+1) as a function of T(n), binned.
    Rows = groups (short / long), columns = outcomes (rewarded / not_rewarded).
    """
    wait    = COLUMN_MAP["wait_time"]
    outcome = COLUMN_MAP["outcome"]
    groups  = _get_groups(processed)

    fig, axes = plt.subplots(len(groups), 2, figsize=(12, 4.5 * len(groups)),
                             sharey="row", squeeze=False)
    fig.suptitle(f"ΔT(n+1) as a function of T(n)\nAnchor: {TIMING_ANCHOR!r}")

    for row, grp in enumerate(groups):
        grp_df    = processed[processed[GROUP_COL] == grp]
        grp_label = GROUP_LABELS.get(grp, grp)

        for col, label in enumerate([REWARDED_LABEL, NOT_REWARDED_LABEL]):
            ax  = axes[row, col]
            clr = OUTCOME_COLORS[label]
            sub = grp_df.loc[(grp_df[outcome] == label) & grp_df["delta_t"].notna()].copy()

            if len(sub) < 20:
                ax.set_title(f"{grp_label} — {label} — insufficient data")
                continue

            try:
                sub["t_bin"] = pd.qcut(sub[wait], q=8, duplicates="drop")
                bin_centers  = sub.groupby("t_bin")[wait].median()
                bin_means    = sub.groupby("t_bin")["excess_delta_t"].mean() * 1000
                bin_sems     = sub.groupby("t_bin")["excess_delta_t"].sem()  * 1000

                ax.plot(bin_centers, bin_means, "o-", color=clr, linewidth=2, markersize=7)
                ax.fill_between(bin_centers, bin_means - bin_sems, bin_means + bin_sems,
                                alpha=0.25, color=clr)
            except Exception as e:
                ax.text(0.5, 0.5, f"Binning error:\n{e}", transform=ax.transAxes, ha="center")

            ax.axhline(0, color="k", linewidth=1, linestyle="--", alpha=0.5)
            ax.set_xlabel(f"T(n) — {TIMING_ANCHOR} wait time (s)", fontsize=10)
            ax.set_ylabel("Excess ΔT (ms)", fontsize=10)
            ax.set_title(f"{grp_label} — after {label.replace('_', ' ')} trial")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_per_mouse(processed: pd.DataFrame, save_path: str = None):
    """
    Figure 3 — Per-mouse excess ΔT, population mean overlaid.
    Rows = groups (short / long), columns = outcomes (rewarded / not_rewarded).
    """
    mouse_col = COLUMN_MAP["mouse_id"]
    outcome   = COLUMN_MAP["outcome"]
    groups    = _get_groups(processed)

    fig, axes = plt.subplots(len(groups), 2, figsize=(10, 4.5 * len(groups)),
                             squeeze=False)
    fig.suptitle(f"Per-mouse excess ΔT — R2M corrected\nAnchor: {TIMING_ANCHOR!r}")

    for row, grp in enumerate(groups):
        grp_df    = processed[processed[GROUP_COL] == grp]
        grp_label = GROUP_LABELS.get(grp, grp)
        mice      = grp_df[mouse_col].unique()

        for col, label in enumerate([REWARDED_LABEL, NOT_REWARDED_LABEL]):
            ax          = axes[row, col]
            clr         = OUTCOME_COLORS[label]
            mouse_means = []

            for mouse in mice:
                sub = grp_df.loc[
                    (grp_df[mouse_col] == mouse) &
                    (grp_df[outcome] == label) &
                    grp_df["excess_delta_t"].notna(),
                    "excess_delta_t"
                ].values * 1000
                if len(sub) > 5:
                    mouse_means.append(sub.mean())
                    ax.plot(0, sub.mean(), "o", color=clr, alpha=0.5,
                            markersize=8, markeredgecolor="gray")

            if mouse_means:
                pop_mean = np.mean(mouse_means)
                pop_sem  = stats.sem(mouse_means)
                ax.errorbar(0, pop_mean, yerr=pop_sem * 1.96,
                            fmt="D", color="k", markersize=10, linewidth=2.5,
                            capsize=5, label="Population mean ± 95% CI", zorder=5)

            ax.axhline(0, color="k", linewidth=1, linestyle="--", alpha=0.5)
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_ylabel("Excess ΔT (ms)", fontsize=10)
            ax.set_title(f"{grp_label} — after {label.replace('_', ' ')} trial"
                         f"\n(n={len(mouse_means)} mice)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


def plot_delta_t_by_Tn_with_reward_prob(processed: pd.DataFrame, save_path: str = None):
    """
    Figure 4 — Excess ΔT(n+1) vs T(n), with empirical p(reward | T) overlaid
    on the not_rewarded panel (right y-axis).

    p(reward | T) is computed from the data: for each T bin, fraction of
    non-miss trials that were rewarded.  This lets you ask whether the update
    magnitude tracks the missed reward probability.

    Rows = groups (short / long), columns = outcomes (rewarded / not_rewarded).
    """
    wait    = COLUMN_MAP["wait_time"]
    outcome = COLUMN_MAP["outcome"]
    groups  = _get_groups(processed)

    # Empirical p(reward | T) per group — use all non-miss trials
    non_miss = processed[processed[outcome] != "miss"].copy()

    fig, axes = plt.subplots(len(groups), 2, figsize=(14, 5 * len(groups)),
                             sharey="row", squeeze=False)
    fig.suptitle(f"ΔT(n+1) vs T(n) with empirical p(reward | T)\nAnchor: {TIMING_ANCHOR!r}")

    for row, grp in enumerate(groups):
        grp_df      = processed[processed[GROUP_COL] == grp]
        grp_nonmiss = non_miss[non_miss[GROUP_COL] == grp].copy()
        grp_label   = GROUP_LABELS.get(grp, grp)

        # Compute empirical p(reward | T) for this group using shared bin edges
        try:
            grp_nonmiss["t_bin"] = pd.qcut(grp_nonmiss[wait], q=10, duplicates="drop")
            p_rew_centers = grp_nonmiss.groupby("t_bin")[wait].median()
            p_rew_vals    = grp_nonmiss.groupby("t_bin")[outcome].apply(
                lambda x: (x == REWARDED_LABEL).mean()
            )
            p_rew_counts  = grp_nonmiss.groupby("t_bin")[outcome].count()
            p_rew_se      = np.sqrt(p_rew_vals * (1 - p_rew_vals) / p_rew_counts)
        except Exception:
            p_rew_centers = p_rew_vals = p_rew_se = None

        for col, label in enumerate([REWARDED_LABEL, NOT_REWARDED_LABEL]):
            ax  = axes[row, col]
            clr = OUTCOME_COLORS[label]
            sub = grp_df.loc[(grp_df[outcome] == label) & grp_df["delta_t"].notna()].copy()

            if len(sub) < 20:
                ax.set_title(f"{grp_label} — {label} — insufficient data")
                continue

            try:
                sub["t_bin"] = pd.qcut(sub[wait], q=10, duplicates="drop")
                bin_centers  = sub.groupby("t_bin")[wait].median()
                bin_means    = sub.groupby("t_bin")["excess_delta_t"].mean() * 1000
                bin_sems     = sub.groupby("t_bin")["excess_delta_t"].sem()  * 1000

                ax.plot(bin_centers, bin_means, "o-", color=clr, linewidth=2.5, markersize=8)
                ax.fill_between(bin_centers, bin_means - bin_sems, bin_means + bin_sems,
                                alpha=0.2, color=clr)
            except Exception as e:
                ax.text(0.5, 0.5, f"Binning error:\n{e}", transform=ax.transAxes, ha="center")

            ax.axhline(0, color="k", linewidth=1, linestyle="--", alpha=0.5)
            ax.set_xlabel(f"T(n) — {TIMING_ANCHOR} wait time (s)", fontsize=10)
            ax.set_ylabel("Excess ΔT (ms)", fontsize=10)
            ax.set_title(f"{grp_label} — after {label.replace('_', ' ')} trial", fontsize=10, color=clr)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Overlay p(reward | T) on the not_rewarded panel only
            if label == NOT_REWARDED_LABEL and p_rew_centers is not None:
                ax2 = ax.twinx()
                ax2.plot(p_rew_centers, p_rew_vals, "s--", color="gray",
                         linewidth=1.5, markersize=5, alpha=0.8, label="p(reward | T)")
                ax2.fill_between(p_rew_centers,
                                 p_rew_vals - p_rew_se,
                                 p_rew_vals + p_rew_se,
                                 color="gray", alpha=0.15)
                ax2.set_ylabel("p(reward | T)", fontsize=10, color="gray")
                ax2.tick_params(axis="y", labelcolor="gray")
                ax2.set_ylim(0, 1)
                ax2.spines["top"].set_visible(False)
                ax2.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig


R2M_METHOD_COLORS = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}
R2M_METHOD_LABELS = {
    "A": "A: decile medians",
    "B": "B: bootstrap decile",
    "C": "C: moving median",
}


def run_all_r2m_methods(df_raw: pd.DataFrame) -> dict:
    """
    Run process_all_sessions for R2M methods A, B, C.
    Returns {method: processed_df} for all three.

    Note: method B runs a bootstrap per block per decile and will be slower.
    """
    processed_by_method = {}
    for method in ["A", "B", "C"]:
        print(f"\nProcessing R2M method {method} ({R2M_METHOD_LABELS[method]})...")
        processed_by_method[method] = process_all_sessions(df_raw, r2m_method=method)
    return processed_by_method


def plot_r2m_comparison(processed_by_method: dict, save_path: str = None):
    """
    Figure 5 — Excess ΔT for R2M methods A, B, C shown as grouped bars.

    Layout: rows = groups (short / long), cols = outcomes (rewarded / not_rewarded).
    Within each panel: three bars (one per method) with 95% CI.
    """
    groups   = _get_groups(next(iter(processed_by_method.values())))
    methods  = list(processed_by_method.keys())
    outcomes = [REWARDED_LABEL, NOT_REWARDED_LABEL]

    n_methods = len(methods)
    bar_width = 0.22

    fig, axes = plt.subplots(len(groups), len(outcomes),
                             figsize=(5 * len(outcomes), 4.5 * len(groups)),
                             squeeze=False)
    fig.suptitle(f"Excess ΔT by R2M method\nAnchor: {TIMING_ANCHOR!r}")

    for row, grp in enumerate(groups):
        grp_label = GROUP_LABELS.get(grp, grp)

        for col, outcome_label in enumerate(outcomes):
            ax = axes[row, col]
            outcome_col = COLUMN_MAP["outcome"]

            for m_idx, method in enumerate(methods):
                grp_df = processed_by_method[method][
                    processed_by_method[method][GROUP_COL] == grp
                ]
                mask        = (grp_df[outcome_col] == outcome_label) & grp_df["delta_t"].notna()
                excess_vals = grp_df.loc[mask, "excess_delta_t"].values
                mean, lo, hi = bootstrap_mean_ci(excess_vals)
                mean_ms, lo_ms, hi_ms = mean * 1000, lo * 1000, hi * 1000

                x = m_idx  # one position per method
                clr = R2M_METHOD_COLORS[method]
                ax.bar(x, mean_ms, width=bar_width * 2.5, color=clr,
                       alpha=0.8, edgecolor="k", linewidth=0.8,
                       label=R2M_METHOD_LABELS[method])
                ax.plot([x, x], [lo_ms, hi_ms], color="k", linewidth=2)
                ax.plot([x - 0.08, x + 0.08], [lo_ms, lo_ms], color="k", linewidth=2)
                ax.plot([x - 0.08, x + 0.08], [hi_ms, hi_ms], color="k", linewidth=2)

                # p= above CI whisker
                clean = excess_vals[~np.isnan(excess_vals)]
                if len(clean) > 10:
                    _, pval = stats.wilcoxon(clean)
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    ax.text(x, hi_ms + y_range * 0.03, f"p={pval:.1e}",
                            ha="center", va="bottom", fontsize=6.5)

            ax.axhline(0, color="k", linewidth=1, linestyle="--", alpha=0.5)
            ax.set_xticks(range(n_methods))
            ax.set_xticklabels([R2M_METHOD_LABELS[m] for m in methods], fontsize=8, rotation=15, ha="right")
            ax.set_ylabel("Excess ΔT (ms)", fontsize=10)
            ax.set_title(f"{grp_label} — after {outcome_label.replace('_', ' ')} trial",
                         fontsize=10, color=OUTCOME_COLORS[outcome_label])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    return fig



# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    DATA_PATH = "/Users/rebekahzhang/data/behavior_data/exp2/trials_training_filtered2.csv"
    out_dir   = "/Users/rebekahzhang/data/behavior_data/delta_t_analysis"
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data once ────────────────────────────────────────────────────────
    print(f"Loading data from:\n  {DATA_PATH}")
    df_loaded = pd.read_csv(DATA_PATH)
    df_loaded = derive_outcome(df_loaded)
    print(f"  Loaded {len(df_loaded):,} rows × {df_loaded.shape[1]} columns")
    print(f"\nOutcome counts:\n{df_loaded['outcome'].value_counts().to_string()}")

    # ── Loop over all timing anchors ──────────────────────────────────────────
    for anchor in TIMING_REFS:
        print(f"\n{'='*60}")
        print(f"TIMING ANCHOR: {anchor!r}  →  '{TIMING_REFS[anchor]}'")
        print(f"{'='*60}")

        COLUMN_MAP["wait_time"] = TIMING_REFS[anchor]

        df_raw = df_loaded.dropna(subset=[COLUMN_MAP["wait_time"]])
        dropped = len(df_loaded) - len(df_raw)
        if dropped:
            print(f"Dropped {dropped:,} rows with NaN in '{COLUMN_MAP['wait_time']}'")

        # patch module-level TIMING_ANCHOR so plot titles reflect current anchor
        globals()["TIMING_ANCHOR"] = anchor

        # ── Run analysis ──────────────────────────────────────────────────────
        results, processed = run_delta_t_analysis(df_raw, r2m_method="A")

        # ── Figures 1–4 ───────────────────────────────────────────────────────
        plot_delta_t_summary(processed,                 save_path=f"{out_dir}/delta_t_summary_{anchor}.png")
        plot_delta_t_by_t_n(processed,                 save_path=f"{out_dir}/delta_t_by_Tn_{anchor}.png")
        plot_per_mouse(processed,                       save_path=f"{out_dir}/delta_t_per_mouse_{anchor}.png")
        plot_delta_t_by_Tn_with_reward_prob(processed, save_path=f"{out_dir}/delta_t_reward_prob_{anchor}.png")

        # ── Figure 5: R2M method comparison (A, B, C) ─────────────────────────
        processed_by_method = run_all_r2m_methods(df_raw)
        plot_r2m_comparison(processed_by_method,        save_path=f"{out_dir}/delta_t_r2m_comparison_{anchor}.png")

        plt.close("all")

    print(f"\nAll anchors complete. Figures saved to: {out_dir}")
    print("Done.")
