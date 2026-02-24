from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

matplotlib.use("Agg")

# =========================
# CONFIG
# =========================
DATA_DIR = "/Users/rebekahzhang/data/behavior_data"
EXP = "exp2"
DATA_FOLDER = os.path.join(DATA_DIR, EXP)
OUTPUT_FOLDER = os.path.join(DATA_DIR, "HMM")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

TRIALS_CSV = os.path.join(DATA_FOLDER, "trials_training_filtered2.csv")
OUTPUT_CSV = os.path.join(OUTPUT_FOLDER, "trials_with_hmm_states.csv")
MODEL_RESULTS_PKL = os.path.join(OUTPUT_FOLDER, "hmm_model_results.pkl")

N_STATES = 3
RANDOM_STATE = 0
N_ITER = 100

# Set to True to force retraining even if saved models exist
FORCE_RETRAIN = False

# Transition "stickiness": higher => states persist longer
STICKY_P = 0.97  # self-transition probability on init
MIN_VAR = 1e-4    # covariance floor for numerical stability
# Plotting config
PLOT_SMOOTHED = True       # whether to smooth state transitions in plots
SMOOTH_MIN_DWELL = 20      # minimum dwell time (trials) for smoothing
SMOOTH_WINDOW = 7          # rolling window size for smoothing

# State labels in canonical order
STATE_LABELS = ["disengaged", "impulsive", "engaged"]

# State colors (matplotlib default cycle)
STATE_COLORS = {
    "disengaged": "#1f77b4",
    "impulsive": "#ff7f0e", 
    "engaged": "#2ca02c"
}

# MAD to standard deviation conversion factor
MAD_TO_SIGMA = 1.4826

# Feature set (engineered below)
FEATURE_COLS = [
    "miss_trial",
    "bg_repeat_rate",
    "mean_bg_lick_phase",
    "is_fast_wait",
    "log_time_waited",
    "log_time_waited_since_last_lick",
    "log_first_lick",
    "log1p_num_consumption_lick",
    "dlog_wait_prevR",
    "dlog_wait_prevU",
    "dlog_wait_valid",
]

REQUIRED_COLS = [
    "mouse",
    "dir",
    "block_num",
    "block_trial_num",
    "miss_trial",
    "bg_repeats",
    "bg_length",
    "mean_bg_lick_phase",
    "time_waited",
    "time_waited_since_last_lick",
    "first_lick",
    "num_consumption_lick",
    "reward",
]

# =========================
# Utilities
# =========================
def _assert_required_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def _safe_log(x: pd.Series, eps: float = 1e-6) -> pd.Series:
    """Log transform with small epsilon; assumes x >= 0 (NaNs preserved)."""
    x = x.astype(float)
    return np.log(np.clip(x, eps, None))

def _robust_standardize(train_values: np.ndarray) -> Tuple[float, float]:
    """Median/MAD-based scaling; returns (center, scale)."""
    med = np.nanmedian(train_values)
    mad = np.nanmedian(np.abs(train_values - med))
    scale = max(mad * MAD_TO_SIGMA, 1e-6)  # MAD -> robust sigma; avoid zero
    return float(med), float(scale)

def _init_sticky_transmat(n_states: int, sticky_p: float) -> np.ndarray:
    """Initialize a sticky transition matrix."""
    if not (0 < sticky_p < 1):
        raise ValueError("sticky_p must be between 0 and 1")
    off = (1.0 - sticky_p) / (n_states - 1)
    A = np.full((n_states, n_states), off, dtype=float)
    np.fill_diagonal(A, sticky_p)
    return A

def _to_rgb(color) -> np.ndarray:
    """Convert matplotlib color to RGB array."""
    if color is None:
        return np.array([0.5, 0.5, 0.5], dtype=float)
    return np.array(matplotlib.colors.to_rgb(color), dtype=float)

def _standardize_features(X: np.ndarray, feature_cols: List[str]) -> np.ndarray:
    """Standardize features using robust (MAD) or sparse-aware scaling."""
    bin_cols = {"miss_trial", "is_fast_wait", "dlog_wait_valid"}
    sparse_cols = {"dlog_wait_prevR", "dlog_wait_prevU"}
    
    Xz = X.copy()
    for j, col in enumerate(feature_cols):
        if col in bin_cols:
            continue
            
        v = Xz[:, j]
        if col in sparse_cols:
            # Scale using non-zero values only; keep center at 0
            nz = v[np.abs(v) > 0]
            scale = float(np.nanstd(nz)) if nz.size > 5 else 1.0
            Xz[:, j] = v / max(scale, 0.1)
        else:
            center, scale = _robust_standardize(v)
            Xz[:, j] = (v - center) / scale
    
    return Xz

def _label_states(summary_stats: pd.DataFrame) -> Dict[int, str]:
    """Label states as disengaged/impulsive/engaged based on signatures."""
    disengaged_state = int(summary_stats["miss_rate"].idxmax())
    remaining = [s for s in range(N_STATES) if s != disengaged_state]
    
    # Impulsive score: fast_wait + bg_repeat_rate + inverse time-since-last-lick
    score = {}
    for s in remaining:
        fast = float(summary_stats.loc[s, "mean_fast_wait"])
        bgr = float(summary_stats.loc[s, "mean_bg_repeat_rate"])
        tsl = float(summary_stats.loc[s, "mean_tsl"]) if pd.notna(summary_stats.loc[s, "mean_tsl"]) else np.inf
        inv_tsl = 1.0 / max(tsl, 1e-3) if np.isfinite(tsl) else 0.0
        score[s] = 1.5 * fast + 1.0 * bgr + 0.5 * inv_tsl
    
    impulsive_state = max(score, key=score.get)
    engaged_state = [s for s in remaining if s != impulsive_state][0]
    
    return {
        disengaged_state: "disengaged",
        impulsive_state: "impulsive",
        engaged_state: "engaged",
    }

def _reorder_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to canonical state order."""
    return df[[c for c in STATE_LABELS if c in df.columns]]

def _save_and_close(fig, path: str, dpi: int = 150) -> None:
    """Save figure and close it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {path}")

# =========================
# Feature engineering
# =========================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for HMM (Round 2 feature list).
    Returns a NEW dataframe (does not mutate caller).
    """
    _assert_required_cols(df, REQUIRED_COLS)
    df = df.copy()

    # Sort to make block-wise shifts correct
    df.sort_values(["mouse", "dir", "block_num", "block_trial_num"], inplace=True)

    # Ensure binary
    df["miss_trial"] = df["miss_trial"].astype(int)

    # Mask timing-like columns on missed trials
    for col in ["time_waited", "time_waited_since_last_lick", "first_lick"]:
        df.loc[df["miss_trial"] == 1, col] = np.nan

    # ---- B) Background inhibition ----
    bg_len = df["bg_length"].astype(float)
    bg_rep = df["bg_repeats"].fillna(0).astype(float)
    df["bg_repeat_rate"] = np.where((bg_len > 0) & np.isfinite(bg_len), bg_rep / bg_len, 0.0)

    df["mean_bg_lick_phase"] = df["mean_bg_lick_phase"].astype(float)
    # If mean phase is NaN for no-violation trials, keep NaN for now; we'll fill later.

    # ---- C) Impulsive/reflex lick marker in wait period ----
    df["is_fast_wait"] = ((df["time_waited"] < 0.5) & (df["miss_trial"] == 0)).astype(int)

    # ---- D) Timing + motor carryover ----
    df["log_time_waited"] = _safe_log(df["time_waited"])
    df["log_time_waited_since_last_lick"] = _safe_log(df["time_waited_since_last_lick"])
    df["log_first_lick"] = _safe_log(df["first_lick"])

    # ---- E) Consummatory vigor ----
    df["log1p_num_consumption_lick"] = np.log1p(df["num_consumption_lick"].fillna(0).astype(float))

    # ---- F) Reward-dependent update ----
    seq_keys = ["mouse", "dir", "block_num"]

    df["prev_miss_trial"] = df.groupby(seq_keys, sort=False)["miss_trial"].shift(1)
    df["prev_reward"] = df.groupby(seq_keys, sort=False)["reward"].shift(1)

    # Delta log wait (current - previous) within each block
    df["dlog_wait"] = df["log_time_waited"] - df.groupby(seq_keys, sort=False)["log_time_waited"].shift(1)

    # Valid only if current + previous trials are not missed and previous reward is known
    valid_delta = (
        (df["miss_trial"] == 0) &
        (df["prev_miss_trial"] == 0) &
        (~df["prev_reward"].isna())
    )
    df["dlog_wait_valid"] = valid_delta.astype(int)

    # Split delta by previous outcome; invalid -> 0
    for reward_val, suffix in [(5, "prevR"), (0, "prevU")]:
        col_name = f"dlog_wait_{suffix}"
        df[col_name] = 0.0
        mask_combined = valid_delta & (df["prev_reward"] == reward_val)
        df.loc[mask_combined, col_name] = df.loc[mask_combined, "dlog_wait"].astype(float)

    # ---- Fill NaNs for HMM-consumed engineered features ----
    fill_zero_cols = [
        "bg_repeat_rate",
        "mean_bg_lick_phase",
        "log_time_waited",
        "log_time_waited_since_last_lick",
        "log_first_lick",
        "log1p_num_consumption_lick",
        "dlog_wait_prevR",
        "dlog_wait_prevU",
    ]
    for c in fill_zero_cols:
        df[c] = df[c].fillna(0.0)

    df["dlog_wait_valid"] = df["dlog_wait_valid"].fillna(0).astype(int)

    return df

# =========================
# Model fitting per mouse
# =========================
@dataclass
class MouseModelResult:
    mouse: str
    model: GaussianHMM
    state_map: Dict[int, str]  # numeric state -> semantic label

def fit_hmm_per_mouse(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, MouseModelResult]]:
    """
    Fits a 3-state sticky Gaussian HMM per mouse, pooling all sessions for that mouse.
    IMPORTANT: uses `lengths` so transitions do NOT cross session boundaries.
    Standardization is done per mouse across all its sessions (robust median/MAD).
    """
    results: Dict[str, MouseModelResult] = {}
    out = df.copy()

    # sanity
    missing_feats = [c for c in FEATURE_COLS if c not in out.columns]
    if missing_feats:
        raise ValueError(f"Missing engineered feature columns: {missing_feats}")

    for mouse, d in out.groupby("mouse", sort=False):
        d = d.copy()
        d.sort_values(["dir", "block_num", "block_trial_num"], inplace=True)

        # Build lengths per (dir, block_num) so sequences don't connect across blocks
        seq_sizes = d.groupby(["dir", "block_num"], sort=False).size().to_numpy()
        lengths = seq_sizes.tolist()

        X = d[FEATURE_COLS].astype(float).to_numpy()
        assert sum(lengths) == X.shape[0], "lengths do not sum to number of rows in X"

        # Robust standardization per mouse
        Xz = _standardize_features(X, FEATURE_COLS)
                
        # Initialize HMM
        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=N_ITER,
            random_state=RANDOM_STATE,
            verbose=False,
            init_params="mc",   # init means/covars only; we set startprob/transmat
            params="stmc",      # train startprob/transmat/means/covars
            min_covar=MIN_VAR,
        )
        model.startprob_ = np.full(N_STATES, 1.0 / N_STATES)
        model.transmat_ = _init_sticky_transmat(N_STATES, STICKY_P)

        # Fit with lengths so sequences don't connect across sessions
        model.fit(Xz, lengths=lengths)
        states = model.predict(Xz, lengths=lengths)
        post = model.predict_proba(Xz, lengths=lengths)

        d_out = d.copy()
        d_out["hmm_state"] = states
        for k in range(N_STATES):
            d_out[f"hmm_p{k}"] = post[:, k]

        # Label states using simple signatures (raw columns, not engineered-only)
        summ = (
            d_out.groupby("hmm_state")
                 .agg(
                     miss_rate=("miss_trial", "mean"),
                     mean_bg_repeat_rate=("bg_repeat_rate", "mean"),
                     mean_fast_wait=("is_fast_wait", "mean"),
                     mean_wait=("time_waited", "mean"),
                     mean_tsl=("time_waited_since_last_lick", "mean"),
                     mean_phase=("mean_bg_lick_phase", "mean"),
                     mean_cons_lick=("num_consumption_lick", "mean"),
                 )
        )

        # Label states using behavioral signatures
        state_map = _label_states(summ)
        d_out["hmm_state_label"] = d_out["hmm_state"].map(state_map)

        # Write back into out by index
        out.loc[d_out.index, "hmm_state"] = d_out["hmm_state"]
        out.loc[d_out.index, "hmm_state_label"] = d_out["hmm_state_label"]
        for k in range(N_STATES):
            out.loc[d_out.index, f"hmm_p{k}"] = d_out[f"hmm_p{k}"]

        results[str(mouse)] = MouseModelResult(mouse=str(mouse), model=model, state_map=state_map)

        print(f"[mouse={mouse}] done. state_map={state_map}")
        print(summ)

    return out, results

# =========================
# Plotting
# =========================

def _state_label_map(df: pd.DataFrame) -> Dict[Tuple[str, int], str]:
    """Map (mouse, hmm_state) -> hmm_state_label using the most common label."""
    if "hmm_state" not in df.columns or "hmm_state_label" not in df.columns:
        raise ValueError("CSV must contain hmm_state and hmm_state_label")

    tmp = (
        df[["mouse", "hmm_state", "hmm_state_label"]]
        .dropna(subset=["mouse", "hmm_state", "hmm_state_label"])
        .copy()
    )
    tmp["hmm_state"] = tmp["hmm_state"].astype(int)
    tmp["hmm_state_label"] = tmp["hmm_state_label"].astype(str)

    # mode label per (mouse, state)
    out: Dict[Tuple[str, int], str] = {}
    for (mouse, st), g in tmp.groupby(["mouse", "hmm_state"], sort=False):
        out[(str(mouse), int(st))] = g["hmm_state_label"].value_counts().idxmax()
    return out


# --- Additional helper functions for posterior smoothing and min-dwell enforcement ---
def _smooth_posteriors(P: np.ndarray, window: int = 7) -> np.ndarray:
    """Simple moving-average smoothing over time (per state)."""
    if window is None or int(window) <= 1:
        return P
    w = int(window)
    if w % 2 == 0:
        w += 1  # prefer odd window
    kernel = np.ones(w, dtype=float) / float(w)

    Ps = np.empty_like(P, dtype=float)
    for k in range(P.shape[1]):
        v = P[:, k].astype(float)
        # pad with edge values so we don't shrink the sequence
        pad = w // 2
        vpad = np.pad(v, (pad, pad), mode="edge")
        Ps[:, k] = np.convolve(vpad, kernel, mode="valid")

    # re-normalize to sum to 1 (numerical safety)
    Z = Ps.sum(axis=1, keepdims=True)
    Z = np.clip(Z, 1e-12, None)
    return Ps / Z


def _rle(states: np.ndarray) -> List[Tuple[int, int, int]]:
    """Run-length encode: returns [(state, start_idx, end_idx_exclusive), ...]."""
    s = np.asarray(states).astype(int)
    if s.size == 0:
        return []
    out: List[Tuple[int, int, int]] = []
    start = 0
    cur = int(s[0])
    for i in range(1, s.size):
        si = int(s[i])
        if si != cur:
            out.append((cur, start, i))
            start = i
            cur = si
    out.append((cur, start, s.size))
    return out


def enforce_min_dwell_from_posterior(
    P: np.ndarray,
    min_len: int = 20,
    init_states: np.ndarray | None = None,
    exclude_states: List[int] | None = None,
) -> np.ndarray:
    """
    Post-process state sequence so no segment is shorter than `min_len` trials.

    Strategy (A):
      - start from MAP states (argmax posterior) unless `init_states` provided
      - iteratively find short segments and "merge" them into the neighboring
        segment (left or right) that has higher mean posterior support.
      - exclude_states: list of state indices to skip min-dwell enforcement for

    This is a heuristic, but it does what you want visually: removes flickery
    1–5 trial islands while respecting local posterior evidence.
    """
    if P.ndim != 2:
        raise ValueError("P must be (T, K)")
    T, K = P.shape
    if T == 0:
        return np.array([], dtype=int)

    s = np.argmax(P, axis=1).astype(int) if init_states is None else np.asarray(init_states).astype(int).copy()
    L = int(min_len)
    if L <= 1:
        return s
    
    exclude_set = set(exclude_states) if exclude_states else set()

    # Iterate a few times; each pass should reduce the number of short segments
    for _ in range(10):
        segs = _rle(s)
        short = [(st, a, b) for (st, a, b) in segs if (b - a) < L and st not in exclude_set]
        if not short:
            break

        # Process shortest segments first
        short.sort(key=lambda t: (t[2] - t[1], t[1]))

        changed = False
        for st, a, b in short:
            if (b - a) >= L:
                continue

            left = a - 1
            right = b

            # If no neighbors (edge case), skip
            has_left = left >= 0
            has_right = right < T
            if not (has_left or has_right):
                continue

            # Candidate neighbor states
            cand = []
            if has_left:
                cand.append(int(s[left]))
            if has_right:
                cand.append(int(s[right]))
            cand = list(dict.fromkeys(cand))  # unique, stable order

            # Choose the neighbor whose posterior is higher on this short segment
            best_state = None
            best_score = -np.inf
            for c in cand:
                score = float(np.nanmean(P[a:b, c]))
                if score > best_score:
                    best_score = score
                    best_state = c

            if best_state is None:
                continue

            s[a:b] = int(best_state)
            changed = True

        if not changed:
            break

    return s


def plot_top_blocks(csv_path: str, top_n: int = 10, smooth: bool = False, min_dwell: int = 20, smooth_window: int = 7, per_mouse: bool = True) -> None:
    """Plot top-N blocks by mean max posterior confidence with posterior shading.
    
    Also enforces a minimum dwell time (in trials) using a posterior-informed
    merge heuristic, so you don't see lots of 1–5 trial state islands.
    
    Args:
        csv_path: Path to the CSV with HMM states
        top_n: Number of top blocks to plot (per mouse if per_mouse=True)
        smooth: If True, apply min-dwell and rolling-window smoothing to posteriors
        min_dwell: Minimum dwell time (trials) for smoothing
        smooth_window: Rolling window size for smoothing
        per_mouse: If True, select top N blocks per mouse; otherwise top N overall
    """
    assert os.path.exists(csv_path), f"Not found: {csv_path}"

    df = pd.read_csv(csv_path)

    # Posterior columns
    p_cols = [c for c in df.columns if c.startswith("hmm_p")]
    if len(p_cols) == 0:
        raise ValueError("No hmm_p* columns found. Did you write posteriors to the CSV?")

    # Score blocks
    df["_pmax"] = df[p_cols].max(axis=1)
    block_score = (
        df.groupby(["mouse", "dir", "block_num"], sort=False)
        .agg(mean_pmax=("_pmax", "mean"), n=("block_trial_num", "size"))
        .reset_index()
    )
    block_score["score"] = block_score["mean_pmax"] + 0.02 * np.log1p(block_score["n"])

    if per_mouse:
        # Select top N blocks per mouse
        top = (
            block_score.sort_values(["mouse", "score"], ascending=[True, False])
            .groupby("mouse", sort=False)
            .head(int(top_n))
            .reset_index(drop=True)
        )
    else:
        # Select top N blocks overall
        top = block_score.sort_values("score", ascending=False).head(int(top_n)).reset_index(drop=True)

    # Build mapping from numeric state -> label, per mouse
    stlab = _state_label_map(df)

    # Use consistent state colors
    color_map = STATE_COLORS.copy()
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    # Output folder structure: base_raw/mouse or base_smoothed/mouse
    base = os.path.splitext(csv_path)[0]
    smooth_suffix = "smoothed" if smooth else "raw"
    out_base_dir = base + f"_top{int(top_n)}_blocks_{smooth_suffix}"
    os.makedirs(out_base_dir, exist_ok=True)

    for rank, row in top.iterrows():
        mouse = str(row["mouse"])
        ddir = row["dir"]
        block_num = int(row["block_num"])
        
        # Create mouse-specific subfolder
        mouse_dir = os.path.join(out_base_dir, mouse)
        os.makedirs(mouse_dir, exist_ok=True)

        d = df[(df["mouse"] == mouse) & (df["dir"] == ddir) & (df["block_num"] == block_num)].copy()
        d.sort_values("block_trial_num", inplace=True)

        x = d["block_trial_num"].to_numpy()
        y = d["time_waited"].to_numpy()
        if "miss_trial" in d.columns:
            y = np.where(d["miss_trial"].to_numpy().astype(int) == 1, np.nan, y)

        # Posterior matrix (n_trials, n_states)
        P = d[p_cols].astype(float).to_numpy()
        n_states = P.shape[1]
        
        # Optionally smooth posteriors to reduce jitter
        P_use = _smooth_posteriors(P, window=smooth_window) if smooth else P

        # Posterior-informed minimum dwell post-processing (only if smoothing)
        if smooth:
            # Find which numeric state is "disengaged" for this mouse (exclude from min-dwell)
            disengaged_states = [
                st for st in range(n_states) 
                if stlab.get((mouse, st), "") == "disengaged"
            ]
            states_map = enforce_min_dwell_from_posterior(
                P_use, min_len=min_dwell, exclude_states=disengaged_states
            )
        else:
            # Raw: just use MAP states
            states_map = np.argmax(P_use, axis=1).astype(int)

        # Convert state posteriors into an RGB mixture based on each state's semantic label
        rgb = np.zeros((len(d), 3), dtype=float)

        for k in range(n_states):
            lab = stlab.get((mouse, k), str(k))
            base_rgb = _to_rgb(color_map.get(lab, cycle[k % len(cycle)] if cycle else None))
            rgb += P_use[:, k:k+1] * base_rgb[None, :]

        # Clip to valid [0, 1] range to avoid numerical precision warnings
        rgb = np.clip(rgb, 0.0, 1.0)

        # Make a 1-pixel-high image that spans the plot area
        fig, ax = plt.subplots(figsize=(14, 4))

        # scatter first to get y-limits from data (then shade)
        ax.scatter(x, y, s=8, linewidths=0, alpha=0.9)
        ylo, yhi = ax.get_ylim()

        # Overlay grouped (min-dwell) state spans so segments are never shorter than ~min_dwell trials
        segs = _rle(states_map)
        for st, a, b in segs:
            lab = stlab.get((mouse, int(st)), str(st))
            c = color_map.get(lab, cycle[int(st) % len(cycle)] if cycle else None)
            ax.axvspan(
                x[a] - 0.5,
                x[b - 1] + 0.5,
                ymin=0,
                ymax=1,
                color=c,
                alpha=0.10,
                zorder=1,
            )

        img = rgb[None, :, :]  # (1, n, 3)
        ax.imshow(
            img,
            extent=[x[0] - 0.5, x[-1] + 0.5, ylo, yhi],
            aspect="auto",
            interpolation="nearest",
            alpha=0.18,
            zorder=0,
        )

        ax.set_xlabel("block_trial_num")
        ax.set_ylabel("time_waited (s)")
        ax.set_title(
            "#{:02d} {} | {} | block {} | mean max posterior={:.3f} | n={}".format(
                rank + 1, mouse, ddir, block_num, float(row["mean_pmax"]), int(row["n"])
            )
        )

        # Legend: show the three semantic labels with their base colors
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_map[lab], alpha=0.18, label=lab)
            for lab in STATE_LABELS
            if lab in set(d.get("hmm_state_label", pd.Series([], dtype=str)).astype(str).unique())
            or any(stlab.get((mouse, k), "") == lab for k in range(P.shape[1]))
        ]
        if handles:
            ax.legend(handles=handles, loc="upper right", frameon=True)

        ax.grid(True, linewidth=0.4, alpha=0.4)
        plt.tight_layout()

        out_png = os.path.join(mouse_dir, f"top{rank+1:02d}_{ddir}_block{block_num}.png")
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved plot: {out_png}")

    print(f"\nWrote {len(top)} plots into: {out_base_dir}")

# =========================
# Model Diagnostics & Analysis
# =========================

def plot_state_occupancy(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot state occupancy (fraction of trials) per mouse as a stacked bar chart.
    """
    if "hmm_state_label" not in df.columns:
        raise ValueError("hmm_state_label column not found")
    
    occupancy = (
        df.groupby(["mouse", "hmm_state_label"])
        .size()
        .unstack(fill_value=0)
    )
    # Normalize to fractions
    occupancy = occupancy.div(occupancy.sum(axis=1), axis=0)
    
    # Reorder columns to canonical state order
    occupancy = _reorder_state_columns(occupancy)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    occupancy.plot(kind="bar", stacked=True, ax=ax, 
                   color=[STATE_COLORS[c] for c in occupancy.columns])
    ax.set_ylabel("Fraction of Trials")
    ax.set_xlabel("Mouse")
    ax.set_title("State Occupancy per Mouse")
    ax.legend(title="State")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "state_occupancy.png")
    _save_and_close(fig, out_path)


def plot_state_occupancy_by_group(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot state occupancy split by group (long vs short) in two subplots.
    """
    if "hmm_state_label" not in df.columns:
        raise ValueError("hmm_state_label column not found")
    if "group" not in df.columns:
        print("Warning: 'group' column not found, skipping group-based occupancy plot")
        return
    
    # Get mouse-to-group mapping
    mouse_groups = df.groupby("mouse")["group"].first()
    
    # Calculate occupancy for all mice
    occupancy = (
        df.groupby(["mouse", "hmm_state_label"])
        .size()
        .unstack(fill_value=0)
    )
    # Normalize to fractions
    occupancy = occupancy.div(occupancy.sum(axis=1), axis=0)
    
    # Reorder columns to canonical state order
    occupancy = _reorder_state_columns(occupancy)
    
    # Split by group
    long_mice = mouse_groups[mouse_groups == "l"].index
    short_mice = mouse_groups[mouse_groups == "s"].index
    
    occupancy_long = occupancy.loc[occupancy.index.isin(long_mice)]
    occupancy_short = occupancy.loc[occupancy.index.isin(short_mice)]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot long group
    if len(occupancy_long) > 0:
        occupancy_long.plot(kind="bar", stacked=True, ax=ax1,
                           color=[STATE_COLORS[c] for c in occupancy_long.columns])
        ax1.set_ylabel("Fraction of Trials")
        ax1.set_xlabel("Mouse")
        ax1.set_title("State Occupancy - Long Group")
        ax1.legend(title="State")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        ax1.grid(True, alpha=0.3, axis="y")
    
    # Plot short group
    if len(occupancy_short) > 0:
        occupancy_short.plot(kind="bar", stacked=True, ax=ax2,
                            color=[STATE_COLORS[c] for c in occupancy_short.columns])
        ax2.set_ylabel("Fraction of Trials")
        ax2.set_xlabel("Mouse")
        ax2.set_title("State Occupancy - Short Group")
        ax2.legend(title="State")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
        ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "state_occupancy_by_group.png")
    _save_and_close(fig, out_path)


def plot_feature_distributions(df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot box plots of raw feature values grouped by state.
    Shows which features differ most across states.
    """
    if "hmm_state_label" not in df.columns:
        raise ValueError("hmm_state_label column not found")
    
    # Use raw features that are interpretable
    plot_features = [
        "time_waited", "miss_trial", "bg_repeat_rate", 
        "mean_bg_lick_phase", "is_fast_wait",
        "time_waited_since_last_lick", "first_lick", 
        "num_consumption_lick"
    ]
    
    available = [f for f in plot_features if f in df.columns]
    if not available:
        print("Warning: No plottable features found")
        return
    
    n_features = len(available)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    for i, feat in enumerate(available):
        ax = axes[i]
        data_to_plot = []
        labels = []
        all_vals = []
        
        for state in STATE_LABELS:
            mask = df["hmm_state_label"] == state
            if mask.any():
                vals = df.loc[mask, feat].dropna()
                if len(vals) > 0:
                    data_to_plot.append(vals)
                    labels.append(state)
                    all_vals.extend(vals.values)
        
        if data_to_plot:
            positions = np.arange(1, len(data_to_plot) + 1)
            parts = ax.violinplot(data_to_plot, positions=positions, 
                                   showmeans=True, showmedians=True, widths=0.7)
            
            # Color the violin plots
            for idx, pc in enumerate(parts['bodies']):
                pc.set_facecolor(STATE_COLORS[labels[idx]])
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            # Style the other elements
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                if partname in parts:
                    vp = parts[partname]
                    vp.set_edgecolor('black')
                    vp.set_linewidth(1)
            
            # Set x-tick labels
            ax.set_xticks(positions)
            ax.set_xticklabels(labels)
            
            # Smart y-limits: use percentiles to avoid extreme outliers squishing the plot
            all_vals = np.array(all_vals)
            p1, p99 = np.percentile(all_vals, [1, 99])
            p25, p75 = np.percentile(all_vals, [25, 75])
            iqr = p75 - p25
            
            # If the range is reasonable, just use percentile-based limits
            # Otherwise keep auto scaling
            if np.isfinite(p1) and np.isfinite(p99) and iqr > 0:
                # Extend slightly beyond p1-p99 for visibility
                margin = 0.1 * (p99 - p1)
                y_min = max(0, p1 - margin) if all_vals.min() >= 0 else p1 - margin
                y_max = p99 + margin
                
                # Only apply if it actually helps (i.e., removes extreme outliers)
                if y_max < all_vals.max() * 0.8:
                    ax.set_ylim(y_min, y_max)
        
        ax.set_ylabel(feat, fontsize=10)
        ax.set_xlabel("State", fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    
    plt.suptitle("Feature Distributions by State", fontsize=14, y=1.00)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "feature_distributions_by_state.png")
    _save_and_close(fig, out_path)


def analyze_feature_separability(
    df: pd.DataFrame, 
    model_results: Dict[str, MouseModelResult],
    output_dir: str
) -> None:
    """
    Analyze which features best separate the states.
    Uses between-state variance / within-state variance as a metric.
    Higher values = better state separation.
    """
    print("\n=== Feature Separability Analysis ===")
    
    # For each mouse, compute per-state means and variances for each feature
    all_results = []
    
    for mouse, res in model_results.items():
        mouse_df = df[df["mouse"] == mouse].copy()
        
        for feat in FEATURE_COLS:
            if feat not in mouse_df.columns:
                continue
            
            # Get values per state
            state_means = []
            state_vars = []
            
            for st_num in range(N_STATES):
                st_label = res.state_map.get(st_num, str(st_num))
                mask = mouse_df["hmm_state_label"] == st_label
                vals = mouse_df.loc[mask, feat].dropna()
                
                if len(vals) > 1:
                    state_means.append(np.mean(vals))
                    state_vars.append(np.var(vals))
            
            if len(state_means) >= 2:
                # Between-state variance
                grand_mean = np.mean(state_means)
                between_var = np.var(state_means)
                
                # Within-state variance (average)
                within_var = np.mean(state_vars)
                
                # Separability = between / within
                if within_var > 1e-9:
                    separability = between_var / within_var
                else:
                    separability = np.inf if between_var > 1e-9 else 0.0
                
                all_results.append({
                    "mouse": mouse,
                    "feature": feat,
                    "separability": separability,
                    "between_var": between_var,
                    "within_var": within_var,
                })
    
    results_df = pd.DataFrame(all_results)
    
    # Average separability across mice per feature
    avg_sep = (
        results_df.groupby("feature")["separability"]
        .mean()
        .sort_values(ascending=False)
    )
    
    print("\nAverage Separability (between-state var / within-state var):")
    for feat, sep in avg_sep.items():
        print(f"  {feat:40s}: {sep:8.3f}")
    
    # Plot top features
    top_features = avg_sep.head(11)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Separability Score (between-var / within-var)")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Separability: Which Features Best Distinguish States?")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "feature_separability.png")
    _save_and_close(fig, out_path)
    
    # Save detailed results
    csv_path = os.path.join(output_dir, "feature_separability.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def analyze_model_results(
    df: pd.DataFrame,
    model_results: Dict[str, MouseModelResult],
    output_dir: str
) -> None:
    """
    Master function to run all diagnostic analyses.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== Running Model Diagnostics ===")
    print(f"Output directory: {output_dir}")
    
    plot_state_occupancy(df, output_dir)
    plot_state_occupancy_by_group(df, output_dir)
    plot_feature_distributions(df, output_dir)
    analyze_feature_separability(df, model_results, output_dir)
    
    # Print per-mouse state summary
    print("\n=== State Summary per Mouse ===")
    for mouse in sorted(df["mouse"].unique()):
        mouse_df = df[df["mouse"] == mouse]
        counts = mouse_df["hmm_state_label"].value_counts()
        total = len(mouse_df)
        print(f"\nMouse {mouse} (n={total} trials):")
        for state in STATE_LABELS:
            if state in counts:
                n = counts[state]
                pct = 100.0 * n / total
                print(f"  {state:12s}: {n:5d} trials ({pct:5.1f}%)")

# =========================
# Main
# =========================
def main() -> None:
    # Check if we should load existing results or retrain
    if os.path.exists(MODEL_RESULTS_PKL) and os.path.exists(OUTPUT_CSV) and not FORCE_RETRAIN:
        print("=== Loading existing HMM results ===")
        print(f"Loading from: {MODEL_RESULTS_PKL}")
        print(f"Loading from: {OUTPUT_CSV}")
        print("(Set FORCE_RETRAIN=True to retrain models)\n")
        
        with open(MODEL_RESULTS_PKL, "rb") as f:
            model_results = pickle.load(f)
        
        df_out = pd.read_csv(OUTPUT_CSV)
        
        print(f"Loaded results for {len(model_results)} mice")
        for mouse in model_results:
            print(f"  - Mouse {mouse}: {model_results[mouse].state_map}")
    else:
        # Step 1: Fit HMM models
        print("=== Step 1: Loading data and engineering features ===")
        trials_filtered = pd.read_csv(TRIALS_CSV)
        df_feat = engineer_features(trials_filtered)
        
        print("\n=== Step 2: Fitting HMM per mouse ===")
        df_out, model_results = fit_hmm_per_mouse(df_feat)

        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"\nWrote: {OUTPUT_CSV}")
        
        # Save model results for future use
        with open(MODEL_RESULTS_PKL, "wb") as f:
            pickle.dump(model_results, f)
        print(f"Saved model results to: {MODEL_RESULTS_PKL}")

        # Quick peek: reward-split delta means per state (in standardized space)
        feat_to_idx = {c: i for i, c in enumerate(FEATURE_COLS)}
        iR = feat_to_idx["dlog_wait_prevR"]
        iU = feat_to_idx["dlog_wait_prevU"]

        print("\n=== Quick peek: state means for reward-split delta features (per mouse, standardized space) ===")
        for mouse, res in model_results.items():
            means = res.model.means_
            print(f"\nMouse {mouse} state_map={res.state_map}")
            for s in range(N_STATES):
                print(f"  state {s}: mean(dlog_prevR)={means[s, iR]: .3f}, mean(dlog_prevU)={means[s, iU]: .3f}")

    # Step 3: Run diagnostic analyses (always run with current data)
    print("\n=== Step 3: Running diagnostic analyses ===")
    diag_dir = os.path.join(OUTPUT_FOLDER, "diagnostics")
    analyze_model_results(df_out, model_results, diag_dir)

    # # Step 4: Plot top blocks (both smoothed and non-smoothed versions)
    # print("\n=== Step 4: Plotting top 20 blocks per mouse ===")
    # print("\n=== Plotting non-smoothed blocks (top 20 per mouse) ===")
    # plot_top_blocks(OUTPUT_CSV, top_n=20, smooth=False, min_dwell=SMOOTH_MIN_DWELL, smooth_window=SMOOTH_WINDOW, per_mouse=True)
    
    # print("\n=== Plotting smoothed blocks (top 20 per mouse) ===")
    # plot_top_blocks(OUTPUT_CSV, top_n=20, smooth=True, min_dwell=SMOOTH_MIN_DWELL, smooth_window=SMOOTH_WINDOW, per_mouse=True)


if __name__ == "__main__":
    main()