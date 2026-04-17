import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import utils

# Configuration
data_dir = '/Users/rebekahzhang/data/behavior_data'
exp = 'exp2'
data_folder = os.path.join(data_dir, exp)
lick_bouts_dir = utils.generate_lick_bouts_path(data_dir)
os.makedirs(lick_bouts_dir, exist_ok=True)

figures_dir = os.path.join(lick_bouts_dir, 'figures')
examples_dir = os.path.join(figures_dir, 'examples')

MIN_LICKS_PER_BOUT = 2
MAX_ILI_FILTER = 1.0   # seconds; only short (within-bout) ILIs used for stats/thresholds
MIN_MEDIAN_ILI = 0.1   # safety net for sensor bounce (should already be caught at step 0 QC)

custom_palette = {'s': '#ffb400', 'l': '#9080ff'}


# =============================================================================
# BOUT DETECTION FUNCTIONS
# =============================================================================

def detect_threshold_from_ilis(ilis, bins=500, smooth_window=35):
    """
    Detect bout ILI threshold (seconds) from pooled per-mouse ILIs by:
    1. Finding the dominant within-bout peak in log10 ILI space (-1.5, -0.4).
    2. Searching for the FIRST local minimum to the right of that peak (up to +0.8 log10 units).
    Falls back to 0.25s if no peak or valley is found.
    """
    ilis = np.asarray(ilis)
    ilis = ilis[ilis > 0]
    log_ilis = np.log10(ilis)
    log_ilis = log_ilis[(log_ilis >= -1.5) & (log_ilis <= 1.5)]

    counts, edges = np.histogram(log_ilis, bins=bins, range=(-1.5, 1.5))
    centers = (edges[:-1] + edges[1:]) / 2

    w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
    smooth = savgol_filter(counts.astype(float), window_length=w, polyorder=3)
    smooth = np.clip(smooth, 0, None)

    # Step 1: find dominant peak in within-bout range
    wb_mask = (centers >= -1.5) & (centers <= -0.4)
    if not wb_mask.any():
        return 0.25
    peak_pos = centers[wb_mask][np.argmax(smooth[wb_mask])]

    # Step 2: find the FIRST local minimum to the right of the peak (up to +0.8 log10)
    valley_mask = (centers > peak_pos) & (centers <= peak_pos + 0.8)
    if not valley_mask.any():
        return 0.25

    valley_smooth = smooth[valley_mask]
    valley_centers = centers[valley_mask]
    deriv = np.diff(valley_smooth)
    sign_changes = np.where((deriv[:-1] < 0) & (deriv[1:] >= 0))[0]

    if len(sign_changes) > 0:
        valley_log10 = valley_centers[sign_changes[0] + 1]
    else:
        valley_log10 = valley_centers[np.argmin(valley_smooth)]

    return float(10 ** valley_log10)


def detect_bouts(lick_onsets, lick_offsets, threshold, min_licks=1):
    """
    Segment licks into bouts using ILI thresholding.

    Parameters
    ----------
    lick_onsets  : sorted array of lick onset times (s)
    lick_offsets : sorted array of lick offset times (s)
    threshold    : max ILI (s) within a bout
    min_licks    : minimum licks to keep a bout

    Returns
    -------
    list of dicts: onset, offset, lick_onsets, n_licks
    """
    lick_onsets = np.asarray(lick_onsets)
    lick_offsets = np.asarray(lick_offsets)
    if len(lick_onsets) == 0:
        return []
    n = min(len(lick_onsets), len(lick_offsets))
    # Off-by-one is expected: the last lick of a block often has an onset but no
    # offset because the block ends while the animal is still mid-lick.
    if abs(len(lick_onsets) - len(lick_offsets)) > 1:
        print(f'  Warning: {len(lick_onsets)} onsets vs {len(lick_offsets)} offsets; truncating to {n}')
    lick_onsets = lick_onsets[:n]
    lick_offsets = lick_offsets[:n]

    bouts = []
    bout_start_idx = 0
    for i in range(1, n):
        if lick_onsets[i] - lick_onsets[i - 1] > threshold:
            if (i - bout_start_idx) >= min_licks:
                bouts.append({
                    'onset': lick_onsets[bout_start_idx],
                    'offset': lick_offsets[i - 1],
                    'lick_onsets': lick_onsets[bout_start_idx:i],
                    'n_licks': i - bout_start_idx,
                })
            bout_start_idx = i
    if (n - bout_start_idx) >= min_licks:
        bouts.append({
            'onset': lick_onsets[bout_start_idx],
            'offset': lick_offsets[-1],
            'lick_onsets': lick_onsets[bout_start_idx:],
            'n_licks': n - bout_start_idx,
        })
    return bouts


# =============================================================================
# STEP 1: PER-SESSION ILI STATS + PER-MOUSE THRESHOLD DETECTION
# =============================================================================

def compute_thresholds(sessions, data_folder):
    """
    Compute per-session lick statistics and per-mouse ILI thresholds.

    Returns
    -------
    mouse_thresholds    : dict  {mouse: threshold_s}
    session_lick_stats  : DataFrame
    mouse_ilis          : dict  {mouse: concatenated ILI array}
    all_ilis            : array of all within-block ILIs across all sessions
    """
    session_lick_stats = []
    mouse_ilis = {}  # mouse -> list of ILI arrays (one per session)

    for _, session_info in sessions.iterrows():
        mouse = session_info['mouse']
        ddir = session_info['dir']
        try:
            events = utils.load_data(utils.generate_events_processed_path(data_folder, session_info))
        except Exception as e:
            print(f'Skipping {ddir}: {e}')
            continue

        lick_events = events.loc[events['key'] == 'lick']
        session_ilis = []
        session_n_licks = 0
        session_active_time = 0.0

        for _, blk in lick_events.groupby('block_num', sort=True):
            block_onsets = blk.loc[blk['value'] == 1, 'session_time'].sort_values().values
            session_n_licks += len(block_onsets)
            if len(block_onsets) < 2:
                continue
            block_ilis = np.diff(block_onsets)
            session_ilis.append(block_ilis)
            session_active_time += block_onsets[-1] - block_onsets[0]

        if not session_ilis or session_active_time == 0:
            continue

        ilis = np.concatenate(session_ilis)
        short_ilis = ilis[ilis < MAX_ILI_FILTER]
        if len(short_ilis) == 0:
            continue

        median_ili_val = float(np.median(short_ilis))
        if median_ili_val < MIN_MEDIAN_ILI:
            print(f'Warning: {ddir} median ILI={median_ili_val:.4f}s (sensor bounce; '
                  f'should have been removed at step 0 QC). Skipping.')
            continue

        mouse_ilis.setdefault(mouse, []).append(ilis)
        session_lick_stats.append({
            'mouse': mouse, 'dir': ddir,
            'n_licks': session_n_licks,
            'session_active_time_s': session_active_time,
            'mean_lick_rate': session_n_licks / session_active_time,
            'mean_ili':   float(np.mean(short_ilis)),
            'median_ili': median_ili_val,
            'p5_ili':     float(np.percentile(short_ilis,  5)),
            'p95_ili':    float(np.percentile(short_ilis, 95)),
        })

    session_lick_stats_df = pd.DataFrame(session_lick_stats)
    session_lick_stats_df = session_lick_stats_df.merge(
        sessions[['dir', 'group']].drop_duplicates(), on='dir', how='left'
    )

    # Compute per-mouse thresholds from pooled ILIs across all sessions
    mouse_thresholds = {}
    all_ilis_list = []
    for mouse, ilis_list in mouse_ilis.items():
        concatenated = np.concatenate(ilis_list)
        mouse_thresholds[mouse] = detect_threshold_from_ilis(concatenated)
        mouse_ilis[mouse] = concatenated
        all_ilis_list.append(concatenated)

    all_ilis = np.concatenate(all_ilis_list) if all_ilis_list else np.array([])

    return mouse_thresholds, session_lick_stats_df, mouse_ilis, all_ilis


# =============================================================================
# STEP 2: BOUT DETECTION
# =============================================================================

def detect_all_bouts(sessions, data_folder, mouse_thresholds, regenerate=False):
    """
    Detect lick bouts for all sessions using per-mouse ILI thresholds.

    Returns
    -------
    bouts_df : DataFrame
    """
    bouts_path = os.path.join(lick_bouts_dir, 'bouts.csv')
    if not regenerate and os.path.isfile(bouts_path):
        print(f'bouts.csv already exists. Loading cached file.')
        print('Pass regenerate=True to recompute.')
        return pd.read_csv(bouts_path)

    all_bouts = []
    total = len(sessions)

    for idx, (_, session_info) in enumerate(sessions.iterrows(), 1):
        mouse = session_info['mouse']
        ddir = session_info['dir']
        threshold = mouse_thresholds.get(mouse, 0.25)
        try:
            events = utils.load_data(utils.generate_events_processed_path(data_folder, session_info))
        except Exception as e:
            print(f'Skipping {ddir}: {e}')
            continue

        lick_events = events.loc[events['key'] == 'lick'].sort_values('session_time')

        for block_num, blk in lick_events.groupby('block_num', sort=True):
            blk_onsets  = blk.loc[blk['value'] == 1, 'session_time'].sort_values().values
            blk_offsets = blk.loc[blk['value'] == 0, 'session_time'].sort_values().values

            if len(blk_onsets) < 1:
                continue

            bouts = detect_bouts(blk_onsets, blk_offsets,
                                  threshold=threshold, min_licks=MIN_LICKS_PER_BOUT)

            for bout in bouts:
                preceding = blk.loc[blk['session_time'] <= bout['onset'], 'session_trial_num']
                trial_num = int(preceding.iloc[-1]) if len(preceding) > 0 else np.nan
                all_bouts.append({
                    'mouse': mouse, 'dir': ddir,
                    'block_num': block_num,
                    'session_trial_num': trial_num,
                    'bout_onset': bout['onset'],
                    'bout_offset': bout['offset'],
                    'bout_duration': bout['offset'] - bout['onset'],
                    'n_licks': bout['n_licks'],
                    'threshold_used': threshold,
                })

        if idx % 100 == 0 or idx == total:
            print(f'Progress: {idx}/{total} sessions')

    return pd.DataFrame(all_bouts)


# =============================================================================
# STEP 3: PLOTS
# =============================================================================

def plot(bouts_df, session_lick_stats, mouse_thresholds, mouse_ilis, all_ilis,
         sessions, mouse_group):
    """
    Generate and save all lick bout summary plots.
    """
    os.makedirs(examples_dir, exist_ok=True)

    mice = sorted(session_lick_stats['mouse'].unique())

    def get_threshold(mouse):
        return mouse_thresholds.get(str(mouse), 0.25)

    # --- Global ILI histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    log_all = np.log10(all_ilis[(all_ilis > 0) & (all_ilis < 30)])
    ax.hist(log_all, bins=300, range=(-1.5, 1.5), color='steelblue', edgecolor='none', alpha=0.85)
    ax.set_xlabel('log\u2081\u2080 ILI (s)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('ILI distribution reveals within-bout and between-bout structure', fontsize=12)
    ax.set_xlim(-1.5, 1.5)
    ax.annotate('Within-bout\n(~5\u20136 Hz)', xy=(-0.75, ax.get_ylim()[1] * 0.6), ha='center', fontsize=10, color='steelblue')
    ax.annotate('Between-bout\n(seconds)',    xy=(0.5,  ax.get_ylim()[1] * 0.4), ha='center', fontsize=10, color='steelblue')
    median_thresh = np.median(list(mouse_thresholds.values()))
    ax.axvline(np.log10(median_thresh), color='crimson', linestyle='--', linewidth=2,
               label=f'Median detected threshold ({median_thresh:.2f} s)')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'global_ili_distribution.png'), dpi=150)
    plt.close()

    # --- Per-mouse ILI histograms ---
    n_cols = min(6, len(mice))
    n_rows = int(np.ceil(len(mice) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.33, 9.5), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for i, mouse in enumerate(mice):
        ax = axes[i]
        ilis_m = mouse_ilis.get(mouse)
        if ilis_m is None:
            ax.axis('off')
            continue
        log_m = np.log10(ilis_m[(ilis_m > 0) & (ilis_m < 30)])
        color = custom_palette.get(mouse_group.get(mouse, 's'), '#888888')
        ax.hist(log_m, bins=150, range=(-1.5, 1.5), color=color, edgecolor='none', alpha=0.8)
        thresh = get_threshold(mouse)
        ax.axvline(np.log10(thresh), color='crimson', linestyle='--', linewidth=1.5,
                   label=f'T={thresh:.3f}s')
        ax.set_xlim(-1.2, 0.5)
        ax.set_title(f'{mouse}', fontsize=5, pad=3)
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('log\u2081\u2080 ILI (s)', fontsize=4)
        ax.set_ylabel('Count', fontsize=4, rotation=90)
        ax.tick_params(labelsize=4, pad=1)
        ax.legend(fontsize=4, loc='upper right', borderpad=0.3, handlelength=1)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.4)
    plt.savefig(os.path.join(figures_dir, 'per_mouse_ili_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Session lick statistics ---
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    metrics = [
        ('mean_lick_rate', 'Mean lick rate (licks/s)', 'Mean Lick Rate per Mouse'),
        ('median_ili',     'Median ILI (s)',            'Median ILI per Mouse (within-bout ILIs only)'),
    ]
    for ax, (col, ylabel, title) in zip(axes, metrics):
        for i, mouse in enumerate(mice):
            grp = session_lick_stats[session_lick_stats['mouse'] == mouse]
            vals = grp[col].dropna().values
            if len(vals) == 0:
                continue
            color = custom_palette.get(mouse_group.get(mouse, 's'), '#888888')
            bp = ax.boxplot(
                vals, positions=[i], widths=0.55, patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(color='#555555', linewidth=1),
                capprops=dict(color='#555555', linewidth=1),
                flierprops=dict(marker='', alpha=0),
                boxprops=dict(linewidth=1),
            )
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.65)
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(i + jitter, vals, color=color, alpha=0.55, s=14, zorder=3, edgecolors='none')
        ax.set_xticks(range(len(mice)))
        ax.set_xticklabels(mice, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.7, label=f'Group {g}')
                   for g, c in custom_palette.items()]
        ax.legend(handles=handles, fontsize=9)
    axes[0].set_ylim(0, 2.7)
    axes[1].set_ylim(0.09, 0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'lick_stats_by_mouse.png'), dpi=150)
    plt.close()

    # --- Bout validation ---
    session_val = (
        bouts_df.groupby(['mouse', 'dir'])
        .agg(
            n_bouts=('n_licks', 'count'),
            mean_bout_size=('n_licks', 'mean'),
            median_bout_size=('n_licks', 'median'),
        ).reset_index()
    )
    rng_v = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    val_metrics = [
        ('n_bouts',        'Bouts per session',       axes[0]),
        ('mean_bout_size', 'Mean bout size (n licks)', axes[1]),
    ]
    for col, ylabel, ax in val_metrics:
        for i, mouse in enumerate(mice):
            grp = session_val[session_val['mouse'] == mouse]
            vals = grp[col].dropna().values
            if len(vals) == 0:
                continue
            color = custom_palette.get(mouse_group.get(mouse, 's'), '#888888')
            bp = ax.boxplot(
                vals, positions=[i], widths=0.55, patch_artist=True,
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(color='#555'), capprops=dict(color='#555'),
                flierprops=dict(marker='', alpha=0),
            )
            bp['boxes'][0].set_facecolor(color)
            bp['boxes'][0].set_alpha(0.65)
            jitter = rng_v.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(i + jitter, vals, color=color, alpha=0.5, s=12, zorder=3, edgecolors='none')
        ax.set_xticks(range(len(mice)))
        ax.set_xticklabels(mice, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle('Bout validation per session (min 2 licks, per-mouse detected thresholds)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'bout_validation_per_session.png'), dpi=150)
    plt.close()

    # --- Bout size distribution & within-bout lick frequency ---
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5))
    ax_a.hist(bouts_df['n_licks'].clip(upper=20), bins=range(2, 22),
              color='steelblue', edgecolor='white', alpha=0.85, rwidth=0.85)
    ax_a.set_xlabel('Licks per bout', fontsize=11)
    ax_a.set_ylabel('Count', fontsize=11)
    ax_a.set_title('Bout size distribution (all mice, min 2 licks)', fontsize=12)
    ax_a.set_xticks(range(2, 21))
    ax_a.grid(True, alpha=0.3, axis='y')

    rng_b = np.random.default_rng(5)
    for i, mouse in enumerate(mice):
        vals = (1 / session_lick_stats.loc[session_lick_stats['mouse'] == mouse, 'median_ili']).dropna().values
        if len(vals) == 0:
            continue
        color = custom_palette.get(mouse_group.get(mouse, 's'), '#888')
        bp = ax_b.boxplot(vals, positions=[i], widths=0.55, patch_artist=True,
                          medianprops=dict(color='black', linewidth=2),
                          whiskerprops=dict(color='#555'), capprops=dict(color='#555'),
                          flierprops=dict(marker='', alpha=0))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.65)
        jitter = rng_b.uniform(-0.18, 0.18, size=len(vals))
        ax_b.scatter(i + jitter, vals, color=color, alpha=0.5, s=12, zorder=3, edgecolors='none')
    ax_b.set_xticks(range(len(mice)))
    ax_b.set_xticklabels(mice, rotation=45, ha='right', fontsize=8)
    ax_b.set_ylabel('Within-bout lick frequency (licks/s)', fontsize=11)
    ax_b.set_title('Within-bout lick frequency per session per mouse', fontsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.75, label=f'Group {g}')
               for g, c in custom_palette.items()]
    ax_b.legend(handles=handles, fontsize=9)
    ax_b.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'bout_size_and_lick_freq.png'), dpi=150)
    plt.close()

    # --- Example bouts (30-second windows) ---
    t_window = 30
    n_examples = 50
    rng_ex = np.random.default_rng(42)
    sample_sessions = sessions.sample(n=n_examples, random_state=42).reset_index(drop=True)

    saved = 0
    for _, session_info in sample_sessions.iterrows():
        mouse = session_info['mouse']
        ddir  = session_info['dir']
        threshold = get_threshold(mouse)
        try:
            events = utils.load_data(utils.generate_events_processed_path(data_folder, session_info))
        except Exception as e:
            print(f'Skipping {ddir}: {e}')
            continue

        ex_onsets  = events.loc[(events['key'] == 'lick') & (events['value'] == 1), 'session_time'].sort_values().values
        ex_offsets = events.loc[(events['key'] == 'lick') & (events['value'] == 0), 'session_time'].sort_values().values

        if len(ex_onsets) < 10 or (ex_onsets[-1] - ex_onsets[0]) < t_window:
            continue

        ex_bouts = detect_bouts(ex_onsets, ex_offsets, threshold=threshold, min_licks=2)
        if len(ex_bouts) == 0:
            continue

        t0 = ex_onsets[0]
        rel_onsets = ex_onsets - t0
        valid_starts = rel_onsets[rel_onsets <= rel_onsets[-1] - t_window]
        if len(valid_starts) == 0:
            continue
        t_start = rng_ex.choice(valid_starts)
        t_end = t_start + t_window

        color_cycle = [custom_palette.get(mouse_group.get(mouse, 's'), '#888'), '#aaaaaa']
        fig, ax = plt.subplots(figsize=(14, 3))
        for k, bout in enumerate(ex_bouts):
            bout_onsets = bout['lick_onsets'] - t0
            if not np.any((bout_onsets >= t_start) & (bout_onsets < t_end)):
                continue
            b_start = bout_onsets[0]
            b_end   = bout_onsets[-1] + 0.15
            ax.axvspan(max(b_start, t_start) - t_start,
                       min(b_end,   t_end)   - t_start,
                       ymin=0, ymax=1, color=color_cycle[k % 2], alpha=0.25)
        onsets_windowed = ex_onsets[(rel_onsets >= t_start) & (rel_onsets < t_end)]
        ax.vlines(onsets_windowed - t0 - t_start, 0.1, 0.9, linewidth=0.6, color='steelblue', alpha=0.9)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_yticks([])
        ax.set_xlim(0, t_window)
        ax.set_title(f'{mouse} | {ddir} | threshold={threshold:.3f}s', fontsize=9)
        ax.grid(True, alpha=0.2, axis='x')
        plt.tight_layout()
        fname = f'{saved + 1:02d}_{mouse}_{ddir[:10]}.png'
        fig.savefig(os.path.join(examples_dir, fname), dpi=100)
        plt.close()
        saved += 1

    print(f'Saved {saved} example bout plots to {examples_dir}')
    print(f'Saved summary figures to {figures_dir}')


# =============================================================================
# MAIN
# =============================================================================

def run_stats(regenerate=False):
    """
    Steps 1 & 2: compute ILI stats, per-mouse thresholds, and detect bouts.
    Saves all outputs to disk. Run this once (or with regenerate=True to recompute).
    """
    sessions = pd.read_csv(os.path.join(data_folder, f'sessions_training_{exp}.csv'))
    mouse_group = sessions.groupby('mouse')['group'].first().to_dict()
    print(f'Loaded {len(sessions)} sessions for {sessions["mouse"].nunique()} mice')

    # Step 1: per-session ILI stats and per-mouse thresholds
    print('\nStep 1: Computing ILI statistics and per-mouse thresholds...')
    mouse_thresholds, session_lick_stats, mouse_ilis, all_ilis = compute_thresholds(sessions, data_folder)
    print(f'Computed thresholds for {len(mouse_thresholds)} mice')

    thresh_df = pd.DataFrame([
        {'mouse': m, 'group': mouse_group.get(m, '?'),
         'threshold_s': round(t, 4),
         'threshold_log10': round(np.log10(t), 3)}
        for m, t in sorted(mouse_thresholds.items())
    ])
    thresh_df.to_csv(os.path.join(lick_bouts_dir, 'mouse_thresholds.csv'), index=False)
    session_lick_stats.to_csv(os.path.join(lick_bouts_dir, 'session_lick_stats.csv'), index=False)
    np.savez(os.path.join(lick_bouts_dir, 'mouse_ilis.npz'), **{str(m): v for m, v in mouse_ilis.items()})
    np.save(os.path.join(lick_bouts_dir, 'all_ilis.npy'), all_ilis)
    print(f'Saved mouse_thresholds.csv ({len(thresh_df)} mice)')
    print(f'Saved session_lick_stats.csv ({len(session_lick_stats)} sessions)')
    print(f'Saved mouse_ilis.npz and all_ilis.npy')

    # Step 2: bout detection
    print('\nStep 2: Detecting lick bouts...')
    bouts_df = detect_all_bouts(sessions, data_folder, mouse_thresholds, regenerate=regenerate)
    bouts_df.to_csv(os.path.join(lick_bouts_dir, 'bouts.csv'), index=False)
    print(f'Detected {len(bouts_df):,} bouts across {bouts_df["dir"].nunique()} sessions, '
          f'{bouts_df["mouse"].nunique()} mice')
    print(f'Saved bouts.csv')


def run_plots():
    """
    Step 3: Load saved outputs and regenerate all plots. No bout detection needed.
    """
    sessions = pd.read_csv(os.path.join(data_folder, f'sessions_training_{exp}.csv'))
    mouse_group = sessions.groupby('mouse')['group'].first().to_dict()

    stats_missing = not all(os.path.isfile(os.path.join(lick_bouts_dir, f))
                            for f in ('bouts.csv', 'session_lick_stats.csv', 'mouse_thresholds.csv'))
    if stats_missing:
        print('Cached stats not found — running full stats pipeline first...')
        run_stats()

    bouts_df = pd.read_csv(os.path.join(lick_bouts_dir, 'bouts.csv'))
    session_lick_stats = pd.read_csv(os.path.join(lick_bouts_dir, 'session_lick_stats.csv'))
    thresh_df = pd.read_csv(os.path.join(lick_bouts_dir, 'mouse_thresholds.csv'))
    mouse_thresholds = dict(zip(thresh_df['mouse'].astype(str), thresh_df['threshold_s']))

    ilis_npz_path = os.path.join(lick_bouts_dir, 'mouse_ilis.npz')
    all_ilis_path = os.path.join(lick_bouts_dir, 'all_ilis.npy')
    if os.path.isfile(ilis_npz_path) and os.path.isfile(all_ilis_path):
        npz = np.load(ilis_npz_path)
        mouse_ilis = {m: npz[m] for m in npz.files}
        all_ilis = np.load(all_ilis_path)
    else:
        print('ILI cache not found — recomputing from sessions (no bout detection)...')
        _, _, mouse_ilis, all_ilis = compute_thresholds(sessions, data_folder)
        np.savez(ilis_npz_path, **{str(m): v for m, v in mouse_ilis.items()})
        np.save(all_ilis_path, all_ilis)
        print('Saved ILI cache for next time.')

    print('Generating plots...')
    plot(bouts_df, session_lick_stats, mouse_thresholds, mouse_ilis, all_ilis,
         sessions, mouse_group)


if __name__ == '__main__':
    # run_stats()
    run_plots()
