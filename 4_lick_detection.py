import os

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

MIN_LICKS_PER_BOUT = 3
MAX_ILI_FILTER = 1.0   # seconds; only short (within-bout) ILIs used for stats/thresholds
MIN_MEDIAN_ILI = 0.1   # safety net for sensor bounce (should already be caught at step 0 QC)

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
    mouse_thresholds = {
        mouse: detect_threshold_from_ilis(np.concatenate(ilis_list))
        for mouse, ilis_list in mouse_ilis.items()
    }

    return mouse_thresholds, session_lick_stats_df


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
# MAIN
# =============================================================================

def main(regenerate=False):
    sessions = pd.read_csv(os.path.join(data_folder, f'sessions_training_{exp}.csv'))
    mouse_group = sessions.groupby('mouse')['group'].first().to_dict()
    print(f'Loaded {len(sessions)} sessions for {sessions["mouse"].nunique()} mice')

    # Step 1: per-session ILI stats and per-mouse thresholds
    print('\nStep 1: Computing ILI statistics and per-mouse thresholds...')
    mouse_thresholds, session_lick_stats = compute_thresholds(sessions, data_folder)
    print(f'Computed thresholds for {len(mouse_thresholds)} mice')

    thresh_df = pd.DataFrame([
        {'mouse': m, 'group': mouse_group.get(m, '?'),
         'threshold_s': round(t, 4),
         'threshold_log10': round(np.log10(t), 3)}
        for m, t in sorted(mouse_thresholds.items())
    ])
    thresh_df.to_csv(os.path.join(lick_bouts_dir, 'mouse_thresholds.csv'), index=False)
    session_lick_stats.to_csv(os.path.join(lick_bouts_dir, 'session_lick_stats.csv'), index=False)
    print(f'Saved mouse_thresholds.csv ({len(thresh_df)} mice)')
    print(f'Saved session_lick_stats.csv ({len(session_lick_stats)} sessions)')

    # Step 2: bout detection
    print('\nStep 2: Detecting lick bouts...')
    bouts_df = detect_all_bouts(sessions, data_folder, mouse_thresholds, regenerate=regenerate)
    bouts_df.to_csv(os.path.join(lick_bouts_dir, 'bouts.csv'), index=False)
    print(f'Detected {len(bouts_df):,} bouts across {bouts_df["dir"].nunique()} sessions, '
          f'{bouts_df["mouse"].nunique()} mice')
    print(f'Saved bouts.csv')


if __name__ == '__main__':
    main(regenerate=False)
