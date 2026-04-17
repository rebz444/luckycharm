import os

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde

import utils

# =============================================================================
# Configuration
# =============================================================================

data_dir = '/Users/rebekahzhang/data/behavior_data'
exp = "exp2"
data_folder = os.path.join(data_dir, exp)
figure_folder = os.path.join(data_dir, 'performance_plots')
os.makedirs(figure_folder, exist_ok=True)

GROUP_LABELS = {'s': 'Short', 'l': 'Long'}
PALETTE = {'s': '#ffb400', 'l': '#9080ff'}
MARKERS = {'s': 'o', 'l': 's'}

MIN_SESSIONS = 20
SESSION_CAP = 70

# =============================================================================
# Load data
# =============================================================================

sessions_training = utils.load_session_log(data_folder, f'sessions_training_{exp}.csv')
sessions_training['period'] = sessions_training['session'] // 10

trials_training = utils.load_data(os.path.join(data_folder, 'trials_training_analyzed.csv'))
trials_training['period'] = trials_training['session'] // 10

# =============================================================================
# Filter: minimum training duration
# =============================================================================

print("\n=== Untrained mice filter ===")
training_duration = sessions_training.groupby('mouse')['session'].max()
valid_mice = training_duration[training_duration >= MIN_SESSIONS].index.tolist()

excluded = set(sessions_training['mouse'].unique()) - set(valid_mice)
print(f"Excluded {len(excluded)} mice with < {MIN_SESSIONS} sessions: {sorted(excluded)}")

sessions_training = sessions_training[sessions_training['mouse'].isin(valid_mice)].reset_index(drop=True)
trials_training = trials_training[trials_training['mouse'].isin(valid_mice)].reset_index(drop=True)

print(f"{sessions_training['mouse'].nunique()} mice, {len(sessions_training)} sessions, {len(trials_training)} trials remaining")

sessions_training = sessions_training[sessions_training['session'] <= SESSION_CAP].reset_index(drop=True)
trials_training = trials_training[trials_training['session'] <= SESSION_CAP].reset_index(drop=True)
print(f"After session cap (<= {SESSION_CAP}): {len(sessions_training)} sessions, {len(trials_training)} trials")

# =============================================================================
# Group summary
# =============================================================================

print("\n=== Group Summary ===")
for group in sorted(sessions_training['group'].unique()):
    s = sessions_training[sessions_training['group'] == group]
    t = trials_training[trials_training['group'] == group]
    mice = s['mouse'].nunique()
    sessions = len(s)
    trials = len(t)
    mean_tw = t.groupby('dir')['time_waited'].mean().mean()
    glabel = GROUP_LABELS.get(group, group)
    print(f"  {glabel} group ({group!r}): {mice} mice, {sessions} sessions, {trials} trials, mean avg_tw={mean_tw:.2f}s")

# =============================================================================
# Filter: outlier mice (group-wise IQR on mean time_waited)
# =============================================================================

# Per-mouse mean time_waited: average per dir first, then across dirs per mouse
mouse_tw = (
    trials_training
    .groupby(['mouse', 'group', 'dir'])['time_waited'].mean()
    .reset_index()
    .groupby(['mouse', 'group'])['time_waited'].mean()
    .reset_index()
    .rename(columns={'time_waited': 'mean_tw'})
)

outlier_mice = []
print("\n=== Outlier Detection (group-wise IQR, 1.5x) ===")
for group, grp in mouse_tw.groupby('group'):
    q1, q3 = grp['mean_tw'].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (grp['mean_tw'] < lower) | (grp['mean_tw'] > upper)
    for _, row in grp[mask].iterrows():
        print(f"  Outlier: {row['mouse']} ({GROUP_LABELS.get(group, group)}) mean_tw={row['mean_tw']:.2f}s")
        outlier_mice.append(row['mouse'])

print(f"\nExcluding {len(outlier_mice)} outlier mice: {sorted(outlier_mice)}")

sessions_training = sessions_training[~sessions_training['mouse'].isin(outlier_mice)].reset_index(drop=True)
trials_training = trials_training[~trials_training['mouse'].isin(outlier_mice)].reset_index(drop=True)

print("\n=== Group Summary (after outlier removal) ===")
_session_tw, _trial_tw = {}, {}
for group in sorted(sessions_training['group'].unique()):
    s = sessions_training[sessions_training['group'] == group]
    t = trials_training[trials_training['group'] == group]
    glabel = GROUP_LABELS.get(group, group)
    session_tw = t.groupby('dir')['time_waited'].mean()
    trial_tw = t['time_waited']
    _session_tw[group] = session_tw
    _trial_tw[group] = trial_tw
    print(f"  {glabel} ({group!r}):")
    print(f"    sessions: n={len(s)},  tw={session_tw.mean():.2f} ± {session_tw.std():.2f}s")
    print(f"    trials:   n={len(t)},  tw={trial_tw.mean():.2f} ± {trial_tw.std():.2f}s")

if set(_session_tw) == {'s', 'l'}:
    _, p_sess = stats.ttest_ind(_session_tw['s'], _session_tw['l'])
    _, p_trial = stats.ttest_ind(_trial_tw['s'], _trial_tw['l'])
    print(f"  t-test session tw: p={p_sess:.3g}")
    print(f"  t-test trial tw:   p={p_trial:.3g}")

# =============================================================================
# Save cleaned dataframes
# =============================================================================

sessions_training.to_csv(os.path.join(data_folder, 'sessions_cleaned.csv'), index=False)
trials_training.to_csv(os.path.join(data_folder, 'trials_cleaned.csv'), index=False)
print(f"\nSaved sessions_cleaned.csv ({len(sessions_training)} sessions)")
print(f"Saved trials_cleaned.csv ({len(trials_training)} trials)")

# =============================================================================
# Plot: time waited per period (per-trial and session-average)
# =============================================================================

def plot_tw_per_period(df, ylabel, title, filename, right_plot='box'):
    """Line plot (mean ± SEM per period) + box or violin plot, with n labels and t-test p-value."""
    agg = (
        df.groupby(['group', 'period'])['mean_tw']
        .agg(mean='mean', sem=lambda x: x.sem())
        .reset_index()
    )

    _, (ax_line, ax_dist) = plt.subplots(1, 2, figsize=(14, 5),
                                            gridspec_kw={'width_ratios': [3, 1]})

    # --- line plot ---
    for group in sorted(agg['group'].unique()):
        d = agg[agg['group'] == group]
        ax_line.plot(d['period'], d['mean'], marker=MARKERS[group],
                     color=PALETTE[group], label=GROUP_LABELS.get(group, group))
        ax_line.fill_between(d['period'], d['mean'] - d['sem'], d['mean'] + d['sem'],
                              color=PALETTE[group], alpha=0.25)

    ax_line.set_xlabel('Period (10-session bins)')
    ax_line.set_ylabel(ylabel)
    ax_line.set_title(title)
    ax_line.legend()

    # --- box or violin plot ---
    groups = sorted(df['group'].unique())
    group_data = [df[df['group'] == g]['mean_tw'].dropna().values for g in groups]
    xlabels = [GROUP_LABELS.get(g, g) for g in groups]

    if right_plot == 'violin':
        parts = ax_dist.violinplot(group_data, positions=range(1, len(groups) + 1),
                                    showmedians=True)
        for pc, group in zip(parts['bodies'], groups):
            pc.set_facecolor(PALETTE[group])
            pc.set_alpha(0.7)
    else:
        bp = ax_dist.boxplot(group_data, patch_artist=True, widths=0.5)
        for patch, group in zip(bp['boxes'], groups):
            patch.set_facecolor(PALETTE[group])
            patch.set_alpha(0.7)

    ax_dist.set_xticks(range(1, len(groups) + 1))
    ax_dist.set_xticklabels(xlabels)
    ax_dist.set_ylabel(ylabel)
    # n labels and t-test
    if len(group_data) == 2 and len(group_data[0]) > 1 and len(group_data[1]) > 1:
        _, p_val = stats.ttest_ind(group_data[0], group_data[1])
        p_str = f"p={p_val:.3g}" if p_val >= 0.001 else "p<0.001"
        y_max = max(d.max() for d in group_data)
        ax_dist.plot([1, 2], [y_max * 1.05, y_max * 1.05], 'k-', linewidth=1)
        ax_dist.text(1.5, y_max * 1.06, p_str, ha='center', va='bottom', fontsize=9)

    for i, (group, gdata) in enumerate(zip(groups, group_data), 1):
        ax_dist.text(i, ax_dist.get_ylim()[0], f"n={len(gdata)}",
                     ha='center', va='bottom', fontsize=8, color=PALETTE[group])

    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, filename), dpi=300)
    plt.close()
    print(f"Saved {filename}")


# Plot 1: per-trial — one row per trial, SEM across trials within period
trials_tw = (
    trials_training[['group', 'period', 'time_waited']]
    .rename(columns={'time_waited': 'mean_tw'})
)
plot_tw_per_period(
    trials_tw,
    ylabel='Mean Time Waited (s)',
    title='Time Waited per Period (per-trial, mean ± SEM)',
    filename='tw_per_period_trials.png',
    right_plot='violin',
)

# Plot 2: session-average — one row per dir, SEM across sessions within period
sessions_tw = (
    trials_training
    .groupby(['group', 'period', 'dir'])['time_waited'].mean()
    .reset_index()
    .rename(columns={'time_waited': 'mean_tw'})
)
plot_tw_per_period(
    sessions_tw,
    ylabel='Mean Time Waited (s)',
    title='Time Waited per Period (session averages, mean ± SEM)',
    filename='tw_per_period_sessions.png',
)
