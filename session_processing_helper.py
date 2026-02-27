import json
import os

import numpy as np
import pandas as pd
import utils

with open('exp_cohort_info.json', 'r') as f:
    training_info = json.load(f)
cohort_info = training_info['cohorts']

meta_change_date = '2024-04-16'

# =============================================================================
# SESSION LOG GENERATION FUNCTIONS
# =============================================================================

def add_cohort_column(sessions_all, cohort_info):
    """Add cohort column based on mouse name and cohort info."""
    # Create reverse mapping
    mouse_to_cohort = {}
    for cohort, mice in cohort_info.items():
        for mouse in mice:
            mouse_to_cohort[mouse] = cohort
    sessions_all['cohort'] = sessions_all['mouse'].map(mouse_to_cohort)
    return sessions_all

def modify_total_trial(row):
    """Modify total trial count based on ending code."""
    ending_code = row['ending_code']
    if pd.isna(ending_code):
        return row['total_trial']

    ending_code = str(ending_code).lower()
    if ending_code == 'pygame' or ending_code == 'manual':
        return row['total_trial'] - 1
    elif ending_code == 'miss':
        return row['total_trial'] - 5
    else:
        return row['total_trial']

def generate_sessions_all(data_folder):
    """Generate DataFrame from session metadata JSON files."""
    data = []

    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.startswith("meta_") and file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    with open(path) as f:
                        session_data = json.load(f)

                    date_str = file.split('_')[1]
                    if date_str < meta_change_date:
                        data.append(session_data)
                    else:
                        data.append(session_data.get('session_config', session_data))

                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    sessions_all = pd.DataFrame(data)
    sessions_all['dir'] = sessions_all['date'] + '_' + sessions_all['time'] + '_' + sessions_all['mouse']
    sessions_all['total_trial'] = sessions_all.apply(modify_total_trial, axis=1)
    sessions_all = add_cohort_column(sessions_all, cohort_info)
    sessions_all['version'] = sessions_all['date'].apply(
        lambda x: 'pre' if x < meta_change_date else 'post'
    )
    sessions_all = sessions_all.drop(columns=['trainer', 'record', 'forward_file', 'pump_ul_per_turn'])
    return sessions_all.sort_values('dir')

def assign_session_numbers(group):
    """Assign sequential session numbers to sessions grouped by mouse."""
    group.sort_values(by=['mouse', 'dir', 'date'], inplace=True)
    group['session'] = list(range(len(group)))
    return group

def generate_sessions_training(sessions_all):
    """Filter for training sessions and assign session numbers."""
    sessions_training = sessions_all.loc[sessions_all.training == 'regular'].reset_index()
    sessions_training = sessions_training.groupby('mouse', group_keys=False).apply(assign_session_numbers)
    return sessions_training

def generate_session_logs(data_folder, save_logs=True):
    """Generate and optionally save session logs."""
    sessions_all = generate_sessions_all(data_folder)
    sessions_training = generate_sessions_training(sessions_all)
    print(f"{len(sessions_training)} sessions in total")

    if save_logs:
        utils.save_as_csv(df=sessions_all, folder=data_folder, filename='sessions_all.csv')
        utils.save_as_csv(df=sessions_training, folder=data_folder, filename='sessions_training.csv')

    return sessions_all, sessions_training

# =============================================================================
# EVENT PROCESSING FUNCTIONS
# =============================================================================

def process_events(session_info, events):
    """Filter events to valid trial range."""
    events = events.loc[events['session_trial_num'].between(0, session_info['total_trial'])]
    return events

def add_trial_time(trial):
    """Add trial_time column relative to trial start."""
    trial['trial_time'] = trial['session_time'] - trial['session_time'].iloc[0]
    return trial

# =============================================================================
# SESSION ANALYSIS FUNCTIONS
# =============================================================================

def get_session_basics(session_df):
    """Extract basic session statistics (blocks, trials, rewards, time)."""
    num_trials = session_df.session_trial_num.max()
    last_trial = session_df.loc[session_df['session_trial_num'] == num_trials]

    num_blocks = last_trial.loc[(last_trial['key'] == 'trial') & (last_trial['value'] == 1), 'block_num'].iloc[0] + 1
    total_reward = round(session_df.reward_size.sum(), 2)

    # Calculate total time as sum of time per block
    total_time = 0
    for block_num in range(int(num_blocks)):
        block_data = session_df[session_df['block_num'] == block_num]
        if not block_data.empty:
            block_time = block_data.session_time.max() - block_data.session_time.min()
            total_time += block_time

    total_time = round(total_time, 2)

    session_basics = {
        'num_blocks': num_blocks,
        'num_trials': num_trials + 1,
        'rewards': total_reward,
        'session_time': total_time
    }
    return session_basics

def stitch_sessions(session_1, session_2):
    """Stitch two sessions by adjusting timing and trial numbers."""
    session_1_basics = get_session_basics(session_1)
    time_offset = session_1_basics['session_time']
    block_offset = session_1_basics['num_blocks']
    trial_offset = session_1_basics['num_trials']

    session_2.session_time = session_2.session_time + time_offset
    session_2.block_num = session_2.block_num + block_offset
    session_2.session_trial_num = session_2.session_trial_num + trial_offset

    stitched_session = pd.concat([session_1, session_2])
    return stitched_session

# =============================================================================
# TRIAL ANALYSIS FUNCTIONS
# =============================================================================

def get_trial_basics(trial):
    """Extract basic trial info (trial numbers, block info, timing)."""
    trial_start = trial.loc[(trial['key'] == 'trial') & (trial['value'] == 1)].iloc[0]
    trial_end = trial.loc[(trial['key'] == 'trial') & (trial['value'] == 0)].iloc[0]

    trial_basics = {
        'session_trial_num': trial_start['session_trial_num'],
        'block_trial_num': trial_start['block_trial_num'],
        'block_num': trial_start['block_num'],
        'start_time': trial_start['session_time'],
        'end_time': trial_end['session_time']
    }
    return trial_basics

def generate_trials(session_info, events):
    """Generate trial information DataFrame from session events."""
    trial_info_list = []

    num_trials = int(session_info.num_trials)
    for t in range(num_trials):
        trial = events.loc[events['session_trial_num'] == t]
        if not trial.empty:
            trial_basics = get_trial_basics(trial)
            trial_info_list.append(trial_basics)

    trials = pd.DataFrame(trial_info_list)
    return trials

def get_trial_bg_data(trial_events):
    """Extract background period metrics from trial events."""
    # Get background period start/end times
    bg_time = trial_events.loc[(trial_events['key'] == 'background') & (trial_events['value'] == 1), 'session_time'].iloc[0]
    wait_time = trial_events.loc[(trial_events['key'] == 'wait') & (trial_events['value'] == 1), 'session_time'].iloc[0]

    bg_events = trial_events.loc[trial_events.state == 'in_background'].copy()
    bg_drawn = float(bg_events.iloc[0]['time_bg']) if not bg_events.empty else 0
    bg_length = wait_time - bg_time
    bg_repeats = trial_events['key'].value_counts().get('background', 0)

    # Extract background licks
    bg_licks = bg_events.loc[(bg_events['key'] == 'lick') & (bg_events['value'] == 1)]

    # Calculate lick rate (handle division by zero)
    bg_repeat_rate = len(bg_licks) / bg_length if bg_length > 0 else 0

    # Calculate mean phase of licks relative to bg_drawn interval
    if len(bg_licks) > 0 and bg_drawn > 0:
        intervals = np.diff(bg_licks['trial_time'], prepend=0)
        proportions = intervals / bg_drawn
        mean_bg_lick_phase = proportions.mean()
    else:
        mean_bg_lick_phase = np.nan

    return {
        'bg_drawn': bg_drawn,
        'bg_length': bg_length,
        'bg_repeats': bg_repeats,
        'num_bg_licks': len(bg_licks),
        'bg_repeat_rate': bg_repeat_rate,
        'mean_bg_lick_phase': mean_bg_lick_phase
    }

def get_trial_wait_data(trial, bg_data):
    """Extract wait trial performance data."""
    wait_start = trial.loc[(trial['key'] == 'wait') & (trial['value'] == 1), 'trial_time']

    wait_start_time = wait_start.iloc[0]
    consumption_events = trial.loc[trial['key'] == 'consumption']

    if consumption_events.empty:
        return {'miss_trial': True, 'time_waited': 60, 'time_waited_since_cue_on': bg_data['bg_length'] + 60,
                'reward': 0, 'num_consumption_lick': 0, 'num_pump': 0}

    consumption_start = consumption_events.iloc[0]['trial_time']
    consumption_data = trial.loc[trial['state'] == 'in_consumption']

    return {
        'miss_trial': False,
        'time_waited': consumption_start - wait_start_time,
        'time_waited_since_cue_on': consumption_start,
        'reward': consumption_events.iloc[0]['reward_size'],
        'num_consumption_lick': len(consumption_data.loc[(consumption_data['key'] == 'lick') & (consumption_data['value'] == 1)]),
        'num_pump': len(consumption_data.loc[(consumption_data['key'] == 'pump') & (consumption_data['value'] == 1)])
    }

def get_trial_lick_data(trial):
    """Extract up to the first four lick times (in-trial time)."""
    licks = trial.loc[(trial['key'] == 'lick') & (trial['value'] == 1), 'trial_time'].reset_index(drop=True)
    return {
        'first_lick': licks.iloc[0] if len(licks) > 0 else np.nan,
        'last_lick': licks.iloc[-1] if len(licks) > 0 else np.nan
    }

def get_trial_performance(t, trial):
    """Get comprehensive trial performance combining background and wait data."""
    bg_data = get_trial_bg_data(trial)
    wait_data = get_trial_wait_data(trial, bg_data)

    lick_data = get_trial_lick_data(trial)
    trial_data = {'session_trial_num': t} | bg_data | wait_data | lick_data

    if (bg_data['num_bg_licks'] == 0) & (wait_data['miss_trial'] == False):
        trial_data['good_trial'] = True
    else:
        trial_data['good_trial'] = False

    return trial_data

def get_trial_data_df(session_by_trial):
    """Generate performance DataFrame for all trials in a session."""
    trial_data_list = []

    for t, trial in session_by_trial:
        # Ensure per-trial relative time exists for lick timing
        trial_with_time = add_trial_time(trial)
        trial_data = get_trial_performance(t, trial_with_time)
        trial_data_list.append(trial_data)

    trials_data = pd.DataFrame(trial_data_list)
    return trials_data

def _compute_time_since_anchor(trials, events, anchor_times_session, anchor_times_by_block, col_name):
    """
    For each trial, compute time from the most recent anchor event to the decision lick
    (hit trials) or trial end (miss trials).

    anchor_times_session  : sorted np.array of all anchor event times in the session
    anchor_times_by_block : dict {block_num: sorted np.array of anchor times}
    col_name              : output column name
    """
    time_map = {}
    decision_licks = events.loc[
        (events['key'] == 'lick') & (events['value'] == 1) & (events['state'] == 'in_wait')
    ]
    first_decision_licks = decision_licks.groupby('session_trial_num')['session_time'].min()

    # Hit trials: O(log n) searchsorted instead of O(n) boolean scan
    for trial_num, dt in first_decision_licks.items():
        pos = np.searchsorted(anchor_times_session, dt, side='left')
        time_map[trial_num] = float(dt - anchor_times_session[pos - 1]) if pos > 0 else np.nan

    # Miss trials: O(1) trial lookup via pre-indexed DataFrame
    trial_lookup = trials.set_index('session_trial_num')
    for t in np.setdiff1d(trials.session_trial_num.unique(), first_decision_licks.index.values):
        row = trial_lookup.loc[t]
        block_times = anchor_times_by_block.get(int(row['block_num']), np.array([]))
        pos = np.searchsorted(block_times, row['end_time'], side='left')
        time_map[int(t)] = float(row['end_time'] - block_times[pos - 1]) if pos > 0 else np.nan

    trials[col_name] = trials['session_trial_num'].map(time_map)
    return trials

def get_time_since_last_lick(trials, events):
    """Calculate time since the previous lick for each trial."""
    licks = events.loc[(events['key'] == 'lick') & (events['value'] == 1)]
    all_times = np.sort(licks['session_time'].values)
    by_block = {b: np.sort(g['session_time'].values) for b, g in licks.groupby('block_num')}
    return _compute_time_since_anchor(trials, events, all_times, by_block, 'time_waited_since_last_lick')

def get_time_since_last_lick_bout(trials, events, session_bouts):
    """Calculate time since the end of the previous lick bout for each trial."""
    if session_bouts is None or session_bouts.empty:
        trials['time_waited_since_last_lick_bout'] = np.nan
        return trials
    all_times = np.sort(session_bouts['bout_offset'].values)
    by_block = {b: np.sort(g['bout_offset'].values) for b, g in session_bouts.groupby('block_num')}
    return _compute_time_since_anchor(trials, events, all_times, by_block, 'time_waited_since_last_lick_bout')

def get_time_since_last_reward(trials, events):
    """Calculate time since the last reward for each trial."""
    rewards = events.loc[
        (events['key'] == 'consumption') & (events['reward_size'].notna()) & (events['reward_size'] > 0)
    ]
    all_times = np.sort(rewards['session_time'].values)
    by_block = {b: np.sort(g['session_time'].values) for b, g in rewards.groupby('block_num')}
    return _compute_time_since_anchor(trials, events, all_times, by_block, 'time_waited_since_last_reward')

def get_previous_trial_performance(trials, rolling_windows=[5, 10]):
    """Add lagged trial features and rolling averages."""
    trials = trials.copy()

    # Lagged features
    lagged_columns = ['bg_drawn', 'bg_length', 'bg_repeats', 'num_bg_licks', 'first_lick', 'last_lick',
                      'time_waited', 'time_waited_since_cue_on', 'time_waited_since_last_lick',
                      'time_waited_since_last_lick_bout', 'time_waited_since_last_reward',
                      'reward', 'num_consumption_lick', 'miss_trial']
    for col in lagged_columns:
        if col in trials.columns:
            fill_value = False if col == 'miss_trial' else 0
            trials[f'previous_trial_{col}'] = trials[col].shift(1).fillna(fill_value)

    # Rolling averages
    rolling_metrics = ['bg_length', 'bg_repeats', 'num_bg_licks', 'first_lick', 'last_lick',
                       'time_waited', 'time_waited_since_cue_on', 'time_waited_since_last_lick',
                       'time_waited_since_last_lick_bout', 'time_waited_since_last_reward',
                       'reward', 'num_consumption_lick']
    for metric in rolling_metrics:
        if metric in trials.columns:
            for window in rolling_windows:
                trials[f'{metric}_rolling_mean_{window}'] = (
                    trials[metric].rolling(window=window, min_periods=1).mean().shift(1, fill_value=0)
                )
    return trials

def get_trial_progress(trials):
    """Add trial progress features."""
    trials = trials.copy()

    trials['trial_fraction_in_session'] = (trials['session_trial_num'] + 1) / len(trials)
    trials['trial_fraction_in_block'] = (
        (trials['block_trial_num'] + 1) /
        trials['block_num'].map(trials['block_num'].value_counts())
    )
    trials['block_fraction_in_session'] = (trials['block_num'] + 1) / trials['block_num'].nunique()

    return trials

def get_rewarded_streak(trials):
    """Add rewarded and unrewarded streak features to trials dataframe."""
    trials = trials.copy()
    rewarded = trials['reward'].fillna(0) > 0

    # Rewarded streak (consecutive rewarded trials)
    rewarded_streak = rewarded.groupby((~rewarded).cumsum()).cumsum()
    trials['rewarded_streak'] = rewarded_streak.shift(1, fill_value=0)

    # Unrewarded streak (consecutive unrewarded trials)
    unrewarded_streak = (~rewarded).groupby(rewarded.cumsum()).cumsum()
    trials['unrewarded_streak'] = unrewarded_streak.shift(1, fill_value=0)

    return trials

def get_block_reward_metrics(trials, events):
    """
    Add reward metrics within blocks to trials dataframe:
    - Reward rate since block start
    - Reward rates for past 1, 5, 10 minutes within block
    - Time since last reward in block
    - Cumulative reward in block
    """
    trials = trials.copy()
    reward_events = events[
        (events['key'] == 'consumption') &
        (events['reward_size'].notna()) &
        (events['reward_size'] > 0)
    ]
    time_windows = [60, 300, 600]  # seconds

    # Initialize columns
    trials['reward_rate_since_block_start'] = 0.0
    trials['time_since_last_reward_in_block'] = np.nan
    trials['cumulative_reward_in_block'] = 0.0
    for w in time_windows:
        trials[f'reward_rate_past_{w//60}min_in_block'] = 0.0

    for block_num, block_trials in trials.groupby('block_num'):
        block_start = block_trials['start_time'].min()
        block_rewards = reward_events[
            (reward_events['block_num'] == block_num) &
            (reward_events['session_time'] >= block_start)
        ].sort_values('session_time')
        r_times = block_rewards['session_time'].values
        r_sizes = block_rewards['reward_size'].values
        idx = block_trials.index
        t_arr = block_trials['start_time'].values

        # Prefix sums for O(log n) window queries via searchsorted
        cumsum = np.concatenate([[0], np.cumsum(r_sizes)])

        # Number of rewards strictly before each trial start
        right_pos = np.searchsorted(r_times, t_arr, side='left')

        # Cumulative reward in block
        trials.loc[idx, 'cumulative_reward_in_block'] = cumsum[right_pos]

        # Reward rate since block start
        elapsed = t_arr - block_start
        safe_elapsed = np.where(elapsed > 0, elapsed, 1.0)
        trials.loc[idx, 'reward_rate_since_block_start'] = np.where(
            elapsed > 0, cumsum[right_pos] / safe_elapsed, 0.0
        )

        # Time since last reward in block
        has_prior = right_pos > 0
        last_r_time = np.where(has_prior, r_times[np.clip(right_pos - 1, 0, len(r_times) - 1)], np.nan)
        trials.loc[idx, 'time_since_last_reward_in_block'] = np.where(has_prior, t_arr - last_r_time, np.nan)

        # Windowed reward rates
        for w in time_windows:
            w_start = np.maximum(t_arr - w, block_start)
            left_pos = np.searchsorted(r_times, w_start, side='left')
            w_rew = cumsum[right_pos] - cumsum[left_pos]
            actual_w = t_arr - w_start
            safe_actual_w = np.where(actual_w > 0, actual_w, 1.0)
            trials.loc[idx, f'reward_rate_past_{w//60}min_in_block'] = np.where(
                actual_w > 0, w_rew / safe_actual_w, 0.0
            )

    trials['cumulative_reward'] = trials['reward'].fillna(0).cumsum().shift(fill_value=0)
    return trials

def get_trial_features(trials_analyzed, events, session_bouts=None):
    trials_with_time_since_last_lick = get_time_since_last_lick(trials_analyzed, events)
    trials_with_time_since_last_lick_bout = get_time_since_last_lick_bout(trials_with_time_since_last_lick, events, session_bouts)
    trials_with_time_since_last_reward = get_time_since_last_reward(trials_with_time_since_last_lick_bout, events)
    trials_with_performance = get_previous_trial_performance(trials_with_time_since_last_reward)
    trials_with_progress = get_trial_progress(trials_with_performance)
    trials_with_streak = get_rewarded_streak(trials_with_progress)
    trials_with_block_reward_metrics = get_block_reward_metrics(trials_with_streak, events)
    return trials_with_block_reward_metrics
