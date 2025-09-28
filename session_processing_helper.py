import json
import os
import pandas as pd
import utils
import shutil
import numpy as np

with open('exp_cohort_info.json', 'r') as f:
    training_info = json.load(f)
cohort_info = training_info['cohorts']

meta_change_date = '2024-04-16'

# =============================================================================
# QUALITY CONTROL FUNCTIONS
# =============================================================================

def check_session_files(data_folder):
    """Check session folders for required files and their status."""
    files_check = []
    
    for entry in os.scandir(data_folder):
        if not entry.is_dir():
            continue
            
        session_path = os.path.join(data_folder, entry.name)
        files = [f for f in os.scandir(session_path) if f.is_file() and not f.name.startswith('.')]
        
        events_files = [f for f in files if f.name.startswith("events_")]
        meta_files = [f for f in files if f.name.startswith("meta_")]
        
        files_check.append({
            'dir': entry.name,
            'events': bool(events_files),
            'meta': bool(meta_files),
            'events_empty': not events_files or all(f.stat().st_size == 0 for f in events_files),
            'meta_empty': not meta_files or all(f.stat().st_size == 0 for f in meta_files)
        })

    files_check_df = pd.DataFrame(files_check).sort_values("dir")
    
    return (
        files_check_df[files_check_df.meta == False],
        files_check_df[files_check_df.events == False],
        files_check_df[(files_check_df.meta == True) & (files_check_df.meta_empty == True)],
        files_check_df[(files_check_df.events == True) & (files_check_df.events_empty == True)]
    )

def backup_directory(source_path):
    """Create backup or update existing with new sessions."""
    backup_path = source_path + "_backup"
    
    if not os.path.exists(backup_path):
        try:
            shutil.copytree(source_path, backup_path)
            print(f"Backup created at {backup_path}")
            return backup_path
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None
    else:
        # Update existing backup with new sessions
        print(f"Backup exists at {backup_path}, checking for new sessions...")
        new_sessions = []
        
        for item in os.listdir(source_path):
            source_item = os.path.join(source_path, item)
            backup_item = os.path.join(backup_path, item)
            
            # Skip if this is the backup directory itself
            if item.endswith('_backup'):
                continue
                
            if os.path.isdir(source_item) and not os.path.exists(backup_item):
                try:
                    shutil.copytree(source_item, backup_item)
                    new_sessions.append(item)
                    print(f"Added to backup: {item}")
                except Exception as e:
                    print(f"Error copying {item} to backup: {e}")
        
        if new_sessions:
            print(f"Updated backup with {len(new_sessions)} new sessions")
        else:
            print("Backup is up-to-date")
        
        return backup_path

def identify_test_folders(data_dir):
    """Identify all folders ending with '_test'."""
    return [os.path.join(data_dir, item) for item in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, item)) and item.endswith('_test')]

def identify_problematic_sessions(data_folder):
    """Identify sessions with missing or empty files."""
    missing_meta, missing_events, empty_meta, empty_events = check_session_files(data_folder)
    
    problematic_sessions = pd.concat([
        missing_meta[['dir']].assign(reason='Missing meta file'),
        missing_events[['dir']].assign(reason='Missing events file'),
        empty_meta[['dir']].assign(reason='Empty meta file'),
        empty_events[['dir']].assign(reason='Empty events file')
    ]).reset_index(drop=True)
    
    return problematic_sessions

def identify_short_or_crashed_sessions(data_folder, short_threshold=20):
    """Identify short or crashed sessions by reading events files."""
    short_sessions, crashed_sessions = [], []
    
    for session_dir in [entry.name for entry in os.scandir(data_folder) if entry.is_dir()]:
        try:
            events_files = [f for f in os.listdir(os.path.join(data_folder, session_dir)) 
                           if f.startswith('events_') and f.endswith('.txt')]
            
            if not events_files:
                short_sessions.append({'dir': session_dir, 'total_trials': 'No events file', 'reason': 'Missing events file'})
                continue
                
            events = pd.read_csv(os.path.join(data_folder, session_dir, events_files[0]), low_memory=False)
            
            # Check for short sessions
            max_trial_num = events['session_trial_num'].max()
            total_trials = max_trial_num + 1 if pd.notna(max_trial_num) else 0
            if pd.isna(total_trials) or total_trials < short_threshold:
                short_sessions.append({'dir': session_dir, 'total_trials': total_trials, 'reason': 'Short'})
            
            # Check for crashed sessions
            session_end = events.loc[(events['key'] == 'session') & (events['value'] == 0)]
            if len(session_end) != 1:
                crashed_sessions.append({'dir': session_dir, 'reason': 'Crashed'})
                
        except Exception as e:
            short_sessions.append({'dir': session_dir, 'total_trials': 'Error', 'reason': f'Cannot read file: {str(e)}'})
    
    all_problematic = short_sessions + crashed_sessions
    if not all_problematic:
        return pd.DataFrame()
    
    # Remove duplicates and combine reasons
    unique_sessions = {}
    for session in all_problematic:
        if session['dir'] not in unique_sessions:
            unique_sessions[session['dir']] = session
        else:
            unique_sessions[session['dir']]['reason'] = f"{unique_sessions[session['dir']]['reason']}; {session['reason']}"
    
    return pd.DataFrame(list(unique_sessions.values()))

def validate_session_directory_names(data_folder):
    """Check if meta JSON files match their directory names."""
    mismatched_sessions = []
    
    for root, _, files in os.walk(data_folder):
        for file in files: 
            if file.startswith("meta_") and file.endswith(".json"):
                try:
                    with open(os.path.join(root, file)) as f:
                        session_data = json.load(f)
                    
                    # Get config data based on date
                    date_str = file.split('_')[1]
                    config_data = session_data.get('session_config', session_data) if date_str >= '2024-04-16' else session_data
                    
                    # Check if metadata matches directory name
                    meta_dir = f"{config_data['date']}_{config_data['time']}_{config_data['mouse']}"
                    if meta_dir != os.path.basename(root):
                        mismatched_sessions.append({
                            'actual_dir': os.path.basename(root),
                            'meta_dir': meta_dir,
                            **config_data
                        })
                        
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    return pd.DataFrame(mismatched_sessions).sort_values('meta_dir') if mismatched_sessions else pd.DataFrame()

def delete_sessions(sessions_df, data_folder, session_type="sessions", auto_delete=False):
    """Delete sessions from the filesystem and return deletion record."""
    if sessions_df.empty:
        print(f"No {session_type} found to delete")
        return pd.DataFrame()
    
    print(f"Found {len(sessions_df)} {session_type} to delete:")
    print(sessions_df.to_string(index=False))
    
    if auto_delete or input(f"\nProceed with deletion of {session_type}? (y/N): ").lower() == 'y':
        deletion_record = []
        for _, row in sessions_df.iterrows():
            session_dir = os.path.join(data_folder, row['dir'])
            if os.path.exists(session_dir):
                try:
                    shutil.rmtree(session_dir)
                    deletion_record.append({
                        'session': row['dir'], 
                        'reason': row.get('reason', f'Deleted {session_type}'),
                        'deleted': True, 
                        'timestamp': pd.Timestamp.now()
                    })
                    print(f"Deleted: {row['dir']}")
                except Exception as e:
                    print(f"Error deleting {row['dir']}: {e}")
        
        print(f"Total {session_type} deleted: {len(deletion_record)}")
        return pd.DataFrame(deletion_record)
    else:
        print("Deletion cancelled")
        return pd.DataFrame()

def sort_sessions_by_experiments(data_dir, exp_info):
    """Sort sessions into experiment folders based on mouse names."""
    # Get the parent directory of raw folder (behavior_data)
    parent_dir = os.path.dirname(data_dir)
    exp_folders = {exp_name: os.path.join(parent_dir, exp_name) for exp_name in exp_info.keys()}
    
    for exp_path in exp_folders.values():
        os.makedirs(exp_path, exist_ok=True)
        print(f"Created experiment folder: {exp_path}")
    
    moved_count = 0
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if not os.path.isdir(item_path) or item in exp_info.keys():
            continue
        
        if len(item.split('_')) == 3:
            _, _, mouse_name = item.split('_')
            for exp_name, mice in exp_info.items():
                if mouse_name in mice:
                    try:
                        shutil.move(item_path, os.path.join(exp_folders[exp_name], item))
                        moved_count += 1
                        print(f"Moved {item} to {exp_name}")
                        break
                    except Exception as e:
                        print(f"Error moving {item}: {e}")
            else:
                print(f"No matching experiment found for {item}")
    
    print(f"Total sessions moved: {moved_count}")
    return moved_count

def update_deletion_record(data_dir, deletion_dfs):
    """Update deletion record CSV file, appending to existing or creating new."""
    deletion_csv_path = os.path.join(data_dir, 'deletion_record.csv')
    all_deletions = [df for df in deletion_dfs if not df.empty]
    
    if all_deletions:
        combined_df = pd.concat(all_deletions, ignore_index=True)
        if os.path.exists(deletion_csv_path):
            existing_df = pd.read_csv(deletion_csv_path)
            pd.concat([existing_df, combined_df], ignore_index=True).to_csv(deletion_csv_path, index=False)
            print(f"Appended to existing: {deletion_csv_path}")
        else:
            combined_df.to_csv(deletion_csv_path, index=False)
            print(f"Created new: {deletion_csv_path}")
    else:
        print("No deletions to record")

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

def get_trial_bg_data(trial):
    """Extract background trial metrics."""
    bg_time = trial.loc[(trial['key'] == 'background') & (trial['value'] == 1), 'session_time'].iloc[0]
    wait_time = trial.loc[(trial['key'] == 'wait') & (trial['value'] == 1), 'session_time'].iloc[0]
    
    bg_events = trial.loc[trial.state == 'in_background']
    
    return {
        'bg_drawn': float(bg_events.iloc[0]['time_bg']) if not bg_events.empty else 0,
        'bg_length': wait_time - bg_time,
        'bg_repeats': trial['key'].value_counts().get('background', 0),
        'num_bg_licks': len(bg_events.loc[(bg_events['key'] == 'lick') & (bg_events['value'] == 1)])
    }

def get_trial_wait_data(trial):
    """Extract wait trial performance data."""
    wait_start = trial.loc[(trial['key'] == 'wait') & (trial['value'] == 1), 'session_time']
    
    wait_start_time = wait_start.iloc[0]
    consumption_events = trial.loc[trial['key'] == 'consumption']
    
    if consumption_events.empty:
        return {'miss_trial': True, 'time_waited': 60, 'reward': 0, 'num_consumption_lick': 0, 'num_pump': 0}
    
    consumption_start = consumption_events.iloc[0]['session_time']
    consumption_data = trial.loc[trial['state'] == 'in_consumption']
    
    return {
        'miss_trial': False,
        'time_waited': consumption_start - wait_start_time,
        'reward': consumption_events.iloc[0]['reward_size'],
        'num_consumption_lick': len(consumption_data.loc[(consumption_data['key'] == 'lick') & (consumption_data['value'] == 1)]),
        'num_pump': len(consumption_data.loc[(consumption_data['key'] == 'pump') & (consumption_data['value'] == 1)])
    }

def get_trial_lick_data(trial):
    """Extract up to the first four lick times (in-trial time)."""
    licks = trial.loc[trial['key'] == 'lick', 'trial_time'].reset_index(drop=True)
    # Use np.nan for missing lick slots to keep numeric dtype when possible
    return {
        'first_lick': licks.iloc[0] if len(licks) > 0 else np.nan,
        'second_lick': licks.iloc[1] if len(licks) > 1 else np.nan,
        'third_lick': licks.iloc[2] if len(licks) > 2 else np.nan,
        'fourth_lick': licks.iloc[3] if len(licks) > 3 else np.nan,
    }

def get_trial_performance(t, trial):
    """Get comprehensive trial performance combining background and wait data."""
    bg_data = get_trial_bg_data(trial)
    wait_data = get_trial_wait_data(trial)
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

def get_previous_trial_performance(trials, rolling_windows=[5, 10]):
    """Add lagged trial features and rolling averages."""
    trials = trials.copy()
    
    # Lagged features
    lagged_columns = ['bg_repeats', 'time_waited', 'reward', 'miss_trial']
    for col in lagged_columns:
        if col in trials.columns:
            trials[f'previous_trial_{col}'] = trials[col].shift(1).fillna(0)
            
    # Rolling averages
    rolling_metrics = ['bg_repeats', 'time_waited']
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

        for idx, row in block_trials.iterrows():
            t = row['start_time']
            mask = r_times < t
            cum_rew = r_sizes[mask].sum()
            trials.at[idx, 'cumulative_reward_in_block'] = cum_rew

            elapsed = t - block_start
            if elapsed > 0:
                trials.at[idx, 'reward_rate_since_block_start'] = cum_rew / elapsed

            if mask.any():
                trials.at[idx, 'time_since_last_reward_in_block'] = t - r_times[mask].max()

            for w in time_windows:
                w_start = max(t - w, block_start)
                mask_w = (r_times >= w_start) & (r_times < t)
                w_rew = r_sizes[mask_w].sum()
                actual_w = t - w_start
                if actual_w > 0:
                    trials.at[idx, f'reward_rate_past_{w//60}min_in_block'] = w_rew / actual_w
    
    trials['cumulative_reward'] = trials['reward'].fillna(0).cumsum().shift(fill_value=0)
    return trials

def get_trial_features(trials_analyzed, events):
    trials_with_performance = get_previous_trial_performance(trials_analyzed)
    trials_with_progress = get_trial_progress(trials_with_performance)
    trials_with_streak = get_rewarded_streak(trials_with_progress)
    trials_with_block_reward_metrics = get_block_reward_metrics(trials_with_streak, events)
    return trials_with_block_reward_metrics