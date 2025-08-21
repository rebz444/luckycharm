import json
import os
import math
import pandas as pd
import utils

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

# =============================================================================
# SESSION LOG GENERATION FUNCTIONS
# =============================================================================

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

def modify_sessions_all(sessions_all):
    """Add dir column, extract exp/group, and modify total_trial."""
    # Create dir column
    sessions_all['dir'] = sessions_all['date'] + '_' + sessions_all['time'] + '_' + sessions_all['mouse']
    
    # Extract exp and group with error handling
    exp_group = sessions_all['exp'].str.extract(r'exp(\d)_(short|long)')
    sessions_all[['exp', 'group']] = exp_group
    
    # Map group values safely
    sessions_all['group'] = sessions_all['group'].map({'short': 's', 'long': 'l'}).fillna('unknown')
    
    # Modify total trial
    sessions_all['total_trial'] = sessions_all.apply(modify_total_trial, axis=1)
    
    # Drop columns safely
    if 'forward_file' in sessions_all.columns:
        sessions_all = sessions_all.drop(['forward_file'], axis=1)
    
    return sessions_all.sort_values('dir')

def generate_sessions_all(data_folder):
    """Generate DataFrame from session metadata JSON files."""
    data = []
    
    for root, _, files in os.walk(data_folder):
        for file in files:
            if not file.startswith("meta_") or not file.endswith(".json"):
                continue
                
            path = os.path.join(root, file)
            try:
                with open(path) as f:
                    session_data = json.load(f)
                
                # Simplified date logic
                date_str = file.split('_')[1]
                if date_str < '2024-04-16':
                    data.append(session_data)
                else:
                    data.append(session_data.get('session_config', session_data))
                    
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    sessions_all = pd.DataFrame(data)
    return modify_sessions_all(sessions_all)

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
    total_time = round((session_df.session_time.max() - session_df.session_time.min()), 2)
    
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
    
    # if wait_start.empty:
    #     return {'miss_trial': True, 'time_waited': 0, 'reward': 0, 'num_consumption_lick': 0, 'num_pump': 0}
    
    wait_start_time = wait_start.iloc[0]
    consumption_events = trial.loc[trial['key'] == 'consumption']
    
    if consumption_events.empty:
        return {'miss_trial': True, 'time_waited': 0, 'reward': 0, 'num_consumption_lick': 0, 'num_pump': 0}
    
    consumption_start = consumption_events.iloc[0]['session_time']
    consumption_data = trial.loc[trial['state'] == 'in_consumption']
    
    return {
        'miss_trial': False,
        'time_waited': consumption_start - wait_start_time,
        'reward': consumption_events.iloc[0]['reward_size'],
        'num_consumption_lick': len(consumption_data.loc[(consumption_data['key'] == 'lick') & (consumption_data['value'] == 1)]),
        'num_pump': len(consumption_data.loc[(consumption_data['key'] == 'pump') & (consumption_data['value'] == 1)])
    }

def get_trial_performance(t, trial):
    """Get comprehensive trial performance combining background and wait data."""
    bg_data = get_trial_bg_data(trial)
    wait_data = get_trial_wait_data(trial)
    trial_data = {'session_trial_num': t} | bg_data | wait_data
    
    if (bg_data['num_bg_licks'] == 0) & (wait_data['miss_trial'] == False):
        trial_data['good_trial'] = True
    else:
        trial_data['good_trial'] = False  
    
    return trial_data

def get_trial_data_df(session_by_trial):
    """Generate performance DataFrame for all trials in a session."""
    trial_data_list = []
    
    for t, trial in session_by_trial:
        trial_data = get_trial_performance(t, trial)
        trial_data_list.append(trial_data)
    
    trials_data = pd.DataFrame(trial_data_list)
    return trials_data