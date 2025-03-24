import json
import os

import utils_c5v2 as utils

import pandas as pd

# Quality control
def check_session_files(data_folder):
    files_check = []
    for entry in os.scandir(data_folder):
        if entry.is_dir():
            dir = entry.name
            session_path = os.path.join(data_folder, dir)
            events_found = False
            meta_found = False
            events_empty = True
            meta_empty = True
            
            required_files = [f for f in os.scandir(session_path) if f.is_file() and not f.name.startswith('.')]
            
            for file in required_files:
                if file.name.startswith("events_"):
                    events_found = True
                    if file.stat().st_size > 0:
                        events_empty = False
                elif file.name.startswith("meta_"):
                    meta_found = True
                    if file.stat().st_size > 0:
                        meta_empty = False
            
            files_check.append({
                'dir': dir,
                'events': events_found,
                'meta': meta_found,
                'events_empty': events_empty if events_found else None,
                'meta_empty': meta_empty if meta_found else None
            })

    files_check_df = pd.DataFrame(files_check).sort_values("dir")
    missing_meta = files_check_df[files_check_df.meta==False]
    missing_events = files_check_df[files_check_df.events==False]
    empty_meta = files_check_df[(files_check_df.meta==True) & (files_check_df.meta_empty==True)]
    empty_events = files_check_df[(files_check_df.events==True) & (files_check_df.events_empty==True)]
    
    return missing_meta, missing_events, empty_meta, empty_events

# generate session logs
def modify_total_trial(row):
    ending_code = row['ending_code']
    if pd.isna(ending_code):
        return row['total_trial']
    
    ending_code = str(ending_code).lower() # convert to string before lowercasing
    if ending_code == 'pygame' or ending_code == 'manual':
        return row['total_trial'] - 1
    elif ending_code == 'miss':
        return row['total_trial'] - 5
    else:
        return row['total_trial']
    
def modify_sessions_all(sessions_all):
    sessions_all['dir'] = sessions_all['date']+ '_' + sessions_all['time'] + '_' + sessions_all['mouse']
    sessions_all = sessions_all.sort_values('dir')
    sessions_all[['exp', 'group']] = sessions_all['exp'].str.extract(r'exp(\d)_(short|long)')
    sessions_all['group'] = sessions_all['group'].map({'short': 's', 'long': 'l'})
    
    sessions_all['total_trial'] = sessions_all.apply(modify_total_trial, axis=1)
    sessions_all = sessions_all.drop(['forward_file'], axis=1)
    return sessions_all

def generate_sessions_all(data_folder):
    """Generates a DataFrame using session metadata from JSON files."""
    data = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.startswith("meta_") and file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    with open(path) as f:
                        session_data = json.load(f)['session_config']
                        data.append(session_data)
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    sessions_all = pd.DataFrame(data)
    sessions_all = modify_sessions_all(sessions_all)
    return sessions_all

def assign_session_numbers(group):
    group.sort_values(by=['mouse', 'dir', 'date'], inplace=True)
    group['session'] = list(range(len(group)))
    return group

def generate_sessions_training(sessions_all):
    sessions_training = sessions_all.loc[sessions_all.training == 'regular'].reset_index()
    sessions_training = sessions_training.groupby('mouse', group_keys=False).apply(assign_session_numbers)
    return sessions_training

def generate_session_logs(data_folder, save_logs=True):
    sessions_all = generate_sessions_all(data_folder)
    sessions_training = generate_sessions_training(sessions_all)
    print(f"{len(sessions_training)} sessions in total")
    if save_logs:
        utils.save_as_csv(df=sessions_all, folder=data_folder, filename='sessions_all.csv')
        utils.save_as_csv(df=sessions_training, folder=data_folder, filename='sessions_training.csv')
    return sessions_all, sessions_training

# process events
def process_events(session_info, events):
    events = events.loc[events['session_trial_num'].between(0, session_info['total_trial'])]
    return events

def add_trial_time(trial):
    trial['trial_time'] = trial['session_time'] - trial['session_time'].iloc[0]
    return trial

# dataset curation
def get_session_basics(session_df):
    num_trials = session_df.session_trial_num.max() 
    last_trial = session_df.loc[session_df['session_trial_num'] == num_trials]

    num_blocks = last_trial.loc[(last_trial['key'] == 'trial') & (last_trial['value'] == 1), 'block_num'].iloc[0] + 1
    total_reward = round(session_df.reward_size.sum(), 2)
    total_time = round((session_df.session_time.max() - session_df.session_time.min()), 2)
    session_basics = {'num_blocks': num_blocks,
                      'num_trials': num_trials + 1,
                      'rewards': total_reward,
                      'session_time': total_time}
    return session_basics  

def stitch_sessions(session_1, session_2):
    session_1_basics = get_session_basics(session_1)
    time_offset = session_1_basics['session_time']
    block_offset = session_1_basics['num_blocks']
    trial_offset = session_1_basics['num_trials']
    
    session_2.session_time = session_2.session_time + time_offset
    session_2.block_num = session_2.block_num + block_offset
    session_2.session_trial_num= session_2.session_trial_num + trial_offset

    stitched_session = pd.concat([session_1, session_2])
    return stitched_session

# analyze events
def generate_trials(session_info, events):
    trial_info_list = []
    for t in range(int(session_info.num_trials)):
        trial = events.loc[events['session_trial_num'] == t]
        trial_basics = get_trial_basics(trial)
        trial_info_list.append(trial_basics)
    trials = pd.DataFrame(trial_info_list)
    return trials

def get_trial_basics(trial):
    """gets the df of a trial, extracts 5 things, and outputs as a dictionary"""
    trial_start = trial.loc[(trial['key'] == 'trial') & (trial['value'] == 1)].iloc[0]
    trial_end = trial.loc[(trial['key'] == 'trial') & (trial['value'] == 0)].iloc[0]

    trial_basics = {'session_trial_num': trial_start['session_trial_num'],
                    'block_trial_num': trial_start['block_trial_num'],
                    'block_num': trial_start['block_num'],
                    'start_time': trial_start['session_time'],
                    'end_time': trial_end['session_time']}
    return trial_basics

def get_trial_bg_data(trial):
    bg_events = trial.loc[trial.state == 'in_background']
    bg_drawn = float(bg_events.iloc[0]['time_bg'])
    bg_length = bg_events.session_time.max() - bg_events.session_time.min()
    bg_repeats = trial['key'].value_counts()['background']
    num_bg_licks = len(bg_events.loc[(bg_events['key'] == 'lick') & (bg_events['value'] == 1)])
    return {'bg_drawn': bg_drawn,
            'bg_length': bg_length,
            'bg_repeats': bg_repeats,
            'num_bg_licks': num_bg_licks}

def get_trial_wait_data(trial):
    """gets 3 values about trial performance, takes trial raw data as input"""
    wait_start_time = trial.loc[(trial['key'] == 'wait') & (trial['value'] == 1), 'session_time'].iloc[0]
    if 'consumption' in trial.key.unique():
        miss_trial = False
        reward = trial.loc[trial['key'] == 'consumption', 'reward_size'].iloc[0]
        consumption_start_time = trial.loc[trial['key'] == 'consumption', 'session_time'].iloc[0]
        time_waited = consumption_start_time - wait_start_time
        consumption = trial.loc[trial['state'] == 'in_consumption']
        num_consumption_lick = len(consumption.loc[(consumption['key'] == 'lick') & (consumption['value'] == 1)])
        num_pump = len(consumption.loc[(consumption['key'] == 'pump') & (consumption['value'] == 1)])
    else:
        miss_trial = True
        reward = math.nan
        time_waited = math.nan
        num_consumption_lick = math.nan
        num_pump = math.nan
    return {'miss_trial': miss_trial,
            'time_waited': time_waited,
            'reward': reward,
            'num_consumption_lick': num_consumption_lick,
            'num_pump': num_pump}

def get_trial_performance(t, trial):
    bg_data = get_trial_bg_data(trial)
    wait_data = get_trial_wait_data(trial)
    trial_data = {'session_trial_num': t} | bg_data | wait_data
    if (bg_data['num_bg_licks'] == 0) & (wait_data['miss_trial'] == False):
        trial_data['good_trial'] = True
    else:
        trial_data['good_trial'] = False  
    return trial_data

def get_trial_data_df(session_by_trial):
    trial_data_list = []
    for t, trial in session_by_trial:
        trial_data = get_trial_performance(t, trial)
        trial_data_list.append(trial_data)
    trials_data = pd.DataFrame(trial_data_list)
    return trials_data