import os
import math

import pandas as pd

# Generate session logs
def generate_all_session_log(data_folder):
    session_log = pd.DataFrame()
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.startswith("all_events_") and file.endswith(".txt"):
                path = os.path.join(root, file)
                session_meta = pd.read_csv(path, nrows=1)
                session_log = pd.concat([session_log, session_meta], ignore_index=True)
    session_log['dir'] = session_log['date'] + '_' + session_log['time'] + '_' + session_log['mouse']
    session_log['exp'] = session_log['exp'].replace({'exp1_short': 's', 'exp1_long': 'l'})
    return session_log

def check_proper_end(last_trial):
    # proper end has the largest trial number to be session ending
    session_end = last_trial.loc[(last_trial['key'] == 'session') & (last_trial['value'] == 0)]
    last_trial_end = last_trial.loc[(last_trial['key'] == 'trial') & (last_trial['value'] == 0)]
    if len(session_end) > 0 and len(last_trial_end) > 0:
        return True
    else:
        return False

def get_session_basics(session_df):
    num_trials = session_df.session_trial_num.max() 
    last_trial = session_df.loc[session_df['session_trial_num'] == num_trials]
    proper_end = check_proper_end(last_trial)
    if proper_end:
        proper_end = True
    else:
        proper_end = False
        num_trials -= 1
        last_trial = session_df.loc[session_df['session_trial_num'] == num_trials]

    num_blocks = last_trial.loc[(last_trial['key'] == 'trial') & (last_trial['value'] == 1), 'block_num'].iloc[0] + 1
    total_reward = round(session_df.reward_size.sum(), 2)
    total_time = round((session_df.session_time.max() - session_df.session_time.min()), 2)
    session_basics = {'num_blocks': num_blocks,
                      'num_trials': num_trials + 1,
                      'rewards': total_reward,
                      'session_time': total_time,
                      'proper_end': proper_end}
    return session_basics  

def assign_session_numbers(group):
    group.sort_values(by=['mouse', 'dir', 'date'], inplace=True)
    group['session'] = list(range(len(group)))
    return group

# Process raw session data
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

# old one can be deleted after testing
def generate_all_trials(sess, session):
    trial_info_list = []
    for t in range(int(sess.num_trials)):
        trial = session.loc[session['session_trial_num'] == t]
        trial_basics = get_trial_basics(trial)
        trial_info_list.append(trial_basics)
    all_trials = pd.DataFrame(trial_info_list)
    return all_trials

def generate_trials(session_info, events):
    trial_info_list = []
    for t in range(int(session_info.num_trials)):
        trial = events.loc[events['session_trial_num'] == t]
        trial_basics = get_trial_basics(trial)
        trial_info_list.append(trial_basics)
    trials = pd.DataFrame(trial_info_list)
    return trials

def align_trial_number(session, all_trials):
    for _, trial_basics in all_trials.iterrows():
        session.loc[session['session_time'].between(trial_basics['start_time'], trial_basics['end_time']), 
                'block_num'] = trial_basics['block_num']
        session.loc[session['session_time'].between(trial_basics['start_time'], trial_basics['end_time']), 
                'session_trial_num'] = trial_basics['session_trial_num']
        session.loc[session['session_time'].between(trial_basics['start_time'], trial_basics['end_time']), 
                'block_trial_num'] = trial_basics['block_trial_num']
    return session

def align_trial_states(trial):
    bg_start_time = trial.loc[(trial['key'] == 'background') & (trial['value'] == 1)].iloc[0]['session_time']
    wait_start_time = trial.loc[(trial['key'] == 'wait') & (trial['value'] == 1)].iloc[0]['session_time']
    if 'consumption' in trial.key.unique():
        consumption_start_time = trial.loc[(trial['key'] == 'consumption') & (trial['value'] == 1)].iloc[0]['session_time']
    else:
        consumption_start_time = math.nan
    
    trial.loc[(trial.session_time > bg_start_time) & (trial.session_time < wait_start_time), 'state'] = 'in_background'
    trial.loc[(trial.session_time > wait_start_time) & (trial.session_time < consumption_start_time), 'state'] = 'in_wait'
    trial.loc[trial.session_time > consumption_start_time, 'state'] = 'in_consumption'
    return trial

def add_trial_time(trial):
    trial['trial_time'] = trial['session_time'] - trial['session_time'].iloc[0]
    return trial

def get_trial_bg_data(trial):
    bg_events = trial.loc[trial.state == 'in_background']
    bg_drawn = float(bg_events.iloc[0]['time_bg'])
    if bg_drawn < 2:
        blk_type = 's'
    elif bg_drawn > 2:
        blk_type = 'l'
    bg_length = bg_events.session_time.max() - bg_events.session_time.min()
    bg_repeats = trial['key'].value_counts()['background']
    num_bg_licks = len(bg_events.loc[(bg_events['key'] == 'lick') & (bg_events['value'] == 1)])
    return {'bg_drawn': bg_drawn,
            'blk_type': blk_type,
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
    all_trials_data = pd.DataFrame(trial_data_list)
    return all_trials_data

# Stitch sessions from the same mouse on the same day
def generate_stitched_session_log(session_log):
    stitched_session_log = session_log[['mouse', 'date', 'exp', 'training', 'rig']].copy()
    stitched_session_log = stitched_session_log.drop_duplicates(subset=['mouse', 'date'], keep='first')
    stitched_session_log['dir'] = stitched_session_log['date'] + '_' + stitched_session_log['mouse']
    return stitched_session_log

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

def stitch_all_trials(all_trials_1, all_trials_2):
    trial_offset = all_trials_1.session_trial_num.max()+1
    block_offset = all_trials_1.block_num.max()+1
    time_offset = all_trials_1.end_time.max()
    
    all_trials_2.session_trial_num = all_trials_2.session_trial_num + trial_offset
    all_trials_2.block_num = all_trials_2.block_num + block_offset
    all_trials_2.start_time = all_trials_2.start_time + time_offset
    all_trials_2.end_time = all_trials_2.end_time + time_offset

    stitched_all_trials = pd.concat([all_trials_1, all_trials_2])
    return stitched_all_trials

# stitched sessions from the same day from all mice
def generate_stitched_all_mice_session_log(session_log):
    stitched_session_log = session_log[['date', 'training']].copy()
    stitched_session_log = stitched_session_log.drop_duplicates(subset=['date'], keep='first')
    return stitched_session_log