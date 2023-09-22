import os
import math
import pandas as pd

def list_folders(path):
    dir_list = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            dir_list.append(item)
    return dir_list

# session based analysis
def generate_all_session_log(path):
    dir_list = list_folders(path)
    date_list = []
    mouse_list = []
    filename_list = []
    for f in dir_list:
        date_list.append(f[0:10])
        mouse = f[-5:]
        mouse_list.append(mouse)
        filename_list.append(f'data_{mouse}_{f[0:19]}.txt')
    session_log = pd.DataFrame({'date': date_list, 'mouse': mouse_list, 
                                'dir': dir_list, 'filename': filename_list})
    session_log = session_log.sort_values(by = ['date', 'mouse'])

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
    session_basic = [num_blocks, num_trials + 1, total_reward, total_time, proper_end]
    return session_basic


# trial based analysis 
# preprocessing of raw data
def generate_total_trial_list(session_log, dir_name):
    """makes a list of 0 to total trial number, used to loop in the session"""
    current_session = session_log.loc[session_log.dir == dir_name]
    total_trial_list = range(int(current_session.num_trials.tolist()[0]))
    return total_trial_list

def get_trial_basics(trial):
    """gets the df of a trial, extracts 5 things, and outputs as a dictionary"""
    trial_start = trial.loc[(trial['key'] == 'trial') & (trial['value'] == 1)].iloc[0]
    trial_end = trial.loc[(trial['key'] == 'trial') & (trial['value'] == 0)].iloc[0]

    trial_basics = {'start_time': trial_start['session_time'],
                    'end_time': trial_end['session_time'],
                    'block_num': trial_start['block_num'],
                    'session_trial_num': trial_start['session_trial_num'],
                    'block_trial_num': trial_start['block_trial_num']}
    return trial_basics

def align_trial_number(session, trial_basics):
    session.loc[session['session_time'].between(trial_basics['start_time'], trial_basics['end_time']), 
                'block_num'] = trial_basics['block_num']
    session.loc[session['session_time'].between(trial_basics['start_time'], trial_basics['end_time']), 
                'session_trial_num'] = trial_basics['session_trial_num']
    session.loc[session['session_time'].between(trial_basics['start_time'], trial_basics['end_time']), 
                'block_trial_num'] = trial_basics['block_trial_num']
    
def get_trial_state_times(trial):
     """gets the df of a trial, extracts start time of each state and outputs as a dictionary"""
     bg_start = trial.loc[(trial['key'] == 'background') & (trial['value'] == 1)].iloc[0]
     wait_start = trial.loc[(trial['key'] == 'wait') & (trial['value'] == 1)].iloc[0]
     if 'consumption' in trial.key.unique():
          consumption_start = trial.loc[(trial['key'] == 'consumption') & (trial['value'] == 1), 'session_time'].iloc[0]
     else:
         consumption_start = math.nan
     
     trial_state_times = {'bg_start': bg_start['session_time'],
                          'wait_start': wait_start['session_time'],
                          'consumption_start': consumption_start}
     
     return trial_state_times

def align_trial_states(session, trial_state_times, trial_basics):
    session.loc[session['session_time'].between(trial_state_times['bg_start'], 
                                                trial_state_times['wait_start']), 'state'] = 'in_background'
    session.loc[session['session_time'].between(trial_state_times['wait_start'], 
                                                trial_state_times['consumption_start']), 'state'] = 'in_wait'
    session.loc[session['session_time'].between(trial_state_times['consumption_start'], 
                                                trial_basics['end_time']), 'state'] = 'in_consumption'

def add_trial_time(session, t, trial, trial_basics):
    trial_time = trial.session_time - trial_basics['start_time']
    trial_time = trial_time.tolist()
    session.loc[session['session_trial_num'] == t, 'trial_time'] = trial_time

# adding data to all trials df
def get_trial_bg_data(trial):
    background = trial.loc[trial['state'] == 'in_background']
    bg_length = background.loc[background['key'] == 'background', 'time_bg'].iloc[0]
    num_bg_licks = len(background.loc[(background['key'] == 'lick') & (background['value'] == 1)])
    if bg_length < 2:
        blk_type = 's'
    elif bg_length > 2:
        blk_type = 'l'
    return [bg_length, blk_type, num_bg_licks]

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
    else:
        miss_trial = True
        reward = math.nan
        time_waited = math.nan
        num_consumption_lick = math.nan

    return [miss_trial, time_waited, reward, num_consumption_lick]

def get_trial_performance(trial):
    bg_data = get_trial_bg_data(trial)
    wait_data = get_trial_wait_data(trial)
    trial_data = bg_data + wait_data
    if (bg_data[2] == 0) & (wait_data[0] == False):
        trial_data.append(True)
    else: 
        trial_data.append(False)
    return trial_data

# stitching functions
def generate_stitched_session_log(session_log):
    unique_date_list = session_log.date.unique().tolist()
    unique_mouse_list = session_log.mouse.unique().tolist()

    date_list = []
    mouse_list = []
    dir_list = []
    filename_list = []
    num_sess_list = []

    for d in unique_date_list:
        log_date = session_log.loc[session_log.date == d]
        for m in unique_mouse_list:
            log_date_mouse = log_date.loc[log_date.mouse == m]
            if len(log_date_mouse) == 0:
                continue
            else:
                date_list.append(d)
                mouse_list.append(m)
                dir_list.append(f'{d}_{m}')
                filename_list.append(f'processed_data_{m}_{d}.csv')
                num_sess_list.append(len(log_date_mouse))

    stitched_session_log = pd.DataFrame({'date': date_list, 'mouse': mouse_list, 'dir': dir_list, 
                                         'filename': filename_list, 'num_sessions': num_sess_list}) 

    return(stitched_session_log)

def generate_stitched_all_mice_session_log(session_log):
    unique_date_list = session_log.date.unique().tolist()

    date_list = []
    filename_list = []
    num_sess_list = []

    for d in unique_date_list:
        log_date = session_log.loc[session_log.date == d]
        if len(log_date) == 0:
            continue
        else:
            date_list.append(d)
            filename_list.append(f'processed_data_{d}.csv')
            num_sess_list.append(len(log_date))

    stitched_all_mice_session_log = pd.DataFrame({'date': date_list, 'filename': filename_list, 
                                         'num_sessions': num_sess_list}) 
    
    return(stitched_all_mice_session_log)

def stitch_sessions(session_1, session_2):
    session_1_basics = get_session_basics(session_1)
    time_offset = session_1_basics[3]
    block_offset = session_1_basics[0]
    trial_offset = session_1_basics[1]
    
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