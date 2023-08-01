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
    return session_log

def check_proper_end(last_trial):
    session_end = last_trial.loc[(last_trial['key'] == 'session') & (last_trial['value'] == 0)]
    last_trial_end = last_trial.loc[(last_trial['key'] == 'trial') & (last_trial['value'] == 0)]
    if len(session_end) > 0 and len(last_trial_end) > 0:
        return True
    else:
        return False

def get_session_basics(session_df):
    last_trial_num = session_df.session_trial_num.max()
    last_trial = session_df.loc[session_df['session_trial_num'] == last_trial_num]
    proper_end = check_proper_end(last_trial)
    if proper_end:
        proper_end = True
    else:
        proper_end = False
        last_trial_num -= 1
        last_trial = session_df.loc[session_df['session_trial_num'] == last_trial_num]

    total_blocks = last_trial.loc[(last_trial['key'] == 'trial') & (last_trial['value'] == 1), 'block_num'].iloc[0]
    total_trials = last_trial_num
    total_reward = round(session_df.reward_size.sum(), 2)
    total_time = round((session_df.session_time.max() - session_df.session_time.min()), 2)
    total_trials -= 1
    session_basic = [total_blocks, total_trials, total_reward, total_time, proper_end]
    return session_basic


# trial based analysis 
# preprocessing of raw data
def generate_total_trial_list(session_log, dir_name):
    """makes a list of 0 to total trial number, used to loop in the session"""
    current_session = session_log.loc[session_log.dir == dir_name]
    total_trial_list = range(int(current_session.num_trials.tolist()[0]) + 1)
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
    bg_start_idx = trial.index[(trial['key'] == 'trial') & (trial['value'] == 1)].tolist()
    bg_end_idx = trial.index[(trial['key'] == 'wait') & (trial['value'] == 1)].tolist()
    trial_bg = trial.loc[bg_start_idx[0] : bg_end_idx[0]]
    bg_drawn = float(trial_bg.iloc[0]['time_bg'])
    if bg_drawn < 2:
        blk_type = 's'
    elif bg_drawn > 2:
        blk_type = 'l'
    bg_length = trial_bg.session_time.max() - trial_bg.session_time.min()
    return [bg_drawn, blk_type, bg_length]  

def get_trial_wait_data(trial):
    """gets 3 values about trial performance, takes trial raw data as input"""
    wait_start_time = trial.loc[(trial['key'] == 'wait') & (trial['value'] == 1), 'session_time'].iloc[0]
    if 'consumption' in trial.key.unique():
        miss_trial = False
        reward = trial.loc[trial['key'] == 'consumption', 'reward_size'].iloc[0]
        consumption_start_time = trial.loc[trial['key'] == 'consumption', 'session_time'].iloc[0]
        time_waited = consumption_start_time - wait_start_time
        consumption = trial.loc[trial['state'] == 'in_consumption']
        num_consumption_lick = len(consumption.loc[(consumption['key'] == 'lick') & (trial['value'] == 1)])
    else:
        miss_trial = True
        reward = math.nan
        time_waited = math.nan
        num_consumption_lick = math.nan

    if (miss_trial == False) & (time_waited > 0.5):
        good_trial = True
    else:
        good_trial = False
    return [miss_trial, good_trial, time_waited, reward, num_consumption_lick]

def get_trial_performance(trial):
    bg_data = get_trial_bg_data(trial)
    wait_data = get_trial_wait_data(trial)
    return [bg_data + wait_data]

# block based analysis
# load analyzed all trials
def generate_total_block_list(session_log, dir_name):
    """makes a list of 0 to total trial number, used to loop in the session"""
    current_session = session_log.loc[session_log.dir == dir_name]
    total_block_list = range(int(current_session.num_blocks.tolist()[0]) + 1)
    return total_block_list

def generate_all_blocks_df(column_names, total_block_list):
    """
    makes an empty df with each row being a trial, and each column with trial info
    trial number is added to the df
    """
    all_blocks = pd.DataFrame(columns=column_names)
    all_blocks['block_num'] = total_block_list
    return all_blocks

def get_block_basics(block):
    """gets the df of a trial, extracts 5 things, and outputs as a dictionary"""
    num_trials = block.block_trial_num.max()
    blk_start = block.loc[block['block_trial_num'] == 0].iloc[0]
    blk_end = block.loc[block['block_trial_num'] == num_trials].iloc[0]

    blk_type = blk_start['blk_type']
    start_time = blk_start['start_time']
    end_time = blk_end['end_time']
    
    return [blk_type, num_trials, start_time, end_time]

def get_block_bg_data(block):
    bg_drawn_mean = block.bg_drawn.mean()
    bg_drawn_std = block.bg_drawn.std()
    bg_length_mean = block.bg_length.mean()
    bg_length_std = block.bg_length.std()
    enl_repeats_mean = block.enl_repeats.mean()
    enl_repeats_std = block.enl_repeats.std()
    return [bg_drawn_mean, bg_drawn_std, bg_length_mean, bg_length_std, enl_repeats_mean, enl_repeats_std]

def get_block_wait_data(block):
    num_miss_trials = block.miss_trial.sum()
    time_waited_mean = block.time_waited.mean()
    time_waited_std = block.time_waited.std()
    reward_mean = block.reward.mean()
    reward_std = block.reward.std()
    num_consumption_lick_mean = block.num_consumption_lick.mean()
    num_consumption_lick_std = block.num_consumption_lick.std()
    return [num_miss_trials, time_waited_mean, time_waited_std, reward_mean, reward_std, 
            num_consumption_lick_mean, num_consumption_lick_std]

def get_block_data(block):
    """
    runs individual functions and consolidate all info to one long list to be added to session log
    """
    block_basics = get_block_basics(block)
    block_bg_data = get_block_bg_data(block)
    block_wait_data = get_block_wait_data(block)
    block_data = block_basics + block_bg_data + block_wait_data
    return block_data