import os
import math
import pandas as pd

# Session basics
def load_session_meta(data_folder, dir_name, file_name):
    path = os.path.join(data_folder, dir_name, file_name)
    session_meta = pd.read_csv(path, nrows=1)
    return session_meta

def load_session(data_folder, dir_name, file_name):
    path = os.path.join(data_folder, dir_name, file_name)
    session_df = pd.read_csv(path, skiprows=3)
    return session_df

def get_session_basic(session_df):
    total_blocks = session_df.block_num.max()
    total_trials = session_df.session_trial_num.max()
    total_reward = round(session_df.reward_size.sum(), 2)
    total_time = round((session_df.session_time.max() - session_df.session_time.min()), 2)
    session_basic = [total_blocks, total_trials, total_reward, total_time]
    return session_basic

def get_session_basic_proper_end(session_df):
    total_blocks = session_df.block_num.max()
    total_trials = session_df.session_trial_num.max()
    total_reward = round(session_df.reward_size.sum(), 2)
    total_time = round((session_df.session_time.max() - session_df.session_time.min()), 2)
    proper_end = session_df.loc[(session_df['key'] == 'session') & (session_df['value'] == 0)]
    if len(proper_end) > 0:
        proper_end = True
        total_trials = total_trials
    else:
        proper_end = False
        total_trials -= 1
    session_basic = [total_blocks, total_trials, total_reward, total_time, proper_end]
    return session_basic

# make all trials df and saves for each ession
def generate_total_trial_list(session_log, dir_name):
    """makes a list of 0 to total trial number, used to loop in the session"""
    current_session = session_log.loc[session_log.dir == dir_name]
    total_trial_list = range(int(current_session.num_trials.tolist()[0]) + 1)
    return total_trial_list

def generate_all_trials_df(column_names, total_trial_list):
    """
    makes an empty df with each row being a trial, and each column with trial info
    trial number is added to the df
    """
    all_trials = pd.DataFrame(columns=column_names)
    all_trials['trial_num'] = total_trial_list
    return all_trials

def get_trial_basics(trial):
    """gets 5 basic things about the trial, takes raw data of each trial as input"""
    block_num = trial.loc[(trial['key'] == 'trial') & (trial['value'] == 1), 'block_num'].iloc[0]
    start_time = trial.loc[(trial['key'] == 'trial') & (trial['value'] == 1), 'session_time'].iloc[0]
    end_time = trial.loc[(trial['key'] == 'trial') & (trial['value'] == 0), 'session_time'].iloc[0]
    enl_repeat = trial['key'].value_counts()['enl']
    blk_bg_avg = float(trial.loc[(trial['key'] == 'trial') & (trial['value'] == 1), 'time_bg'].iloc[0])
    if blk_bg_avg < 2:
        blk_type = 's'
    elif blk_bg_avg > 2:
        blk_type = 'l'
    return [block_num, start_time, end_time, enl_repeat, blk_bg_avg, blk_type]

def get_trial_bg_length(trial):
    """gets total time in background, takes raw data of each trial as input"""
    bg_start_idx = trial.index[(trial['key'] == 'trial') & (trial['value'] == 1)].tolist()
    bg_end_idx = trial.index[(trial['key'] == 'wait') & (trial['value'] == 1)].tolist()
    trial_bg = trial.loc[bg_start_idx[0] : bg_end_idx[0]]
    trial_bg_length = trial_bg.session_time.max() - trial_bg.session_time.min()
    return [trial_bg_length]

def get_trial_performance(trial):
    """gets 3 values about trial performance, takes trial raw data as input"""
    wait_start_time = trial.loc[(trial['key'] == 'wait') & (trial['value'] == 1), 'session_time'].iloc[0]
    if 'in_consumption' in trial.state.unique() :
        miss_trial = False
        reward = trial.loc[trial['key'] == 'reward', 'reward_size'].iloc[0]
        consumption_start_time = trial.loc[trial['state'] == 'in_consumption', 'session_time'].iloc[0]
        time_waited = consumption_start_time - wait_start_time
    else :
        miss_trial = True
        reward = math.nan
        time_waited = math.nan
    return [reward, miss_trial, time_waited]

def get_num_consumption_licks(trial):
    """gets the number of consumption licks of each trial, takes raw data as input"""
    consumption = trial.loc[trial['state'] == 'in_consumption']
    num_consumption_lick = len(consumption.loc[(consumption['key'] == 'lick') & (trial['value'] == 1)])
    return [num_consumption_lick]

def get_trial_data(trial):
    """
    runs individual functions and consolidate all info to one long list to be added to session log
    """
    trial_basics = get_trial_basics(trial)
    trial_bg_length = get_trial_bg_length(trial)
    trial_performance = get_trial_performance(trial)
    num_consumption_lick = get_num_consumption_licks(trial)
    trial_data = [trial_basics + trial_bg_length + trial_performance + num_consumption_lick]
    return trial_data

# session based analysis and add to training log based on all trials df
def load_all_trials(data_folder, dir_name):
    filename = f'{dir_name}_all_trials.csv'
    path = os.path.join(data_folder, dir_name, filename)
    all_trials_df = pd.read_csv(path, index_col=0)
    return all_trials_df

def select_good_trials(all_trials):
    good_trials = all_trials.loc[(all_trials['miss_trial'] == False) & (all_trials['enl_repeats'] == 1)]
    return good_trials

def get_session_performance(all_trials):
    num_miss_trials = all_trials.miss_trial.values.sum()
    good_trials = all_trials.loc[(all_trials['miss_trial'] == False) & (all_trials['enl_repeats'] == 1)]
    num_good_trials = len(good_trials)
    return [num_miss_trials, num_good_trials]

def get_session_mistakes(all_trials):
    num_bg_repeats_mean = all_trials.enl_repeats.mean()
    num_bg_repeats_med = all_trials.enl_repeats.median()
    num_bg_repeats_std = all_trials.enl_repeats.std()
    return [num_bg_repeats_mean, num_bg_repeats_med, num_bg_repeats_std]

def get_session_time_waited(all_trials):
    tw_mean = all_trials.time_waited.mean()
    tw_med = all_trials.time_waited.median()
    tw_std = all_trials.time_waited.std()
    return [tw_mean, tw_med, tw_std]