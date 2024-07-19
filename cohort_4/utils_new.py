import os

import pandas as pd

def get_filename(session_info):
    return f'all_events_{session_info.dir}.txt'

def save_as_csv(df, folder, filename):
    path = os.path.join(folder, filename)
    df.to_csv(path)

def trim_session(session_info, session):
    return session[session.session_trial_num.between(0, session_info.num_trials-1)]

def generate_processed_session_path(data_folder, session_info):
    filename = get_filename(session_info)
    return os.path.join(data_folder, session_info.dir, f'processed_{filename[:-4]}.csv')

def generate_stitched_processed_session_path(data_folder, session_info):
    dir = f'{session_info.date}_{session_info.mouse}'
    return os.path.join(data_folder, dir, f'stitched_processed_{dir}.csv')

def generate_stitched_all_mice_processed_session_path(data_folder, session_info):
    return os.path.join(data_folder, session_info.date, 
                        f'stitched_all_mice_processed_session_{session_info.date}.csv')

def generate_all_trials_path(data_folder, session_info):
    filename = f'all_trials_{session_info.dir}.csv'
    return os.path.join(data_folder, session_info.dir, filename)

def generate_all_trials_analyzed_path(data_folder, session_info):
    filename = f'all_trials_analyzed_{session_info.dir}.csv'
    return os.path.join(data_folder, session_info.dir, filename)

def generate_stitched_all_trials_path(data_folder, session_info):
    dir = f'{session_info.date}_{session_info.mouse}'
    return os.path.join(data_folder, dir, f'all_trials_analyzed_{dir}.csv')

def generate_stitched_all_mice_all_trials_analyzed_path(data_folder, session_info):
    return os.path.join(data_folder, session_info.date, 
                        f'stitched_all_mice_all_trials_analyzed_{session_info.date}.csv')

def load_session_log(data_folder, log_name):
    path = os.path.join(data_folder, log_name)
    session_log = pd.read_csv(path, index_col=0)
    return session_log

def load_session(data_folder, session_info):
    path = os.path.join(data_folder, session_info.dir, get_filename(session_info))
    session_df = pd.read_csv(path, skiprows=3)
    return session_df

def load_processed_session(data_folder, session_info):
    path = generate_processed_session_path(data_folder, session_info)
    session_df = pd.read_csv(path, index_col=0)
    return session_df

def load_stitched_processed_session(data_folder, session_info):
    path = generate_stitched_processed_session_path(data_folder, session_info)
    session_df = pd.read_csv(path, index_col=0)
    return session_df

def load_stitched_all_mice_processed_session(data_folder, session_info):
    path = generate_stitched_all_mice_processed_session_path(data_folder, session_info)
    session_df = pd.read_csv(path, index_col=0)
    return session_df

def load_all_trials(data_folder, session_info):
    path = generate_all_trials_path(data_folder, session_info)
    all_trials_df = pd.read_csv(path, index_col=0)
    return all_trials_df

def load_all_trials_analyzed(data_folder, session_info):
    path = generate_all_trials_analyzed_path(data_folder, session_info)
    all_trials_df = pd.read_csv(path, index_col=0)
    return all_trials_df

def load_stitched_all_trials_analyzed(data_folder, session_info):
    path = generate_stitched_all_trials_path(data_folder, session_info)
    all_trials_df = pd.read_csv(path, index_col=0)
    return all_trials_df

def load_stitched_all_mice_all_trials_analyzed(data_folder, session_info):
    path = generate_stitched_all_mice_all_trials_analyzed_path(data_folder, session_info)
    all_trials_df = pd.read_csv(path, index_col=0)
    return all_trials_df

