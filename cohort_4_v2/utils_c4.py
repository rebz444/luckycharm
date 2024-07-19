import os

import pandas as pd

def generate_events_path(data_folder, session_info):
    return os.path.join(data_folder, session_info.dir, f'events_{session_info.dir}.txt')

def generate_events_processed_path(data_folder, session_info):
    filename = f'events_processed_{session_info.dir}.csv'
    return os.path.join(data_folder, session_info.dir, filename)

def generate_trials_path(data_folder, session_info):
    filename = f'trials_{session_info.dir}.csv'
    return os.path.join(data_folder, session_info.dir, filename)

def generate_trials_analyzed_path(data_folder, session_info):
    filename = f'trials_analyzed_{session_info.dir}.csv'
    return os.path.join(data_folder, session_info.dir, filename)

def load_data(path):
   df = pd.read_csv(path, index_col=0)
   return df

def save_as_csv(df, folder, filename):
    path = os.path.join(folder, filename)
    df.to_csv(path)

def trim_session(session_info, session):
    return session[session.session_trial_num.between(0, session_info.num_trials-1)]