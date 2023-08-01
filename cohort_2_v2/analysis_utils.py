import os

import pandas as pd
import statsmodels.api as sm

def list_folders(path):
    dir_list = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            dir_list.append(item)
    return dir_list

def load_session_meta(data_folder, dir_name, file_name):
    path = os.path.join(data_folder, dir_name, file_name)
    session_meta = pd.read_csv(path, nrows=1)
    return session_meta

def load_session(data_folder, dir_name, file_name):
    path = os.path.join(data_folder, dir_name, file_name)
    session_df = pd.read_csv(path, skiprows=3)
    return session_df

def load_processed_session(data_folder, dir_name, file_name):
    path = os.path.join(data_folder, dir_name, f'processed_{file_name[:-4]}.csv')
    session_df = pd.read_csv(path, index_col=0)
    return session_df

def load_session_log(data_folder, log_name):
    path = os.path.join(data_folder, log_name)
    session_log = pd.read_csv(path, index_col=0)
    return session_log

def get_session_info_by_dirname(session_log, dir_name, column_name):
    return session_log.loc[session_log['dir'] == dir_name, column_name].tolist()[0]

def trim_session(session_log, dir_name, session):
    num_trials = get_session_info_by_dirname(session_log, dir_name, 'num_trials')
    return session[session.session_trial_num.between(0, num_trials)]

def load_all_trials(data_folder, dir_name):
    filename = f'{dir_name}_all_trials.csv'
    path = os.path.join(data_folder, dir_name, filename)
    all_trials_df = pd.read_csv(path, index_col=0)
    return all_trials_df

def load_all_trials_analyzed(data_folder, dir_name):
    filename = f'{dir_name}_all_trials_analyzed.csv'
    path = os.path.join(data_folder, dir_name, filename)
    all_trials_analyzed_df = pd.read_csv(path, index_col=0)
    return all_trials_analyzed_df

def load_all_blocks(data_folder, dir_name):
    filename = f'{dir_name}_all_blocks.csv'
    path = os.path.join(data_folder, dir_name, filename)
    all_blocks_df = pd.read_csv(path)
    return all_blocks_df

def generate_mouse_list(session_log):
    mouse_list = session_log.mouse.unique().tolist()
    mouse_list.sort()
    print(mouse_list)
    return mouse_list

def select_good_trials(all_trials):
    good_trials = all_trials.loc[all_trials['good_trial'] == True]
    return good_trials

def get_test_session(data_folder, training_session_log, session_num):
    dir_name = training_session_log.dir[session_num]
    file_name = training_session_log.filename[session_num]
    session = load_session(data_folder, dir_name, file_name)
    return session

def get_test_trial(session, trial_num):
    trial = session.loc[session['session_trial_num'] == trial_num]
    return trial

def linear_fit(df, x_column_name, y_column_name):
    x = df[x_column_name]
    y = df[y_column_name]
    x = sm.add_constant(x)

    model = sm.OLS(y, x)
    results = model.fit()

    slope = results.params['bg_length']
    intercept = results.params['const']
    rsquared = results.rsquared

    return [slope, intercept, rsquared]