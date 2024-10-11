import json
import os

import pandas as pd

def assign_session_numbers(group):
    group.sort_values(by=['mouse', 'dir', 'date'], inplace=True)
    group['session'] = list(range(len(group)))
    return group

def modify_total_trial(row):
    if row['ending_code'] == 'pygame':
        return row['total_trial'] - 1
    elif row['ending_code'] == 'miss':
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
    """Generates a DataFrame using session metadata from JSON files.
    Args:
        data_folder (str): Path to the directory containing JSON files.
    Returns:
        pd.DataFrame: DataFrame containing session metadata, sorted by 'dir' column.
    """

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

def generate_sessions_training(sessions_all):
    sessions_training = sessions_all.loc[sessions_all.training == 'regular'].reset_index()
    sessions_training = sessions_training.groupby('mouse', group_keys=False).apply(assign_session_numbers)
    return sessions_training