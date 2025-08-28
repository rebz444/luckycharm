import json
import os
import pandas as pd

with open('exp_cohort_info.json', 'r') as f:
    training_info = json.load(f)
cohort_info = training_info['cohorts']

def add_cohort_column(sessions_all, cohort_info):
    """Add cohort column based on mouse name and cohort info."""
    # Create reverse mapping
    mouse_to_cohort = {}
    for cohort, mice in cohort_info.items():
        for mouse in mice:
            mouse_to_cohort[mouse] = cohort
    sessions_all['cohort'] = sessions_all['mouse'].map(mouse_to_cohort)
    return sessions_all

def generate_sessions_all(data_folder):
    """Generate DataFrame from session metadata JSON files."""
    data = []
    
    for root, _, files in os.walk(data_folder):
        for file in files: 
            if file.startswith("meta_") and file.endswith(".json"):
                path = os.path.join(root, file)
                try:
                    with open(path) as f:
                        session_data = json.load(f)

                    date_str = file.split('_')[1]
                    if date_str < '2024-04-16':
                        data.append(session_data)
                    else:
                        data.append(session_data.get('session_config', session_data))
                        
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    sessions_all = pd.DataFrame(data)
    sessions_all['dir'] = sessions_all['date'] + '_' + sessions_all['time'] + '_' + sessions_all['mouse']
    sessions_all = add_cohort_column(sessions_all, cohort_info)
    sessions_all = sessions_all.drop(columns=['trainer', 'record', 'forward_file', 'pump_ul_per_turn'])
    return sessions_all.sort_values('dir')

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

def add_trial_time(trial):
    """Add trial_time column relative to trial start."""
    trial['trial_time'] = trial['session_time'] - trial['session_time'].iloc[0]
    return trial