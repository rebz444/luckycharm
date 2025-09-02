import json
import os
import pandas as pd

with open('exp_cohort_info.json', 'r') as f:
    training_info = json.load(f)
cohort_info = training_info['cohorts']

meta_change_date = '2024-04-16'

def add_cohort_column(sessions_all, cohort_info):
    """Add cohort column based on mouse name and cohort info."""
    # Create reverse mapping
    mouse_to_cohort = {}
    for cohort, mice in cohort_info.items():
        for mouse in mice:
            mouse_to_cohort[mouse] = cohort
    sessions_all['cohort'] = sessions_all['mouse'].map(mouse_to_cohort)
    return sessions_all

def modify_total_trial(row):
    """Modify total trial count based on ending code."""
    ending_code = row['ending_code']
    if pd.isna(ending_code):
        return row['total_trial']
    
    ending_code = str(ending_code).lower()
    if ending_code == 'pygame' or ending_code == 'manual':
        return row['total_trial'] - 1
    elif ending_code == 'miss':
        return row['total_trial'] - 5
    else:
        return row['total_trial']

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
                    if date_str < meta_change_date:
                        data.append(session_data)
                    else:
                        data.append(session_data.get('session_config', session_data))
                        
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    sessions_all = pd.DataFrame(data)
    sessions_all['dir'] = sessions_all['date'] + '_' + sessions_all['time'] + '_' + sessions_all['mouse']
    sessions_all['total_trial'] = sessions_all.apply(modify_total_trial, axis=1)
    sessions_all = add_cohort_column(sessions_all, cohort_info)
    
    # Add version column based on date
    sessions_all['version'] = sessions_all['date'].apply(
        lambda x: 'pre' if x < meta_change_date else 'post'
    )
    
    sessions_all = sessions_all.drop(columns=['trainer', 'record', 'forward_file', 'pump_ul_per_turn'])
    return sessions_all.sort_values('dir')