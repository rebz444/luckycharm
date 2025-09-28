import os
import pandas as pd
import warnings

import session_processing_helper as helper
import utils

# Suppress pandas FutureWarning about groupby.apply
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrameGroupBy.apply operated on the grouping columns.*')

# Configuration
data_dir = '/Users/rebekahzhang/data/behavior_data'
exp = 'exp2'
data_folder = os.path.join(data_dir, exp)

def correct_sessions_training(data_folder, save_log=True):
    """Generate and correct sessions training data for both pre and post meta change."""
    print("\nCorrecting sessions training data...")
    sessions_all = helper.generate_sessions_all(data_folder)
    sessions_training = sessions_all.loc[sessions_all.training == 'regular'].reset_index()
    
    # Process all sessions (no need to separate pre/post)
    session_basics_list = []
    
    for _, session_info in sessions_training.iterrows():
        try:
            events_processed = pd.read_csv(utils.generate_events_processed_path(data_folder, session_info), low_memory=False)
            session_basics = helper.get_session_basics(events_processed)
            session_basics['dir'] = session_info['dir']
            session_basics_list.append(session_basics)
        except Exception as e:
            print(f"Error processing {session_info.get('dir', 'unknown')}: {e}")
            continue
    
    # Merge session info with session basics
    session_basics_df = pd.DataFrame(session_basics_list)
    corrected_sessions_training = pd.merge(sessions_training, session_basics_df, on="dir")
    
    # Clean up columns (remove duplicates and unnecessary columns)
    columns_to_drop = ['index', 'total_reward', 'total_trial']
    corrected_sessions_training = corrected_sessions_training.drop(columns=[col for col in columns_to_drop if col in corrected_sessions_training.columns])
    corrected_sessions_training['group'] = corrected_sessions_training['exp'].str.extract(r'exp\d+_(short|long)').replace({'short': 's', 'long': 'l'})
    corrected_sessions_training = corrected_sessions_training.groupby('mouse', group_keys=False).apply(helper.assign_session_numbers)
    
    if save_log:
        utils.save_log(corrected_sessions_training, data_folder, f'sessions_training_{exp}.csv')
        print("Session log saved\n")
    
    return corrected_sessions_training
    
if __name__ == "__main__":
    sessions_training = correct_sessions_training(data_folder)
