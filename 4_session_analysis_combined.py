import os
import shutil
import pandas as pd
import warnings

import processing_helper_0827 as helper_pre
import session_processing_helper as helper
import utils

# Suppress pandas FutureWarning about groupby.apply
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrameGroupBy.apply operated on the grouping columns.*')

# Configuration
data_dir = '/Users/rebekahzhang/data/behavior_data'
exp = 'exp2'
data_folder = os.path.join(data_dir, exp)
meta_change_date = '2024-04-16'

def correct_sessions_training(data_folder, save_log=True):
    """Generate and correct sessions training data for both pre and post meta change."""
    sessions_all = helper_pre.generate_sessions_all(data_folder)
    
    # Split sessions by meta change date
    sessions_pre_meta = sessions_all.loc[
        (sessions_all.training == 'regular') & 
        (sessions_all.date < meta_change_date)
    ].reset_index()
    
    sessions_post_meta = sessions_all.loc[
        (sessions_all.training == 'regular') & 
        (sessions_all.date >= meta_change_date)
    ].reset_index()
    
    print(f"Found {len(sessions_pre_meta)} training sessions before {meta_change_date}")
    print(f"Found {len(sessions_post_meta)} training sessions after {meta_change_date}")
    
    # Process pre meta change sessions
    pre_meta_list = []
    for _, session_info in sessions_pre_meta.iterrows():
        try:
            events_processed = pd.read_csv(utils.generate_events_processed_path(data_folder, session_info), low_memory=False)
            session_basics = helper.get_session_basics(events_processed)
            session_basics['dir'] = session_info['dir']
            session_basics['meta_change_period'] = 'pre'
            pre_meta_list.append(session_basics)
        except Exception as e:
            print(f"Error processing pre meta session {session_info.get('dir', 'unknown')}: {e}")
            continue
    
    # Process post meta change sessions
    post_meta_list = []
    for _, session_info in sessions_post_meta.iterrows():
        try:
            events_processed = pd.read_csv(utils.generate_events_processed_path(data_folder, session_info), low_memory=False)
            session_basics = helper.get_session_basics(events_processed)
            session_basics['dir'] = session_info['dir']
            session_basics['meta_change_period'] = 'post'
            post_meta_list.append(session_basics)
        except Exception as e:
            print(f"Error processing post meta session {session_info.get('dir', 'unknown')}: {e}")
            continue
    
    # Combine both periods
    all_sessions = pd.concat([sessions_pre_meta, sessions_post_meta], ignore_index=True)
    all_session_info = pd.concat([pd.DataFrame(pre_meta_list), pd.DataFrame(post_meta_list)], ignore_index=True)
    
    # Merge and process
    corrected_sessions_training = pd.merge(all_sessions, all_session_info, on="dir")
    corrected_sessions_training = corrected_sessions_training.drop(columns=['index', 'total_reward', 'total_trial', 'total_reward'])
    
    # Extract group information from exp column (exp2_short -> s, exp2_long -> l)
    corrected_sessions_training['group'] = corrected_sessions_training['exp'].str.extract(r'exp\d+_(short|long)').replace({'short': 's', 'long': 'l'})
    
    # Assign session numbers across all sessions
    corrected_sessions_training = corrected_sessions_training.groupby('mouse', group_keys=False).apply(helper.assign_session_numbers)
    
    if save_log:
        utils.save_as_csv(df=corrected_sessions_training, folder=data_folder, filename=f'sessions_training_{exp}.csv')
    
    return corrected_sessions_training

def process_trials_generation(sessions_training, data_folder, regenerate=False):
    """Generate trials for all sessions and return problematic sessions."""
    problematic_sessions = pd.DataFrame(columns=sessions_training.columns)
    
    for _, session_info in sessions_training.iterrows():
        try: 
            trials_path = utils.generate_trials_path(data_folder, session_info)
            if os.path.isfile(trials_path) and not regenerate:
                continue
            
            events_processed = pd.read_csv(utils.generate_events_processed_path(data_folder, session_info))
            trials = helper.generate_trials(session_info, events_processed)
            trials.to_csv(trials_path)
            if regenerate:
                print(f"Regenerated trials for session {session_info.get('dir', 'unknown')}")
        except:
            problematic_sessions = pd.concat([problematic_sessions, session_info.to_frame().T], ignore_index=True)
    
    return problematic_sessions

def analyze_trials(sessions_training, data_folder, regenerate=False):
    """Analyze trials for all sessions and return problematic sessions."""
    problematic_sessions = pd.DataFrame(columns=sessions_training.columns)
    
    for _, session_info in sessions_training.iterrows():
        try:
            trials_analyzed_path = utils.generate_trials_analyzed_path(data_folder, session_info)
            if os.path.isfile(trials_analyzed_path) and not regenerate:
                continue
            
            # Check if trials file exists before trying to analyze
            trials_path = utils.generate_trials_path(data_folder, session_info)
            if not os.path.isfile(trials_path):
                print(f"Warning: Trials file not found for session {session_info.get('dir', 'unknown')}, skipping analysis")
                continue
            
            session_by_trial = utils.load_data(utils.generate_events_processed_path(data_folder, session_info)).groupby('session_trial_num')
            trials = utils.load_data(utils.generate_trials_path(data_folder, session_info))
            trials_data = helper.get_trial_data_df(session_by_trial)
            trials_analyzed = pd.merge(trials, trials_data, on='session_trial_num')
            trials_analyzed['group'] = session_info['group'] #assigning trial type manually
            trials_analyzed['meta_change_period'] = session_info['meta_change_period'] #preserve meta change period info
            trials_analyzed.to_csv(trials_analyzed_path)
            if regenerate:
                print(f"Regenerated analysis for session {session_info.get('dir', 'unknown')} ({session_info.get('meta_change_period', 'unknown')})")
            else:
                print(f"Analyzed trials for session {session_info.get('dir', 'unknown')} ({session_info.get('meta_change_period', 'unknown')})")
        except Exception as e:
            print(f"Error analyzing trials for session {session_info.get('dir', 'unknown')}: {e}")
            problematic_sessions = pd.concat([problematic_sessions, session_info.to_frame().T], ignore_index=True)
    
    return problematic_sessions

def main(test_mode=False, test_sessions=5, regenerate=False):
    """Main function to run the combined session analysis pipeline."""
    # Generate sessions for both periods
    print("Generating combined sessions...")
    sessions_training = correct_sessions_training(data_folder)
    
    if len(sessions_training) == 0:
        print("No sessions generated, exiting.")
        return
    
    # Apply test mode if requested
    if test_mode:
        sessions_training = sessions_training.head(test_sessions)
        print(f"TEST MODE: Processing only {len(sessions_training)} sessions for testing")
    else:
        print(f"Processing {len(sessions_training)} total sessions...")
        pre_count = len(sessions_training[sessions_training['meta_change_period'] == 'pre'])
        post_count = len(sessions_training[sessions_training['meta_change_period'] == 'post'])
        print(f"  - Pre meta change: {pre_count} sessions")
        print(f"  - Post meta change: {post_count} sessions")
    
    if regenerate:
        print("REGENERATE MODE: Will regenerate existing trials and analysis files")
    
    # Generate trials
    print("Generating trials...")
    problematic_trials = process_trials_generation(sessions_training, data_folder, regenerate=regenerate)
    
    if len(problematic_trials) > 0:
        print(f"Found {len(problematic_trials)} problematic sessions during trial generation:")
        print(problematic_trials.to_string())
        return
    else:
        print("All sessions processed successfully for trial generation!")
    
    # Analyze trials
    print("Analyzing trials...")
    problematic_analysis = analyze_trials(sessions_training, data_folder, regenerate=regenerate)
    
    if len(problematic_analysis) > 0:
        print(f"Found {len(problematic_analysis)} problematic sessions during trial analysis:")
        print(problematic_analysis.to_string())
        return
    else:
        print("All sessions analyzed successfully!")
    
    print("Combined session analysis complete!")

if __name__ == "__main__":
    # Set test_mode=True to process only a few sessions for testing
    # Set test_sessions=N to control how many sessions to test
    # Set regenerate=True to force regeneration of existing files
    main(test_mode=False, test_sessions=10, regenerate=False)
    
    # For full processing, use:
    # main(test_mode=False, regenerate=False)
    
    # For regeneration mode, use:
    # main(test_mode=False, regenerate=True)
