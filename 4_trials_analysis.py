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

def batch_generate_trials(sessions_training, data_folder, regenerate_trials=False):
    """Generate trials for all sessions and return problematic sessions."""
    print("\nGenerating trials data...")
    
    problematic_sessions = pd.DataFrame(columns=sessions_training.columns)
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for idx, (_, session_info) in enumerate(sessions_training.iterrows(), 1):
        try: 
            trials_path = utils.generate_trials_path(data_folder, session_info)
            if os.path.isfile(trials_path) and not regenerate_trials:
                skipped_count += 1
                continue
            
            events_processed = pd.read_csv(utils.generate_events_processed_path(data_folder, session_info))
            trials = helper.generate_trials(session_info, events_processed)
            trials.to_csv(trials_path)
            processed_count += 1
            
            # Show progress every 100 sessions
            if idx % 100 == 0 or idx == len(sessions_training):
                print(f"  Progress: {idx}/{len(sessions_training)} sessions processed")
                
        except Exception as e:
            error_count += 1
            print(f"  Error generating trials for {session_info.get('dir', 'unknown')}: {e}")
            problematic_sessions = pd.concat([problematic_sessions, session_info.to_frame().T], ignore_index=True)
    
    print(f"Trials generation complete: {processed_count} processed, {skipped_count} skipped, {error_count} errors")
    
    return problematic_sessions

def analyze_trials(sessions_training, data_folder, regenerate_analysis=False):
    """Analyze trials for all sessions and return problematic sessions."""
    print("\nAnalyzing trials data...")
    
    problematic_sessions = pd.DataFrame(columns=sessions_training.columns)
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for idx, (_, session_info) in enumerate(sessions_training.iterrows(), 1):
        try:
            trials_analyzed_path = utils.generate_trials_analyzed_path(data_folder, session_info)
            if os.path.isfile(trials_analyzed_path) and not regenerate_analysis:
                skipped_count += 1
                continue
            
            # Check if trials file exists before trying to analyze
            trials_path = utils.generate_trials_path(data_folder, session_info)
            if not os.path.isfile(trials_path):
                print(f"  Warning: Trials file not found for {session_info.get('dir', 'unknown')}, skipping analysis")
                error_count += 1
                continue
            
            session_by_trial = utils.load_data(utils.generate_events_processed_path(data_folder, session_info)).groupby('session_trial_num')
            trials = utils.load_data(utils.generate_trials_path(data_folder, session_info))
            trials_data = helper.get_trial_data_df(session_by_trial)
            trials_analyzed = pd.merge(trials, trials_data, on='session_trial_num')
            
            # Add time since last reward calculation
            events = utils.load_data(utils.generate_events_processed_path(data_folder, session_info))
            trials_analyzed = helper.get_trial_reward_outcome(trials_analyzed, events)
            
            trials_analyzed[['group', 'dir']] = session_info[['group', 'dir']].values
            trials_analyzed.to_csv(trials_analyzed_path)
            processed_count += 1
            
            # Show progress every 100 sessions
            if idx % 100 == 0 or idx == len(sessions_training):
                print(f"  Progress: {idx}/{len(sessions_training)} sessions processed")
                
        except Exception as e:
            error_count += 1
            print(f"  Error analyzing trials for {session_info.get('dir', 'unknown')}: {e}")
            problematic_sessions = pd.concat([problematic_sessions, session_info.to_frame().T], ignore_index=True)
    
    print(f"Trials analysis complete: {processed_count} processed, {skipped_count} skipped, {error_count} errors")
    
    return problematic_sessions

def main(sessions_training, test_mode=False, test_sessions=5, regenerate_trials=False, regenerate_analysis=False):
    """Main function to run the combined session analysis pipeline."""
    # Apply test mode if requested
    if test_mode:
        sessions_training = sessions_training.head(test_sessions)
        print(f"Test mode: Processing only {len(sessions_training)} sessions")
        print(sessions_training.to_string())
    else:
        print(f"Processing {len(sessions_training)} total sessions")
    
    # Generate trials
    problematic_trials = batch_generate_trials(sessions_training, data_folder, regenerate_trials)
    if len(problematic_trials) > 0:
        print(f"Found {len(problematic_trials)} problematic sessions during trial generation")
        print(problematic_trials.to_string())
        return
    else:
        print("All sessions processed successfully for trial generation")
    
    # Analyze trials
    problematic_analysis = analyze_trials(sessions_training, data_folder, regenerate_analysis)
    if len(problematic_analysis) > 0:
        print(f"Found {len(problematic_analysis)} problematic sessions during trial analysis")
        print(problematic_analysis.to_string())
        return
    else:
        print("All sessions analyzed successfully")

def stitch_all_trials(data_folder, sessions_training):
    trials_list = []
    for _, session_info in sessions_training.iterrows():
        trials = utils.load_data(utils.generate_trials_analyzed_path(data_folder, session_info))
        trials_list.append(trials)
    trials_training = pd.concat(trials_list, ignore_index=True)
    sessions_info_to_merge = sessions_training[['dir', 'mouse', 'date', 'time', 'avg_tw', 'session']]
    trials_training_analyzed = trials_training.merge(sessions_info_to_merge, on='dir', how='left')
    trials_training_analyzed.to_csv(os.path.join(data_folder, 'trials_training_analyzed.csv'))


if __name__ == "__main__":
    sessions_training = correct_sessions_training(data_folder)

    # sessions_training = utils.load_session_log(data_folder, f'sessions_training_{exp}.csv')
    # # main(sessions_training, test_mode=False, test_sessions=20, regenerate_trials=False, regenerate_analysis=True)

    # stitch_all_trials(data_folder, sessions_training)