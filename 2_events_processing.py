import os
import math
from pickle import FALSE
import warnings

import session_processing_helper as helper
import utils

import pandas as pd

# Suppress pandas FutureWarning about groupby.apply behavior
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrameGroupBy.apply operated on the grouping columns.*')

data_dir = '/Users/rebekahzhang/data/behavior_data'
exp = 'exp2'
data_folder = os.path.join(data_dir, exp)

# Generate all sessions
sessions_all = helper.generate_sessions_all(data_folder)

# Save sessions_all as CSV in data_folder
sessions_all_path = os.path.join(data_folder, 'sessions_all_pre_processing.csv')
sessions_all.to_csv(sessions_all_path, index=False)
print(f"Saved sessions_all.csv to {sessions_all_path}")

# Split sessions by metadata change date (2024-04-16)
sessions_pre_meta = sessions_all.loc[sessions_all.version == 'pre'].reset_index()
sessions_post_meta = sessions_all.loc[sessions_all.version == 'post'].reset_index()

print(f"Found {len(sessions_pre_meta)} sessions before metadata change to process")
print(f"Found {len(sessions_post_meta)} sessions after metadata change to process")

# =============================================================================
# PRE-METADATA CHANGE PROCESSING FUNCTIONS
# =============================================================================

def get_max_trial_num(events):
    """Get the maximum valid trial number, accounting for incomplete last trials."""
    max_trial_num = events['session_trial_num'].max()
    last_trial = events.loc[events['session_trial_num'] == max_trial_num]
    session_end = last_trial.loc[(last_trial['key'] == 'session') & (last_trial['value'] == 0)]
    last_trial_end = last_trial.loc[(last_trial['key'] == 'trial') & (last_trial['value'] == 0)]
    
    if len(session_end) > 0 and len(last_trial_end) > 0:
        return int(max_trial_num)
    else:
        return int(max_trial_num - 1)

def generate_trials(events, max_trial_num):
    """Generate trial information for all trials in the session."""
    trial_info_list = []
    for t in range(int(max_trial_num)+1):
        trial = events.loc[events['session_trial_num'] == t]
        if not trial.empty:
            trial_basics = helper.get_trial_basics(trial)
            trial_info_list.append(trial_basics)
    trials = pd.DataFrame(trial_info_list)
    return trials

def align_trial_number(session, trials):
    """Align trial numbers with session events based on time windows."""
    for _, trial_basics in trials.iterrows():
        # Create time window mask
        time_mask = session['session_time'].between(trial_basics['start_time'], trial_basics['end_time'])
        
        # Apply trial information to events within the time window
        session.loc[time_mask, 'block_num'] = trial_basics['block_num']
        session.loc[time_mask, 'session_trial_num'] = trial_basics['session_trial_num']
        session.loc[time_mask, 'block_trial_num'] = trial_basics['block_trial_num']
    
    return session

def align_trial_states(trial):
    """Align trial states for pre-metadata change sessions."""
    bg_start_time = trial.loc[(trial['key'] == 'background') & (trial['value'] == 1)].iloc[0]['session_time']
    wait_start_time = trial.loc[(trial['key'] == 'wait') & (trial['value'] == 1)].iloc[0]['session_time']
    
    if 'consumption' in trial.key.unique():
        consumption_start_time = trial.loc[(trial['key'] == 'consumption') & (trial['value'] == 1)].iloc[0]['session_time']
    else:
        consumption_start_time = math.nan
    
    trial.loc[(trial.session_time > bg_start_time) & (trial.session_time < wait_start_time), 'state'] = 'in_background'
    trial.loc[(trial.session_time > wait_start_time) & (trial.session_time < consumption_start_time), 'state'] = 'in_wait'
    trial.loc[trial.session_time > consumption_start_time, 'state'] = 'in_consumption'
    
    return trial

def process_events_pre_meta_change(data_folder, session_info):
    """Process events for sessions before metadata change (pre 2024-04-16)."""
    # Load and process events
    events = pd.read_csv(utils.generate_events_path(data_folder, session_info), low_memory=False)
    max_trial_num = get_max_trial_num(events)
    
    # Generate trial information and align data
    trials = generate_trials(events, max_trial_num)
    events = align_trial_number(events, trials)
    events = events.loc[events['session_trial_num'].between(0, max_trial_num)]
    
    # Add trial states and timing
    events = events.groupby('session_trial_num', group_keys=False).apply(align_trial_states)
    events_processed = events.groupby('session_trial_num', group_keys=False).apply(helper.add_trial_time)
    
    return events_processed

# =============================================================================
# POST-METADATA CHANGE PROCESSING FUNCTIONS
# =============================================================================

def process_events_post_meta_change(data_folder, session_info):
    """Process events for sessions after metadata change (post 2024-04-16)."""
    # Load and process events using post-metadata helper functions
    events = pd.read_csv(utils.generate_events_path(data_folder, session_info), low_memory=False)
    events = helper.process_events(session_info, events)
    events_processed = events.groupby('session_trial_num', group_keys=False).apply(helper.add_trial_time)
    
    return events_processed

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def process_session_batch(sessions_df, process_function, batch_name, regenerate):
    """Process a batch of sessions with the specified processing function."""
    total_sessions = len(sessions_df)
    processed_count = 0
    skipped_count = 0
    problematic_sessions = []
    
    for idx, (_, session_info) in enumerate(sessions_df.iterrows(), 1):
        # Check if file already exists
        output_path = utils.generate_events_processed_path(data_folder, session_info)
        
        # Skip existing files unless regenerating
        if not regenerate and os.path.isfile(output_path):
            skipped_count += 1
            continue
        
        # Process the session
        try:
            events_processed = process_function(data_folder, session_info)
            events_processed.to_csv(output_path)
            processed_count += 1
        except:
            problematic_sessions.append(session_info)
        
        # Show progress every 100 sessions
        if idx % 100 == 0 or idx == total_sessions:
            print(f"📊 Progress: {idx}/{total_sessions} | ✅ {processed_count} | ⏭️ {skipped_count} | ❌ {len(problematic_sessions)}")
    
    print(f"\n📊 {batch_name} Complete: {processed_count} processed, {skipped_count} skipped, {len(problematic_sessions)} errors")
    return problematic_sessions

def main(regenerate):
    """Main function to process all sessions before and after metadata change."""
    all_problematic_sessions = []
    
    # Process pre-metadata change sessions
    if len(sessions_pre_meta) > 0:
        print(f"\n🔄 Processing {len(sessions_pre_meta)} sessions before metadata change...")
        pre_problematic = process_session_batch(sessions_pre_meta, process_events_pre_meta_change, "Pre-Metadata Change", regenerate)
        all_problematic_sessions.extend(pre_problematic)
    
    # Process post-metadata change sessions
    if len(sessions_post_meta) > 0:
        print(f"\n🔄 Processing {len(sessions_post_meta)} sessions after metadata change...")
        post_problematic = process_session_batch(sessions_post_meta, process_events_post_meta_change, "Post-Metadata Change", regenerate)
        all_problematic_sessions.extend(post_problematic)
    
    # Show final results
    if all_problematic_sessions:
        print(f"\n❌ Problematic Sessions ({len(all_problematic_sessions)} total):")
        problematic_df = pd.DataFrame(all_problematic_sessions)
        print(problematic_df.to_string())
    else:
        print("\n🎉 All sessions processed successfully!\n")

if __name__ == "__main__":
    regenerate = False
    main(regenerate)
