import os
import shutil
import pandas as pd

import processing_helper_0827 as helper_pre
import session_processing_helper as helper_post
import utils

# Configuration
data_dir = '/Users/rebekahzhang/data/behavior_data'
exp = 'exp2'
data_folder = os.path.join(data_dir, exp)
meta_change_date = '2024-04-16'
second_sessions_dir = os.path.join(data_dir, 'exp2_second_sessions')

# Create destination directory for second sessions if it doesn't exist
os.makedirs(second_sessions_dir, exist_ok=True)

def stitch_sessions(session_1, session_2):
    """Stitch two sessions together with proper time and trial offsets"""
    session_1_basics = helper_post.get_session_basics(session_1)
    time_offset = session_1_basics['session_time']
    block_offset = session_1_basics['num_blocks']
    trial_offset = session_1_basics['num_trials']
    
    # Apply offsets to session 2
    session_2_copy = session_2.copy()
    session_2_copy['session_time'] = session_2_copy['session_time'] + time_offset
    session_2_copy['block_num'] = session_2_copy['block_num'] + block_offset
    session_2_copy['session_trial_num'] = session_2_copy['session_trial_num'] + trial_offset

    # Concatenate sessions
    stitched_session = pd.concat([session_1, session_2_copy], ignore_index=True)
    return stitched_session

def stitch_events_for_period(data_folder, sessions_period, second_sessions_dir, period_name):
    """
    Stitch events from sessions for a specific time period.
    Moves second sessions to a separate directory instead of deleting them.
    
    Args:
        data_folder: Path to the main data folder
        sessions_period: DataFrame of sessions for the period
        second_sessions_dir: Directory to move second sessions to
        period_name: Name of the period for logging (e.g., "pre-meta-change", "post-meta-change")
    """
    print(f"\n{'='*60}")
    print(f"Processing {period_name} sessions...")
    print(f"{'='*60}")
    
    # Split sessions by date
    sessions_by_date = sessions_period.groupby('date')
    mouse_list = sessions_period['mouse'].unique()
    
    days_to_stitch = []
    mice_to_stitch = []
    for date, data in sessions_by_date:
        for mouse in mouse_list:
            mouse_by_date = data.loc[data['mouse'] == mouse]
            if len(mouse_by_date) > 1:
                days_to_stitch.append(date)
                mice_to_stitch.append(mouse)
                print(f"on {date}, {mouse} has {len(mouse_by_date)} sessions")
    
    if not days_to_stitch:
        print(f"No {period_name} sessions to stitch!")
        return
    
    print(f"Found {len(days_to_stitch)} days with multiple sessions to stitch")
    
    # Stitch sessions from the same mouse on the same day
    for d, m in zip(days_to_stitch, mice_to_stitch):
        day = sessions_by_date.get_group(d)
        sessions_to_stitch = day[day['mouse'] == m]
        
        print(f"\nProcessing {len(sessions_to_stitch)} sessions for {m} on {d}")
        
        if len(sessions_to_stitch) < 2:
            print(f"Only {len(sessions_to_stitch)} session(s) for {m} on {d}, skipping")
            continue
            
        # Start with the first session as our base
        base_session_dir = utils.generate_events_processed_path(data_folder, sessions_to_stitch.iloc[0])
        base_session = pd.read_csv(base_session_dir)
        
        # Iteratively stitch each additional session to the base
        for i in range(1, len(sessions_to_stitch)):
            current_session_dir = utils.generate_events_processed_path(data_folder, sessions_to_stitch.iloc[i])
            
            if os.path.exists(base_session_dir) and os.path.exists(current_session_dir):
                current_session = pd.read_csv(current_session_dir)
                
                # Stitch current session to the base
                stitched_session = stitch_sessions(base_session, current_session)
                
                # Update the base session with the stitched result
                stitched_session.to_csv(base_session_dir, index=False)
                base_session = stitched_session
                
                # Move the current session folder to backup directory
                source_path = os.path.join(data_folder, sessions_to_stitch.iloc[i].dir)
                dest_path = os.path.join(second_sessions_dir, sessions_to_stitch.iloc[i].dir)
                shutil.move(source_path, dest_path)
                print(f"  ✓ {d} {m} session {i+1} stitched and moved to backup")
            else:
                print(f"  ❌ Session files not found for {m} on {d}")
                break
        
        print(f"  ✓ Completed stitching {len(sessions_to_stitch)} sessions for {m} on {d}")
    
    print(f"\n{period_name} processing complete!")

def stitch_all_events(data_folder, meta_change_date, second_sessions_dir):
    """
    Main function to stitch all events from both pre and post meta-change periods.
    
    Args:
        data_folder: Path to the main data folder
        meta_change_date: Date of metadata change (YYYY-MM-DD)
        second_sessions_dir: Directory to move second sessions to
    """
    print("🧵 Starting comprehensive events stitching process...")
    print(f"📁 Data folder: {data_folder}")
    print(f"📅 Meta change date: {meta_change_date}")
    print(f"💾 Backup directory: {second_sessions_dir}")
    
    # Generate all sessions
    print("\n📊 Generating session list...")
    sessions_all = helper_pre.generate_sessions_all(data_folder)
    print(f"Found {len(sessions_all)} total sessions")
    
    # Split sessions by metadata change date
    print(f"\n📅 Splitting sessions by meta-change date: {meta_change_date}")
    
    # Pre-meta-change sessions (before the date)
    sessions_pre_meta = sessions_all.loc[
        (sessions_all.training == 'regular') & 
        (sessions_all.date < meta_change_date)
    ].reset_index()
    
    # Post-meta-change sessions (on or after the date)
    sessions_post_meta = sessions_all.loc[
        (sessions_all.training == 'regular') & 
        (sessions_all.date >= meta_change_date)
    ].reset_index()
    
    print(f"Pre-meta-change sessions: {len(sessions_pre_meta)}")
    print(f"Post-meta-change sessions: {len(sessions_post_meta)}")
    
    # Process pre-meta-change sessions
    if len(sessions_pre_meta) > 0:
        stitch_events_for_period(data_folder, sessions_pre_meta, second_sessions_dir, "Pre-meta-change")
    else:
        print("\nNo pre-meta-change sessions found.")
    
    # Process post-meta-change sessions
    if len(sessions_post_meta) > 0:
        stitch_events_for_period(data_folder, sessions_post_meta, second_sessions_dir, "Post-meta-change")
    else:
        print("\nNo post-meta-change sessions found.")
    
    print(f"\n{'='*60}")
    print("🎉 COMPREHENSIVE EVENTS STITCHING COMPLETE!")
    print(f"{'='*60}")
    print(f"All stitched sessions are in: {data_folder}")
    print(f"All backup sessions are in: {second_sessions_dir}")

if __name__ == "__main__":
    # Backup the data folder before processing
    print("🔄 Creating backup before processing...")
    utils.backup(data_folder)
    
    # Run the comprehensive stitching
    stitch_all_events(data_folder, meta_change_date, second_sessions_dir)
