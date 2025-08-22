#!/usr/bin/env python3
"""
Session Quality Control Script
Backs up raw data
removes test sessions, sessions with corrupted files, short sessions, and crashed sessions
sorts sessions by experiments
"""

import os
import shutil
import json
import pandas as pd

def backup_directory(source_path):
    """Create backup or update existing with new sessions"""
    backup_path = source_path + "_backup"
    
    if not os.path.exists(backup_path):
        try:
            shutil.copytree(source_path, backup_path)
            print(f"Backup created at {backup_path}")
            return backup_path
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None
    else:
        # Update existing backup with new sessions
        print(f"Backup exists at {backup_path}, checking for new sessions...")
        new_sessions = []
        
        for item in os.listdir(source_path):
            source_item = os.path.join(source_path, item)
            backup_item = os.path.join(backup_path, item)
            
            if os.path.isdir(source_item) and not os.path.exists(backup_item):
                try:
                    shutil.copytree(source_item, backup_item)
                    new_sessions.append(item)
                    print(f"Added to backup: {item}")
                except Exception as e:
                    print(f"Error copying {item} to backup: {e}")
        
        if new_sessions:
            print(f"Updated backup with {len(new_sessions)} new sessions")
        else:
            print("Backup is up-to-date")
        
        return backup_path

def delete_test_folders(data_dir):
    """Delete all folders ending with '_test'"""
    test_folders = [os.path.join(data_dir, item) for item in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, item)) and item.endswith('_test')]
    
    if not test_folders:
        print("No test folders found to delete")
        return
    
    print(f"Found {len(test_folders)} test folders to delete:")
    if input("Proceed with deletion? (y/N): ").lower() == 'y':
        for folder in test_folders:
            try:
                shutil.rmtree(folder)
                print(f"Deleted: {os.path.basename(folder)}")
            except Exception as e:
                print(f"Error deleting {os.path.basename(folder)}: {e}")
    else:
        print("Deletion cancelled")

def check_and_clean_sessions_with_corrupted_files(data_folder):
    """Check session files and delete problematic sessions"""
    files_check = []
    for entry in os.scandir(data_folder):
        if entry.is_dir():
            dir_name = entry.name
            session_path = os.path.join(data_folder, dir_name)
            events_found = False
            meta_found = False
            events_empty = True
            meta_empty = True
            
            required_files = [f for f in os.scandir(session_path) if f.is_file() and not f.name.startswith('.')]
            
            for file in required_files:
                if file.name.startswith("events_"):
                    events_found = True
                    if file.stat().st_size > 0:
                        events_empty = False
                elif file.name.startswith("meta_"):
                    meta_found = True
                    if file.stat().st_size > 0:
                        meta_empty = False
            
            files_check.append({
                'dir': entry.name, 'events': events_found, 'meta': meta_found,
                'events_empty': events_empty if events_found else None,
                'meta_empty': meta_empty if meta_found else None
            })
    
    files_check_df = pd.DataFrame(files_check).sort_values("dir")
    missing_meta = files_check_df[files_check_df.meta == False]
    missing_events = files_check_df[files_check_df.events == False]
    empty_meta = files_check_df[(files_check_df.meta == True) & (files_check_df.meta_empty == True)]
    empty_events = files_check_df[(files_check_df.events == True) & (files_check_df.events_empty == True)]
    
    problematic_sessions = pd.concat([
        missing_meta[['dir']].assign(reason='Missing meta file'),
        missing_events[['dir']].assign(reason='Missing events file'),
        empty_meta[['dir']].assign(reason='Empty meta file'),
        empty_events[['dir']].assign(reason='Empty events file')
    ]).reset_index(drop=True)
    
    if not problematic_sessions.empty:
        print(f"\nFound {len(problematic_sessions)} problematic sessions to delete:")
        print(problematic_sessions.to_string(index=False))
        
        if input("\nProceed with deletion? (y/N): ").lower() == 'y':
            deletion_record = []
            for _, row in problematic_sessions.iterrows():
                session_dir = os.path.join(data_folder, row['dir'])
                if os.path.exists(session_dir):
                    try:
                        shutil.rmtree(session_dir)
                        deletion_record.append({
                            'session': row['dir'], 'reason': row['reason'],
                            'deleted': True, 'timestamp': pd.Timestamp.now()
                        })
                        print(f"Deleted: {row['dir']} - {row['reason']}")
                    except Exception as e:
                        print(f"Error deleting {row['dir']}: {e}")
        else:
            print("Deletion cancelled")
    else:
        print("No problematic sessions found - all sessions have valid files.")
    
    # Create deletion record DataFrame
    deletion_df = pd.DataFrame(deletion_record)
    
    return missing_meta, missing_events, empty_meta, empty_events, deletion_df

def identify_and_clean_short_or_crashed_sessions(data_folder, short_threshold=20):
    """Identify and delete short/crashed sessions by reading events files once"""
    short_sessions, crashed_sessions = [], []
    
    for session_dir in [entry.name for entry in os.scandir(data_folder) if entry.is_dir()]:
        try:
            events_files = [f for f in os.listdir(os.path.join(data_folder, session_dir)) 
                           if f.startswith('events_') and f.endswith('.txt')]
            events = pd.read_csv(os.path.join(data_folder, session_dir, events_files[0]), low_memory=False)
            
            # Check for short sessions
            max_trial_num = events['session_trial_num'].max()
            total_trials = max_trial_num + 1 if pd.notna(max_trial_num) else 0
            if pd.isna(total_trials) or total_trials < short_threshold:
                short_sessions.append({'dir': session_dir, 'total_trials': total_trials, 'reason': 'Short'})
            
            # Check for crashed sessions
            session_end = events.loc[(events['key'] == 'session') & (events['value'] == 0)]
            if len(session_end) != 1:
                crashed_sessions.append({'dir': session_dir, 'reason': 'Crashed'})
                
        except Exception as e:
            short_sessions.append({'dir': session_dir, 'total_trials': 'Error', 'reason': f'Cannot read file: {str(e)}'})
    
    all_problematic = short_sessions + crashed_sessions
    if not all_problematic:
        print("No short or crashed sessions found.")
        return pd.DataFrame()
    
    # Remove duplicates and combine reasons
    unique_sessions = {}
    for session in all_problematic:
        if session['dir'] not in unique_sessions:
            unique_sessions[session['dir']] = session
        else:
            unique_sessions[session['dir']]['reason'] = f"{unique_sessions[session['dir']]['reason']}; {session['reason']}"
    
    problematic_sessions_df = pd.DataFrame(list(unique_sessions.values()))
    print(f"\nFound {len(problematic_sessions_df)} problematic sessions to delete:")
    print(problematic_sessions_df.to_string(index=False))
    
    if input("\nProceed with deletion? (y/N): ").lower() == 'y':
        deletion_record = []
        for _, row in problematic_sessions_df.iterrows():
            session_dir = os.path.join(data_folder, row['dir'])
            if os.path.exists(session_dir):
                try:
                    shutil.rmtree(session_dir)
                    deletion_record.append({
                        'session': row['dir'], 'reason': row['reason'],
                        'deleted': True, 'timestamp': pd.Timestamp.now()
                    })
                    print(f"Deleted: {row['dir']} - {row['reason']}")
                except Exception as e:
                    print(f"Error deleting {row['dir']}: {e}")
        else:
            print("Deletion cancelled")
    
    return pd.DataFrame(deletion_record)

def sort_sessions_by_experiments(data_dir, exp_info):
    """Sort sessions into experiment folders based on mouse names"""
    # Get the parent directory of raw folder (behavior_data)
    parent_dir = os.path.dirname(data_dir)
    exp_folders = {exp_name: os.path.join(parent_dir, exp_name) for exp_name in exp_info.keys()}
    
    for exp_path in exp_folders.values():
        os.makedirs(exp_path, exist_ok=True)
        print(f"Created experiment folder: {exp_path}")
    
    moved_count = 0
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if not os.path.isdir(item_path) or item in exp_info.keys():
            continue
        
        if len(item.split('_')) == 3:
            _, _, mouse_name = item.split('_')
            for exp_name, mice in exp_info.items():
                if mouse_name in mice:
                    try:
                        shutil.move(item_path, os.path.join(exp_folders[exp_name], item))
                        moved_count += 1
                        break
                    except Exception as e:
                        print(f"Error moving {item}: {e}")
            else:
                print(f"No matching experiment found for {item}")
    
    print(f"Total sessions moved: {moved_count}")

def update_deletion_record(data_dir, deletion_dfs):
    """Update deletion record CSV file, appending to existing or creating new"""
    deletion_csv_path = os.path.join(data_dir, 'deletion_record.csv')
    all_deletions = [df for df in deletion_dfs if not df.empty]
    
    if all_deletions:
        combined_df = pd.concat(all_deletions, ignore_index=True)
        if os.path.exists(deletion_csv_path):
            pd.concat([pd.read_csv(deletion_csv_path), combined_df], ignore_index=True).to_csv(deletion_csv_path, index=False)
            print(f"Appended to existing: {deletion_csv_path}")
        else:
            combined_df.to_csv(deletion_csv_path, index=False)
            print(f"Created new: {deletion_csv_path}")

def main():
    data_dir = '/Users/rebekahzhang/data/behavior_data/raw'
    with open('exp_cohort_info.json', 'r') as f:
        exp_info = json.load(f)['experiments']

    print("=== Session Quality Control Script ===")
    print(f"Data directory: {data_dir}")
    print(f"Found {len(exp_info)} experiments: {list(exp_info.keys())}\n")
    
    # Step 1: Backup raw data
    print("Step 1: Backing up raw data...")
    if not backup_directory(data_dir):
        print("Backup failed. Aborting.")
        return
    print()
    
    # Step 2: Delete test folders
    print("Step 2: Removing test folders...")
    delete_test_folders(data_dir)
    print()
    
    # Step 3: Check session file quality and clean problematic sessions
    print("Step 3: Checking session file quality and cleaning problematic sessions...")
    missing_meta, missing_events, empty_meta, empty_events, deletion_df_3 = check_and_clean_sessions_with_corrupted_files(data_dir)
    print()
    
    # Step 4: Identify and clean both short sessions and crashed sessions
    print("Step 4: Identifying and cleaning short sessions and crashed sessions...")
    deletion_df_4 = identify_and_clean_short_or_crashed_sessions(data_dir, short_threshold=20)
    print()
    
    # Update deletion record
    update_deletion_record(data_dir, [deletion_df_3, deletion_df_4])

    # Step 5: Sort sessions by experiments
    print("Step 5: Sorting sessions by experiments...")
    sort_sessions_by_experiments(data_dir, exp_info)
    print()

    print("Session quality control completed!")

if __name__ == "__main__":
    main()
