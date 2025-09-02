#!/usr/bin/env python3
"""
Session Quality Control Script
Backs up raw data, removes test sessions, sessions with corrupted files, 
short sessions, and crashed sessions, validates session directory names 
match metadata, and sorts sessions by experiments.
"""

import os
import json
from session_processing_helper import (
    backup_directory, identify_test_folders, 
    identify_problematic_sessions, identify_short_or_crashed_sessions,
    validate_session_directory_names, delete_sessions, 
    sort_sessions_by_experiments, update_deletion_record
)

def delete_test_folders(data_dir, auto_delete=False):
    """Delete all folders ending with '_test'."""
    test_folders = identify_test_folders(data_dir)
    
    if not test_folders:
        print("No test folders found to delete")
        return
    
    print(f"Found {len(test_folders)} test folders to delete:")
    for folder in test_folders:
        print(f"  - {os.path.basename(folder)}")
    
    if auto_delete or input("Proceed with deletion? (y/N): ").lower() == 'y':
        deletion_count = 0
        for folder in test_folders:
            try:
                import shutil
                shutil.rmtree(folder)
                deletion_count += 1
                print(f"Deleted: {os.path.basename(folder)}")
            except Exception as e:
                print(f"Error deleting {os.path.basename(folder)}: {e}")
        print(f"Total test folders deleted: {deletion_count}")
    else:
        print("Deletion cancelled")

def main(auto_delete=False):
    """Main quality control workflow."""
    data_dir = '/Users/rebekahzhang/data/behavior_data/raw'
    
    with open('exp_cohort_info.json', 'r') as f:
        exp_info = json.load(f)['experiments']

    print("\n=== Session Quality Control Script ===")
    print(f"Data directory: {data_dir}")
    print(f"Found {len(exp_info)} experiments: {list(exp_info.keys())}")
    if auto_delete:
        print("AUTO DELETE MODE: All deletions will proceed automatically\n")
    else:
        print()
    
    deletion_records = []
    
    # Step 1: Backup raw data
    print("\nStep 1: Backing up raw data...")
    if not backup_directory(data_dir):
        print("Backup failed. Aborting.")
        return
    
    # Step 2: Delete test folders
    print("\nStep 2: Removing test folders...")
    delete_test_folders(data_dir, auto_delete)
    
    # Step 3: Check session file quality and clean problematic sessions
    print("\nStep 3: Checking session file quality and cleaning problematic sessions...")
    problematic_sessions = identify_problematic_sessions(data_dir)
    deletion_df_3 = delete_sessions(problematic_sessions, data_dir, "problematic sessions", auto_delete)
    if not deletion_df_3.empty:
        deletion_records.append(deletion_df_3)
    
    # Step 4: Identify and clean short/crashed sessions
    print("\nStep 4: Identifying and cleaning short sessions and crashed sessions...")
    short_crashed_sessions = identify_short_or_crashed_sessions(data_dir, short_threshold=20)
    deletion_df_4 = delete_sessions(short_crashed_sessions, data_dir, "short/crashed sessions", auto_delete)
    if not deletion_df_4.empty:
        deletion_records.append(deletion_df_4)
    
    # Update deletion record
    update_deletion_record(data_dir, deletion_records)
    
    # Step 5: Validate session directory names match metadata
    print("\nStep 5: Validating session directory names match metadata...")
    mismatched_df = validate_session_directory_names(data_dir)
    if not mismatched_df.empty:
        print(f"Found {len(mismatched_df)} sessions with mismatched directory names:")
        print(mismatched_df[['actual_dir', 'meta_dir', 'total_reward', 'total_trial', 'avg_tw']].to_string(index=False))
        print("Please fix the mismatched sessions before proceeding.\n")
        return
    else:
        print("All session directory names match their metadata.")
    
    # Step 6: Sort sessions by experiments
    print("\nStep 6: Sorting sessions by experiments...")
    sort_sessions_by_experiments(data_dir, exp_info)
    
    print("\nSession quality control completed!\n")

if __name__ == "__main__":
    import sys
    # Check if auto delete mode is enabled via command line argument
    auto_delete = True
    main(auto_delete)