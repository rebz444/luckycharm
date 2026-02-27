import json
import os
import shutil

import numpy as np
import pandas as pd

# =============================================================================
# QUALITY CONTROL FUNCTIONS
# =============================================================================

def check_session_files(data_folder):
    """Check session folders for required files and their status."""
    files_check = []

    for entry in os.scandir(data_folder):
        if not entry.is_dir():
            continue

        session_path = os.path.join(data_folder, entry.name)
        files = [f for f in os.scandir(session_path) if f.is_file() and not f.name.startswith('.')]

        events_files = [f for f in files if f.name.startswith("events_")]
        meta_files = [f for f in files if f.name.startswith("meta_")]

        files_check.append({
            'dir': entry.name,
            'events': bool(events_files),
            'meta': bool(meta_files),
            'events_empty': not events_files or all(f.stat().st_size == 0 for f in events_files),
            'meta_empty': not meta_files or all(f.stat().st_size == 0 for f in meta_files)
        })

    if not files_check:
        empty_df = pd.DataFrame(
            columns=["dir", "events", "meta", "events_empty", "meta_empty"]
        )
        return empty_df, empty_df, empty_df, empty_df

    files_check_df = pd.DataFrame(files_check).sort_values("dir")

    return (
        files_check_df[files_check_df.meta == False],
        files_check_df[files_check_df.events == False],
        files_check_df[(files_check_df.meta == True) & (files_check_df.meta_empty == True)],
        files_check_df[(files_check_df.events == True) & (files_check_df.events_empty == True)]
    )

def backup_directory(source_path):
    """Create backup or update existing with new sessions."""
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

            # Skip if this is the backup directory itself
            if item.endswith('_backup'):
                continue

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

def identify_test_folders(data_dir):
    """Identify all folders ending with '_test'."""
    return [os.path.join(data_dir, item) for item in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, item)) and item.endswith('_test')]

def identify_problematic_sessions(data_folder):
    """Identify sessions with missing or empty files."""
    missing_meta, missing_events, empty_meta, empty_events = check_session_files(data_folder)

    problematic_sessions = pd.concat([
        missing_meta[['dir']].assign(reason='Missing meta file'),
        missing_events[['dir']].assign(reason='Missing events file'),
        empty_meta[['dir']].assign(reason='Empty meta file'),
        empty_events[['dir']].assign(reason='Empty events file')
    ]).reset_index(drop=True)

    return problematic_sessions

def identify_short_or_crashed_sessions(data_folder, short_threshold=20):
    """Identify short or crashed sessions by reading events files."""
    short_sessions, crashed_sessions = [], []

    for session_dir in [entry.name for entry in os.scandir(data_folder) if entry.is_dir()]:
        try:
            events_files = [f for f in os.listdir(os.path.join(data_folder, session_dir))
                           if f.startswith('events_') and f.endswith('.txt')]

            if not events_files:
                short_sessions.append({'dir': session_dir, 'total_trials': 'No events file', 'reason': 'Missing events file'})
                continue

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
        return pd.DataFrame()

    # Remove duplicates and combine reasons
    unique_sessions = {}
    for session in all_problematic:
        if session['dir'] not in unique_sessions:
            unique_sessions[session['dir']] = session
        else:
            unique_sessions[session['dir']]['reason'] = f"{unique_sessions[session['dir']]['reason']}; {session['reason']}"

    return pd.DataFrame(list(unique_sessions.values()))

def identify_sensor_bounce_sessions(data_folder, min_median_ili=0.1, max_ili_filter=1.0):
    """
    Identify sessions with sensor bounce artifacts.

    Sensor bounce produces physiologically implausible lick ILIs (< 0.1s / > 10 Hz).
    Computes median of short ILIs (< max_ili_filter) across all lick onsets in session.
    Sessions with median ILI below min_median_ili are flagged.

    Args:
        data_folder: Path to directory containing session subdirectories.
        min_median_ili: ILI threshold in seconds; sessions below this are flagged (default 0.1).
        max_ili_filter: Upper ILI cutoff for median computation (default 1.0s).

    Returns:
        DataFrame with columns ['dir', 'median_ili', 'reason'], or empty DataFrame.
    """
    bounce_sessions = []

    for session_dir in [e.name for e in os.scandir(data_folder) if e.is_dir()]:
        try:
            events_files = [f for f in os.listdir(os.path.join(data_folder, session_dir))
                            if f.startswith('events_') and f.endswith('.txt')]
            if not events_files:
                continue

            events = pd.read_csv(os.path.join(data_folder, session_dir, events_files[0]),
                                 low_memory=False)

            lick_onsets = (events.loc[(events['key'] == 'lick') & (events['value'] == 1),
                                      'session_time']
                                 .sort_values()
                                 .values)

            if len(lick_onsets) < 2:
                continue

            ilis = np.diff(lick_onsets)
            short_ilis = ilis[ilis < max_ili_filter]

            if len(short_ilis) < 2:
                continue

            median_ili = float(np.median(short_ilis))
            if median_ili < min_median_ili:
                bounce_sessions.append({
                    'dir': session_dir,
                    'median_ili': round(median_ili, 4),
                    'reason': f'Sensor bounce (median ILI={round(median_ili, 4)}s)'
                })

        except Exception:
            pass  # Skip unreadable sessions; already caught by identify_problematic_sessions

    return pd.DataFrame(bounce_sessions) if bounce_sessions else pd.DataFrame()

def validate_session_directory_names(data_folder):
    """Check if meta JSON files match their directory names."""
    mismatched_sessions = []

    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.startswith("meta_") and file.endswith(".json"):
                try:
                    with open(os.path.join(root, file)) as f:
                        session_data = json.load(f)

                    # Get config data based on date
                    date_str = file.split('_')[1]
                    config_data = session_data.get('session_config', session_data) if date_str >= '2024-04-16' else session_data

                    # Check if metadata matches directory name
                    meta_dir = f"{config_data['date']}_{config_data['time']}_{config_data['mouse']}"
                    if meta_dir != os.path.basename(root):
                        mismatched_sessions.append({
                            'actual_dir': os.path.basename(root),
                            'meta_dir': meta_dir,
                            **config_data
                        })

                except Exception as e:
                    print(f"Error processing file {file}: {e}")

    return pd.DataFrame(mismatched_sessions).sort_values('meta_dir') if mismatched_sessions else pd.DataFrame()

def delete_sessions(sessions_df, data_folder, session_type="sessions", auto_delete=False):
    """Delete sessions from the filesystem and return deletion record."""
    if sessions_df.empty:
        print(f"No {session_type} found to delete")
        return pd.DataFrame()

    print(f"Found {len(sessions_df)} {session_type} to delete:")
    print(sessions_df.to_string(index=False))

    if auto_delete or input(f"\nProceed with deletion of {session_type}? (y/N): ").lower() == 'y':
        deletion_record = []
        for _, row in sessions_df.iterrows():
            session_dir = os.path.join(data_folder, row['dir'])
            if os.path.exists(session_dir):
                try:
                    shutil.rmtree(session_dir)
                    deletion_record.append({
                        'session': row['dir'],
                        'reason': row.get('reason', f'Deleted {session_type}'),
                        'deleted': True,
                        'timestamp': pd.Timestamp.now()
                    })
                    print(f"Deleted: {row['dir']}")
                except Exception as e:
                    print(f"Error deleting {row['dir']}: {e}")

        print(f"Total {session_type} deleted: {len(deletion_record)}")
        return pd.DataFrame(deletion_record)
    else:
        print("Deletion cancelled")
        return pd.DataFrame()

def sort_sessions_by_experiments(data_dir, exp_info):
    """Sort sessions into experiment folders based on mouse names."""
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
                        print(f"Moved {item} to {exp_name}")
                        break
                    except Exception as e:
                        print(f"Error moving {item}: {e}")
            else:
                print(f"No matching experiment found for {item}")

    print(f"Total sessions moved: {moved_count}")
    return moved_count

def update_deletion_record(data_dir, deletion_dfs):
    """Update deletion record CSV file, appending to existing or creating new."""
    deletion_csv_path = os.path.join(data_dir, 'deletion_record.csv')
    all_deletions = [df for df in deletion_dfs if not df.empty]

    if all_deletions:
        combined_df = pd.concat(all_deletions, ignore_index=True)
        if os.path.exists(deletion_csv_path):
            existing_df = pd.read_csv(deletion_csv_path)
            pd.concat([existing_df, combined_df], ignore_index=True).to_csv(deletion_csv_path, index=False)
            print(f"Appended to existing: {deletion_csv_path}")
        else:
            combined_df.to_csv(deletion_csv_path, index=False)
            print(f"Created new: {deletion_csv_path}")
    else:
        print("No deletions to record")
