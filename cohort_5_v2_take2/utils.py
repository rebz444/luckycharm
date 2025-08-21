import os
import shutil

import pandas as pd

# =============================================================================
# File and folder operations
# =============================================================================

def delete_folders(folder_list, data_folder):
    """Delete folders from the given list."""
    if folder_list:
        for folder in folder_list:
            full_path = os.path.join(data_folder, folder)
            if os.path.exists(full_path):
                shutil.rmtree(full_path)
                print(f"Deleted folder: {full_path}")
            else:
                print(f"Folder not found: {full_path}")
    else:
        print("no sessions to delete")

def backup(source_dir):
    """Create a copy for source_dir in the same path parallel to source_dir."""
    data_folder = os.path.dirname(source_dir)
    source_name = os.path.basename(source_dir)
    destination_dir = os.path.join(data_folder, f"{source_name}_copy")
    if not os.path.isdir(destination_dir):
        shutil.copytree(source_dir, destination_dir)
        print(f"{os.path.basename(source_dir)} backed up")
    else:
        print(f"{os.path.basename(destination_dir)} already exist")

def remove_sessions(sessions_to_remove_df, data_folder):
    """Remove sessions based on the provided DataFrame."""
    if len(sessions_to_remove_df) == 0:
        print('no sessions to delete')
    else:
        for _, session_info in sessions_to_remove_df.iterrows():
            shutil.rmtree(os.path.join(data_folder, session_info.dir))
            print(f'{session_info.dir} deleted')

# =============================================================================
# Data loading functions
# =============================================================================

def load_data(path):
    """Load data from CSV file with index column."""
    df = pd.read_csv(path, index_col=0)
    return df

def load_session_log(data_folder, log_name):
    """Load session log data."""
    path = os.path.join(data_folder, log_name)
    session_log = pd.read_csv(path, index_col=0)
    return session_log

# =============================================================================
# Data saving functions
# =============================================================================

def save_log(log_df, data_folder, filename):
    """Save log DataFrame to CSV without index."""
    path = os.path.join(data_folder, filename)
    log_df.to_csv(path, index=False)

def save_as_csv(df, folder, filename):
    """Save DataFrame to CSV with index."""
    path = os.path.join(folder, filename)
    df.to_csv(path)

# =============================================================================
# Path generation functions
# =============================================================================

def generate_events_path(data_folder, session_info):
    """Generate path for events file."""
    return os.path.join(data_folder, session_info.dir, f'events_{session_info.dir}.txt')

def generate_events_processed_path(data_folder, session_info):
    """Generate path for processed events file."""
    filename = f'events_processed_{session_info.dir}.csv'
    return os.path.join(data_folder, session_info.dir, filename)

def generate_trials_path(data_folder, session_info):
    """Generate path for trials file."""
    filename = f'trials_{session_info.dir}.csv'
    return os.path.join(data_folder, session_info.dir, filename)

def generate_trials_analyzed_path(data_folder, session_info):
    """Generate path for analyzed trials file."""
    filename = f'trials_analyzed_{session_info.dir}.csv'
    return os.path.join(data_folder, session_info.dir, filename)

# =============================================================================
# Data analysis and utility functions
# =============================================================================

def generate_mouse_list(session_log):
    """Generate sorted list of unique mouse IDs."""
    mouse_list = session_log.mouse.unique().tolist()
    mouse_list.sort()
    return mouse_list

def select_good_trials(all_trials):
    """Select only good trials from the dataset."""
    good_trials = all_trials.loc[all_trials['good_trial'] == True]
    return good_trials
