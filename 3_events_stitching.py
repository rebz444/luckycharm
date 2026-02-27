import os
import math
import pandas as pd
import shutil
import warnings

import session_processing_helper as helper
import utils

# Suppress pandas FutureWarning about groupby.apply
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DataFrameGroupBy.apply operated on the grouping columns.*')

# Configuration
data_dir = '/Users/rebekahzhang/data/behavior_data'
exp = 'exp2'
data_folder = os.path.join(data_dir, exp)
meta_change_date = '2024-04-16'
second_sessions_dir = os.path.join(data_dir, 'exp2_second_sessions')

# ===== PRE-META CHANGE FUNCTIONS =====

def get_max_trial_num_pre(events):
    max_trial_num = events['session_trial_num'].max()
    last_trial = events.loc[events['session_trial_num'] == max_trial_num]
    session_end = last_trial.loc[(last_trial['key'] == 'session') & (last_trial['value'] == 0)]
    last_trial_end = last_trial.loc[(last_trial['key'] == 'trial') & (last_trial['value'] == 0)]
    if len(session_end) > 0 and len(last_trial_end) > 0:
        return max_trial_num
    else:
        return max_trial_num - 1

def generate_trials_pre(events, max_trial_num):
    trial_info_list = []
    for t in range(int(max_trial_num)+1):
        trial = events.loc[events['session_trial_num'] == t]
        trial_basics = helper.get_trial_basics(trial)
        trial_info_list.append(trial_basics)
    trials = pd.DataFrame(trial_info_list)
    return trials

def align_trial_number_pre(session, trials):
    for _, trial_basics in trials.iterrows():
        session.loc[session['session_time'].between(trial_basics['start_time'], trial_basics['end_time']),
                'block_num'] = trial_basics['block_num']
        session.loc[session['session_time'].between(trial_basics['start_time'], trial_basics['end_time']),
                'session_trial_num'] = trial_basics['session_trial_num']
        session.loc[session['session_time'].between(trial_basics['start_time'], trial_basics['end_time']),
                'block_trial_num'] = trial_basics['block_trial_num']
    return session

def align_trial_states_pre(trial):
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

def stitch_sessions_pre(session_1, session_2):
    session_1_basics = helper.get_session_basics(session_1)
    time_offset = session_1_basics['session_time']
    block_offset = session_1_basics['num_blocks']
    trial_offset = session_1_basics['num_trials']

    session_2.session_time = session_2.session_time + time_offset
    session_2.block_num = session_2.block_num + block_offset
    session_2.session_trial_num= session_2.session_trial_num + trial_offset

    stitched_session = pd.concat([session_1, session_2])
    return stitched_session

# ===== POST-META CHANGE FUNCTIONS =====

def stitch_sessions_post(session_1, session_2):
    """Use the helper function for post-meta change stitching"""
    return helper.stitch_sessions(session_1, session_2)

# ===== STITCHING FUNCTIONS =====

def get_sessions_to_stitch(sessions_by_date, mouse_list):
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
        print("no sessions to stitch!")
    return days_to_stitch, mice_to_stitch

def stitching_combined(days_to_stitch, mice_to_stitch, sessions_by_date, is_pre_meta=True):
    """Combined stitching function that handles both pre and post meta change data"""
    if not days_to_stitch:
        return

    stitch_func = stitch_sessions_pre if is_pre_meta else stitch_sessions_post
    period_name = "pre-meta change" if is_pre_meta else "post-meta change"

    print(f"\n=== Processing {period_name} sessions ===")

    for d, m in zip(days_to_stitch, mice_to_stitch):
        day = sessions_by_date.get_group(d)
        sessions_to_stitch = day[day['mouse'] == m]
        num_sessions = len(sessions_to_stitch)

        print(f"\nProcessing {num_sessions} sessions for {m} on {d}")

        if num_sessions < 2:
            print(f"Only {num_sessions} session(s) for {m} on {d}, skipping")
            continue

        base_session_dir = utils.generate_events_processed_path(data_folder, sessions_to_stitch.iloc[0])
        base_session = pd.read_csv(base_session_dir)

        for i in range(1, num_sessions):
            current_session_dir = utils.generate_events_processed_path(data_folder, sessions_to_stitch.iloc[i])

            if os.path.exists(base_session_dir) and os.path.exists(current_session_dir):
                current_session = pd.read_csv(current_session_dir)
                stitched_session = stitch_func(base_session, current_session)

                stitched_session.to_csv(base_session_dir, index=False)
                base_session = stitched_session

                source_path = os.path.join(data_folder, sessions_to_stitch.iloc[i].dir)
                dest_path = os.path.join(second_sessions_dir, sessions_to_stitch.iloc[i].dir)
                shutil.move(source_path, dest_path)
                print(f"  ✓ {d} {m} session {i+1} stitched and moved to backup")
            else:
                print(f"  ❌ Session files not found for {m} on {d}")
                break

        print(f"  ✓ Completed stitching {num_sessions} sessions for {m} on {d}")

def process_stitching():
    """Stitch together multiple sessions from the same mouse on the same day."""
    utils.backup(data_folder)

    sessions_all = pd.read_csv(os.path.join(data_folder, 'sessions_all_pre_processing.csv'))

    print("\n" + "="*50)
    print("PROCESSING PRE-META CHANGE SESSIONS")
    print("="*50)

    sessions_pre_meta = sessions_all.loc[
        (sessions_all.training == 'regular') &
        (sessions_all.version == 'pre')
    ].reset_index()

    if not sessions_pre_meta.empty:
        sessions_by_date_pre = sessions_pre_meta.groupby('date')
        mouse_list_pre = sessions_pre_meta['mouse'].unique()
        days_to_stitch_pre, mice_to_stitch_pre = get_sessions_to_stitch(sessions_by_date_pre, mouse_list_pre)
        stitching_combined(days_to_stitch_pre, mice_to_stitch_pre, sessions_by_date_pre, is_pre_meta=True)
    else:
        print("No pre-meta change sessions found.")

    print("\n" + "="*50)
    print("PROCESSING POST-META CHANGE SESSIONS")
    print("="*50)

    sessions_post_meta = sessions_all.loc[
        (sessions_all.training == 'regular') &
        (sessions_all.version == 'post')
    ].reset_index()

    if not sessions_post_meta.empty:
        sessions_by_date_post = sessions_post_meta.groupby('date')
        mouse_list_post = sessions_post_meta['mouse'].unique()
        days_to_stitch_post, mice_to_stitch_post = get_sessions_to_stitch(sessions_by_date_post, mouse_list_post)
        stitching_combined(days_to_stitch_post, mice_to_stitch_post, sessions_by_date_post, is_pre_meta=False)
    else:
        print("No post-meta change sessions found.")

    print("\n" + "="*50)
    print("ALL SESSIONS STITCHED")
    print("="*50)

# ===== SESSION LOG FUNCTIONS =====

def correct_sessions_training(save_log=True):
    """Generate and correct sessions training data for both pre and post meta change."""
    print("\nCorrecting sessions training data...")
    sessions_all = helper.generate_sessions_all(data_folder)
    sessions_training = sessions_all.loc[sessions_all.training == 'regular'].reset_index()

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

    session_basics_df = pd.DataFrame(session_basics_list)
    corrected_sessions_training = pd.merge(sessions_training, session_basics_df, on="dir")

    columns_to_drop = ['index', 'total_reward', 'total_trial']
    corrected_sessions_training = corrected_sessions_training.drop(columns=[col for col in columns_to_drop if col in corrected_sessions_training.columns])
    corrected_sessions_training['group'] = corrected_sessions_training['exp'].str.extract(r'exp\d+_(short|long)').replace({'short': 's', 'long': 'l'})
    corrected_sessions_training = corrected_sessions_training.groupby('mouse', group_keys=False).apply(helper.assign_session_numbers)

    if save_log:
        utils.save_log(corrected_sessions_training, data_folder, f'sessions_training_{exp}.csv')
        print("Session log saved\n")

    return corrected_sessions_training

# ===== MAIN EXECUTION =====

if __name__ == "__main__":
    process_stitching()
    correct_sessions_training()
