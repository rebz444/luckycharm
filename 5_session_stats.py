"""
Session Statistics Analysis for LuckyCharm Experiment
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

import utils

# Setup
data_dir = '/Users/rebekahzhang/data/behavior_data'
exp = "exp2"
data_folder = os.path.join(data_dir, exp)
figure_folder = os.path.join(data_dir, exp, 'timing_analysis')
os.makedirs(figure_folder, exist_ok=True)

two_colors = ["#ffb400", "#9080ff"]
custom_palette = {'s': two_colors[0], 'l': two_colors[1]}

# Load and process data
sessions_training = utils.load_session_log(data_folder, f'sessions_training_{exp}.csv')

def get_lick_times(trial):
    licks = trial.loc[trial['key'] == "lick", "trial_time"].reset_index(drop=True)
    return {
        "first_lick": licks.iloc[0] if len(licks) > 0 else None,
        "second_lick": licks.iloc[1] if len(licks) > 1 else None,
        "third_lick": licks.iloc[2] if len(licks) > 2 else None,
        "fourth_lick": licks.iloc[3] if len(licks) > 3 else None,
    }

def process_trials_data(trials_list, sessions_training):
    # Process trials data
    trials_list = []
    for _, session_info in sessions_training.iterrows():
        trials = utils.load_data(utils.generate_trials_analyzed_path(data_folder, session_info))
        trials['dir'] = session_info['dir']
        events = utils.load_data(utils.generate_events_processed_path(data_folder, session_info))
        
        lick_times_dict = {}
        for t, trial_events in events.groupby("session_trial_num"):
            lick_times_dict[t] = get_lick_times(trial_events)
        
        lick_times_df = pd.DataFrame.from_dict(lick_times_dict, orient='index')
        lick_times_df['session_trial_num'] = lick_times_df.index
        trials_with_lick_time = trials.merge(lick_times_df, on='session_trial_num', how='left')
        trials_list.append(trials_with_lick_time)

    trials_training = pd.concat(trials_list, ignore_index=True)
    sessions_info_to_merge = sessions_training[['dir', 'mouse', 'date', 'time', 'avg_tw', 'session']]
    trials_training_analyzed = trials_training.merge(sessions_info_to_merge, on='dir', how='left')
    trials_training_analyzed.to_csv(os.path.join(data_folder, 'trials_training_analyzed.csv'))

    # Clean data for analysis
    bg_vs_tw = trials_training_analyzed[trials_training_analyzed['time_waited'].notna()].copy()
    bg_vs_tw['bg_length_rounded'] = bg_vs_tw['bg_length'].round(1)
    bg_vs_tw['period'] = bg_vs_tw['session'] // 10
    bg_vs_tw.loc[bg_vs_tw['period'] == 10, 'period'] = 9
    bg_vs_tw['z_time'] = bg_vs_tw['bg_length'] + bg_vs_tw['time_waited']
    bg_vs_tw.to_csv(os.path.join(data_folder, 'trials_bg_vs_tw_analysis.csv'))

# Remove outliers for LMEM
def remove_outliers(df, group_col='group', value_col='time_waited'):
    filtered_data = []
    for group_name in df[group_col].unique():
        group_df = df[df[group_col] == group_name]
        Q1, Q3 = group_df[value_col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        bounds = [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
        filtered_data.append(group_df[(group_df[value_col] >= bounds[0]) & (group_df[value_col] <= bounds[1])])
    return pd.concat(filtered_data, ignore_index=True)



trials_training_analyzed = utils.load_data(os.path.join(data_folder, 'trials_training_analyzed.csv'))
df_filtered = remove_outliers(trials_training_analyzed)
mouse_avg = df_filtered.groupby(['mouse', 'group'])['time_waited'].mean().reset_index()
mice_to_keep = []
for group in mouse_avg['group'].unique():
    group_avg = mouse_avg[mouse_avg['group'] == group]
    Q1, Q3 = group_avg['time_waited'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    bounds = [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
    mice_to_keep.extend(group_avg[(group_avg['time_waited'] >= bounds[0]) & (group_avg['time_waited'] <= bounds[1])]['mouse'].tolist())
df_clean = df_filtered[df_filtered['mouse'].isin(mice_to_keep)]

# LMEM Analysis
print("Running LMEM Analysis...")

# Linear model
linear_model = smf.mixedlm("time_waited ~ C(group) * session", data=df_clean, groups=df_clean['mouse']).fit()
print("Linear Model:")
print(linear_model.summary())

# Quadratic model  
quadratic_model = smf.mixedlm("time_waited ~ C(group) * session + C(group) * I(session**2)", data=df_clean, groups=df_clean['mouse']).fit()
print("\nQuadratic Model:")
print(quadratic_model.summary())

# Model comparison
print(f"\nModel Comparison:")
print(f"Linear - AIC: {linear_model.aic:.2f}, BIC: {linear_model.bic:.2f}")
print(f"Quadratic - AIC: {quadratic_model.aic:.2f}, BIC: {quadratic_model.bic:.2f}")

# Z-time analysis
print(f"\nZ-Time Analysis:")
print(f"Mean: {bg_vs_tw['z_time'].mean():.3f} ± {bg_vs_tw['z_time'].std():.3f}")
for group in ['s', 'l']:
    group_data = bg_vs_tw[bg_vs_tw['group'] == group]['z_time']
    print(f"{group.upper()}: {group_data.mean():.3f} ± {group_data.std():.3f}")

print(f"\nAnalysis complete! Data shapes:")
print(f"Raw trials: {len(trials_training_analyzed)}")
print(f"Clean trials: {len(df_clean)}")
print(f"BG vs TW: {len(bg_vs_tw)}")