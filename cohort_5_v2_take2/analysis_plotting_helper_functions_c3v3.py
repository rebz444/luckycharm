def get_session_performance(all_trials):
    num_miss_trials = all_trials.miss_trial.sum()
    num_good_trials = all_trials.good_trial.sum()
    return [num_miss_trials, num_good_trials]

def get_session_mistakes(all_trials):
    num_bg_repeats_mean = all_trials.bg_repeats.mean()
    num_bg_repeats_med = all_trials.bg_repeats.median()
    num_bg_repeats_std = all_trials.bg_repeats.std()
    return [num_bg_repeats_mean, num_bg_repeats_med, num_bg_repeats_std]

def get_session_tw(all_trials):
    tw_mean = all_trials.time_waited.mean()
    tw_med = all_trials.time_waited.median()
    tw_std = all_trials.time_waited.std()
    return [tw_mean, tw_med, tw_std]

def get_block_tw(all_trials):
    tw_list = []
    blk_type_list = ['l', 's']
    for blk_type in blk_type_list: 
        blk = all_trials.loc[all_trials['blk_type']==blk_type]
        tw_list.append(blk.time_waited.mean())
        tw_list.append(blk.time_waited.median())
        tw_list.append(blk.time_waited.std())
    return tw_list



