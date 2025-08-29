# Function Organization - LUCKYCHARM Codebase

## 📁 File Structure Overview
```
0_session_quality_control.py      # Quality control and data cleaning
1_data_set_curation.ipynb        # Missing session analysis
2_events_processing.py            # Event processing pipeline
3_events_stitching.py            # Session stitching
4_session_analysis.py             # Pre-meta change session analysis
4b_session_analysis_post_meta_change.py  # Post-meta change session analysis
4_session_analysis_combined.py    # Combined pre/post analysis
session_processing_helper.py      # Core helper functions
processing_helper_0827.py         # Legacy helper functions
utils.py                         # Utility functions (referenced but not shown)
```

## 🔧 Core Helper Functions (`session_processing_helper.py`)

### Quality Control
- `check_session_files(data_folder)` - Check session folders for required files
- `modify_total_trial(row)` - Modify total trial count based on ending code
- `modify_sessions_all(sessions_all)` - Add dir column, extract exp/group, modify total_trial

### Session Generation
- `generate_sessions_all(data_folder)` - Generate DataFrame from session metadata JSON files
- `assign_session_numbers(group)` - Assign sequential session numbers to sessions grouped by mouse
- `generate_sessions_training(sessions_all)` - Filter for training sessions and assign session numbers
- `generate_session_logs(data_folder, save_logs=True)` - Generate and optionally save session logs

### Event Processing
- `process_events(session_info, events)` - Filter events to valid trial range
- `add_trial_time(trial)` - Add trial_time column relative to trial start

### Session Analysis
- `get_session_basics(session_df)` - Extract basic session statistics (blocks, trials, rewards, time)
- `stitch_sessions(session_1, session_2)` - Stitch two sessions by adjusting timing and trial numbers

### Trial Analysis
- `get_trial_basics(trial)` - Extract basic trial info (trial numbers, block info, timing)
- `generate_trials(session_info, events)` - Generate trial information DataFrame from session events
- `get_trial_bg_data(trial)` - Extract background trial metrics
- `get_trial_wait_data(trial)` - Extract wait trial performance data
- `get_trial_performance(t, trial)` - Get comprehensive trial performance combining background and wait data
- `get_trial_data_df(session_by_trial)` - Generate performance DataFrame for all trials in a session

## 🏗️ Legacy Helper Functions (`processing_helper_0827.py`)

### Session Generation
- `add_cohort_column(sessions_all, cohort_info)` - Add cohort column based on mouse name and cohort info
- `generate_sessions_all(data_folder)` - Generate DataFrame from session metadata JSON files (legacy version)

### Trial Analysis
- `get_trial_basics(trial)` - Extract basic trial info (legacy version)
- `add_trial_time(trial)` - Add trial_time column relative to trial start (legacy version)

## 🧹 Quality Control Functions (`0_session_quality_control.py`)

### Data Management
- `backup_directory(source_path)` - Create backup or update existing with new sessions
- `delete_test_folders(data_dir)` - Delete all folders ending with '_test'
- `update_deletion_record(data_dir, deletion_dfs)` - Update deletion record CSV file

### Session Validation
- `check_and_clean_sessions_with_corrupted_files(data_folder)` - Check session files and delete problematic sessions
- `identify_and_clean_short_or_crashed_sessions(data_folder, short_threshold=20)` - Identify and delete short/crashed sessions
- `validate_session_directory_names(data_folder)` - Check if meta JSON files match their directory names
- `sort_sessions_by_experiments(data_dir, exp_info)` - Sort sessions into experiment folders based on mouse names

## 📊 Data Curation Functions (`1_data_set_curation.ipynb`)

### Missing Session Analysis
- `should_ignore_mouse_due_to_dod(mouse, session_date, mouse_dod_df)` - Check if mouse should be ignored due to death
- `has_documented_reason(mouse, date, missing_log_df)` - Check if mouse has documented reason for missing

## ⚙️ Event Processing Functions (`2_events_processing.py`)

### Pre-Meta Change Processing
- `get_max_trial_num(events)` - Get the maximum valid trial number, accounting for incomplete last trials
- `generate_trials(events, max_trial_num)` - Generate trial information for all trials in the session
- `align_trial_number(session, trials)` - Align trial numbers with session events based on time windows
- `align_trial_states(trial)` - Align trial states for pre-metadata change sessions
- `process_events_pre_meta_change(data_folder, session_info)` - Process events for sessions before metadata change

### Post-Meta Change Processing
- `process_events_post_meta_change(data_folder, session_info)` - Process events for sessions after metadata change

### Batch Processing
- `process_session_batch(sessions_df, process_function, batch_name, regenerate)` - Process a batch of sessions with the specified processing function

## 🧵 Event Stitching Functions (`3_events_stitching.py`)

### Session Stitching
- `stitch_sessions(session_1, session_2)` - Stitch two sessions together with proper time and trial offsets
- `stitch_events_for_period(data_folder, sessions_period, second_sessions_dir, period_name)` - Stitch events from sessions for a specific time period
- `stitch_all_events(data_folder, meta_change_date, second_sessions_dir)` - Main function to stitch all events from both pre and post meta-change periods

## 📈 Session Analysis Functions

### Pre-Meta Change (`4_session_analysis.py`)
- `correct_sessions_training(data_folder, save_log=True)` - Generate and correct sessions training data
- `get_trial_basics(trial)` - Extract basic trial info
- `generate_trials(session_info, events)` - Generate trials DataFrame for a given session
- `process_trials_generation(sessions_training, data_folder)` - Generate trials for all sessions
- `analyze_trials(sessions_training, data_folder)` - Analyze trials for all sessions

### Post-Meta Change (`4b_session_analysis_post_meta_change.py`)
- `correct_sessions_training(data_folder, save_log=True)` - Generate and correct sessions training data for post meta change
- `process_trials_generation(sessions_training, data_folder)` - Generate trials for all sessions
- `analyze_trials(sessions_training, data_folder)` - Analyze trials for all sessions

### Combined (`4_session_analysis_combined.py`)
- `correct_sessions_training(data_folder, save_log=True)` - Generate and correct sessions training data for both pre and post meta change
- `process_trials_generation(sessions_training, data_folder, regenerate=False)` - Generate trials for all sessions with regeneration option
- `analyze_trials(sessions_training, data_folder, regenerate=False)` - Analyze trials for all sessions with regeneration option

## 🔄 Function Usage Patterns

### Session Generation Flow
1. **Raw Data** → `generate_sessions_all()` → `sessions_all`
2. **Filter** → `sessions_all.loc[training == 'regular']` → `sessions_training`
3. **Process** → `correct_sessions_training()` → `corrected_sessions_training`
4. **Save** → `utils.save_as_csv()` → CSV file

### Event Processing Flow
1. **Load Events** → `pd.read_csv(events_path)` → `events`
2. **Process** → `process_events_pre_meta_change()` or `process_events_post_meta_change()` → `events_processed`
3. **Save** → `events_processed.to_csv()` → processed events file

### Trial Analysis Flow
1. **Generate Trials** → `generate_trials()` → `trials`
2. **Analyze** → `get_trial_data_df()` → `trials_data`
3. **Merge** → `pd.merge(trials, trials_data)` → `trials_analyzed`
4. **Save** → `trials_analyzed.to_csv()` → analyzed trials file

## 🚨 Duplicate Functions

### Functions Defined in Multiple Files
- `get_trial_basics()` - Defined in `processing_helper_0827.py` and `session_processing_helper.py`
- `add_trial_time()` - Defined in `processing_helper_0827.py` and `session_processing_helper.py`
- `generate_trials()` - Defined in `4_session_analysis.py` and `session_processing_helper.py`

### Recommendations
1. **Consolidate** duplicate functions into `session_processing_helper.py`
2. **Remove** legacy versions from `processing_helper_0827.py`
3. **Update** imports to use unified helper functions
4. **Standardize** function signatures across the codebase

## 📋 Function Categories by Purpose

### Data Loading & Generation
- Session metadata parsing
- Cohort information mapping
- File existence checking

### Data Processing
- Event filtering and alignment
- Trial state assignment
- Time calculations

### Data Analysis
- Session statistics extraction
- Trial performance metrics
- Background and wait analysis

### Data Management
- File backup and restoration
- Session stitching and merging
- Quality control and cleaning

### Data Export
- CSV file generation
- Progress reporting
- Error handling and logging
