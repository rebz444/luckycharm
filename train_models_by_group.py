"""
Train Random Forest models for waiting behavior prediction.
- Separate models per group (Long BG vs Short BG)
- Three target variables: time from cue_on, cue_off, and last_lick
- No current-trial background features (prevents leakage)
- Both raw and log-transformed targets
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = '/Users/rebekahzhang/data/behavior_data'
EXP = "exp2"
DATA_PATH = os.path.join(DATA_DIR, EXP, "trials_training_filtered2.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, f'{EXP}_modeling_by_group')

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/models', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/results', exist_ok=True)

# Random state for reproducibility
RNG = 43

# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

# Session/trial structure features
SESSION_FEATURES = [
    'session', 'session_trial_num', 'block_trial_num', 'block_num', 'mouse'
]

# Previous trial background features (safe - no leakage)
PREV_BACKGROUND_FEATURES = [
    'previous_trial_bg_drawn', 'previous_trial_bg_length', 
    'previous_trial_bg_repeats', 'previous_trial_num_bg_licks',
    'bg_length_rolling_mean_5', 'bg_length_rolling_mean_10',
    'bg_repeats_rolling_mean_5', 'bg_repeats_rolling_mean_10'
]

# Wait time history features
WAIT_HISTORY_FEATURES = [
    'previous_trial_time_waited', 'time_waited_rolling_mean_5',
    'time_waited_rolling_mean_10', 'previous_trial_miss_trial'
]

# Reward history features
REWARD_FEATURES = [
    'previous_trial_reward', 'rewarded_streak', 'unrewarded_streak',
    'time_since_last_reward_in_block', 'cumulative_reward_in_block',
    'cumulative_reward', 'reward_rate_since_block_start',
    'reward_rate_past_1min_in_block', 'reward_rate_past_5min_in_block',
    'reward_rate_past_10min_in_block'
]

# Feature set combinations (NO current trial background features)
FEATURE_SETS = {
    'all': (SESSION_FEATURES + PREV_BACKGROUND_FEATURES + 
            WAIT_HISTORY_FEATURES + REWARD_FEATURES),
    'no_session': (PREV_BACKGROUND_FEATURES + WAIT_HISTORY_FEATURES + 
                   REWARD_FEATURES),
    'no_prev_bg': (SESSION_FEATURES + WAIT_HISTORY_FEATURES + 
                   REWARD_FEATURES),
    'no_wait_history': (SESSION_FEATURES + PREV_BACKGROUND_FEATURES + 
                        REWARD_FEATURES),
    'no_reward': (SESSION_FEATURES + PREV_BACKGROUND_FEATURES + 
                  WAIT_HISTORY_FEATURES),
    'only_wait_history': WAIT_HISTORY_FEATURES,
    'only_reward': REWARD_FEATURES,
    'only_prev_bg': PREV_BACKGROUND_FEATURES
}

CATEGORICAL_FEATURES = ['mouse', 'previous_trial_reward', 'previous_trial_miss_trial']

# Target definitions
TARGET_ANCHORS = ['cue_on', 'cue_off', 'last_lick']

# Groups
GROUPS = ['l', 's']  # Long BG, Short BG

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_preprocess_data(data_path):
    """Load data and prepare target variables for different timing anchors."""
    df = pd.read_csv(data_path)
    
    # Filter to non-missed trials only
    df = df.loc[(df['miss_trial'] == False) & (~df['time_waited'].isna())].copy()
    
    # Fill missing values
    df["time_since_last_reward_in_block"] = df["time_since_last_reward_in_block"].fillna(2000)
    
    # Create target variables for different anchors
    # 1. Time from cue_on
    df['time_from_cue_on'] = df['time_waited_since_cue_on']

    # 2. Time from cue_off (time_waited IS the time since cue_off)
    df['time_from_cue_off'] = df['time_waited']

    # 3. Time from last lick
    if 'time_waited_since_last_lick' in df.columns:
        df['time_from_last_lick'] = df['time_waited_since_last_lick']
    else:
        print("WARNING: 'time_waited_since_last_lick' column not found. Defaulting to time_from_cue_on.")
        df['time_from_last_lick'] = df['time_waited_since_cue_on']
    
    # Create log-transformed versions (adding small constant to handle zeros)
    for anchor in TARGET_ANCHORS:
        col = f'time_from_{anchor}'
        # Ensure no negative values before log transform
        df[f'{col}_clipped'] = df[col].clip(lower=0.001)
        df[f'{col}_log'] = np.log1p(df[f'{col}_clipped'])
    
    return df


def get_available_features(df, feature_list):
    """Return only features that exist in the dataframe."""
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"    Warning: Missing features: {missing}")
    return available


def prepare_features(df, feature_list):
    """Encode categorical features and standardize numeric features."""
    # Get only available features
    available_features = get_available_features(df, feature_list)
    
    categorical = [c for c in CATEGORICAL_FEATURES if c in available_features]
    numeric = [c for c in available_features if c not in categorical]
    
    # Make a copy with only needed columns
    df_subset = df[available_features].copy()
    
    # One-hot encode categorical features
    if categorical:
        df_encoded = pd.get_dummies(df_subset, columns=categorical, drop_first=False)
    else:
        df_encoded = df_subset.copy()
    
    # Identify encoded columns
    dummy_cols = [c for c in df_encoded.columns if c not in df_subset.columns]
    numeric_cols = [c for c in numeric if c in df_encoded.columns]
    
    # Handle missing values in numeric columns
    for col in numeric_cols:
        if df_encoded[col].isna().any():
            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].median())
    
    # Standardize numeric features
    scaler = None
    if numeric_cols:
        scaler = StandardScaler()
        df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    
    # Return encoded columns in sorted order
    encoded_columns = sorted(dummy_cols + numeric_cols)
    
    return df_encoded, encoded_columns, scaler


def compute_metrics(y_true, y_pred):
    """Calculate regression metrics."""
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': mean_squared_error(y_true, y_pred, squared=False)
    }


def train_rf_model(X_train, y_train, X_test, y_test):
    """Train Random Forest and return model with predictions and metrics."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=50,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=RNG
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    
    return model, y_pred, metrics


def save_model_results(model, feature_names, metrics, group, anchor, transform, feature_set):
    """Save trained model and metadata."""
    filename = f'{OUTPUT_DIR}/models/rf_{group}_{anchor}_{transform}_{feature_set}.pkl'
    
    joblib.dump({
        'model': model,
        'feature_names': feature_names,
        'metrics': metrics,
        'group': group,
        'anchor': anchor,
        'transform': transform,
        'feature_set': feature_set
    }, filename)
    
    return filename


def get_feature_importances(model, feature_names):
    """Extract feature importances as a sorted dataframe."""
    return pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("TRAINING RANDOM FOREST MODELS BY GROUP")
    print("Targets: time from cue_on, cue_off, last_lick")
    print("Groups: Long BG (l), Short BG (s)")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {DATA_PATH}")
    df = load_and_preprocess_data(DATA_PATH)
    print(f"Loaded {len(df)} total trials")
    
    # Check group distribution
    print(f"\nGroup distribution:")
    print(df['group'].value_counts())
    
    # Store all results
    all_results = []
    all_importances = {}
    
    # Train models for each group
    for group in GROUPS:
        group_name = "Long BG" if group == 'l' else "Short BG"
        df_group = df[df['group'] == group].copy()
        
        print(f"\n{'#' * 80}")
        print(f"GROUP: {group_name} ({group}) - {len(df_group)} trials")
        print(f"{'#' * 80}")
        
        # Train for each anchor point
        for anchor in TARGET_ANCHORS:
            print(f"\n{'=' * 60}")
            print(f"TARGET ANCHOR: time_from_{anchor}")
            print(f"{'=' * 60}")
            
            # Train for both transforms
            for transform in ['raw', 'log']:
                target_col = f'time_from_{anchor}' if transform == 'raw' else f'time_from_{anchor}_log'
                
                print(f"\n  Transform: {transform}")
                
                # Check target validity
                if df_group[target_col].isna().any():
                    print(f"    WARNING: {df_group[target_col].isna().sum()} NaN values in target")
                    df_valid = df_group.dropna(subset=[target_col])
                else:
                    df_valid = df_group
                
                # Train for each feature set
                for feature_set_name, feature_list in FEATURE_SETS.items():
                    # Prepare features
                    df_encoded, encoded_columns, scaler = prepare_features(df_valid, feature_list)
                    
                    if len(encoded_columns) == 0:
                        print(f"    {feature_set_name}: No valid features, skipping")
                        continue
                    
                    # Prepare X and y
                    X = df_encoded[encoded_columns]
                    y = df_valid[target_col]
                    
                    # Align indices
                    common_idx = X.index.intersection(y.index)
                    X = X.loc[common_idx]
                    y = y.loc[common_idx]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=RNG
                    )
                    
                    # Train model
                    model, y_pred, metrics = train_rf_model(X_train, y_train, X_test, y_test)
                    
                    # Save model
                    save_model_results(model, encoded_columns, metrics, 
                                      group, anchor, transform, feature_set_name)
                    
                    # Store feature importances for 'all' feature set
                    if feature_set_name == 'all':
                        fi_df = get_feature_importances(model, encoded_columns)
                        key = f'{group}_{anchor}_{transform}'
                        all_importances[key] = fi_df
                        fi_df.to_csv(
                            f'{OUTPUT_DIR}/results/feature_importances_{key}.csv', 
                            index=False
                        )
                    
                    # Store results
                    all_results.append({
                        'group': group,
                        'group_name': group_name,
                        'anchor': anchor,
                        'transform': transform,
                        'feature_set': feature_set_name,
                        'n_features': len(encoded_columns),
                        'r2': metrics['r2'],
                        'mae': metrics['mae'],
                        'rmse': metrics['rmse'],
                        'n_train': len(X_train),
                        'n_test': len(X_test)
                    })
                    
                    print(f"    {feature_set_name:20s} | R²={metrics['r2']:.4f} | "
                          f"MAE={metrics['mae']:.3f} | RMSE={metrics['rmse']:.3f}")
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_path = f'{OUTPUT_DIR}/results/model_comparison_all.csv'
    results_df.to_csv(results_path, index=False)
    
    # ========================================================================
    # GENERATE OUTPUT FILES
    # ========================================================================
    
    print(f"\n{'=' * 80}")
    print("GENERATING OUTPUT FILES")
    print(f"{'=' * 80}")
    
    # ------------------------------------------------------------------------
    # 1. R² SUMMARY TABLES BY ANCHOR AND TRANSFORM
    # ------------------------------------------------------------------------
    
    # Filter to 'all' feature set for cleaner comparison
    summary = results_df[results_df['feature_set'] == 'all'].copy()
    
    # Create R² pivot tables for each transform
    r2_tables = {}
    for transform in ['raw', 'log']:
        subset = summary[summary['transform'] == transform]
        pivot = subset.pivot(index='anchor', columns='group', values='r2')
        pivot.columns = ['Long_BG', 'Short_BG']
        pivot['diff'] = pivot['Long_BG'] - pivot['Short_BG']
        r2_tables[transform] = pivot
        
        # Save individual table
        pivot.to_csv(f'{OUTPUT_DIR}/results/r2_by_anchor_{transform}.csv')
    
    # Combined R² summary
    r2_combined = pd.concat([
        r2_tables['raw'].add_suffix('_raw'),
        r2_tables['log'].add_suffix('_log')
    ], axis=1)
    r2_combined.to_csv(f'{OUTPUT_DIR}/results/r2_summary_by_anchor.csv')
    
    print(f"  Saved: r2_summary_by_anchor.csv")
    print(f"  Saved: r2_by_anchor_raw.csv, r2_by_anchor_log.csv")
    
    # ------------------------------------------------------------------------
    # 2. ABLATION TABLES (feature set comparison)
    # ------------------------------------------------------------------------
    
    for transform in ['raw', 'log']:
        for anchor in TARGET_ANCHORS:
            ablation = results_df[
                (results_df['transform'] == transform) & 
                (results_df['anchor'] == anchor)
            ].pivot(
                index='feature_set', 
                columns='group', 
                values='r2'
            )
            ablation.columns = ['Long_BG', 'Short_BG']
            ablation['diff'] = ablation['Long_BG'] - ablation['Short_BG']
            ablation = ablation.sort_values('Long_BG', ascending=False)
            ablation.to_csv(f'{OUTPUT_DIR}/results/ablation_{anchor}_{transform}.csv')
    
    print(f"  Saved: ablation_{{anchor}}_{{transform}}.csv (6 files)")
    
    # ------------------------------------------------------------------------
    # 3. FEATURE IMPORTANCE COMPARISONS ACROSS GROUPS
    # ------------------------------------------------------------------------
    
    for anchor in TARGET_ANCHORS:
        for transform in ['raw', 'log']:
            fi_long = all_importances.get(f'l_{anchor}_{transform}')
            fi_short = all_importances.get(f's_{anchor}_{transform}')
            
            if fi_long is not None and fi_short is not None:
                # Merge on feature name
                comparison = fi_long.merge(
                    fi_short, on='feature', suffixes=('_long', '_short'), how='outer'
                ).fillna(0)
                comparison['diff'] = comparison['importance_long'] - comparison['importance_short']
                comparison['abs_diff'] = comparison['diff'].abs()
                comparison['ratio'] = (comparison['importance_long'] / 
                                       comparison['importance_short'].replace(0, np.nan))
                comparison = comparison.sort_values('abs_diff', ascending=False)
                
                comparison.to_csv(
                    f'{OUTPUT_DIR}/results/fi_group_comparison_{anchor}_{transform}.csv',
                    index=False
                )
    
    print(f"  Saved: fi_group_comparison_{{anchor}}_{{transform}}.csv (6 files)")
    
    # ------------------------------------------------------------------------
    # 4. FEATURE IMPORTANCE COMPARISONS ACROSS ANCHORS (within group)
    # ------------------------------------------------------------------------
    
    for group in GROUPS:
        group_name = "long" if group == 'l' else "short"
        for transform in ['raw', 'log']:
            # Get importances for all anchors
            fi_dfs = {}
            for anchor in TARGET_ANCHORS:
                key = f'{group}_{anchor}_{transform}'
                if key in all_importances:
                    fi_dfs[anchor] = all_importances[key].set_index('feature')['importance']
            
            if len(fi_dfs) == len(TARGET_ANCHORS):
                # Combine into single dataframe
                anchor_comparison = pd.DataFrame(fi_dfs)
                anchor_comparison.columns = [f'importance_{a}' for a in TARGET_ANCHORS]
                
                # Add variance across anchors (high variance = anchor-sensitive feature)
                anchor_comparison['std_across_anchors'] = anchor_comparison.std(axis=1)
                anchor_comparison['mean_importance'] = anchor_comparison[
                    [f'importance_{a}' for a in TARGET_ANCHORS]
                ].mean(axis=1)
                anchor_comparison = anchor_comparison.sort_values(
                    'mean_importance', ascending=False
                )
                
                anchor_comparison.to_csv(
                    f'{OUTPUT_DIR}/results/fi_anchor_comparison_{group_name}_{transform}.csv'
                )
    
    print(f"  Saved: fi_anchor_comparison_{{group}}_{{transform}}.csv (4 files)")
    
    # ------------------------------------------------------------------------
    # 5. TOP FEATURES SUMMARY (easy reference)
    # ------------------------------------------------------------------------
    
    top_n = 15
    top_features_summary = []
    
    for group in GROUPS:
        group_name = "Long_BG" if group == 'l' else "Short_BG"
        for anchor in TARGET_ANCHORS:
            for transform in ['raw', 'log']:
                key = f'{group}_{anchor}_{transform}'
                if key in all_importances:
                    top = all_importances[key].head(top_n).copy()
                    top['rank'] = range(1, len(top) + 1)
                    top['group'] = group_name
                    top['anchor'] = anchor
                    top['transform'] = transform
                    top_features_summary.append(top)
    
    top_features_df = pd.concat(top_features_summary, ignore_index=True)
    top_features_df = top_features_df[['group', 'anchor', 'transform', 'rank', 'feature', 'importance']]
    top_features_df.to_csv(f'{OUTPUT_DIR}/results/top_{top_n}_features_all_conditions.csv', index=False)
    
    print(f"  Saved: top_{top_n}_features_all_conditions.csv")
    
    # ------------------------------------------------------------------------
    # 6. WIDE FORMAT: Feature ranks across conditions
    # ------------------------------------------------------------------------
    
    # Create wide table showing rank of each feature in each condition
    rank_data = []
    for group in GROUPS:
        for anchor in TARGET_ANCHORS:
            key = f'{group}_{anchor}_raw'  # Use raw for this comparison
            if key in all_importances:
                fi = all_importances[key].copy()
                fi['rank'] = range(1, len(fi) + 1)
                fi = fi.set_index('feature')['rank']
                fi.name = f'{group}_{anchor}'
                rank_data.append(fi)
    
    if rank_data:
        rank_wide = pd.concat(rank_data, axis=1)
        rank_wide['mean_rank'] = rank_wide.mean(axis=1)
        rank_wide['rank_std'] = rank_wide[[c for c in rank_wide.columns if c != 'mean_rank']].std(axis=1)
        rank_wide = rank_wide.sort_values('mean_rank')
        rank_wide.to_csv(f'{OUTPUT_DIR}/results/feature_ranks_across_conditions.csv')
        
        print(f"  Saved: feature_ranks_across_conditions.csv")
    
    # ------------------------------------------------------------------------
    # 7. COMPREHENSIVE MARKDOWN REPORT
    # ------------------------------------------------------------------------
    
    report_lines = [
        "# Random Forest Model Results",
        f"\nGenerated from: {DATA_PATH}",
        f"Output directory: {OUTPUT_DIR}",
        "",
        "## Overview",
        f"- Total trials: {len(df)}",
        f"- Long BG trials: {len(df[df['group'] == 'l'])}",
        f"- Short BG trials: {len(df[df['group'] == 's'])}",
        "",
        "## R² Performance Summary (all features)",
        "",
        "### Raw Target",
        "| Anchor | Long BG | Short BG | Diff |",
        "|--------|---------|----------|------|",
    ]
    
    for anchor in TARGET_ANCHORS:
        row = r2_tables['raw'].loc[anchor]
        report_lines.append(
            f"| {anchor} | {row['Long_BG']:.4f} | {row['Short_BG']:.4f} | {row['diff']:.4f} |"
        )
    
    report_lines.extend([
        "",
        "### Log-Transformed Target",
        "| Anchor | Long BG | Short BG | Diff |",
        "|--------|---------|----------|------|",
    ])
    
    for anchor in TARGET_ANCHORS:
        row = r2_tables['log'].loc[anchor]
        report_lines.append(
            f"| {anchor} | {row['Long_BG']:.4f} | {row['Short_BG']:.4f} | {row['diff']:.4f} |"
        )
    
    report_lines.extend([
        "",
        "## Top 10 Features by Condition (raw target)",
        "",
    ])
    
    for group in GROUPS:
        group_name = "Long BG" if group == 'l' else "Short BG"
        report_lines.append(f"### {group_name}")
        
        for anchor in TARGET_ANCHORS:
            key = f'{group}_{anchor}_raw'
            if key in all_importances:
                report_lines.append(f"\n#### Anchor: {anchor}")
                report_lines.append("| Rank | Feature | Importance |")
                report_lines.append("|------|---------|------------|")
                
                for i, row in all_importances[key].head(10).iterrows():
                    report_lines.append(
                        f"| {i+1} | {row['feature']} | {row['importance']:.4f} |"
                    )
        report_lines.append("")
    
    # Feature set ablation
    report_lines.extend([
        "## Feature Set Ablation (raw target, cue_off anchor)",
        "",
        "| Feature Set | Long BG R² | Short BG R² | Diff |",
        "|-------------|------------|-------------|------|",
    ])
    
    ablation_cue_off = results_df[
        (results_df['transform'] == 'raw') & 
        (results_df['anchor'] == 'cue_off')
    ].pivot(index='feature_set', columns='group', values='r2')
    ablation_cue_off.columns = ['Long_BG', 'Short_BG']
    ablation_cue_off['diff'] = ablation_cue_off['Long_BG'] - ablation_cue_off['Short_BG']
    ablation_cue_off = ablation_cue_off.sort_values('Long_BG', ascending=False)
    
    for fs, row in ablation_cue_off.iterrows():
        report_lines.append(
            f"| {fs} | {row['Long_BG']:.4f} | {row['Short_BG']:.4f} | {row['diff']:.4f} |"
        )
    
    report_lines.extend([
        "",
        "## Output Files",
        "",
        "### Model Files",
        "- `models/rf_{group}_{anchor}_{transform}_{feature_set}.pkl`",
        "",
        "### Results Files",
        "- `model_comparison_all.csv` - All model metrics",
        "- `r2_summary_by_anchor.csv` - R² by anchor (combined)",
        "- `ablation_{anchor}_{transform}.csv` - Feature set ablation",
        "- `fi_group_comparison_{anchor}_{transform}.csv` - Feature importance: Long vs Short",
        "- `fi_anchor_comparison_{group}_{transform}.csv` - Feature importance across anchors",
        "- `top_15_features_all_conditions.csv` - Top features reference",
        "- `feature_ranks_across_conditions.csv` - Feature rank consistency",
        "- `feature_importances_{group}_{anchor}_{transform}.csv` - Raw importances",
    ])
    
    with open(f'{OUTPUT_DIR}/results/RESULTS_REPORT.md', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"  Saved: RESULTS_REPORT.md")
    
    # ========================================================================
    # CONSOLE OUTPUT SUMMARY
    # ========================================================================
    
    print(f"\n{'=' * 80}")
    print("R² COMPARISON: ALL FEATURES SET")
    print(f"{'=' * 80}")
    
    for transform in ['raw', 'log']:
        print(f"\n--- {transform.upper()} TARGET ---")
        print(r2_tables[transform].round(4))
    
    print(f"\n{'=' * 80}")
    print("TOP 10 FEATURE IMPORTANCES BY GROUP AND ANCHOR (raw target)")
    print(f"{'=' * 80}")
    
    for group in GROUPS:
        group_name = "Long BG" if group == 'l' else "Short BG"
        print(f"\n{'=' * 40}")
        print(f"GROUP: {group_name}")
        print(f"{'=' * 40}")
        
        for anchor in TARGET_ANCHORS:
            key = f'{group}_{anchor}_raw'
            if key in all_importances:
                print(f"\n  Anchor: {anchor}")
                print(all_importances[key].head(10).to_string(index=False))
    
    print(f"\n{'=' * 80}")
    print("FEATURE SET ABLATION SUMMARY (raw target, cue_off anchor)")
    print(f"{'=' * 80}")
    print(ablation_cue_off.round(4))
    
    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 80}")
    print(f"\nResults saved to: {OUTPUT_DIR}/results/")
    print(f"Models saved to: {OUTPUT_DIR}/models/")
    print(f"\nKey output files:")
    print(f"  - RESULTS_REPORT.md (comprehensive summary)")
    print(f"  - model_comparison_all.csv (all metrics)")
    print(f"  - r2_summary_by_anchor.csv (R² comparison)")
    print(f"  - fi_group_comparison_*.csv (group differences)")
    print(f"  - feature_ranks_across_conditions.csv (rank consistency)")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
