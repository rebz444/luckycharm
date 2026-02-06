"""
Simple script to train Random Forest models for waiting behavior prediction.
Compares performance across different feature sets and target transformations.
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
OUTPUT_DIR = os.path.join(DATA_DIR, f'{EXP}_modeling_simple')

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/models', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/results', exist_ok=True)

# Random state for reproducibility
RNG = 43

# Feature definitions (same as notebook)
GROUP_FEATURES = ['group']
SESSION_FEATURES = ['session', 'session_trial_num', 'block_trial_num', 'block_num', 'mouse']
BACKGROUND_FEATURES = [
    'bg_drawn', 'bg_length', 'bg_repeats', 'num_bg_licks', 'previous_trial_bg_repeats',
    'bg_repeats_rolling_mean_5', 'bg_repeats_rolling_mean_10'
]
BACKGROUND_HISTORY_FEATURES = [
    'previous_trial_bg_drawn', 'previous_trial_bg_length', 'previous_trial_num_bg_licks',
    'bg_drawn_rolling_mean_5', 'bg_drawn_rolling_mean_10'
]
BACKGROUND_FEATURES = ['bg_drawn', 'bg_length', 'bg_repeats', 'num_bg_licks']
WAIT_HISTORY_FEATURES = [
    'previous_trial_time_waited', 'time_waited_rolling_mean_5',
    'time_waited_rolling_mean_10', 'previous_trial_miss_trial'
]
REWARD_FEATURES = [
    'previous_trial_reward', 'rewarded_streak', 'unrewarded_streak',
    'time_since_last_reward_in_block', 'cumulative_reward_in_block',
    'cumulative_reward', 'reward_rate_since_block_start',
    'reward_rate_past_1min_in_block', 'reward_rate_past_5min_in_block',
    'reward_rate_past_10min_in_block'
]

# Feature set combinations
FEATURE_SETS = {
    'all': GROUP_FEATURES + SESSION_FEATURES + REWARD_FEATURES + BACKGROUND_FEATURES + WAIT_HISTORY_FEATURES,
    'no_session': GROUP_FEATURES + REWARD_FEATURES + BACKGROUND_FEATURES + WAIT_HISTORY_FEATURES,
    'no_background': GROUP_FEATURES + SESSION_FEATURES + REWARD_FEATURES + WAIT_HISTORY_FEATURES,
    'no_wait': GROUP_FEATURES + SESSION_FEATURES + REWARD_FEATURES + BACKGROUND_FEATURES,
    'only_wait': GROUP_FEATURES + WAIT_HISTORY_FEATURES,
    'no_reward': GROUP_FEATURES + SESSION_FEATURES + BACKGROUND_FEATURES + WAIT_HISTORY_FEATURES,
    'only_reward': GROUP_FEATURES + REWARD_FEATURES
}

CATEGORICAL_FEATURES = ["group", "mouse", "previous_trial_reward", 'previous_trial_miss_trial']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_and_preprocess_data(data_path):
    """Load data and prepare for modeling."""
    df = pd.read_csv(data_path)
    
    # Filter to non-missed trials only
    df = df.loc[(df['miss_trial'] == False) & (~df['time_waited'].isna())].copy()
    
    # Fill missing values
    df["time_since_last_reward_in_block"] = df["time_since_last_reward_in_block"].fillna(2000)
    
    # Create transformed target
    df['time_waited_log'] = np.log1p(df['time_waited'])
    
    return df


def prepare_features(df, feature_list):
    """Encode categorical features and standardize numeric features."""
    categorical = [c for c in CATEGORICAL_FEATURES if c in feature_list]
    numeric = [c for c in feature_list if c not in categorical]
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=categorical, drop_first=False)
    
    # Identify encoded columns
    dummy_cols = [c for c in df_encoded.columns if c not in df.columns]
    numeric_cols = [c for c in numeric if c in df_encoded.columns]
    
    # Standardize numeric features
    if numeric_cols:
        scaler = StandardScaler()
        df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    
    # Return encoded columns in sorted order
    encoded_columns = sorted(dummy_cols + numeric_cols)
    
    return df_encoded, encoded_columns


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
        n_jobs=-1,
        random_state=RNG
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    
    return model, y_pred, metrics


def save_model_results(model, feature_names, metrics, feature_set, target_type):
    """Save trained model and metadata."""
    filename = f'{OUTPUT_DIR}/models/rf_{target_type}_{feature_set}.pkl'
    
    joblib.dump({
        'model': model,
        'feature_names': feature_names,
        'metrics': metrics,
        'feature_set': feature_set,
        'target_type': target_type
    }, filename)
    
    return filename


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("TRAINING RANDOM FOREST MODELS FOR WAITING BEHAVIOR PREDICTION")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {DATA_PATH}")
    df = load_and_preprocess_data(DATA_PATH)
    print(f"Loaded {len(df)} trials")
    
    # Store results
    results = []
    
    # Train models for both target types
    for target_type in ['raw', 'log']:
        target_col = 'time_waited' if target_type == 'raw' else 'time_waited_log'
        
        print(f"\n{'=' * 80}")
        print(f"TARGET: {target_col}")
        print(f"{'=' * 80}")
        
        # Train models for each feature set
        for feature_set_name, feature_list in FEATURE_SETS.items():
            print(f"\n  Feature Set: {feature_set_name} ({len(feature_list)} features)")
            
            # Prepare features
            df_encoded, encoded_columns = prepare_features(df, feature_list)
            
            # Split data
            X = df_encoded[encoded_columns]
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RNG
            )
            
            print(f"    Train size: {len(X_train)}, Test size: {len(X_test)}")
            
            # Train model
            model, y_pred, metrics = train_rf_model(X_train, y_train, X_test, y_test)
            
            # Save model
            save_model_results(model, encoded_columns, metrics, feature_set_name, target_type)
            
            # Store results
            results.append({
                'target_type': target_type,
                'feature_set': feature_set_name,
                'n_features': len(encoded_columns),
                'r2': metrics['r2'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'n_train': len(X_train),
                'n_test': len(X_test)
            })
            
            print(f"    R² = {metrics['r2']:.4f}, MAE = {metrics['mae']:.3f}, RMSE = {metrics['rmse']:.3f}")
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_path = f'{OUTPUT_DIR}/results/model_comparison.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {results_path}")
    print(f"Models saved to: {OUTPUT_DIR}/models/")
    
    # Display comparison
    print(f"\n{'=' * 80}")
    print("R² COMPARISON ACROSS FEATURE SETS")
    print(f"{'=' * 80}")
    pivot = results_df.pivot(index='feature_set', columns='target_type', values='r2')
    print(pivot.round(4))
    
    # Feature importance analysis
    print(f"\n{'=' * 80}")
    print("TOP 15 FEATURE IMPORTANCES (all features, raw target)")
    print(f"{'=' * 80}")
    
    model_data = joblib.load(f'{OUTPUT_DIR}/models/rf_raw_all.pkl')
    importances = model_data['model'].feature_importances_
    feature_names = model_data['feature_names']
    
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(15)
    
    print(fi_df.to_string(index=False))
    
    # Save feature importances
    fi_full = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    fi_full.to_csv(f'{OUTPUT_DIR}/results/feature_importances_raw_all.csv', index=False)
    
    # Also for log target
    model_data_log = joblib.load(f'{OUTPUT_DIR}/models/rf_log_all.pkl')
    fi_log = pd.DataFrame({
        'feature': model_data_log['feature_names'],
        'importance': model_data_log['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    fi_log.to_csv(f'{OUTPUT_DIR}/results/feature_importances_log_all.csv', index=False)
    
    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
