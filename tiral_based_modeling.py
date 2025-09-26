# predict_time_waited.py
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

# -----------------------
# 0) Config
# -----------------------
data_dir = '/Users/rebekahzhang/data/behavior_data'
exp = "exp2"
data_folder = os.path.join(data_dir, exp)
DATA_PATH = os.path.join(data_folder, "trials_training_filtered2.csv")   # change if needed
TARGET = "time_waited"

# Features you specified (normalized to straight quotes & stripped)
FEATURES = [
    "session_trial_num", "block_trial_num", "block_num",
    "bg_drawn", "bg_length", "bg_repeats", "num_bg_licks", "first_lick",
    "time_since_last_reward", "cumulative_reward",
    "mouse", "session", "running_reward_rate",
    "previous_trial_reward_outcome", "group"
]

# Which columns should be treated as categorical?
CATEGORICAL = ["mouse", "session", "previous_trial_reward_outcome", "group"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

# Evaluation modes: "random", "by_session", "by_mouse"
EVAL_MODES = ["random", "by_session", "by_mouse"]

# Output files
OUT_METRICS_CSV = "model_metrics.csv"
OUT_IMPORTANCE_CSV = "feature_importance.csv"

# Random state
RNG = 42

# -----------------------
# 1) Utilities
# -----------------------
# def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
#     """Fix curly quotes/spaces and ensure required columns exist."""
#     # Normalize column names: strip spaces, replace fancy quotes with ASCII
#     mapping = {c: (c.replace("’", "'").replace("`", "'").strip()) for c in df.columns}
#     df = df.rename(columns=mapping)
#     # Also fix any stray spaces in FEATURES
#     global FEATURES, CATEGORICAL, NUMERIC
#     FEATURES = [f.replace("’", "'").replace("`", "'").strip() for f in FEATURES]
#     CATEGORICAL = [f.replace("’", "'").replace("`", "'").strip() for f in CATEGORICAL]
#     NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
#     # Quick check
#     missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
#     if missing:
#         raise ValueError(f"Missing columns in CSV: {missing}")
#     return df

def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return dict(r2=r2, mae=mae, rmse=rmse)

def summarize_permutation_importance(perm, feature_names, top_k=30):
    mean = perm.importances_mean
    std = perm.importances_std
    df = pd.DataFrame({
        "feature": feature_names,
        "perm_importance_mean": mean,
        "perm_importance_std": std
    }).sort_values("perm_importance_mean", ascending=False)
    return df.head(top_k)

def get_feature_names_from_ct(ct: ColumnTransformer) -> list:
    """
    Retrieve transformed feature names from a ColumnTransformer.
    Works for OneHotEncoder/OrdinalEncoder + passthrough/scalers.
    """
    feature_names = []
    for name, transformer, cols in ct.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps"):
            # Pipeline inside ColumnTransformer
            last = list(transformer.named_steps.values())[-1]
        else:
            last = transformer

        if isinstance(last, OneHotEncoder):
            # Detailed names
            ohe_names = last.get_feature_names_out(cols)
            feature_names.extend(ohe_names)
        elif hasattr(last, "get_feature_names_out"):
            try:
                feature_names.extend(last.get_feature_names_out(cols))
            except:
                feature_names.extend(cols)
        else:
            # No name expansion
            feature_names.extend(cols)
    return list(feature_names)

# -----------------------
# 2) Build pipelines
# -----------------------
def make_linear_pipeline():
    """
    L1-regularized linear regression (Lasso) with:
      - Numeric: StandardScaler
      - Categorical: OneHotEncoder (min_frequency reduces huge one-hot explosions)
    """
    numeric_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    ])
    categorical_pipe = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=20,   # collapse rare categories
        sparse_output=False
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL)
        ],
        remainder="drop"
    )
    model = Lasso(alpha=0.05, random_state=RNG, max_iter=10000)
    pipe = Pipeline([
        ("preproc", preproc),
        ("model", model)
    ])
    return pipe

def make_rf_pipeline():
    """
    RandomForest with:
      - Numeric: passthrough
      - Categorical: OrdinalEncoder (unknown -> -1).
    Trees handle ordinal codes fine and avoid huge one-hots.
    """
    categorical_pipe = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    preproc = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL)
        ],
        remainder="drop"
    )
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=RNG
    )
    pipe = Pipeline([
        ("preproc", preproc),
        ("model", rf)
    ])
    return pipe

# -----------------------
# 3) Evaluation modes
# -----------------------
def evaluate_random_split(df, model_name, pipe):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    # Permutation importance on held-out set
    # We need transformed feature names for readability.
    preproc = pipe.named_steps["preproc"]
    # Fit the preprocessor on train to retrieve feature names (already fit)
    feat_names = get_feature_names_from_ct(preproc)
    perm = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=RNG, n_jobs=-1)
    perm_df = summarize_permutation_importance(perm, feat_names, top_k=50)
    return metrics, perm_df

def evaluate_group_cv(df, group_col, model_name, pipe, n_splits=5):
    """
    Leave-groups-out style evaluation with GroupKFold:
      - group_col = 'session' or 'mouse'
    Aggregates metrics across folds and computes permutation importance on the last fold.
    """
    X = df[FEATURES]
    y = df[TARGET]
    groups = df[group_col].astype(str).values
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    fold_metrics = []
    last_perm_df = None

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pipe_fold = make_linear_pipeline() if model_name == "lasso" else make_rf_pipeline()
        pipe_fold.fit(X_tr, y_tr)
        y_pred = pipe_fold.predict(X_te)
        m = compute_metrics(y_te, y_pred)
        m["fold"] = fold_idx + 1
        fold_metrics.append(m)

        # Permutation importance on one representative fold
        if fold_idx == 0:
            preproc = pipe_fold.named_steps["preproc"]
            feat_names = get_feature_names_from_ct(preproc)
            perm = permutation_importance(pipe_fold, X_te, y_te, n_repeats=10, random_state=RNG, n_jobs=-1)
            last_perm_df = summarize_permutation_importance(perm, feat_names, top_k=50)

    metrics_df = pd.DataFrame(fold_metrics)
    agg = metrics_df[["r2", "mae", "rmse"]].mean().to_dict()
    agg["folds"] = len(fold_metrics)
    return agg, last_perm_df, metrics_df

# -----------------------
# 4) Main
# -----------------------
def main():
    print(f"Loading: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    # df = sanitize_columns(df)

    # Drop rows with missing target; for features, we'll let the transformers handle
    df = df[~df[TARGET].isna()].copy()
    print(df.head())
    exit()

    # Ensure categorical dtypes (helps encoders & memory)
    for c in CATEGORICAL:
        df[c] = df[c].astype("string")

    all_metrics_rows = []
    all_importance_rows = []

    # ---- Models to run
    models = {
        "lasso": make_linear_pipeline(),
        "random_forest": make_rf_pipeline()
    }

    # ---- EVAL: RANDOM SPLIT
    for name, pipe in models.items():
        print(f"\n=== {name} | random split ===")
        metrics, perm_df = evaluate_random_split(df, name, pipe)
        metrics_row = {"model": name, "eval_mode": "random", **metrics}
        all_metrics_rows.append(metrics_row)
        if perm_df is not None:
            perm_df = perm_df.assign(model=name, eval_mode="random")
            all_importance_rows.append(perm_df)

    # ---- EVAL: LEAVE SESSIONS OUT
    for name, _ in models.items():
        print(f"\n=== {name} | by_session (GroupKFold) ===")
        pipe = models[name]
        agg_metrics, perm_df, fold_df = evaluate_group_cv(df, "session", name, pipe)
        metrics_row = {"model": name, "eval_mode": "by_session", **agg_metrics}
        all_metrics_rows.append(metrics_row)
        if perm_df is not None:
            perm_df = perm_df.assign(model=name, eval_mode="by_session")
            all_importance_rows.append(perm_df)

    # ---- EVAL: LEAVE MICE OUT
    for name, _ in models.items():
        print(f"\n=== {name} | by_mouse (GroupKFold) ===")
        pipe = models[name]
        agg_metrics, perm_df, fold_df = evaluate_group_cv(df, "mouse", name, pipe)
        metrics_row = {"model": name, "eval_mode": "by_mouse", **agg_metrics}
        all_metrics_rows.append(metrics_row)
        if perm_df is not None:
            perm_df = perm_df.assign(model=name, eval_mode="by_mouse")
            all_importance_rows.append(perm_df)

    # ---- Save outputs
    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_df = metrics_df[["model", "eval_mode", "r2", "rmse", "mae"]]
    metrics_df.to_csv(OUT_METRICS_CSV, index=False)
    print(f"\nSaved metrics → {OUT_METRICS_CSV}\n")
    print(metrics_df)

    if all_importance_rows:
        imp_df = pd.concat(all_importance_rows, ignore_index=True)
        # Reorder nicely
        cols = ["model", "eval_mode", "feature", "perm_importance_mean", "perm_importance_std"]
        imp_df = imp_df[cols].sort_values(["model", "eval_mode", "perm_importance_mean"], ascending=[True, True, False])
        imp_df.to_csv(OUT_IMPORTANCE_CSV, index=False)
        print(f"Saved permutation importance → {OUT_IMPORTANCE_CSV}")
    else:
        print("No permutation importance results to save.")

if __name__ == "__main__":
    main()
