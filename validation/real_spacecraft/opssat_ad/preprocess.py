"""
preprocess.py — OPS-SAT-AD preprocessing utilities.

Responsibilities:
  - channel one-hot encoding (optional)
  - sampling flag as binary feature
  - feature scaling (StandardScaler, fitted on TRAIN only)
  - no data leakage between train and test partitions
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .loader import load_dataset, get_official_split, get_feature_matrix

FEATURE_COLS = [
    'duration', 'len', 'mean', 'var', 'std', 'kurtosis', 'skew',
    'n_peaks', 'smooth10_n_peaks', 'smooth20_n_peaks',
    'diff_peaks', 'diff2_peaks', 'diff_var', 'diff2_var',
    'gaps_squared', 'len_weighted', 'var_div_duration', 'var_div_len',
]


def add_channel_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the channel column (drop_first=True to avoid collinearity)."""
    dummies = pd.get_dummies(df['channel'], prefix='ch', drop_first=True)
    return pd.concat([df, dummies], axis=1)


def add_sampling_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if sampling=5, 0 if sampling=1."""
    df = df.copy()
    df['sampling_5s'] = (df['sampling'] == 5).astype(int)
    return df


def build_feature_matrix(df: pd.DataFrame, use_channel_dummies: bool = True,
                          use_sampling_flag: bool = True) -> tuple:
    """
    Build (X, y) from a dataset.csv DataFrame.
    Returns (X: DataFrame, y: Series).
    """
    df = df.copy()
    if use_sampling_flag:
        df = add_sampling_flag(df)
    if use_channel_dummies:
        df = add_channel_dummies(df)

    cols = list(FEATURE_COLS)
    if use_sampling_flag:
        cols.append('sampling_5s')
    if use_channel_dummies:
        ch_cols = [c for c in df.columns if c.startswith('ch_')]
        cols += ch_cols

    X = df[cols].copy()
    y = df['anomaly'].copy()
    return X, y


def prepare_train_test(use_channel_dummies: bool = True,
                       use_sampling_flag: bool = True,
                       scale: bool = True) -> tuple:
    """
    Return (X_train, y_train, X_test, y_test, scaler).

    The scaler is fitted on X_train only and applied to both sets.
    If scale=False, scaler is None and features are unscaled.

    Guarantees:
      - official train/test split (train column)
      - no segment leakage
      - scaler never sees test data during fit
    """
    ds = load_dataset()
    train_df, test_df = get_official_split(ds)

    X_train, y_train = build_feature_matrix(train_df, use_channel_dummies, use_sampling_flag)
    X_test,  y_test  = build_feature_matrix(test_df,  use_channel_dummies, use_sampling_flag)

    # Align columns — test might miss a dummy column if a channel is absent
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )

    return X_train, y_train, X_test, y_test, scaler
