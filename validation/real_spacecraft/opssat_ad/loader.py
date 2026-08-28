"""
loader.py — OPS-SAT-AD dataset loader.

Loads the two canonical files:
  data/real_spacecraft/opssat_ad/segments.csv   (303,493 samples x 8 columns)
  data/real_spacecraft/opssat_ad/dataset.csv    (2,123 segments x 23 columns)

Column reference
  segments.csv:
    channel   : str  — telemetry channel ID (9 unique: CADC0872–CADC0894)
    timestamp : str  — UTC ISO-8601 timestamps (timezone-aware)
    value     : f64  — raw telemetry value
    label     : str  — always 'anomaly' (data-source tag, NOT the target)
    sampling  : int  — sampling period in seconds (1 or 5)
    anomaly   : int  — ground-truth binary label (0=nominal, 1=anomaly)
    segment   : int  — segment ID, 1-indexed, links to dataset.csv
    train     : int  — official split (1=train, 0=test)

  dataset.csv:
    segment   : int  — segment ID (1–2123, unique)
    anomaly   : int  — ground-truth label (0/1)
    train     : int  — official split (1=train, 0=test)
    channel   : str  — channel ID
    sampling  : int  — sampling period (1 or 5)
    duration  : int  — duration in seconds
    len       : int  — number of samples in segment
    mean,var,std,kurtosis,skew  : scalar statistics
    n_peaks,smooth10_n_peaks,smooth20_n_peaks : peak counts
    diff_peaks,diff2_peaks,diff_var,diff2_var : derivative statistics
    gaps_squared,len_weighted   : gap metrics
    var_div_duration,var_div_len: ratio features

NOTE — the `label` column in segments.csv is not the target variable.
       The target is the `anomaly` column (0 or 1).
"""

import os
import pandas as pd

_DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', '..', 'data', 'real_spacecraft', 'opssat_ad'
)

SEG_PATH = os.path.join(_DATA_ROOT, 'segments.csv')
DS_PATH  = os.path.join(_DATA_ROOT, 'dataset.csv')


def load_segments(parse_timestamps: bool = True) -> pd.DataFrame:
    """Return the full segments.csv (303,493 rows)."""
    df = pd.read_csv(SEG_PATH)
    if parse_timestamps:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


def load_dataset() -> pd.DataFrame:
    """Return the full dataset.csv (2,123 rows, one row per segment)."""
    return pd.read_csv(DS_PATH)


def get_official_split(df: pd.DataFrame = None):
    """
    Return (train_df, test_df) using the official train/test column.
    Works on either segments.csv or dataset.csv DataFrames.
    If df is None, loads dataset.csv.
    """
    if df is None:
        df = load_dataset()
    train = df[df['train'] == 1].copy()
    test  = df[df['train'] == 0].copy()
    return train, test


def get_feature_matrix(df: pd.DataFrame = None):
    """
    Return (X, y, groups) from dataset.csv for ML experiments.
      X      : DataFrame of 18 engineered features
      y      : Series of binary anomaly labels (0/1)
      groups : Series of channel IDs (for stratification)
    """
    if df is None:
        df = load_dataset()
    FEATURE_COLS = [
        'duration', 'len', 'mean', 'var', 'std', 'kurtosis', 'skew',
        'n_peaks', 'smooth10_n_peaks', 'smooth20_n_peaks',
        'diff_peaks', 'diff2_peaks', 'diff_var', 'diff2_var',
        'gaps_squared', 'len_weighted', 'var_div_duration', 'var_div_len',
    ]
    X = df[FEATURE_COLS].copy()
    y = df['anomaly'].copy()
    groups = df['channel'].copy()
    return X, y, groups


def describe_split() -> dict:
    """Return a dict summarising the official split."""
    ds = load_dataset()
    train, test = get_official_split(ds)
    return {
        'total_segments'       : len(ds),
        'train_segments'       : len(train),
        'test_segments'        : len(test),
        'train_anomaly_pos'    : int(train['anomaly'].sum()),
        'train_anomaly_neg'    : int((train['anomaly'] == 0).sum()),
        'test_anomaly_pos'     : int(test['anomaly'].sum()),
        'test_anomaly_neg'     : int((test['anomaly'] == 0).sum()),
        'train_anomaly_rate'   : float(train['anomaly'].mean()),
        'test_anomaly_rate'    : float(test['anomaly'].mean()),
        'channels'             : sorted(ds['channel'].unique().tolist()),
        'leakage_segments'     : int(len(set(train['segment']) & set(test['segment']))),
    }
