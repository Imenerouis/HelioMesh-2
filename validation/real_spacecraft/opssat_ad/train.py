"""
train.py — OPS-SAT-AD baseline model training.

Three models trained on the OFFICIAL TRAIN partition only:
  1. Majority baseline (always predict 0)
  2. Random Forest Classifier (supervised)
  3. Isolation Forest (unsupervised anomaly detector)

No HelioMesh simulation models are used here.
No frozen ML artifacts are modified.

Returns a dict mapping model_name -> fitted model (or baseline descriptor).
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from .preprocess import prepare_train_test


def train_all_models(n_estimators: int = 200, random_state: int = 42) -> dict:
    """
    Fit all three baselines on the official train partition.

    Returns
    -------
    dict with keys:
      'majority_baseline'   : dict(predict_class=int)
      'random_forest'       : fitted RandomForestClassifier
      'isolation_forest'    : fitted IsolationForest
      'X_train', 'y_train'  : training features/labels
      'X_test',  'y_test'   : test features/labels (NOT seen during fit)
      'scaler'              : fitted StandardScaler
    """
    X_train, y_train, X_test, y_test, scaler = prepare_train_test(
        use_channel_dummies=True,
        use_sampling_flag=True,
        scale=True,
    )

    # 1. Majority baseline
    majority_class = int(np.bincount(y_train.values).argmax())

    # 2. Random Forest
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',   # handles class imbalance
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # 3. Isolation Forest — trained on TRAIN set (including anomalies)
    # contamination = observed anomaly rate in train
    contamination = float(y_train.mean())
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_train)

    return {
        'majority_baseline' : {'predict_class': majority_class},
        'random_forest'     : rf,
        'isolation_forest'  : iso,
        'X_train'           : X_train,
        'y_train'           : y_train,
        'X_test'            : X_test,
        'y_test'            : y_test,
        'scaler'            : scaler,
        'contamination'     : contamination,
    }
