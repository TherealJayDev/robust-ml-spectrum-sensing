"""
models.py
=========
Train and tune six ML classifiers on the 4-modulation dataset:

    Simple baselines:
        Logistic Regression  (linear baseline)
        Decision Tree        (interpretable baseline)
        k-NN                 (distance-based baseline)

    Existing (already used in the undergraduate report):
        SVM (RBF kernel)
        Random Forest
        MLP

Design disciplines enforced structurally (unchanged from earlier version):

1. FEATURE COLUMN DISCIPLINE: features are selected via the explicit
   HYBRID_FEATURES list. Metadata columns are NEVER included.
2. LEAK-FREE SCALING: every model is wrapped in a Pipeline of
   [StandardScaler, model]. The scaler is fit on training folds only.
3. TRAIN/VAL/TEST SEPARATION: stratified 70/15/15. Tuning uses 5-fold CV
   on the training set only. The test set is touched exactly once.
4. STRATIFICATION ACROSS CONDITIONS: composite key (label + snr + channel)
   preserves the proportions of each in every split.
5. REPRODUCIBILITY: all seeds pinned.

Storage layout expected:
    data/features.csv   -> index + 21 feature columns
    data/metadata.csv   -> index + label + snr_db + channel + modulation + seed

Both files are merged on 'index' at load time. This keeps metadata separate
from features per the project document (section 9.4).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .features import HYBRID_FEATURES


# Model list is exposed at module level so notebook code can iterate easily.
ALL_MODELS = ["logreg", "dtree", "knn", "svm", "rf", "mlp"]


def load_features_and_labels(features_csv_path='data/features.csv',
                              metadata_csv_path='data/metadata.csv'):
    """Load features and metadata from separate files, merge on 'index'.

    Returns
    -------
    X    : DataFrame of 21 feature columns (order matches HYBRID_FEATURES)
    y    : ndarray of int labels (0 or 1)
    meta : DataFrame of metadata columns (index, snr_db, channel, modulation, seed)
           NOTE: label is removed from meta since it goes into y.
    """
    features = pd.read_csv(features_csv_path)
    metadata = pd.read_csv(metadata_csv_path)

    # Safety: both files should have 'index' and same row count.
    if 'index' not in features.columns:
        raise ValueError("features.csv missing 'index' column")
    if 'index' not in metadata.columns:
        raise ValueError("metadata.csv missing 'index' column")
    if len(features) != len(metadata):
        raise ValueError(
            f"row count mismatch: features={len(features)}, metadata={len(metadata)}"
        )

    df = metadata.merge(features, on='index', how='inner')
    if len(df) != len(features):
        raise ValueError("merge dropped rows; check 'index' alignment")

    X = df[HYBRID_FEATURES].copy()
    y = df['label'].values.astype(int)
    meta = df[['index', 'snr_db', 'channel', 'modulation', 'seed']].copy()
    return X, y, meta


def stratified_split(X, y, meta, test_size=0.15, val_size=0.15, seed=42):
    """Stratified 70/15/15 split by (label, snr_db, channel).

    Two-step split: first carve out the test set, then split the remainder
    into train and validation. Stratification key ensures every (label, SNR,
    channel) combination appears in all three splits proportionally.
    """
    strat = (
        pd.Series(y, index=meta.index).astype(str) + "_"
        + meta["snr_db"].astype(str) + "_"
        + meta["channel"].astype(str)
    )

    X_tv, X_te, y_tv, y_te, meta_tv, meta_te = train_test_split(
        X, y, meta, test_size=test_size, stratify=strat, random_state=seed
    )

    strat_tv = (
        pd.Series(y_tv, index=meta_tv.index).astype(str) + "_"
        + meta_tv["snr_db"].astype(str) + "_"
        + meta_tv["channel"].astype(str)
    )
    val_rel = val_size / (1.0 - test_size)
    X_tr, X_va, y_tr, y_va, meta_tr, meta_va = train_test_split(
        X_tv, y_tv, meta_tv, test_size=val_rel, stratify=strat_tv, random_state=seed
    )

    return {
        "train": (X_tr, y_tr, meta_tr),
        "val":   (X_va, y_va, meta_va),
        "test":  (X_te, y_te, meta_te),
    }


def make_pipeline(model_type, seed=42):
    """Build a [StandardScaler, classifier] pipeline for the named model."""
    if model_type == "logreg":
        clf = LogisticRegression(max_iter=1000, random_state=seed, n_jobs=-1)
    elif model_type == "dtree":
        clf = DecisionTreeClassifier(random_state=seed)
    elif model_type == "knn":
        clf = KNeighborsClassifier(n_jobs=-1)
    elif model_type == "svm":
        clf = SVC(probability=True, random_state=seed)
    elif model_type == "rf":
        clf = RandomForestClassifier(random_state=seed, n_jobs=-1)
    elif model_type == "mlp":
        clf = MLPClassifier(max_iter=500, random_state=seed)
    else:
        raise ValueError(f"unknown model_type: {model_type}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def get_param_grid(model_type):
    """Hyperparameter grid per model.

    Grid sizes are chosen so total fit time stays reasonable while still
    exploring the tuning axes named in the project document (Section 14).

    Choices explained:

    LOGREG (12 combinations):
        - Two penalty types (L1 and L2) capture the main regularisation
          shapes the document names. ElasticNet dropped because it needs
          special solver handling and rarely helps for this feature count.
        - Three C values span three orders of magnitude, standard spread.
        - Two solvers: 'liblinear' handles both L1 and L2 for binary
          classification; 'saga' is a fallback for larger datasets.
          Both work with our 21-dim features.

    DTREE (36 combinations):
        - Four depth values from shallow (3) to deep (30). Undergraduate
          document names max_depth as a key tuning axis; wide range chosen
          because tree depth interacts strongly with class boundary shape.
        - Three min_samples_split values control overfitting.
        - Three min_samples_leaf values are a second overfitting control.

    KNN (30 combinations):
        - Six neighbor counts from very local (3) to very smooth (25).
          Document names 'number of neighbors' as a tuning axis.
        - Two distance metrics: 'euclidean' and 'manhattan' (as the
          document says 'distance metric'). Cosine dropped because our
          features have very different magnitudes even after scaling.
        - Two weight options: 'uniform' and 'distance' (as the document
          says 'weighting method').

    SVM, RF, MLP: unchanged from previous version so results stay
    directly comparable to the undergraduate report.
    """
    if model_type == "logreg":
        return {
            "clf__penalty": ["l1", "l2"],
            "clf__C":       [0.1, 1.0, 10.0],
            "clf__solver":  ["liblinear", "saga"],
        }
    elif model_type == "dtree":
        return {
            "clf__max_depth":         [3, 10, 20, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf":  [1, 2, 5],
        }
    elif model_type == "knn":
        return {
            "clf__n_neighbors": [3, 5, 7, 11, 15, 25],
            "clf__metric":      ["euclidean", "manhattan"],
            "clf__weights":     ["uniform", "distance"],
        }
    elif model_type == "svm":
        return {
            "clf__C":      [0.1, 1, 10],
            "clf__gamma":  ["scale", 0.01, 0.1],
            "clf__kernel": ["rbf"],
        }
    elif model_type == "rf":
        return {
            "clf__n_estimators":     [100, 300],
            "clf__max_depth":        [None, 10, 20],
            "clf__min_samples_split": [2, 5],
        }
    elif model_type == "mlp":
        return {
            "clf__hidden_layer_sizes": [(64,), (128,), (64, 32)],
            "clf__alpha":              [1e-4, 1e-3],
            "clf__learning_rate_init": [1e-3],
        }
    raise ValueError(f"unknown model_type: {model_type}")


def train_and_tune(model_type, X_train, y_train, seed=42, cv_folds=5, verbose=True):
    """Grid-search CV on the training set, refit on winning params. Returns the search."""
    pipe = make_pipeline(model_type, seed=seed)
    grid = get_param_grid(model_type)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1,
        verbose=(2 if verbose else 0), refit=True,
    )
    search.fit(X_train, y_train)
    return search


def train_all_models(features_csv_path='data/features.csv',
                     metadata_csv_path='data/metadata.csv',
                     output_dir="outputs/models",
                     models=None,
                     seed=42):
    """Train (and tune) all requested models, save artefacts, return summary DataFrame.

    Parameters
    ----------
    models : list of str or None
        Model names to train. If None, trains all six in ALL_MODELS.
        Useful for iterative work (e.g. train only the 3 new ones first).
    """
    if models is None:
        models = ALL_MODELS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, meta = load_features_and_labels(features_csv_path, metadata_csv_path)
    splits = stratified_split(X, y, meta, seed=seed)

    # Save split indices so downstream experiments (mismatch, ablation) can
    # reuse the exact same train/val/test partition.
    for name in ["train", "val", "test"]:
        _, _, m = splits[name]
        m[["index"]].to_csv(output_dir / f"split_{name}_indices.csv", index=False)

    X_tr, y_tr, _ = splits["train"]
    X_va, y_va, _ = splits["val"]
    X_te, y_te, _ = splits["test"]
    print(f"Split sizes: train={len(y_tr)}, val={len(y_va)}, test={len(y_te)}")
    print(f"Training models: {models}")

    results_summary = []
    for name in models:
        print(f"\n=== {name.upper()} ===")
        search = train_and_tune(name, X_tr, y_tr, seed=seed, verbose=False)

        joblib.dump(search.best_estimator_, output_dir / f"{name}_pipeline.joblib")
        with open(output_dir / f"{name}_best_params.json", "w") as f:
            json.dump(search.best_params_, f, indent=2)
        pd.DataFrame(search.cv_results_).to_csv(
            output_dir / f"{name}_cv_results.csv", index=False
        )

        results_summary.append({
            "model":       name,
            "best_cv_auc": search.best_score_,
            "best_params": json.dumps(search.best_params_),
        })
        print(f"  best CV AUC = {search.best_score_:.4f}")
        print(f"  best params = {search.best_params_}")

    summary = pd.DataFrame(results_summary)
    summary.to_csv(output_dir / "training_summary.csv", index=False)
    print(f"\nAll models saved to {output_dir}/")
    return summary


def load_trained_model(output_dir, model_name):
    """Load a fitted pipeline previously saved by train_all_models."""
    return joblib.load(Path(output_dir) / f"{model_name}_pipeline.joblib")