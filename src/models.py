"""
models.py
=========
Defines and trains the machine-learning classifiers for the minimum version:
    - SVM (RBF kernel)
    - Random Forest
    - ANN / MLP

Handles the full, leak-free training workflow:
    1. split into train / validation / test
    2. FIT the StandardScaler on training data only, then transform all splits
    3. train each model with cross-validated hyperparameter tuning
    4. save trained model AND the fitted scaler together

This explicitly fixes the earlier bugs: the scaler is always instantiated and
fit before use, and the save/load order is well defined.

Planned functions:
    get_models(...)
    train_and_tune(...)
    save_model(...)
    load_model(...)
"""

"""
models.py
=========
Train and tune the three ML classifiers for the minimum-version project:
    SVM (RBF kernel), Random Forest, MLP.

Design disciplines enforced structurally:

1. FEATURE COLUMN DISCIPLINE: features are selected via the explicit
   HYBRID_FEATURES list. Metadata columns are NEVER included.
2. LEAK-FREE SCALING: every model is wrapped in a Pipeline of
   [StandardScaler, model]. The scaler is fit on training folds only.
3. TRAIN/VAL/TEST SEPARATION: stratified 70/15/15. Tuning uses 5-fold CV
   on the training set only. The test set is touched exactly once.
4. STRATIFICATION ACROSS CONDITIONS: composite key (label + snr + channel)
   preserves the proportions of each in every split.
5. REPRODUCIBILITY: all seeds pinned.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection  import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline         import Pipeline
from sklearn.preprocessing    import StandardScaler
from sklearn.svm              import SVC
from sklearn.ensemble         import RandomForestClassifier
from sklearn.neural_network   import MLPClassifier

from .features import HYBRID_FEATURES


def load_features_and_labels(features_csv_path):
    df = pd.read_csv(features_csv_path)
    X    = df[HYBRID_FEATURES].copy()
    y    = df["label"].values.astype(int)
    meta = df.drop(columns=HYBRID_FEATURES + ["label"]).copy()
    return X, y, meta


def stratified_split(X, y, meta, test_size=0.15, val_size=0.15, seed=42):
    strat = (pd.Series(y, index=meta.index).astype(str) + "_"
             + meta["snr_db"].astype(str) + "_"
             + meta["channel"].astype(str))

    X_tv, X_te, y_tv, y_te, meta_tv, meta_te = train_test_split(
        X, y, meta, test_size=test_size, stratify=strat, random_state=seed
    )

    strat_tv = (pd.Series(y_tv, index=meta_tv.index).astype(str) + "_"
                + meta_tv["snr_db"].astype(str) + "_"
                + meta_tv["channel"].astype(str))
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
    if model_type == "svm":
        clf = SVC(probability=True, random_state=seed)
    elif model_type == "rf":
        clf = RandomForestClassifier(random_state=seed, n_jobs=-1)
    elif model_type == "mlp":
        clf = MLPClassifier(max_iter=500, random_state=seed)
    else:
        raise ValueError(f"unknown model_type: {model_type}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def get_param_grid(model_type):
    if model_type == "svm":
        return {"clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", 0.01, 0.1],
                "clf__kernel": ["rbf"]}
    elif model_type == "rf":
        return {"clf__n_estimators": [100, 300],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5]}
    elif model_type == "mlp":
        return {"clf__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "clf__alpha": [1e-4, 1e-3],
                "clf__learning_rate_init": [1e-3]}
    raise ValueError(f"unknown model_type: {model_type}")


def train_and_tune(model_type, X_train, y_train, seed=42, cv_folds=5, verbose=True):
    pipe  = make_pipeline(model_type, seed=seed)
    grid  = get_param_grid(model_type)
    cv    = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1,
        verbose=(2 if verbose else 0), refit=True,
    )
    search.fit(X_train, y_train)
    return search


def train_all_models(features_csv_path, output_dir="outputs/models", seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, meta = load_features_and_labels(features_csv_path)
    splits = stratified_split(X, y, meta, seed=seed)

    for name in ["train", "val", "test"]:
        _, _, m = splits[name]
        m[["index"]].to_csv(output_dir / f"split_{name}_indices.csv", index=False)

    X_tr, y_tr, _ = splits["train"]
    print(f"Training on {len(y_tr)}, val {len(splits['val'][1])}, test {len(splits['test'][1])}.")

    results_summary = []
    for name in ["svm", "rf", "mlp"]:
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
    return joblib.load(Path(output_dir) / f"{model_name}_pipeline.joblib")