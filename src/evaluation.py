# """
# evaluation.py
# =============
# Computes performance metrics and produces the report figures.

# Communication-oriented metrics (the heart of a sensing study):
#     Pd  = TP / (TP + FN)     probability of detection
#     Pfa = FP / (FP + TN)     probability of false alarm
#     Pm  = FN / (TP + FN)     probability of missed detection (= 1 - Pd)

# Standard ML metrics:
#     accuracy, precision, recall, F1-score, ROC-AUC

# Plots:
#     confusion matrices, accuracy/Pd/Pfa vs SNR, ROC curves,
#     feature-ablation bar charts, channel-mismatch charts.

# Planned functions:
#     confusion_counts(...)
#     comms_metrics(...)
#     ml_metrics(...)
#     plot_roc(...)
#     plot_vs_snr(...)
# """

# """
# evaluation.py
# =============
# Compute performance metrics and produce the report figures/tables for Phase 7.

# Communication metrics (confusion-matrix-based):
#     Pd  = TP / (TP + FN)   probability of detection
#     Pfa = FP / (FP + TN)   probability of false alarm
#     Pm  = FN / (TP + FN)   probability of missed detection = 1 - Pd

# ML metrics: accuracy, precision, recall, F1, ROC-AUC.

# Key discipline: all detectors are compared at the SAME operating point.
# Standard convention: fix Pfa at a regulatory target (Pfa=0.1) and report Pd.
# For ML models this means: threshold predict_proba at whatever value achieves
# Pfa=0.1 on H0 samples in the evaluated subset, then compute Pd on H1.
# """

# import numpy as np
# import pandas as pd
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
#     confusion_matrix, roc_curve,
# )


# def comms_metrics(y_true, y_pred):
#     y_true = np.asarray(y_true).astype(int)
#     y_pred = np.asarray(y_pred).astype(int)
#     cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
#     tn, fp, fn, tp = cm.ravel()
#     total_h1 = tp + fn
#     total_h0 = fp + tn
#     pd_val = tp / total_h1 if total_h1 > 0 else np.nan
#     pfa = fp / total_h0 if total_h0 > 0 else np.nan
#     pm = 1 - pd_val if not np.isnan(pd_val) else np.nan
#     return {"Pd": pd_val, "Pfa": pfa, "Pm": pm,
#             "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


# def ml_metrics(y_true, y_pred, y_score):
#     y_true = np.asarray(y_true).astype(int)
#     y_pred = np.asarray(y_pred).astype(int)
#     metrics = {
#         "accuracy":  accuracy_score(y_true, y_pred),
#         "precision": precision_score(y_true, y_pred, zero_division=0),
#         "recall":    recall_score(y_true, y_pred, zero_division=0),
#         "f1":        f1_score(y_true, y_pred, zero_division=0),
#     }
#     if len(np.unique(y_true)) == 2:
#         metrics["auc"] = roc_auc_score(y_true, y_score)
#     else:
#         metrics["auc"] = np.nan
#     return metrics


# def pd_at_target_pfa_score(y_true, y_score, target_pfa=0.1):
#     y_true  = np.asarray(y_true).astype(int)
#     y_score = np.asarray(y_score, dtype=float)
#     s_h0 = y_score[y_true == 0]
#     s_h1 = y_score[y_true == 1]
#     if len(s_h0) == 0 or len(s_h1) == 0:
#         return np.nan, np.nan, np.nan
#     threshold    = float(np.quantile(s_h0, 1.0 - target_pfa))
#     achieved_pfa = float(np.mean(s_h0 > threshold))
#     pd_val       = float(np.mean(s_h1 > threshold))
#     return pd_val, threshold, achieved_pfa


# def evaluate_detector_by_condition(y_true, y_score, meta, target_pfa=0.1,
#                                     detector_name="detector"):
#     y_true  = np.asarray(y_true).astype(int)
#     y_score = np.asarray(y_score, dtype=float)
#     meta = meta.reset_index(drop=True)
#     rows = []
#     for snr in sorted(meta["snr_db"].unique()):
#         for ch in sorted(meta["channel"].unique()):
#             mask = ((meta["snr_db"] == snr) & (meta["channel"] == ch)).values
#             if mask.sum() == 0:
#                 continue
#             pd_val, thr, apfa = pd_at_target_pfa_score(
#                 y_true[mask], y_score[mask], target_pfa
#             )
#             rows.append({
#                 "detector":     detector_name,
#                 "snr_db":       snr,
#                 "channel":      ch,
#                 "Pd":           pd_val,
#                 "Pfa_achieved": apfa,
#                 "threshold":    thr,
#             })
#     return pd.DataFrame(rows)


# def confusion_by_condition(y_true, y_pred, meta, detector_name="detector"):
#     y_true = np.asarray(y_true).astype(int)
#     y_pred = np.asarray(y_pred).astype(int)
#     meta = meta.reset_index(drop=True)
#     rows = []
#     for snr in sorted(meta["snr_db"].unique()):
#         for ch in sorted(meta["channel"].unique()):
#             mask = ((meta["snr_db"] == snr) & (meta["channel"] == ch)).values
#             if mask.sum() == 0: continue
#             m = comms_metrics(y_true[mask], y_pred[mask])
#             m.update({"detector": detector_name, "snr_db": snr, "channel": ch})
#             rows.append(m)
#     return pd.DataFrame(rows)

# def run_feature_ablation(features_csv_path, output_dir="outputs/tables", seed=42,
#                           model_types=("svm", "rf", "mlp")):
#     """
#     Retrain each model on each feature subset (FS1..FS4), evaluate on the
#     held-out test split at Pfa=0.1 per (SNR, channel).

#     Uses the SAME stratified split as Phase 6 so results are comparable
#     across feature sets and against the main comparison.

#     Uses 3-fold CV (not 5) to keep runtime reasonable with 12 model fits.
#     """
#     from pathlib import Path
#     from .models import load_features_and_labels, stratified_split, make_pipeline, get_param_grid
#     from .features import FEATURE_SETS
#     from sklearn.model_selection import GridSearchCV, StratifiedKFold

#     Path(output_dir).mkdir(parents=True, exist_ok=True)

#     X_all, y_all, meta = load_features_and_labels(features_csv_path)
#     splits = stratified_split(X_all, y_all, meta, seed=seed)
#     X_tr, y_tr, meta_tr = splits["train"]
#     X_te, y_te, meta_te = splits["test"]

#     rows = []
#     for fs_name, feat_cols in FEATURE_SETS.items():
#         print(f"\n--- Feature set: {fs_name} ({len(feat_cols)} features) ---")
#         X_tr_sub = X_tr[feat_cols]
#         X_te_sub = X_te[feat_cols]

#         for model_type in model_types:
#             print(f"  training {model_type.upper()}...", end=" ", flush=True)
#             pipe  = make_pipeline(model_type, seed=seed)
#             grid  = get_param_grid(model_type)
#             cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
#             search = GridSearchCV(
#                 pipe, grid, scoring="roc_auc", cv=cv,
#                 n_jobs=-1, verbose=0, refit=True
#             )
#             search.fit(X_tr_sub, y_tr)
#             y_score = search.predict_proba(X_te_sub)[:, 1]
#             per_condition = evaluate_detector_by_condition(
#                 y_te, y_score, meta_te, target_pfa=0.1,
#                 detector_name=f"{model_type.upper()}_{fs_name}"
#             )
#             per_condition.insert(0, "feature_set", fs_name)
#             per_condition.insert(0, "model",       model_type.upper())
#             rows.append(per_condition)
#             print(f"CV AUC = {search.best_score_:.4f}")

#     all_results = pd.concat(rows, ignore_index=True)
#     all_results.to_csv(Path(output_dir) / "feature_ablation.csv", index=False)
#     print(f"\nSaved feature_ablation.csv to {output_dir}/")
#     return all_results

# def run_channel_mismatch(features_csv_path, output_dir="outputs/tables", seed=42,
#                           model_types=("svm","rf","mlp"),
#                           train_channel="awgn"):
#     """
#     Train each model on ONE channel only (default: AWGN), then evaluate on
#     ALL three channels of the held-out test set.

#     Also computes a MATCHED baseline (trained on the full multi-channel
#     training set) for direct comparison per (test_channel, snr).
#     """
#     from pathlib import Path
#     from .models import load_features_and_labels, stratified_split, make_pipeline, get_param_grid
#     from sklearn.model_selection import GridSearchCV, StratifiedKFold

#     Path(output_dir).mkdir(parents=True, exist_ok=True)

#     X_all, y_all, meta = load_features_and_labels(features_csv_path)
#     splits = stratified_split(X_all, y_all, meta, seed=seed)
#     X_tr, y_tr, meta_tr = splits["train"]
#     X_te, y_te, meta_te = splits["test"]

#     train_mask = (meta_tr["channel"] == train_channel).values
#     X_tr_ch = X_tr[train_mask]
#     y_tr_ch = y_tr[train_mask]
#     print(f"Mismatch training set: {len(y_tr_ch)} samples (channel={train_channel})")
#     print(f"Matched training set:  {len(y_tr)} samples (all channels)")

#     rows = []
#     for mt in model_types:
#         print(f"\n=== {mt.upper()} ===")
#         print(f"  training on {train_channel} only... ", end="", flush=True)
#         pipe = make_pipeline(mt, seed=seed); grid = get_param_grid(mt)
#         cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
#         srch_mm = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0, refit=True)
#         srch_mm.fit(X_tr_ch, y_tr_ch)
#         y_score_mm = srch_mm.predict_proba(X_te)[:,1]
#         pc_mm = evaluate_detector_by_condition(y_te, y_score_mm, meta_te, 0.1,
#                                                 f"{mt.upper()}_mismatch")
#         pc_mm = pc_mm.rename(columns={"channel": "test_channel"})
#         pc_mm.insert(0, "training", f"{train_channel}_only")
#         pc_mm.insert(0, "model", mt.upper())
#         rows.append(pc_mm)
#         print(f"CV AUC = {srch_mm.best_score_:.4f}")

#         print(f"  training on all channels (matched baseline)... ", end="", flush=True)
#         pipe2 = make_pipeline(mt, seed=seed)
#         srch_m = GridSearchCV(pipe2, grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0, refit=True)
#         srch_m.fit(X_tr, y_tr)
#         y_score_m = srch_m.predict_proba(X_te)[:,1]
#         pc_m = evaluate_detector_by_condition(y_te, y_score_m, meta_te, 0.1,
#                                                f"{mt.upper()}_matched")
#         pc_m = pc_m.rename(columns={"channel": "test_channel"})
#         pc_m.insert(0, "training", "matched")
#         pc_m.insert(0, "model", mt.upper())
#         rows.append(pc_m)
#         print(f"CV AUC = {srch_m.best_score_:.4f}")

#     all_r = pd.concat(rows, ignore_index=True)
#     all_r.to_csv(Path(output_dir)/"channel_mismatch.csv", index=False)
#     return all_r


"""
evaluation.py
=============
Compute performance metrics and produce the report figures/tables.

Communication metrics (confusion-matrix-based):
    Pd  = TP / (TP + FN)   probability of detection
    Pfa = FP / (FP + TN)   probability of false alarm
    Pm  = FN / (TP + FN)   probability of missed detection = 1 - Pd

ML metrics: accuracy, precision, recall, F1, ROC-AUC.

Key discipline: all detectors are compared at the SAME operating point.
Standard convention: fix Pfa at a regulatory target (Pfa=0.1) and report Pd.
For ML models this means: threshold predict_proba at whatever value achieves
Pfa=0.1 on H0 samples in the evaluated subset, then compute Pd on H1.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)


# Default six-model list. Update here if new models are added later.
DEFAULT_MODELS = ("logreg", "dtree", "knn", "svm", "rf", "mlp")


def comms_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total_h1 = tp + fn
    total_h0 = fp + tn
    pd_val = tp / total_h1 if total_h1 > 0 else np.nan
    pfa = fp / total_h0 if total_h0 > 0 else np.nan
    pm = 1 - pd_val if not np.isnan(pd_val) else np.nan
    return {"Pd": pd_val, "Pfa": pfa, "Pm": pm,
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


def ml_metrics(y_true, y_pred, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        metrics["auc"] = roc_auc_score(y_true, y_score)
    else:
        metrics["auc"] = np.nan
    return metrics


def pd_at_target_pfa_score(y_true, y_score, target_pfa=0.1):
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    s_h0 = y_score[y_true == 0]
    s_h1 = y_score[y_true == 1]
    if len(s_h0) == 0 or len(s_h1) == 0:
        return np.nan, np.nan, np.nan
    threshold    = float(np.quantile(s_h0, 1.0 - target_pfa))
    achieved_pfa = float(np.mean(s_h0 > threshold))
    pd_val       = float(np.mean(s_h1 > threshold))
    return pd_val, threshold, achieved_pfa


def evaluate_detector_by_condition(y_true, y_score, meta, target_pfa=0.1,
                                    detector_name="detector"):
    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    meta = meta.reset_index(drop=True)
    rows = []
    for snr in sorted(meta["snr_db"].unique()):
        for ch in sorted(meta["channel"].unique()):
            mask = ((meta["snr_db"] == snr) & (meta["channel"] == ch)).values
            if mask.sum() == 0:
                continue
            pd_val, thr, apfa = pd_at_target_pfa_score(
                y_true[mask], y_score[mask], target_pfa
            )
            rows.append({
                "detector":     detector_name,
                "snr_db":       snr,
                "channel":      ch,
                "Pd":           pd_val,
                "Pfa_achieved": apfa,
                "threshold":    thr,
            })
    return pd.DataFrame(rows)


def confusion_by_condition(y_true, y_pred, meta, detector_name="detector"):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    meta = meta.reset_index(drop=True)
    rows = []
    for snr in sorted(meta["snr_db"].unique()):
        for ch in sorted(meta["channel"].unique()):
            mask = ((meta["snr_db"] == snr) & (meta["channel"] == ch)).values
            if mask.sum() == 0:
                continue
            m = comms_metrics(y_true[mask], y_pred[mask])
            m.update({"detector": detector_name, "snr_db": snr, "channel": ch})
            rows.append(m)
    return pd.DataFrame(rows)


def run_feature_ablation(features_csv_path='data/features.csv',
                          metadata_csv_path='data/metadata.csv',
                          output_dir="outputs/tables",
                          seed=42,
                          model_types=DEFAULT_MODELS):
    """
    Retrain each model on each feature subset (FS1..FS4), evaluate on the
    held-out test split at Pfa=0.1 per (SNR, channel).

    Now defaults to all 6 models. Pass a shorter tuple if you want fewer.

    Uses 3-fold CV (not 5) to keep runtime reasonable across many fits.
    """
    from pathlib import Path
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    from .features import FEATURE_SETS
    from .models import (get_param_grid, load_features_and_labels,
                          make_pipeline, stratified_split)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X_all, y_all, meta = load_features_and_labels(features_csv_path, metadata_csv_path)
    splits = stratified_split(X_all, y_all, meta, seed=seed)
    X_tr, y_tr, meta_tr = splits["train"]
    X_te, y_te, meta_te = splits["test"]

    rows = []
    for fs_name, feat_cols in FEATURE_SETS.items():
        print(f"\n--- Feature set: {fs_name} ({len(feat_cols)} features) ---")
        X_tr_sub = X_tr[feat_cols]
        X_te_sub = X_te[feat_cols]

        for model_type in model_types:
            print(f"  training {model_type.upper()}...", end=" ", flush=True)
            pipe  = make_pipeline(model_type, seed=seed)
            grid  = get_param_grid(model_type)
            cv    = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            search = GridSearchCV(
                pipe, grid, scoring="roc_auc", cv=cv,
                n_jobs=-1, verbose=0, refit=True
            )
            search.fit(X_tr_sub, y_tr)
            y_score = search.predict_proba(X_te_sub)[:, 1]
            per_condition = evaluate_detector_by_condition(
                y_te, y_score, meta_te, target_pfa=0.1,
                detector_name=f"{model_type.upper()}_{fs_name}"
            )
            per_condition.insert(0, "feature_set", fs_name)
            per_condition.insert(0, "model",       model_type.upper())
            rows.append(per_condition)
            print(f"CV AUC = {search.best_score_:.4f}")

    all_results = pd.concat(rows, ignore_index=True)
    all_results.to_csv(Path(output_dir) / "feature_ablation.csv", index=False)
    print(f"\nSaved feature_ablation.csv to {output_dir}/")
    return all_results


def run_channel_mismatch(features_csv_path='data/features.csv',
                          metadata_csv_path='data/metadata.csv',
                          output_dir="outputs/tables",
                          seed=42,
                          model_types=DEFAULT_MODELS,
                          train_channel="awgn"):
    """
    Train each model on ONE channel only (default: AWGN), then evaluate on
    ALL three channels of the held-out test set.

    Also computes a MATCHED baseline (trained on the full multi-channel
    training set) for direct comparison per (test_channel, snr).
    """
    from pathlib import Path
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    from .models import (get_param_grid, load_features_and_labels,
                          make_pipeline, stratified_split)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X_all, y_all, meta = load_features_and_labels(features_csv_path, metadata_csv_path)
    splits = stratified_split(X_all, y_all, meta, seed=seed)
    X_tr, y_tr, meta_tr = splits["train"]
    X_te, y_te, meta_te = splits["test"]

    train_mask = (meta_tr["channel"] == train_channel).values
    X_tr_ch = X_tr[train_mask]
    y_tr_ch = y_tr[train_mask]
    print(f"Mismatch training set: {len(y_tr_ch)} samples (channel={train_channel})")
    print(f"Matched training set:  {len(y_tr)} samples (all channels)")

    rows = []
    for mt in model_types:
        print(f"\n=== {mt.upper()} ===")
        print(f"  training on {train_channel} only... ", end="", flush=True)
        pipe = make_pipeline(mt, seed=seed); grid = get_param_grid(mt)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        srch_mm = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0, refit=True)
        srch_mm.fit(X_tr_ch, y_tr_ch)
        y_score_mm = srch_mm.predict_proba(X_te)[:, 1]
        pc_mm = evaluate_detector_by_condition(y_te, y_score_mm, meta_te, 0.1,
                                                f"{mt.upper()}_mismatch")
        pc_mm = pc_mm.rename(columns={"channel": "test_channel"})
        pc_mm.insert(0, "training", f"{train_channel}_only")
        pc_mm.insert(0, "model", mt.upper())
        rows.append(pc_mm)
        print(f"CV AUC = {srch_mm.best_score_:.4f}")

        print(f"  training on all channels (matched baseline)... ", end="", flush=True)
        pipe2 = make_pipeline(mt, seed=seed)
        srch_m = GridSearchCV(pipe2, grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0, refit=True)
        srch_m.fit(X_tr, y_tr)
        y_score_m = srch_m.predict_proba(X_te)[:, 1]
        pc_m = evaluate_detector_by_condition(y_te, y_score_m, meta_te, 0.1,
                                               f"{mt.upper()}_matched")
        pc_m = pc_m.rename(columns={"channel": "test_channel"})
        pc_m.insert(0, "training", "matched")
        pc_m.insert(0, "model", mt.upper())
        rows.append(pc_m)
        print(f"CV AUC = {srch_m.best_score_:.4f}")

    all_r = pd.concat(rows, ignore_index=True)
    all_r.to_csv(Path(output_dir) / "channel_mismatch.csv", index=False)
    return all_r

"""
Updated evaluation.py - adds run_modulation_mismatch for Experiment 7.

Rest of the file is unchanged from the previous version.
Only additions shown here; use this to update src/evaluation.py.
"""

def run_modulation_mismatch(features_csv_path='data/features.csv',
                             metadata_csv_path='data/metadata.csv',
                             output_dir="outputs/tables",
                             seed=42,
                             model_types=DEFAULT_MODELS):
    """
    Two modulation-mismatch configurations per document Section 17:

    Config 1: Train on {BPSK, QPSK, FSK}, test on same test set.
              Transfer loss measured on 16-QAM subset of test.

    Config 2: Train on {BPSK, QPSK, 16-QAM}, test on same test set.
              Transfer loss measured on FSK subset of test.

    Plus matched baseline: train on all 4 modulations, test on same test set.
    All three regimes evaluated on the SAME held-out test set for direct
    per-condition comparability.

    H0 (noise-only) windows have modulation='none' and are INCLUDED in every
    training configuration. Excluding them would break H0/H1 balance and make
    threshold calibration meaningless.

    Two Pd measurements per (SNR, channel) cell:
      - Pd_all: fraction of ALL H1 windows in the cell above threshold
      - Pd_unseen: fraction of H1 windows with the config's unseen modulation
                    above threshold (the transfer-loss measurement that matters)

    For the matched baseline both '16qam' and 'fsk' get their Pd_unseen
    computed so they can serve as the baseline for both configs.
    """
    from pathlib import Path

    import numpy as np
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    from .models import (get_param_grid, load_features_and_labels,
                         make_pipeline, stratified_split)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X_all, y_all, meta = load_features_and_labels(features_csv_path, metadata_csv_path)
    splits = stratified_split(X_all, y_all, meta, seed=seed)
    X_tr, y_tr, meta_tr = splits["train"]
    X_te, y_te, meta_te = splits["test"]

    # Definitions: each config keeps H0 (modulation='none') in training.
    CONFIGS = [
        {"name": "matched",  "train_mods": ["bpsk", "qpsk", "16qam", "fsk", "none"]},
        {"name": "config1_no_16qam", "train_mods": ["bpsk", "qpsk", "fsk", "none"]},
        {"name": "config2_no_fsk",   "train_mods": ["bpsk", "qpsk", "16qam", "none"]},
    ]

    print(f"Test set: {len(y_te)} windows total")
    print(f"  16-QAM H1 in test: "
      f"{int(((y_te == 1) & (meta_te['modulation'] == '16qam')).sum())}")
    print(f"  FSK H1 in test:    "
      f"{int(((y_te == 1) & (meta_te['modulation'] == 'fsk')).sum())}")

    rows = []
    for cfg in CONFIGS:
        train_mask = meta_tr['modulation'].isin(cfg['train_mods']).values
        X_tr_cfg = X_tr[train_mask]
        y_tr_cfg = y_tr[train_mask]
        print(f"\n=== {cfg['name']} training set: {len(y_tr_cfg)} windows ===")

        for mt in model_types:
            print(f"  training {mt.upper()}... ", end="", flush=True)
            pipe = make_pipeline(mt, seed=seed)
            grid = get_param_grid(mt)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv,
                                   n_jobs=-1, verbose=0, refit=True)
            search.fit(X_tr_cfg, y_tr_cfg)
            y_score = search.predict_proba(X_te)[:, 1]
            print(f"CV AUC = {search.best_score_:.4f}")

            # Per-condition evaluation.
            df_meta = meta_te.reset_index(drop=True).copy()
            df_meta['score'] = y_score
            df_meta['label'] = y_te

            for snr in sorted(df_meta['snr_db'].unique()):
                for ch in sorted(df_meta['channel'].unique()):
                    cell = df_meta[(df_meta['snr_db'] == snr) &
                                    (df_meta['channel'] == ch)]
                    h0 = cell[cell['label'] == 0]
                    h1_all = cell[cell['label'] == 1]
                    h1_16q = h1_all[h1_all['modulation'] == '16qam']
                    h1_fsk = h1_all[h1_all['modulation'] == 'fsk']

                    if len(h0) == 0:
                        thr = np.nan
                    else:
                        thr = float(np.quantile(h0['score'].values, 0.9))

                    def pd_at(subset):
                        if len(subset) == 0 or np.isnan(thr):
                            return np.nan
                        return float((subset['score'].values > thr).mean())

                    rows.append({
                        'model': mt.upper(),
                        'training': cfg['name'],
                        'snr_db': int(snr),
                        'channel': ch,
                        'threshold': thr,
                        'n_h0': len(h0),
                        'n_h1_all': len(h1_all),
                        'n_h1_16qam': len(h1_16q),
                        'n_h1_fsk': len(h1_fsk),
                        'Pd_all': pd_at(h1_all),
                        'Pd_16qam_only': pd_at(h1_16q),
                        'Pd_fsk_only': pd_at(h1_fsk),
                    })

    result = pd.DataFrame(rows)
    out_path = Path(output_dir) / "modulation_mismatch.csv"
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return result