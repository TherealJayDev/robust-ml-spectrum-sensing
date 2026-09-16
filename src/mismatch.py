"""Modulation mismatch experiment.

Trains classifiers with and without 16-QAM windows in the training set, tests
on the same held-out set (which includes 16-QAM), and reports the transfer
loss per (classifier, SNR, channel) at fixed probability of false alarm 0.1.

Analogous to the channel-mismatch experiment in Chapter 4 of the FYP report,
but along the modulation dimension.

Design:
- Test set: 20% of all 8,400 windows, stratified by (label, snr_db, channel,
  modulation), fixed seed. IDENTICAL between matched and mismatch regimes so
  per-condition transfer loss is directly comparable.
- Matched training: the other 80% (all modulations, including 16-QAM H1).
- Mismatch training: the same 80% MINUS all H1 windows with modulation '16qam'.
  H0 (label=0) windows have modulation='none' and are unaffected by this
  exclusion, so the H0 statistics used for threshold calibration are the same
  under both training regimes.

Two Pd measurements per (SNR, channel) cell:
- 'aggregate': computed over all H1 modulations in the test cell.
- '16qam_only': computed over H1 windows with modulation='16qam' in the cell.
  This is the actually-unseen-modulation Pd and is the transfer-loss quantity
  that matters.

Hyperparameters are fixed to the FYP cross-validation winners; they are NOT
re-tuned in this experiment, so the mismatch and matched results are directly
comparable to the Chapter 4 numbers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================================
# Configuration
# ============================================================================

SEED = 42
TARGET_PFA = 0.1
TEST_SIZE = 0.20
TEST_MOD = '16qam'          # the modulation the mismatch regime never sees

# Best hyperparameters from the FYP cross-validation. Fixed here so the
# mismatch experiment is directly comparable to Chapter 4 results.
HPARAMS = {
    'svm': {
        'C': 10.0, 'gamma': 0.01, 'kernel': 'rbf',
        'probability': True, 'random_state': SEED,
    },
    'rf': {
        'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 5,
        'random_state': SEED, 'n_jobs': -1,
    },
    'mlp': {
        'hidden_layer_sizes': (128,), 'alpha': 0.001,
        'learning_rate_init': 0.001, 'max_iter': 500,
        'random_state': SEED,
    },
}

META_COLS = ['index', 'label', 'snr_db', 'channel', 'modulation', 'seed']


# ============================================================================
# Data loading and splitting
# ============================================================================

def load_data(features_path='data/features.csv',
              metadata_path='data/metadata.csv'):
    """Load features (and metadata if needed), return merged DataFrame + feature column list.

    Handles two possible schemas:
      1. features.csv already contains metadata columns (label, snr_db, channel, modulation).
         In this case, features.csv is used as-is.
      2. features.csv has only features + index; metadata is in metadata.csv.
         In this case, the two files are merged on 'index'.
    """
    features = pd.read_csv(features_path)

    if 'label' in features.columns:
        # Schema 1: features already has metadata columns.
        df = features.copy()
        # Fill in any missing metadata columns from metadata.csv.
        needed = ['snr_db', 'channel', 'modulation']
        missing = [c for c in needed if c not in df.columns]
        if missing:
            metadata = pd.read_csv(metadata_path)
            df = df.merge(metadata[['index'] + missing], on='index', how='inner')
    else:
        # Schema 2: features.csv has only features + index; merge with metadata.
        metadata = pd.read_csv(metadata_path)
        df = metadata.merge(features, on='index', how='inner')

    feat_cols = [c for c in df.columns if c not in META_COLS]
    return df, feat_cols


def make_splits(df, feat_cols, test_size=TEST_SIZE, seed=SEED):
    """Create the matched training set, mismatch training set, and shared test set.

    Returns dict with keys: train_matched, train_mismatch, test, feature_names.
    Each of train_matched / train_mismatch / test is itself a dict with X, y, meta.
    """
    strat_key = (
        df['label'].astype(str) + '|' +
        df['snr_db'].astype(str) + '|' +
        df['channel'] + '|' +
        df['modulation']
    )
    train_idx, test_idx = train_test_split(
        df.index.values,
        test_size=test_size,
        stratify=strat_key,
        random_state=seed,
    )
    train_all = df.loc[train_idx].reset_index(drop=True)
    test = df.loc[test_idx].reset_index(drop=True)

    # Mismatch training: drop H1 windows with modulation=16qam.
    # H0 windows have modulation='none' so they are all retained.
    train_mm = train_all[train_all['modulation'] != TEST_MOD].reset_index(drop=True)

    def pack(frame):
        return {
            'X': frame[feat_cols].values,
            'y': frame['label'].values,
            'meta': frame[META_COLS].copy(),
        }

    return {
        'train_matched':  pack(train_all),
        'train_mismatch': pack(train_mm),
        'test':           pack(test),
        'feature_names':  feat_cols,
    }


# ============================================================================
# Model training
# ============================================================================

def build_pipeline(name):
    """Build a StandardScaler + classifier pipeline for name in {svm, rf, mlp}."""
    if name == 'svm':
        clf = SVC(**HPARAMS['svm'])
    elif name == 'rf':
        clf = RandomForestClassifier(**HPARAMS['rf'])
    elif name == 'mlp':
        clf = MLPClassifier(**HPARAMS['mlp'])
    else:
        raise ValueError(f'unknown model: {name!r}')
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def train_all(X, y, verbose=True):
    """Train SVM, RF, MLP on (X, y). Returns dict of fitted pipelines."""
    models = {}
    for name in ('svm', 'rf', 'mlp'):
        if verbose:
            print(f'  training {name} on {len(y)} windows...')
        pipe = build_pipeline(name)
        pipe.fit(X, y)
        models[name] = pipe
    return models


# ============================================================================
# Per-condition evaluation at fixed Pfa = 0.1
# ============================================================================

def pd_at_fixed_pfa(scores, y_true, target_pfa=TARGET_PFA):
    """Set threshold on H0 samples for target Pfa; return Pd on H1 samples.

    Returns np.nan if either H0 or H1 subset is empty (per-condition safety).
    """
    h0 = scores[y_true == 0]
    h1 = scores[y_true == 1]
    if len(h0) == 0 or len(h1) == 0:
        return np.nan
    threshold = np.quantile(h0, 1.0 - target_pfa)
    return float((h1 > threshold).mean())


def evaluate(model, test_pack, model_name, training_regime):
    """Compute per-condition Pd and 16qam-only Pd at fixed Pfa = 0.1.

    Returns a DataFrame with columns:
      model, training, snr_db, channel, n_h0, n_h1, n_h1_16qam,
      Pd_aggregate, Pd_16qam_only.

    Threshold is calibrated once per (snr, channel) cell using ALL H0 windows
    in that cell (H0 modulation is always 'none'). Two Pd values are reported:
      - Pd_aggregate: fraction of ALL H1 windows in that cell above threshold.
      - Pd_16qam_only: fraction of H1 windows with modulation=16qam above
        threshold. This is the unseen-modulation Pd for the mismatch regime
        and is what transfer loss is really measuring.
    """
    scores = model.predict_proba(test_pack['X'])[:, 1]
    df = test_pack['meta'].copy()
    df['score'] = scores

    rows = []
    for (snr, ch), group in df.groupby(['snr_db', 'channel']):
        h0_mask = group['label'] == 0
        h1_mask = group['label'] == 1
        h1_16q_mask = h1_mask & (group['modulation'] == '16qam')

        threshold = np.quantile(group.loc[h0_mask, 'score'].values, 1.0 - TARGET_PFA) \
            if h0_mask.any() else np.nan

        if np.isnan(threshold):
            pd_agg = np.nan
            pd_16q = np.nan
        else:
            pd_agg = float((group.loc[h1_mask, 'score'].values > threshold).mean()) \
                if h1_mask.any() else np.nan
            pd_16q = float((group.loc[h1_16q_mask, 'score'].values > threshold).mean()) \
                if h1_16q_mask.any() else np.nan

        rows.append({
            'model': model_name,
            'training': training_regime,
            'snr_db': int(snr),
            'channel': ch,
            'n_h0': int(h0_mask.sum()),
            'n_h1': int(h1_mask.sum()),
            'n_h1_16qam': int(h1_16q_mask.sum()),
            'Pd_aggregate': pd_agg,
            'Pd_16qam_only': pd_16q,
        })
    return pd.DataFrame(rows)


# ============================================================================
# Main experiment driver
# ============================================================================

def run_experiment(features_path='data/features.csv',
                   metadata_path='data/metadata.csv',
                   output_dir='outputs/tables',
                   verbose=True):
    """Full modulation-mismatch experiment.

    1. Load features + metadata, build stratified splits.
    2. Train SVM/RF/MLP on matched training set (all modulations).
    3. Train SVM/RF/MLP on mismatch training set (no 16qam H1).
    4. Evaluate both on the SHARED test set at fixed Pfa = 0.1 per condition.
    5. Save per-condition Pd (matched + mismatch) and transfer-loss summary.

    Returns (per_condition_df, loss_df).
    """
    if verbose:
        print(f'Loading data from {features_path} and {metadata_path}...')
    df, feat_cols = load_data(features_path, metadata_path)
    if verbose:
        print(f'  {len(df)} total windows, {len(feat_cols)} features')

    splits = make_splits(df, feat_cols)
    n_matched = len(splits['train_matched']['y'])
    n_mismatch = len(splits['train_mismatch']['y'])
    n_test = len(splits['test']['y'])
    n_16qam_test = int((splits['test']['meta']['modulation'] == '16qam').sum())
    if verbose:
        print(f'\nSplits:')
        print(f'  matched training:  {n_matched} windows')
        print(f'  mismatch training: {n_mismatch} windows'
              f' ({n_matched - n_mismatch} 16-QAM H1 excluded)')
        print(f'  test:              {n_test} windows'
              f' ({n_16qam_test} of which are 16-QAM H1)')

    if verbose:
        print('\nTraining MATCHED models (all modulations)...')
    models_matched = train_all(
        splits['train_matched']['X'],
        splits['train_matched']['y'],
        verbose=verbose,
    )

    if verbose:
        print('\nTraining MISMATCH models (no 16-QAM H1)...')
    models_mismatch = train_all(
        splits['train_mismatch']['X'],
        splits['train_mismatch']['y'],
        verbose=verbose,
    )

    if verbose:
        print('\nEvaluating per (SNR, channel) cell at Pfa = 0.1...')
    frames = []
    for name, model in models_matched.items():
        frames.append(evaluate(model, splits['test'], name, 'matched'))
    for name, model in models_mismatch.items():
        frames.append(evaluate(model, splits['test'], name, 'mismatch'))
    per_condition = pd.concat(frames, ignore_index=True)

    # Transfer loss: matched Pd minus mismatch Pd, per (model, snr, channel).
    piv_agg = per_condition.pivot_table(
        index=['model', 'snr_db', 'channel'],
        columns='training',
        values='Pd_aggregate',
    ).reset_index()
    piv_agg['loss_aggregate'] = piv_agg['matched'] - piv_agg['mismatch']

    piv_16q = per_condition.pivot_table(
        index=['model', 'snr_db', 'channel'],
        columns='training',
        values='Pd_16qam_only',
    ).reset_index()
    piv_16q['loss_16qam_only'] = piv_16q['matched'] - piv_16q['mismatch']

    loss = piv_agg[['model', 'snr_db', 'channel', 'loss_aggregate']].merge(
        piv_16q[['model', 'snr_db', 'channel', 'loss_16qam_only']],
        on=['model', 'snr_db', 'channel'],
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_condition_path = output_dir / 'modulation_mismatch.csv'
    loss_path = output_dir / 'modulation_mismatch_loss.csv'
    per_condition.to_csv(per_condition_path, index=False)
    loss.to_csv(loss_path, index=False)
    if verbose:
        print(f'\nWrote {per_condition_path}')
        print(f'Wrote {loss_path}')

    return per_condition, loss


if __name__ == '__main__':
    per_condition, loss = run_experiment()

    print('\n' + '=' * 70)
    print('SUMMARY: mean 16-QAM-only transfer loss per (model, SNR) across channels')
    print('=' * 70)
    summary = (
        loss.groupby(['model', 'snr_db'])['loss_16qam_only']
        .mean()
        .unstack('model')
        .round(3)
    )
    print(summary.to_string())

    print('\n' + '=' * 70)
    print('SUMMARY: mean 16-QAM-only transfer loss per model, low SNR only (-20, -15, -10 dB)')
    print('=' * 70)
    low = loss[loss['snr_db'].isin([-20, -15, -10])]
    print(low.groupby('model')['loss_16qam_only'].mean().round(3).to_string())