"""
dataset.py
==========
Builds the labelled dataset of received signal windows for both classes,
across the full SNR / channel / modulation sweep.

Each window is one received signal y[n] of length `n_samples` that a detector
might see. The sweep grid is:
    SNR levels   (e.g. -20 to +10 dB step 5)
  x channels     (AWGN, Rayleigh, Rician)
  x both classes (H1 signal present, H0 noise only)
With n_per_class windows per (SNR, channel) cell per class, and for H1 the
modulation is randomly drawn from the configured modulation list.

CRITICAL DATASET RULES
----------------------
1. Noise statistics are identical for H0 and H1: complex Gaussian, unit
   variance, drawn the same way. The two classes differ ONLY by whether a
   scaled signal is also present.
2. SNR, modulation, and channel are stored as METADATA only. They must not be
   used as features at training time (the model would learn label shortcuts).
3. Each window has its own derived seed so it is independently reproducible.

Returned structure
------------------
A dict with three aligned-by-index pieces:
    'windows'  : np.ndarray (complex), shape (N, n_samples)
    'labels'   : np.ndarray (int),     shape (N,)   - 1 for H1, 0 for H0
    'metadata' : pd.DataFrame with columns
                 ['index', 'label', 'snr_db', 'channel', 'modulation', 'seed']
"""

import numpy as np
import pandas as pd

from .signals  import generate_bpsk, generate_qpsk, generate_16qam
from .channels import apply_awgn, apply_rayleigh, apply_rician, make_noise


MODULATION_FNS = {
    "bpsk":   generate_bpsk,
    "qpsk":   generate_qpsk,
    "16qam":  generate_16qam,
}

CHANNEL_FNS = {
    "awgn":     lambda s, snr_db, rng: apply_awgn    (s, snr_db, rng),
    "rayleigh": lambda s, snr_db, rng: apply_rayleigh(s, snr_db, rng),
    "rician":   lambda s, snr_db, rng: apply_rician  (s, snr_db, rng, K_dB=4.0),
}


def _seed_for(master_seed, index):
    """Per-window deterministic seed from a master seed and the window index."""
    return (master_seed * 1_000_003 + index) & 0xFFFFFFFF


def build_dataset(
    snr_db_list  = (-20, -15, -10, -5, 0, 5, 10),
    channels     = ("awgn", "rayleigh", "rician"),
    modulations  = ("bpsk", "qpsk", "16qam"),
    n_per_class  = 500,
    n_samples    = 1024,
    master_seed  = 42,
    verbose      = True,
):
    """
    Generate the full labelled dataset.

    For every (snr, channel) cell:
      - n_per_class H1 windows (modulation drawn uniformly from `modulations`)
      - n_per_class H0 windows (pure noise, same noise statistics)

    Returns
    -------
    dict with 'windows' (complex ndarray), 'labels' (int ndarray),
    'metadata' (pandas DataFrame), all aligned by row index.
    """
    snr_db_list = list(snr_db_list)
    channels    = list(channels)
    modulations = list(modulations)

    n_cells = len(snr_db_list) * len(channels)
    n_total = n_cells * 2 * n_per_class

    windows = np.empty((n_total, n_samples), dtype=np.complex128)
    labels  = np.empty(n_total, dtype=np.int8)
    meta    = []

    idx = 0
    for snr_db in snr_db_list:
        for ch_name in channels:
            ch_fn = CHANNEL_FNS[ch_name]

            # H1 windows: signal present
            for k in range(n_per_class):
                seed = _seed_for(master_seed, idx)
                rng  = np.random.default_rng(seed)
                mod_name = modulations[rng.integers(0, len(modulations))]
                s_tx     = MODULATION_FNS[mod_name](n_samples, rng)
                y        = ch_fn(s_tx, snr_db, rng)
                windows[idx] = y
                labels [idx] = 1
                meta.append({
                    "index": idx, "label": 1,
                    "snr_db": snr_db, "channel": ch_name,
                    "modulation": mod_name, "seed": int(seed),
                })
                idx += 1

            # H0 windows: noise only, same noise statistics
            for k in range(n_per_class):
                seed = _seed_for(master_seed, idx)
                rng  = np.random.default_rng(seed)
                y    = make_noise(n_samples, rng)
                windows[idx] = y
                labels [idx] = 0
                meta.append({
                    "index": idx, "label": 0,
                    "snr_db": snr_db, "channel": ch_name,
                    "modulation": "none", "seed": int(seed),
                })
                idx += 1

            if verbose:
                print(f"  SNR={snr_db:+4d} dB  channel={ch_name:<8s}  "
                      f"done {idx}/{n_total} windows")

    metadata = pd.DataFrame(meta)
    return {"windows": windows, "labels": labels, "metadata": metadata}