"""
energy_detector.py
==================
Classical energy detection: the compulsory baseline every spectrum-sensing
study must include.

Test statistic:
    T_ED = (1/N) * sum(|y[n]|^2)

is the average per-sample power over one received window. Decision:
    T_ED >  lambda  -> declare H1 (occupied)
    T_ED <= lambda  -> declare H0 (free)

Two operational metrics of primary interest:
    Pd  = probability of detection    (correctly flag H1)
    Pfa = probability of false alarm  (wrongly flag H0 as H1)

We do NOT choose a "best" threshold. Instead we sweep threshold over its full
range and produce the (Pfa, Pd) curve. Real deployments then pick a threshold
that achieves a regulatorily-required Pfa target (e.g. Pfa = 0.1), and the
detector is reported by its Pd at that target.

Design rule: this module operates on the SAME received windows y[n] that the
ML detectors see. Fair comparison requires identical inputs.


energy_statistic(y): one window in, one scalar out. Direct implementation of T_ED.
decide(t_ed, threshold): applies the decision rule. Trivial.
sweep_threshold(t_ed_values, labels): takes a bunch of test statistics and their true labels, sweeps the threshold from low to high, and returns arrays of (Pfa, Pd) pairs. This is what generates a ROC curve.
pd_at_target_pfa(t_ed_values, labels, target_pfa): given a Pfa target, finds the threshold that achieves it on H0 windows, then reports Pd at that threshold. This is the operational metric.
"""

import numpy as np


def energy_statistic(y):
    """T_ED for one window."""
    return float(np.mean(np.abs(y) ** 2))


def energy_statistics(windows):
    """Vectorised T_ED for a batch of windows, shape (M, N) -> (M,)."""
    return np.mean(np.abs(windows) ** 2, axis=1)


def decide(t_ed, threshold):
    """Boolean H1/H0 decision."""
    return t_ed > threshold


def sweep_threshold(t_ed_values, labels, n_points=200):
    """
    Sweep the decision threshold, return arrays for a ROC curve.

    Returns
    -------
    pfa : ndarray of Pfa values
    pd  : ndarray of Pd values  (aligned with pfa)
    thresholds : ndarray of threshold values used
    """
    t_ed_values = np.asarray(t_ed_values)
    labels      = np.asarray(labels).astype(int)

    lo = np.min(t_ed_values) - 1e-6
    hi = np.max(t_ed_values) + 1e-6
    thresholds = np.linspace(lo, hi, n_points)

    t_h1 = t_ed_values[labels == 1]
    t_h0 = t_ed_values[labels == 0]

    pd_arr  = np.array([np.mean(t_h1 > th) for th in thresholds])
    pfa_arr = np.array([np.mean(t_h0 > th) for th in thresholds])
    return pfa_arr, pd_arr, thresholds


def pd_at_target_pfa(t_ed_values, labels, target_pfa=0.1):
    """
    Set the threshold to achieve target_pfa on H0 windows, report Pd on H1.

    Returns
    -------
    pd : float, achieved Pd
    threshold : float, the threshold used
    achieved_pfa : float, actual Pfa (may differ slightly from target)
    """
    t_ed_values = np.asarray(t_ed_values)
    labels      = np.asarray(labels).astype(int)

    t_h0 = t_ed_values[labels == 0]
    t_h1 = t_ed_values[labels == 1]

    threshold    = float(np.quantile(t_h0, 1.0 - target_pfa))
    achieved_pfa = float(np.mean(t_h0 > threshold))
    pd_val       = float(np.mean(t_h1 > threshold))
    return pd_val, threshold, achieved_pfa