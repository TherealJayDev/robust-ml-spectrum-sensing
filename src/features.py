"""
features.py
===========
Turns each raw received signal window y[n] into a fixed-length feature vector.

Three feature families (the "hybrid" feature set), defined so we can run the
feature-ablation study later:
    Time-domain        -> mean, variance, std, RMS, energy, skewness,
                          kurtosis, peak, PAPR
    Frequency-domain   -> FFT peak, dominant bin, spectral centroid/bandwidth,
                          spectral entropy, spectral flatness, PSD max/mean
    Correlation/eigen  -> autocorrelation peak, covariance trace,
                          max eigenvalue, eigenvalue ratio

Feature sets for the ablation study:
    FS1 = time only
    FS2 = frequency only
    FS3 = correlation/eigenvalue only
    FS4 = hybrid (all combined)

Planned functions:
    extract_time_features(...)
    extract_freq_features(...)
    extract_corr_features(...)
    extract_all_features(...)
"""

"""
features.py
===========
Turn each raw complex window y[n] into a fixed-length feature vector.

Three feature families (the "hybrid" set):

    TIME_FEATURES     : mean_abs, variance, std, rms, energy, skewness,
                        kurtosis, peak, papr
    FREQ_FEATURES     : fft_peak, dominant_bin, spectral_centroid,
                        spectral_bandwidth, spectral_entropy,
                        spectral_flatness, psd_max, psd_mean
    CORR_FEATURES     : autocorr_peak, cov_trace, max_eigenvalue,
                        eigenvalue_ratio

Feature sets for the ablation study (Phase 7):
    FS1 = TIME_FEATURES
    FS2 = FREQ_FEATURES
    FS3 = CORR_FEATURES
    FS4 = HYBRID = TIME + FREQ + CORR
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch


def extract_time_features(y):
    mag = np.abs(y)
    N = len(y)
    return {
        "mean_abs":  np.mean(mag),
        "variance":  np.var(mag),
        "std":       np.std(mag),
        "rms":       np.sqrt(np.mean(mag ** 2)),
        "energy":    np.sum(mag ** 2) / N,
        "skewness":  float(skew(mag)),
        "kurtosis":  float(kurtosis(mag)),
        "peak":      np.max(mag),
        "papr":      (np.max(mag) ** 2) / (np.mean(mag ** 2) + 1e-12),
    }


def extract_freq_features(y):
    N = len(y)
    Y = np.fft.fftshift(np.fft.fft(y))
    mag_spec = np.abs(Y)
    freqs = np.fft.fftshift(np.fft.fftfreq(N))

    fft_peak = np.max(mag_spec)
    dominant_bin = int(np.argmax(mag_spec) - N // 2)
    total_mag = np.sum(mag_spec) + 1e-12
    spectral_centroid = float(np.sum(freqs * mag_spec) / total_mag)
    spectral_bandwidth = float(
        np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * mag_spec) / total_mag)
    )

    _, psd = welch(y, nperseg=min(256, N), return_onesided=False)
    psd = np.abs(psd) + 1e-12
    psd_norm = psd / np.sum(psd)
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm)))
    spectral_flatness = float(np.exp(np.mean(np.log(psd))) / np.mean(psd))

    return {
        "fft_peak":           float(fft_peak),
        "dominant_bin":       dominant_bin,
        "spectral_centroid":  spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_entropy":   spectral_entropy,
        "spectral_flatness":  spectral_flatness,
        "psd_max":            float(np.max(psd)),
        "psd_mean":           float(np.mean(psd)),
    }


def extract_corr_features(y, L=8):
    N = len(y)
    mag = np.abs(y)

    ac = np.correlate(mag - np.mean(mag), mag - np.mean(mag), mode="full")
    ac = ac[ac.size // 2:]
    ac = ac / (ac[0] + 1e-12)
    autocorr_peak = float(np.max(np.abs(ac[1:20])))

    M = N - L + 1
    Y = np.lib.stride_tricks.sliding_window_view(y, L)
    Y_c = Y - Y.mean(axis=0, keepdims=True)
    R = (Y_c.conj().T @ Y_c) / M
    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.maximum(eigvals, 1e-12)

    return {
        "autocorr_peak":    autocorr_peak,
        "cov_trace":        float(np.real(np.trace(R))),
        "max_eigenvalue":   float(eigvals[-1]),
        "eigenvalue_ratio": float(eigvals[-1] / eigvals[0]),
    }


def extract_all_features(y):
    d = {}
    d.update(extract_time_features(y))
    d.update(extract_freq_features(y))
    d.update(extract_corr_features(y))
    return d


TIME_FEATURES = ["mean_abs","variance","std","rms","energy",
                 "skewness","kurtosis","peak","papr"]
FREQ_FEATURES = ["fft_peak","dominant_bin","spectral_centroid","spectral_bandwidth",
                 "spectral_entropy","spectral_flatness","psd_max","psd_mean"]
CORR_FEATURES = ["autocorr_peak","cov_trace","max_eigenvalue","eigenvalue_ratio"]
HYBRID_FEATURES = TIME_FEATURES + FREQ_FEATURES + CORR_FEATURES

FEATURE_SETS = {
    "FS1_time":   TIME_FEATURES,
    "FS2_freq":   FREQ_FEATURES,
    "FS3_corr":   CORR_FEATURES,
    "FS4_hybrid": HYBRID_FEATURES,
}


def build_feature_matrix(windows, verbose=False):
    rows = []
    for i, y in enumerate(windows):
        rows.append(extract_all_features(y))
        if verbose and (i + 1) % 1000 == 0:
            print(f"  extracted features for {i + 1}/{len(windows)} windows")
    return pd.DataFrame(rows)
