"""
channels.py
===========
Applies channel effects and noise to a transmitted signal s[n], producing the
received signal y[n].

Signal model (binary hypothesis):
    H0: y[n] = w[n]                  (channel free, noise only)
    H1: y[n] = alpha * h[n] * s[n] + w[n]   (channel occupied)

Where:
    s[n]   : transmitted signal from signals.py (unit average power)
    h[n]   : channel coefficient (1 for AWGN, Rayleigh, or Rician)
    w[n]   : complex Gaussian noise (unit variance, our fixed noise floor)
    alpha  : signal scaling derived from target SNR

KEY SNR CONVENTION
------------------
Noise has FIXED unit variance. We set SNR by SCALING THE SIGNAL with
    alpha = sqrt(10^(SNR_dB / 10))
This way both H0 and H1 share identical noise statistics, so a classifier
cannot 'cheat' by looking at noise level differences between classes.

Block fading: one random channel coefficient h is drawn per window and held
constant across the window's samples. This is standard for spectrum sensing
studies where windows are short.
"""

import numpy as np


def make_noise(n_samples, rng, noise_var=1.0):
    """
    Generate complex Gaussian noise (the w[n] term).
    
    Real and imaginary parts are independent zero-mean Gaussians, each with
    variance noise_var/2, so the total variance per complex sample is noise_var.
    'White' means samples are independent (no temporal correlation).
    
    This is also the H0 sample: under H0 the received signal IS the noise.
    """
    sigma = np.sqrt(noise_var / 2.0)
    real = rng.normal(0.0, sigma, size=n_samples)
    imag = rng.normal(0.0, sigma, size=n_samples)
    return (real + 1j * imag).astype(np.complex128)


def _snr_scaling(snr_db, noise_var=1.0):
    """
    Signal-scaling factor alpha for a target SNR with unit-power signal
    and noise variance noise_var.
    
        SNR_linear = (alpha^2 * P_s) / noise_var,  P_s = 1
        alpha      = sqrt(noise_var * 10^(SNR_dB / 10))
    """
    snr_linear = 10.0 ** (snr_db / 10.0)
    return np.sqrt(noise_var * snr_linear)


def apply_awgn(s, snr_db, rng, noise_var=1.0):
    """
    AWGN channel: no fading, h = 1.
        y[n] = alpha * s[n] + w[n]
    """
    alpha = _snr_scaling(snr_db, noise_var)
    w = make_noise(len(s), rng, noise_var)
    return alpha * s + w


def apply_rayleigh(s, snr_db, rng, noise_var=1.0):
    """
    Rayleigh fading (no line-of-sight). h ~ CN(0, 1), so E[|h|^2] = 1.
        y[n] = alpha * h * s[n] + w[n]
    Block fading: one h drawn per window.
    """
    alpha = _snr_scaling(snr_db, noise_var)
    h_real = rng.normal(0.0, np.sqrt(0.5))
    h_imag = rng.normal(0.0, np.sqrt(0.5))
    h = h_real + 1j * h_imag
    w = make_noise(len(s), rng, noise_var)
    return alpha * h * s + w


def apply_rician(s, snr_db, rng, K_dB=4.0, noise_var=1.0):
    """
    Rician fading (multipath WITH line-of-sight).
    
    h = sqrt(K/(K+1))  +  sqrt(1/(K+1)) * h_scatter,  K = 10^(K_dB/10)
    
    E[|h|^2] = 1 (same average power as AWGN/Rayleigh), so it is a fair
    comparison across channels at the same nominal SNR.
    """
    alpha = _snr_scaling(snr_db, noise_var)
    K = 10.0 ** (K_dB / 10.0)
    los = np.sqrt(K / (K + 1.0))
    sca_std = np.sqrt(1.0 / (2.0 * (K + 1.0)))
    h_scatter_re = rng.normal(0.0, sca_std)
    h_scatter_im = rng.normal(0.0, sca_std)
    h = los + (h_scatter_re + 1j * h_scatter_im)
    w = make_noise(len(s), rng, noise_var)
    return alpha * h * s + w