"""
signals.py
==========
Generates the transmitted primary-user signal s[n] for the occupied (H1) case.

This is the s[n] term in the binary hypothesis model:
    H0: y[n] = w[n]                  (noise only, channel free)
    H1: y[n] = h[n] * s[n] + w[n]    (signal present, channel occupied)

This module produces s[n] only. Channel effects h[n] and noise w[n] are added
later in channels.py. Keeping signal generation separate from channel/noise is
what lets us mix any modulation with any channel cleanly.

Planned functions (minimum version, 4 modulations):
    generate_sine(...)    -> sinusoidal carrier
    generate_bpsk(...)    -> Binary Phase Shift Keying
    generate_qpsk(...)    -> Quadrature Phase Shift Keying
    generate_16qam(...)   -> 16-point Quadrature Amplitude Modulation

Design rule: every signal is returned as complex baseband and normalised to
unit average power, so that SNR is controlled entirely by the noise stage.
"""
import numpy as np


def generate_sine(n_samples, rng, f_norm=0.1):
    """
    Generate an unmodulated sinusoidal carrier (the simplest 'occupied' signal).

    A pure complex tone:  s[n] = A * exp(j * 2*pi * f_norm * n)

    Parameters
    ----------
    n_samples : int
        Number of samples in the window (e.g. 1024).
    rng : numpy.random.Generator
        Random generator. Not used for the tone itself, but accepted so every
        signal function has the SAME call signature (keeps dataset.py simple).
        A random starting phase is drawn so each sample window is not identical.
    f_norm : float
        Normalised frequency (cycles per sample), 0 < f_norm < 0.5.
        0.1 means the tone completes one cycle every 10 samples.

    Returns
    -------
    s : np.ndarray (complex), shape (n_samples,)
        Unit-average-power complex baseband tone.
    """
    n = np.arange(n_samples)                      # sample indices 0,1,2,...
    phase = rng.uniform(0, 2 * np.pi)             # random start phase per window
    s = np.exp(1j * (2 * np.pi * f_norm * n + phase))
    # |s[n]| = 1 for all n already, so average power = 1. Normalise anyway,
    # defensively, so this function obeys the same rule as the others.
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    return s


def generate_bpsk(n_samples, rng, sps=8):
    """
    Generate a BPSK-modulated signal carrying random data bits.

    BPSK maps each bit to one of two constellation points:
        bit 0 -> -1 + 0j   (phase 180 degrees)
        bit 1 -> +1 + 0j   (phase   0 degrees)
    Each symbol is held for `sps` samples (samples-per-symbol).

    Parameters
    ----------
    n_samples : int
        Total number of samples in the output window.
    rng : numpy.random.Generator
        Seeded random generator (used to draw the random bits).
    sps : int
        Samples per symbol. With sps=8 and n_samples=1024, we transmit
        1024 / 8 = 128 symbols.

    Returns
    -------
    s : np.ndarray (complex), shape (n_samples,)
        Unit-average-power BPSK signal.
    """
    n_symbols = n_samples // sps                   # how many symbols fit
    bits = rng.integers(0, 2, size=n_symbols)      # random 0/1 bits
    symbols = 2 * bits - 1                         # map: 0 -> -1, 1 -> +1
    # Hold each symbol for `sps` samples ("rectangular pulse shaping")
    s = np.repeat(symbols, sps).astype(np.complex128)
    # Normalise to unit average power (already ~1 for BPSK, defensive line)
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    return s

def generate_qpsk(n_samples, rng, sps=8):
    """
    Generate a QPSK-modulated signal carrying random data bits.

    QPSK maps each pair of bits to one of four constellation points using
    Gray coding (adjacent points differ in only one bit):
        bits 00 -> +1 + j
        bits 01 -> -1 + j
        bits 11 -> -1 - j
        bits 10 -> +1 - j

    Each symbol carries 2 bits and is held for `sps` samples.
    Symbols have magnitude sqrt(2), so we divide by sqrt(2) to get unit power.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the output window.
    rng : numpy.random.Generator
        Seeded random generator (used to draw the random bits).
    sps : int
        Samples per symbol. With sps=8 and n_samples=1024, we transmit
        1024 / 8 = 128 symbols carrying 256 bits.

    Returns
    -------
    s : np.ndarray (complex), shape (n_samples,)
        Unit-average-power QPSK signal.
    """
    n_symbols = n_samples // sps
    # Draw 2 bits per symbol -> shape (n_symbols, 2)
    bits = rng.integers(0, 2, size=(n_symbols, 2))
    # Map bit pairs to constellation points (Gray coded).
    # I component depends only on the first bit (0 -> +1, 1 -> -1).
    # Q component depends only on the second bit (0 -> +1, 1 -> -1).
    I = 1 - 2 * bits[:, 0]
    Q = 1 - 2 * bits[:, 1]
    symbols = (I + 1j * Q)
    # Hold each symbol for `sps` samples
    s = np.repeat(symbols, sps).astype(np.complex128)
    # Normalise to unit average power (divides out the sqrt(2) automatically)
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    return s

def generate_16qam(n_samples, rng, sps=8):
    """
    Generate a 16-QAM modulated signal carrying random data bits.

    16-QAM uses both phase AND amplitude. The constellation is a 4x4 grid:
    I and Q each take one of four Gray-coded levels {-3, -1, +1, +3},
    giving 16 distinct symbols carrying 4 bits each.

    Bit-to-level mapping (per axis):
        bits 00 -> -3
        bits 01 -> -1
        bits 11 -> +1
        bits 10 -> +3

    Unnormalised symbols have an average power of 10 (averaged over the 16
    equally likely points), so the unit-power normalisation divides by sqrt(10).
    Our defensive normalise line handles this automatically.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the output window.
    rng : numpy.random.Generator
        Seeded random generator.
    sps : int
        Samples per symbol. With sps=8 and n_samples=1024, we transmit
        1024 / 8 = 128 symbols carrying 512 bits.

    Returns
    -------
    s : np.ndarray (complex), shape (n_samples,)
        Unit-average-power 16-QAM signal.
    """
    n_symbols = n_samples // sps
    # 4 bits per symbol: first 2 -> I level, last 2 -> Q level
    bits = rng.integers(0, 2, size=(n_symbols, 4))

    # Map a pair of bits to one of {-3, -1, +1, +3} using Gray coding.
    # Pair (b1, b0) interpreted as: b1 picks the sign, b0 picks inner/outer.
    #   00 -> -3   01 -> -1   11 -> +1   10 -> +3
    def pair_to_level(b1, b0):
        sign = 2 * b1 - 1
        magnitude = 3 - 2 * b0
        return sign * magnitude

    I = pair_to_level(bits[:, 0], bits[:, 1])
    Q = pair_to_level(bits[:, 2], bits[:, 3])
    symbols = I + 1j * Q

    # Hold each symbol for `sps` samples (rectangular pulse shaping)
    s = np.repeat(symbols, sps).astype(np.complex128)
    # Normalise to unit average power (divides by sqrt(10) automatically)
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    return s

def generate_fsk(n_samples, rng, sps=8, f_dev=0.0625):
    """
    Generate a binary FSK (2-FSK) modulated signal carrying random data bits.

    Binary FSK maps each bit to one of two carrier frequencies:
        bit 0 -> exp(j * 2*pi * (-f_dev) * t)   (frequency -f_dev)
        bit 1 -> exp(j * 2*pi * (+f_dev) * t)   (frequency +f_dev)
    where t is a local time index within each symbol (0 to sps-1).

    Each symbol carries 1 bit and is held for `sps` samples. The two 
    frequencies are symmetric around DC (0), so the signal has zero mean 
    frequency. With the default f_dev = 0.0625 and sps = 8, the frequency
    spacing (2 * f_dev = 0.125) equals 1/sps, giving orthogonal FSK.

    Phase is discontinuous at symbol boundaries. This is simpler than
    continuous-phase FSK but has slightly broader spectrum, which is 
    fine for a spectrum sensing feature extraction study.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the output window.
    rng : numpy.random.Generator
        Seeded random generator (used to draw the random bits).
    sps : int
        Samples per symbol. With sps=8 and n_samples=1024, we transmit
        1024 / 8 = 128 symbols.
    f_dev : float
        Frequency deviation from DC in normalised units (cycles per sample).
        Default 0.0625 gives orthogonal FSK when sps=8.

    Returns
    -------
    s : np.ndarray (complex), shape (n_samples,)
        Unit-average-power binary FSK signal.
    """
    n_symbols = n_samples // sps
    bits = rng.integers(0, 2, size=n_symbols)      # random 0/1 bits
    freqs = (2 * bits - 1) * f_dev                 # 0 -> -f_dev, 1 -> +f_dev

    # For each symbol, generate sps samples of exp(j * 2*pi * freq * t)
    # where t is the local time index within the symbol.
    t = np.arange(sps)
    symbol_signals = np.exp(1j * 2 * np.pi * freqs[:, None] * t[None, :])
    s = symbol_signals.flatten()

    # Random starting phase per window (matches sine convention)
    phase_start = rng.uniform(0, 2 * np.pi)
    s = s * np.exp(1j * phase_start)

    # Normalise to unit average power (|s[n]| = 1 already, defensive)
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    return s