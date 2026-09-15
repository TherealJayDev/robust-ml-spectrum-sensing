# Robust Machine Learning-Based Spectrum Occupancy Detection Under Low-SNR and Channel-Impairment Conditions

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![arXiv](https://img.shields.io/badge/arXiv-coming_soon-lightgrey.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

A simulation-based study comparing feature-based machine learning against classical energy detection for **spectrum occupancy detection** in cognitive radio, with a focus on **robustness** under varying SNR and channel conditions.

---

## Overview

Cognitive radio systems must decide whether a licensed primary user is transmitting on a given frequency band, so that unlicensed users can opportunistically access unused spectrum without causing interference. This decision, called **spectrum occupancy detection**, is the enabling function for the entire cognitive radio paradigm.

Classical energy detection, the industry-standard baseline, degrades sharply at low signal-to-noise ratio and under multipath fading. These are precisely the conditions where deployment matters most. Feature-based machine learning has been proposed as an alternative, but existing studies typically report accuracy without stating the operating point, evaluate on AWGN alone, and do not test channel mismatch.

This project provides a **reproducible, fairly-compared benchmark** of three machine learning classifiers against the classical energy detector at a fixed operating point, evaluated across three modulations, three channel models, and seven SNR levels, with dedicated experiments for feature ablation and channel mismatch.

## Research question

Not merely *"can machine learning detect occupied spectrum?"* but the stronger question: **how robust are feature-based machine learning models when SNR and channel conditions differ from training?**

## Key results

Evaluated on 8,400 signal windows at fixed probability of false alarm (Pfa) = 0.1:

- **Cross-validated ROC-AUC:** SVM 0.9005, Random Forest 0.8981, MLP 0.8998. All three within 0.003 of each other on the training distribution.
- **Main comparison:** ML classifiers meaningfully outperform energy detection in the -15 to -10 dB transition region under fading channels, with gains up to +20 percentage points on Rayleigh at -15 dB and +17 percentage points on Rician at -10 dB.
- **Honest negative result:** Random Forest drops 7 percentage points below classical energy detection on Rician at -10 dB. Machine learning is not uniformly superior; different classifiers have different failure modes.
- **Feature ablation:** The correlation-and-eigenvalue family (only 4 features) is the strongest single family for all three classifiers, matching the full 21-feature set within 2-3 percentage points.
- **Channel mismatch:** Classifiers trained on AWGN alone transfer to fading channels with bounded loss of 2-17 percentage points at low SNR. Random Forest is the most transfer-robust; MLP is the least, even though MLP wins the matched-training comparison.

## Publications

- **arXiv preprint:** *coming soon* (target October 2026, category `eess.SP`)
- Final year thesis submitted to the Federal University of Technology Akure, September 2026

## Signal model

Binary hypothesis test on each 1024-sample window of received complex baseband data:

```
H0: y[n] = w[n]                    (channel free, noise only, label 0)
H1: y[n] = h[n] * s[n] + w[n]      (channel occupied, signal present, label 1)
```

Where:
- `s[n]` is the transmitted signal (BPSK, QPSK, or 16-QAM) with unit average power
- `h[n]` is the channel realisation (AWGN, Rayleigh, or Rician with K = 4 dB)
- `w[n]` is complex Gaussian noise with unit variance
- SNR is set by scaling the signal: `s'[n] = alpha * s[n]` where `alpha = sqrt(10^(SNR_dB / 10))`

## Scope

| Dimension        | Values                                                          |
|------------------|-----------------------------------------------------------------|
| Modulations      | BPSK, QPSK, 16-QAM                                              |
| Channels         | AWGN, Rayleigh, Rician (K = 4 dB)                               |
| SNR range        | -20 dB to +10 dB (5 dB steps, 7 levels)                         |
| Baseline         | Classical energy detection (Urkowitz, 1967)                     |
| ML models        | SVM (RBF kernel), Random Forest, MLP                            |
| Feature families | Time-domain (9), Frequency-domain (8), Correlation-eigenvalue (4) |
| Total features   | 21 across four feature sets (FS1, FS2, FS3, FS4)                |
| Total windows    | 8,400 (200 per SNR-channel cell, balanced H0/H1)                |
| Operating point  | Fixed Pfa = 0.1 per condition                                   |
| Metrics          | Pd at fixed Pfa, ROC AUC, F1, accuracy                          |

## Reproducibility

Every random draw is derived from a single master seed (`SEED = 42`). The dataset can be regenerated bit-identically from the source code. Package versions are pinned in `requirements.txt`.

Design decisions to prevent shortcut learning:
- Noise statistics are identical across H0 and H1 by construction.
- SNR, modulation, and channel type are stored as metadata only; they are never presented as input features.
- Thresholds are set on H0 samples of each specific test condition and applied to H1 samples of the same condition, preventing operating-point leakage.
- The scaler is fit on training data only and applied to validation and test folds.

## Getting started

```bash
git clone https://github.com/TherealJayDev/robust-ml-spectrum-sensing.git
cd robust-ml-spectrum-sensing
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Run the notebook cells in order. Every result in the paper regenerates identically.

To skip signal generation and jump directly to feature-level experiments, use the included `data/features.csv` and `data/metadata.csv`.

## Repository structure

```
├── src/
│   ├── signals.py         # BPSK, QPSK, 16-QAM signal generators
│   ├── channels.py        # AWGN, Rayleigh, Rician channels + SNR scaling
│   ├── dataset.py         # H0/H1 window generation with metadata
│   ├── features.py        # 21 features across four feature sets (FS1-FS4)
│   ├── energy_detector.py # Classical energy detection baseline
│   ├── models.py          # Training pipelines, CV splits, hyperparameter tuning
│   └── evaluation.py      # Pd/Pfa metrics, ablation and mismatch experiments
├── data/
│   ├── features.csv       # Feature vectors (small, included)
│   ├── metadata.csv       # SNR / modulation / channel labels (included)
│   └── windows.npz        # Raw signal windows (regenerate from notebook)
├── outputs/
│   ├── figures/           # Publication-quality plots
│   ├── tables/            # Result tables as CSV
│   └── models/            # Trained models (regenerate from notebook)
├── notebook.ipynb         # Main narrative and driver
├── requirements.txt
├── LICENSE
└── README.md
```

## Feature set definitions

| Set | Family                        | Features                                                                                                                     | Count |
|-----|-------------------------------|------------------------------------------------------------------------------------------------------------------------------|-------|
| FS1 | Time-domain                   | mean_abs, variance, std, rms, energy, skewness, kurtosis, peak, papr                                                         | 9     |
| FS2 | Frequency-domain              | fft_peak, dominant_bin, spectral_centroid, spectral_bandwidth, spectral_entropy, spectral_flatness, psd_max, psd_mean        | 8     |
| FS3 | Correlation and eigenvalue    | autocorr_peak, cov_trace, max_eigenvalue, eigenvalue_ratio                                                                   | 4     |
| FS4 | Hybrid                        | Union of FS1, FS2, and FS3                                                                                                   | 21    |

The `energy` feature in FS1 is mathematically identical to the classical energy detector statistic. This means all four detectors (classical energy plus the three ML classifiers) consume identical input data, so any performance difference is attributable to the discriminative computation itself rather than to differences in the information available to each detector.

## Environment

Python 3.11 with:
- NumPy, SciPy, pandas
- scikit-learn (SVM, Random Forest, MLP)
- Matplotlib

See `requirements.txt` for pinned versions.

## Author

**Jackson A. Joshua**  
B.Eng Final Year Project, 2026  
Department of Information and Communication Technology  
Federal University of Technology Akure (FUTA), Nigeria  
Supervisor: Dr C. Udekwe

## License

MIT (see [`LICENSE`](./LICENSE))

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{joshua2026robust,
  author       = {Joshua, Jackson A.},
  title        = {Robust Machine Learning-Based Spectrum Occupancy Detection
                  Under Low-SNR and Channel-Impairment Conditions},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/TherealJayDev/robust-ml-spectrum-sensing}},
  note         = {Undergraduate thesis, Federal University of Technology Akure}
}
```

The BibTeX entry will be updated once the arXiv preprint is available.