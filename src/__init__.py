"""
Spectrum Sensing Project - source package.

This package holds all reusable code for the project. The Jupyter notebook
imports from these modules and narrates what is happening, while the heavy
logic lives here in clean, testable files.

Module map:
    signals.py          -> generate primary-user signals (sine, BPSK, QPSK, 16-QAM)
    channels.py         -> apply channel effects (AWGN, Rayleigh, Rician)
    dataset.py          -> combine signals + channels into labelled H0/H1 samples
    features.py         -> extract time, frequency, correlation/eigenvalue features
    energy_detector.py  -> classical energy-detection baseline
    models.py           -> define and train ML models (SVM, Random Forest, ANN/MLP)
    evaluation.py       -> metrics (Pd, Pfa, Pm, ROC/AUC, F1) and plots
"""
