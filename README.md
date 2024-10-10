# Signal Denoising and Detrending
This repository contains two main components for processing signals contaminated with noise: high-frequency denoising using MATLAB and low-frequency detrending using Python. The workflow involves first applying high-frequency denoising to the original signal and then removing low-frequency components from the denoised signal.

Workflow Overview
High-Frequency Denoising: Implemented in MATLAB using a CPO (Cuckoo Search Optimization) algorithm to optimize Variational Mode Decomposition (VMD) for effective denoising.
Low-Frequency Detrending: Implemented in Python, this step removes low-frequency trends from the denoised signal using polynomial detrending and VMD.
Key Features
High-Frequency Denoising:
Utilizes CPO optimization for VMD to effectively remove high-frequency noise from the signal.
Low-Frequency Detrending:
Polynomial detrending to eliminate low-frequency trends.
VMD to decompose the signal into Intrinsic Mode Functions (IMFs).
SNR-based reconstruction to select the most relevant IMFs for signal reconstruction.
