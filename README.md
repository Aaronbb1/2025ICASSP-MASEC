# Signal adaptive processing
This repository contains two main components for processing signals contaminated with noise: high-frequency denoising using MATLAB and low-frequency removal of low-frequency components using Python. The workflow involves first combining with the CEEMDAN method that IMF components containing high-frequency information are further extracted from the original signal applying high-frequency denoising and using adaptive weighted sliding window method to remove the low-frequency components from the denoising signal. Finally, the method implements an adaptive signal processing strategy.

<div style="text-align: center">
  <img alt="p1" src="https://github.com/user-attachments/assets/8efb1333-7cb2-4c9c-ae56-24e850bd49d1">
</div>

## Workflow Overview
#### High-Frequency Denoising: 
Implemented in MATLAB using a CPO (Crown Porcupine Optimization) algorithm to optimize Variational Mode Decomposition (VMD) for effective denoise of our needed IMFs combining with the CEEMDAN method that we search.
#### Low-Frequency Detrending: 
Implemented in Python, this step removes low-frequency trends from the denoised signal using polynomial detrending and VMD. In the process, we used the mean, higher-order polynomial fitting and the double traversal method to further optimize our strategy
## Key Features
#### High-Frequency Denoising:
·Utilizes CPO optimization for VMD to effectively remove high-frequency noise from the signal.
#### Low-Frequency Detrending:
·Polynomial detrending to eliminate low-frequency trends.
·VMD to decompose the signal into Intrinsic Mode Functions (IMFs).
·SNR-based reconstruction to select the most relevant IMFs for signal reconstruction.
## Dependencies
### MATLAB
Ensure you have MATLAB installed with the necessary toolboxes for signal processing.
### Python
The Python script requires the following libraries:
·numpy
·pandas
·matplotlib
·scipy
·VMDpy (a Python implementation of the VMD algorithm)...
#### You can install the required libraries using pip:
```bash
pip install numpy pandas matplotlib scipy
```
#### VMDpy can be installed from its GitHub repository:
```bash
pip install git+https://github.com/2025ICASSP-MASEC/Lowf processing/vmdpy.py
````
## Usage
#### Step 1: High-Frequency Denoising (MATLAB)
1.Prepare your data: Ensure your original signal data is in a suitable format for MATLAB.
2.Run the MATLAB script: Execute the MATLAB script to perform high-frequency denoising.
run('main1.m');
3.Output: The denoised signal will be saved as a .csv file (e.g., S_real.csv).
#### Step 2: Low-Frequency Detrending (Python)
1.Prepare your data: Ensure the denoised signal is saved as S_real.csv and the clean original signal (if available) as S0.csv.
2.Run the Python script: Execute the Python script in your terminal or command prompt.
VMD_denoise.py
3.View the results: The script will generate plots and save them as .png files. It will also output the reconstructed signal to a .csv file named S_rec.csv.
#### Output
Plots: The Python script generates several plots to visualize the detrending process, the VMD decomposition, and the SNR of each IMF.
Reconstructed Signal: A .csv file containing the reconstructed signal after detrending and denoising.

![PSD补充](https://github.com/user-attachments/assets/9cc908f7-0f9a-47b4-b6b8-ce132454cac5)

## Example
Here's an example of how the complete workflow processes the signal:
#### High-Frequency Denoising:
The MATLAB script reads the original signal, applies CPO-optimized VMD, and outputs a denoised signal.
#### Low-Frequency Detrending:
The Python script reads the denoised signal, applies polynomial detrending, performs VMD to decompose the signal into IMFs, and reconstructs the signal based on SNR criteria.
## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please create an issue or submit a pull request.
## License
This project is licensed under the MIT License - see the LICENSE file for details.**





