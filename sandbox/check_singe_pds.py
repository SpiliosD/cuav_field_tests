import numpy as np 
import matplotlib.pyplot as plt 

def main_test():
    file = r'D:\Raymetrics_Tests\BOMA2025\RawSpectra_txt\20250923\Spectra\User\20250923\Wind\2025-09-23\09-23_12h\09-23_12-50\spectra_2025-09-23_12-58-47.21.txt'
    data = np.loadtxt(file, delimiter='\t', skiprows=13)
    freq = np.arange(64) * 400/128
    for i, d in enumerate(data):
        fig = plt.figure()
        plt.plot(freq, d)
        plt.title(f'Spectrum {i+1}')
        plt.show()


if __name__ == '__main__':
    main_test()
    