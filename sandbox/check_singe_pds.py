import numpy as np 
import matplotlib.pyplot as plt 

def main_test():
    file = r'D:\Raymetrics_Tests\BOMA2025\20250926\Spectra\User\20250926\Wind\2025-09-26\09-26_07h\09-26_07-20\spectra_2025-09-26_07-20-00.65.txt'
    data = np.loadtxt(file, delimiter=' ', skiprows=13)
    for i, d in enumerate(data):
        fig = plt.figure()
        plt.plot(d)
        plt.title(f'Spectrum {i}')
        plt.show()


if __name__ == '__main__':
    main_test()
    