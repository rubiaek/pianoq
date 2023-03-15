import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pianoq.lab import Edac40
from pianoq.lab.spectrometer_ccs import Spectrometer
from pianoq_results.QPPickleResult import QPPickleResult


class SLDSpectrumResult(QPPickleResult):
    def __init__(self, path=None):
        super().__init__(path=path)
        self.integration_time = 0
        self.wavelengths = np.array([])
        self.data = []
        self.comment = ''

    def show_spectrum(self, index):
        fig, ax = plt.subplots()
        #plot data
        ax.plot(self.wavelengths, self.data[index, :])
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Intensity [a.u.]")
        ax.grid(True)
        fig.show()


def main(run_name='test', integration_time=3e-3):
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    saveto_path = f"G:\\My Drive\\Projects\\Quantum Piano\\Results\\SLD\\{timestamp}_SLD_statistics_{run_name}.sldqp"

    s = Spectrometer(integration_time=integration_time)
    dac = Edac40(max_piezo_voltage=120)
    res = SLDSpectrumResult()
    res.integration_time = integration_time
    res.comment = 'SLD 180mA, V polarized, single speckle, ND1.0, integration 3ms'

    wl, a = s.get_data()
    wl, a = wl[2400: -700], a[2400: -700]
    N = 10000
    res.data = np.zeros((N, len(a)))
    res.wavelengths = wl
    for i in range(N):
        amps = np.random.rand(40)
        dac.set_amplitudes(amps)
        _, a = s.get_data()
        res.data[i, :] = a[2400: -700]

        if i % 10 == 0:
            res.saveto(saveto_path)
            print(i, end=' ')

    s.close()
    dac.close()

if __name__ == "__main__":
    main()
