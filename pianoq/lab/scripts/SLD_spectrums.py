import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pianoq.lab import Edac40
from pianoq.lab.spectrometer_ccs import Spectrometer
from pianoq.lab.spectrometer_yokogawa import YokogawaSpectrometer
from pianoq_results.QPPickleResult import QPPickleResult


class SLDSpectrumResult(QPPickleResult):
    def __init__(self, path=None):
        super().__init__(path=path)
        self.integration_time = 0
        self.wavelengths = np.array([])
        self.data = np.array([])
        self.comment = ''
        self.delta_wl = None
        self._fourier_threshold = 1

    def _init_normzlizations(self):
        self.mean_spectrum = self.data.mean(axis=0)
        index_mean_wl = self.mean_spectrum.argmax()
        self._mean_wl = self.wavelengths[index_mean_wl]
        self.normalized_data = self.data / self.mean_spectrum

        self.filtered_data = np.zeros_like(self.data)
        for i, row in enumerate(self.data):
            self.filtered_data[i, :] = self._filter_fourier(row)

        self.filtered_mean_spectrum = self.filtered_data.mean(axis=0)
        self.filterd_normalized_data = self.filtered_data / self.filtered_mean_spectrum

    def set_fourier_threshold(self, threshold=1):
        self._fourier_threshold = threshold
        self._init_normzlizations()

    def _clean_zeros(self):
        means = self.data.mean(axis=1)
        zero_index = np.where(means == 0)[0][0]
        self.data = self.data[:zero_index, :]

    def show_spectrum(self, amps):
        fig, ax = plt.subplots()
        ax.plot(self.wavelengths, amps)
        ax.set_xlabel("Wavelength [nm]")
        ax.set_ylabel("Intensity [a.u.]")
        ax.grid(True)
        fig.show()

    def show_mean_spectrum(self, is_filtered):
        if is_filtered:
            data = self.filtered_mean_spectrum
        else:
            data = self.mean_spectrum
        self.show_spectrum(data)

    def get_slice_x_nm_filter(self, x_nm):
        indexes = np.where(np.abs(self.wavelengths - self._mean_wl) < x_nm / 2)[0]
        slc = np.index_exp[indexes[0]: indexes[-1]]
        return slc

    def _contrast(self, v):
        return v.std() / v.mean()

    def contrast_per_bandwidth(self, is_normalized=True, is_filtered=False):
        if is_filtered and is_normalized:
            data = self.filterd_normalized_data
        elif is_filtered and not is_normalized:
            data = self.filtered_data
        elif not is_filtered and is_normalized:
            data = self.normalized_data
        else:  # not filtered and not normalized
            data = self.data

        filters = np.linspace(0.02, 25, 150)
        contrasts = np.zeros_like(filters)
        for i, x_nm in enumerate(filters):
            slc = self.get_slice_x_nm_filter(x_nm)
            filterd_out = data[:, slc[0]]
            mean_of_filter = filterd_out.mean(axis=1)
            contrasts[i] = self._contrast(mean_of_filter)

        return filters, contrasts

    def _filter_fourier(self, amps):
        fourier = np.fft.fftshift(np.fft.fft(np.fft.fftshift(amps)))
        fourier_X = np.fft.fftshift(np.fft.fftfreq(len(amps), np.diff(self.wavelengths)[0]))

        mask = np.where(np.abs(fourier_X) > self._fourier_threshold)
        fourier2 = fourier
        fourier2[mask] = 0

        filtered_amps = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(fourier2)))
        return filtered_amps

    def fourier_check(self, amps, threshold=1):
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].set_title('original')
        axes[0, 0].plot(sr.wavelengths, amps)
        axes[0, 0].set_xlabel('wavelength (nm)')

        fourier = np.fft.fftshift(np.fft.fft(np.fft.fftshift(amps)))
        fourier_X = np.fft.fftshift(np.fft.fftfreq(len(amps), np.diff(self.wavelengths)[0]))

        axes[0, 1].plot(fourier_X, fourier)
        axes[0, 1].set_title('fourier space')

        mask = np.where(np.abs(fourier_X) > threshold)
        fourier2 = fourier
        fourier2[mask] = 0

        filtered_amps = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(fourier2)))
        axes[1, 0].plot(sr.wavelengths, filtered_amps)
        axes[1, 0].set_title('filtered signal')

        axes[1, 1].plot(fourier_X, fourier2)
        axes[1, 1].set_title('filtered fourier')

        fig.show()

    def show_contrast_per_bandwidth(self, is_normalized=True, is_filtered=False):
        filters, contrasts = self.contrast_per_bandwidth(is_normalized, is_filtered)
        fig, ax = plt.subplots()
        color = 'tab:red'
        ax.plot(filters, contrasts, '*', color=color)
        ax.set_xlabel('Filter width (nm)')
        ax.set_ylabel('contrast', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim(None, 1)

        ax2 = ax.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('N modes', color=color)
        ax2.plot(filters, 1/contrasts**2, '*', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(1)

        fig.show()

    def loadfrom(self, path):
        super().loadfrom(path)
        self._clean_zeros()
        self.delta_wl = np.diff(self.wavelengths)[0]
        self.wavelengths = np.array(self.wavelengths)
        if self.wavelengths.min() < 1e-3:  # Means this is in [m] and I want it in [nm]
            self.wavelengths = self.wavelengths * 1e9
        self._init_normzlizations()


def main(run_name='first_long_yokagawa', integration_time=3e-3, yokogawa=True):
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    saveto_path = f"G:\\My Drive\\Projects\\Quantum Piano\\Results\\SLD\\{timestamp}_SLD_statistics_{run_name}.sldqp"

    if not yokogawa:
        s = Spectrometer(integration_time=integration_time)
    else:
        s = YokogawaSpectrometer()

    dac = Edac40(max_piezo_voltage=120)
    res = SLDSpectrumResult()
    res.integration_time = integration_time
    res.comment = 'SLD 180mA, V polarized, single speckle, ND1.0, integration 3ms'

    wl, a = s.get_data()
    if not yokogawa:
        wl, a = wl[2400: -700], a[2400: -700]
    N = 4000
    res.data = np.zeros((N, len(a)))
    res.wavelengths = wl
    for i in range(N):
        amps = np.random.rand(40)
        dac.set_amplitudes(amps)
        _, a = s.get_data()
        if not yokogawa:
            a = a[2400: -700]
        res.data[i, :] = a

        if i % 10 == 0:
            res.saveto(saveto_path)
            print(i, end=' ')


    res.saveto(saveto_path)

    s.close()
    dac.close()


if __name__ == "__main__":
    pass
    # TODO: look with camera that piezos do what we want them
    # TODO: repeat with light closed
    # TODO: try some kind of better parameters in spectrometer (resolution etc.)
    # main()

sr = SLDSpectrumResult()
# sr.loadfrom(r"G:\My Drive\Projects\Quantum Piano\Results\SLD\2023_03_15_10_04_05_SLD_statistics_test_good.sldqp")
# sr.show_conrast_per_bandwidth()
