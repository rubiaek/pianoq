import datetime
import numpy as np
import matplotlib.pyplot as plt
from pianoq.lab.photon_counter import PhotonCounter
from pianoq.lab.Edac40 import Edac40
from pianoq.lab.time_tagger import QPTimeTagger
from pianoq_results.QPPickleResult import QPPickleResult

LOGS_DIR = r"G:\My Drive\Projects\Quantum Piano\Results\temp"


class SpeckleStatisticsResult(QPPickleResult):
    def __init__(self, path=None, single1s=None, single2s=None, coincidences=None,
                 coin_window=None, integration_time=None, is_timetagger=None):
        super().__init__(path=path)
        self.path = path
        self.single1s = single1s
        self.single2s = single2s
        self.coincidences = coincidences
        self.integration_time = integration_time
        self.coin_window = coin_window
        self.is_timetagger = is_timetagger

    @property
    def accidentals(self):
        return 2*self.single1s*self.single2s*self.coin_window

    @property
    def real_coin(self):
        return self.coincidences - self.accidentals

    def _get_contrast(self, V):
        self.reload()
        contrast = V.std() / V.mean()
        N = V.size
        contrast_err = contrast * np.sqrt(1/(2*N-2) + (contrast**2)/N)
        return contrast, contrast_err

    def _contrast_to_N_speckles(self, contrast):
        return 1/contrast**2

    def print_contrasts(self):
        cc, cdc = self._get_contrast(self.real_coin)
        s1c, s1dc = self._get_contrast(self.single1s)
        s2c, s2dc = self._get_contrast(self.single2s)

        print(f'Contrast for coincidence: {cc:.2f}+-{cdc:.2f} ~ {self._contrast_to_N_speckles(cc):.1f} speckle patterns')
        print(f'Contrast for single1s: {s1c:.2f}+-{s1dc:.2f} ~ {self._contrast_to_N_speckles(s1c):.1f} speckle patterns')
        print(f'Contrast for single2s: {s2c:.2f}+-{s2dc:.2f} ~ {self._contrast_to_N_speckles(s2c):.1f} speckle patterns')

    def histogram(self, normalized=True):
        self.reload()
        fig_hist, ax_hist = plt.subplots()
        coin = self.real_coin / self.real_coin.mean() if normalized else self.real_coin
        ax_hist.hist(coin, bins=100, density=True) #, range=(0, 500))
        # ax_hist.set_ylim(0, 8)
        fig_hist.show()

    def singles_histogram(self, second=False):
        self.reload()
        fig_hist, ax_hist = plt.subplots()
        if not second:
            ax_hist.hist(self.single1s, bins=100) #, range=(0, 500))
        else:
            ax_hist.hist(self.single2s, bins=100) #, range=(0, 500))
        # ax_hist.set_ylim(0, 8)
        fig_hist.show()

    def loadfrom(self, path):
        super().loadfrom(path)
        self.path = path
        self.single1s = np.array(self.single1s)
        self.single2s = np.array(self.single2s)
        self.coincidences = np.array(self.coincidences)


def main(is_timetagger=True, integration_time=5, coin_window=1e-9, saveto_path=None, run_name='noname'):
    # Technical
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    saveto_path = saveto_path or f"{LOGS_DIR}\\{timestamp}_speckle_statistics_{run_name}.spst"
    res = SpeckleStatisticsResult()
    res.coin_window = coin_window
    res.integration_time = integration_time
    res.is_timetagger = is_timetagger


    # Get hardware
    dac = Edac40(max_piezo_voltage=120)
    dac.SLEEP_AFTER_SEND = 0.3
    print('got DAC')

    if not is_timetagger:
        ph = PhotonCounter(integration_time=integration_time)
    else:
        ph = QPTimeTagger(integration_time=integration_time, coin_window=coin_window)
    print('got photon counter')

    single1s = []
    single2s = []
    coincidences = []
    stds = []

    for i in range(1000):
        amps = np.random.rand(40)
        dac.set_amplitudes(amps)

        s1, s2, coin = ph.read_interesting()
        print(f'{s1}, {s2}, {coin}')
        single1s.append(s1)
        single2s.append(s2)
        coincidences.append(coin)

        if i % 5 == 0:
            res.single1s = single1s
            res.single2s = single2s
            res.coincidences = coincidences
            res.saveto(saveto_path)

if __name__ == "__main__":
    main(is_timetagger=False, integration_time=7, coin_window=4e-9, saveto_path=None, run_name='filter=10nm_no_ND_heralded_D=105um')
