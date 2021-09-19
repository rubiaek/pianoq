import numpy as np
import matplotlib.pyplot as plt

from pianoq.results.polarization_meas_result import PolarizationMeasResult
import traceback


class MultiPolarizationMeasResult(object):
    def __init__(self):
        self.roi = None
        self.exposure_time = None

        self.meas1s = []  # qwp.angle = 0,  hwp.angle = 0
        self.meas2s = []  # qwp.angle = 45, hwp.angle = 22.5
        self.meas3s = []  # qwp.angle = 0,  hwp.angle = 22.5

        self.mask_of_interest = None

        self.dac_amplitudes = []

        self.pol_meass = []

    def saveto(self, path):
        try:
            f = open(path, 'wb')
            np.savez(f,
                     roi=self.roi,
                     exposure_time=self.exposure_time,
                     meas1s=self.meas1s,
                     meas2s=self.meas2s,
                     meas3s=self.meas3s,
                     mask_of_interest=self.mask_of_interest,
                     dac_amplitudes=self.dac_amplitudes,
                     )
            f.close()
        except Exception as e:
            print("ERROR!!")
            print(e)
            traceback.print_exc()

    def loadfrom(self, path):
        # path = path or self.DEFAULT_PATH
        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        self.roi = data.get('roi', None)
        self.exposure_time = data.get('exposure_time', None)

        self.meas1s = data.get('meas1s', None)
        self.meas2s = data.get('meas2s', None)
        self.meas3s = data.get('meas3s', None)

        self.mask_of_interest = data.get('mask_of_interest', None)
        self.dac_amplitudes = data.get('dac_amplitudes', None)

        for i in range(len(self.meas1s)):
            pol_meas = PolarizationMeasResult()

            pol_meas.roi = self.roi
            pol_meas.exposure_time = self.exposure_time
            pol_meas.mask_of_interest = self.mask_of_interest

            pol_meas.meas1 = self.meas1s[i]
            pol_meas.meas2 = self.meas2s[i]
            pol_meas.meas3 = self.meas3s[i]

            pol_meas.dac_amplitudes = self.dac_amplitudes[i]

            self.pol_meass.append(pol_meas)


if __name__ == "__main__":
    mp = MultiPolarizationMeasResult()
    # mp.loadfrom(r"G:\My Drive\Projects\Quantum Piano\Results\PolarizationMeasurements\Second Set\polarized_2021_09_12_13_04_16.polms")
    mp.loadfrom(r"G:\My Drive\Projects\Quantum Piano\Results\PolarizationMeasurements\Second Set\2021_09_12_12_47_45.polms")
    q = mp.pol_meass[10]
    assert isinstance(q, PolarizationMeasResult)
    q.plot_poincare()
    q.plot_stokes_params()
    q.plot_polarization_speckle()
    print(q.get_degree_of_polarization(True))
    print(q.get_degree_of_polarization(False))

    plt.show()
