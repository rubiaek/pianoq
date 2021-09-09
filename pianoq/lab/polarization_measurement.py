import numpy as np
import datetime

from pianoq.lab.Edac40 import Edac40
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.misc.calc_correlation import get_correlations_mask
from pianoq.misc.consts import DEFAULT_BORDERS
from pianoq.results.polarization_meas_result import PolarizationMeasResult
from results.multi_polarization_meas_result import MultiPolarizationMeasResult

LOGS_DIR = 'C:\\temp'


class MeasurePolarization(object):
    def __init__(self, exposure_time=900, saveto_path=None, roi=None, multi=False):
        self.cam = VimbaCamera(2, exposure_time=exposure_time)
        self.dac = Edac40(max_piezo_voltage=30, ip=Edac40.DEFAULT_IP)
        self.is_multi = multi

        roi = roi or DEFAULT_BORDERS
        self.cam.set_borders(DEFAULT_BORDERS)

        if not self.is_multi:
            self.res = PolarizationMeasResult()
        else:
            self.res = MultiPolarizationMeasResult()

        self.res.exposure_time = exposure_time
        self.res.roi = roi
        self.res.mask_of_interest = get_correlations_mask()

        self.saveto_path = saveto_path
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    def run(self, amplitudes=None):
        amplitudes = amplitudes or np.ones(self.dac.NUM_OF_PIEZOS)
        amplitudes = np.random.uniform(0, 1, size=self.dac.NUM_OF_PIEZOS)
        self.res.dac_amplitudes = amplitudes
        self.dac.set_amplitudes(amplitudes)

        # So H and V get to wollaston prism, and WPs won't bother
        print("Make sure the fast axis of both QWP and HWP is on 0 degrees")
        input()
        self.res.meas1 = self.cam.get_image()

        # So +-45 will turb to H and V, and QWP won't bother
        print("Make sure the fast axis of QWP is on 45 degrees and HWP is on 22.5 degrees")
        input()
        self.res.meas2 = self.cam.get_image()

        # So QWP will change R,L to +-45m and then HWP will turn them to H, V
        print("Make sure the fast axis of QWP is on 0 degrees and HWP is on 22.5 degrees")
        input()
        self.res.meas3 = self.cam.get_image()
        self._save_result()

    def run_multi(self):
        all_amplitudes = []
        for a in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            all_amplitudes.append(a * np.ones(self.dac.NUM_OF_PIEZOS))

        for i in range(5):
            amplitudes = np.random.uniform(0, 1, size=self.dac.NUM_OF_PIEZOS)
            all_amplitudes.append(amplitudes)

        self.res.dac_amplitudes = all_amplitudes

        # So H and V get to wollaston prism, and WPs won't bother
        input("Make sure the fast axis of both QWP and HWP is on 0 degrees\n")
        for i, amps in enumerate(all_amplitudes):
            print(f'{i}/{len(all_amplitudes)}')
            self.dac.set_amplitudes(amps)
            im = self.cam.get_image()
            self.res.meas1s.append(im)

        # So +-45 will turb to H and V, and QWP won't bother
        input("Make sure the fast axis of QWP is on 45 degrees and HWP is on 22.5 degrees\n")
        for i, amps in enumerate(all_amplitudes):
            print(f'{i}/{len(all_amplitudes)}')
            self.dac.set_amplitudes(amps)
            im = self.cam.get_image()
            self.res.meas2s.append(im)

        # So QWP will change R,L to +-45m and then HWP will turn them to H, V
        input("Make sure the fast axis of QWP is on 0 degrees and HWP is on 22.5 degrees\n")
        for i, amps in enumerate(all_amplitudes):
            print(f'{i}/{len(all_amplitudes)}')
            self.dac.set_amplitudes(amps)
            im = self.cam.get_image()
            self.res.meas3s.append(im)

        self._save_result()

    def _save_result(self):
        if not self.is_multi:
            suffix = 'polm'
        else:
            suffix = 'polms'

        saveto_path = self.saveto_path or f"{LOGS_DIR}\\{self.timestamp}.{suffix}"
        self.res.saveto(saveto_path)

    def close(self):
        self.cam.close()


if __name__ == "__main__":
    # mp = MeasurePolarization(multi=False)
    # mp.run()

    mp = MeasurePolarization(multi=True)
    mp.run_multi()
    mp.close()
