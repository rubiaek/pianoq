import numpy as np
import datetime

from pianoq import Borders
from pianoq.lab import ThorlabsRotatingServoMotor
from pianoq.lab.Edac40 import Edac40
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.lab.thorlabs_motor import ManualMotor
from pianoq.misc.calc_correlation import get_correlations_mask
from pianoq.misc.consts import DEFAULT_BORDERS, DEFAULT_CAM_NO
from pianoq.results.polarization_meas_result import PolarizationMeasResult
from pianoq.results.multi_polarization_meas_result import MultiPolarizationMeasResult

LOGS_DIR = 'C:\\temp'


class MeasurePolarization(object):

    MY_QWP_ZERO = 2

    def __init__(self, exposure_time=900, saveto_path=None, roi=None, multi=False):
        self.cam = VimbaCamera(DEFAULT_CAM_NO, exposure_time=exposure_time)
        self.dac = Edac40(max_piezo_voltage=30, ip=Edac40.DEFAULT_IP)
        self.qwp_motor = ManualMotor()
        self.hwp_motor = ThorlabsRotatingServoMotor()

        self.is_multi = multi

        roi = roi or DEFAULT_BORDERS
        self.cam.set_borders(roi)

        if not self.is_multi:
            self.res = PolarizationMeasResult()
        else:
            self.res = MultiPolarizationMeasResult()

        self.res.exposure_time = exposure_time
        self.res.roi = roi
        self.res.mask_of_interest = get_correlations_mask()
        self.res.version = 2
        self.res.start_first = 50
        self.res.end_first = 130
        self.res.dist_x = 269
        self.res.dist_y = 3

        self.saveto_path = saveto_path
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    def run(self, amplitudes=None):
        amplitudes = amplitudes or 0.2 * np.ones(self.dac.NUM_OF_PIEZOS)
        # amplitudes = np.random.uniform(0, 1, size=self.dac.NUM_OF_PIEZOS)
        self.res.dac_amplitudes = amplitudes
        self.dac.set_amplitudes(amplitudes)

        # So H and V get to wollaston prism, and WPs won't bother
        self.hwp_motor.move_absolute(0)
        self.qwp_motor.move_absolute(0)
        self.res.meas1 = self.cam.get_image()

        # So QWP will change R,L to +-45 and then HWP will turn them to H, V
        self.hwp_motor.move_absolute(22.5)
        self.qwp_motor.move_absolute(0)
        self.res.meas3 = self.cam.get_image()

        # So +-45 will turn to H and V, and QWP won't bother
        self.hwp_motor.move_absolute(22.5)
        self.qwp_motor.move_absolute(45)
        self.res.meas2 = self.cam.get_image()

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
        self.qwp_motor.move_absolute(0)
        self.hwp_motor.move_absolute(0)

        for i, amps in enumerate(all_amplitudes):
            print(f'{i+1}/{len(all_amplitudes)}')
            self.dac.set_amplitudes(amps)
            im = self.cam.get_image()
            self.res.meas1s.append(im)

        # So +-45 will turn to H and V, and QWP won't bother
        self.qwp_motor.move_absolute(45)
        self.hwp_motor.move_absolute(22.5)

        for i, amps in enumerate(all_amplitudes):
            print(f'{i+1}/{len(all_amplitudes)}')
            self.dac.set_amplitudes(amps)
            im = self.cam.get_image()
            self.res.meas2s.append(im)

        # So QWP will change R,L to +-45 and then HWP will turn them to H, V
        self.qwp_motor.move_absolute(0)
        self.hwp_motor.move_absolute(22.5)

        for i, amps in enumerate(all_amplitudes):
            print(f'{i+1}/{len(all_amplitudes)}')
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
        print(f'Saved result to {saveto_path}')

    def close(self):
        self.cam.close()
        self.qwp_motor.close()
        self.hwp_motor.close()


if __name__ == "__main__":
    # mp = MeasurePolarization(multi=False, exposure_time=500, roi=Borders(330, 550, 800, 640)) # When inserting PBS
    mp = MeasurePolarization(multi=False, exposure_time=300, roi=Borders(330, 520, 800, 615))
    mp.run()

    # mp = MeasurePolarization(multi=True, exposure_time=900)
    # mp.run_multi()
    mp.close()
