import numpy as np

from pianoq import Borders
from pianoq.lab import VimbaCamera
from pianoq.lab.elliptec_stage import ElliptecMotor
from pianoq.lab.thorlabs_motor import ThorlabsRotatingServoMotor
from pianoq.misc.consts import DEFAULT_CAM_NO, DEFAULT_BORDERS, DEFAULT_ELLO_PORTNO
from pianoq.results.waveplates_optimization_result import WavePlateOptimizationResult

LOGS_DIR = "C:\\temp"


class GlobalRotationOptimization(object):
    def __init__(self, H_angles=None, Q_angles=None):
        print("Getting HWP Thorlabs Motor")
        self.hwm = ThorlabsRotatingServoMotor()
        print("Getting QWP Elliptec Motor")
        self.qwm = ElliptecMotor(port_no=DEFAULT_ELLO_PORTNO)  # quicker
        print("Getting vimba camera")
        self.cam = VimbaCamera(DEFAULT_CAM_NO)
        # self.cam.set_borders(Borders(100, 400, 650, 800))
        # self.cam.set_borders(Borders(150, 570, 550, 690))
        self.res = WavePlateOptimizationResult()

        self.H_angles = H_angles or np.linspace(0, 180, 19)
        self.Q_angles = Q_angles or np.linspace(0, 180, 19)

        self.res.H_angles = self.H_angles
        self.res.Q_angles = self.Q_angles
        self.res.heatmap = np.zeros((self.H_angles.shape[0], self.Q_angles.shape[0]))
        self.res.path = f"{LOGS_DIR}\\{self.res.timestamp}.wpscan"

    def run(self):
        print("Beginning Scan")
        for i, h_angle in enumerate(self.H_angles):
            self.hwm.move_absolute(h_angle)

            for j, q_angle in enumerate(self.Q_angles):
                self.qwm.move_absolute(q_angle)

                im = self.cam.get_image()
                p = self._energy_percent_in_H(im)
                self.res.heatmap[i, j] = p
                self.res.saveto()

    def _energy_percent_in_H(self, im):
        # get the most light into H
        mid_col = im.shape[1] // 2

        pol1_energy = im[:, :mid_col].sum()
        pol2_energy = im[:, mid_col:].sum()
        tot_energy = pol1_energy + pol2_energy
        return pol1_energy / tot_energy

    def close(self):
        self.hwm.close()
        self.qwm.close()
        self.cam.close()


if __name__ == "__main__":
    gro = GlobalRotationOptimization()
    gro.run()
    gro.close()


