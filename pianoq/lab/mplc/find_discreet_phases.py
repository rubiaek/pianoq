import datetime
import time

import numpy as np
import matplotlib.pyplot as plt
from pianoq.lab.thorlabs_motor import ThorlabsKcubeDC, ThorlabsKcubeStepper
from pianoq.lab.zaber_motor import ZaberMotors
from pianoq.lab.time_tagger import QPTimeTagger
from pianoq.lab.mplc.consts import thorlabs_x_serial, thorlabs_y_serial
from pianoq.lab.mplc.discrete_scan_result import DiscreetScanResult
from pianoq.lab.mplc.mplc_device import MPLCDevice
from pianoq.lab.mplc.wfm_res_to_masks import matlab_WFM_masks_to_masks

LOGS_DIR = r"G:\My Drive\People\Ronen\PHD\MPLC\results"

class PhaseFinderResult(object):
    def __init__(self):
        self.integration_time = 0
        self.phases = np.zeros(50)


class PhaseFinder(object):
    def __init__(self, integration_time=1, remote_tagger=True):
        self._get_hardware(remote_tagger=remote_tagger)

    def _get_hardware(self, remote_tagger=True):
        self.zaber_ms = ZaberMotors()
        self.m_sig_x = self.zaber_ms.motors[1]
        self.m_sig_y = self.zaber_ms.motors[0]
        print("Got Zaber motors!")

        self.m_idl_x = ThorlabsKcubeDC(thorlabs_x_serial)
        self.m_idl_y = ThorlabsKcubeStepper(thorlabs_y_serial)
        print("Got Thorlabs motors!")

        self.time_tagger = QPTimeTagger(integration_time=self.integration_time, remote=remote_tagger)
        print("Got TimeTagger!")

    def close(self):
        self.zaber_ms.close()
        self.time_tagger.close()
        self.m_sig_x.close()
        self.m_sig_y.close()

def main():
    m = MPLCDevice()

    masks_path = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\Masks_31_10_23_QKD5d_MUB2_mm_33_3_conjbases.mat"
    phases_path = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\phase_align_QKD5d_10_11_23_3.mat"
    masks = matlab_WFM_masks_to_masks(out_path=None, wfm_masks_path=masks_path, phases_path=phases_path)
    m.load_masks(masks, linear_tilts=True)

    locs_idler = np.array(
        [(9.078127869934999, 3.1259338321488515),
         (9.052534495520005, 2.7471518908069443),
         (9.011585096456015, 2.383725974114033),
         (8.978313709716524, 2.0049440327721255),
         (8.95272033530153, 1.6287214288717176)]
    )

    locs_signal = np.array(
        [(11.528657427216665, 8.774409886706874),
         (11.56436061831159, 9.141642709397514),
         (11.587312669729753, 9.508875532088153),
         (11.62811631669538, 9.878658582714145),
         (11.656168823984249, 10.245891405404784)]
    )

    i = 0
    j = 0
    pf = PhaseFinder(integration_time=4, remote_tagger=True, run_name='QKD_row3_phases')
    pf.m_idl_x.move_absolute(locs_idler[i, 0])
    pf.m_idl_y.move_absolute(locs_idler[j, 1])
    pf.m_sig_x.move_absolute(locs_signal[i, 0])
    pf.m_sig_y.move_absolute(locs_signal[j, 1])
    time.sleep(0.5)

    pf.find_phases(modes_keep)

    pf.close()
    m.close()


if __name__ == "__main__":
    main()
