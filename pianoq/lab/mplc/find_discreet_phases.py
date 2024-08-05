import datetime
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from pianoq.lab.thorlabs_motor import ThorlabsKcubeDC, ThorlabsKcubeStepper
from pianoq.lab.zaber_motor import ZaberMotors
from pianoq.lab.time_tagger import QPTimeTagger
from pianoq.lab.mplc.consts import thorlabs_x_serial, thorlabs_y_serial
from pianoq.lab.mplc.mplc_device import MPLCDevice
from pianoq.lab.mplc.mask_utils import get_masks_matlab, add_phase_input_spots, remove_input_modes
from pianoq.lab.mplc.consts import N_SPOTS
from pianoq.lab.mplc.phase_finder_result import PhaseFinderResult

LOGS_DIR = r"G:\My Drive\People\Ronen\PHD\MPLC\results"


class PhaseFinder(object):
    def __init__(self, mplc, modes_to_keep, integration_time=1, remote_tagger=True, run_name='', N_phases=10, intial_phases=None, coin_window=2e-9, saveto_path=None):
        self.mplc = mplc
        self.orig_masks = mplc.masks.copy()
        self.res = PhaseFinderResult()
        self.res.path = saveto_path or f"{LOGS_DIR}\\{self.res.timestamp}_{run_name}.phases"
        self.res.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.res.integration_time = integration_time
        self.res.run_name = run_name
        self.res.modes_to_keep = modes_to_keep
        self.res.N_phases = N_phases
        self.res.phase_vec_step = 2*np.pi / self.res.N_phases
        self.res.phase_vec = np.linspace(0, 2*np.pi - self.res.phase_vec_step, self.res.N_phases)
        self.res.coincidences = np.zeros((len(self.res.modes_to_keep), self.res.N_phases))
        self.res.single1s = np.zeros((len(self.res.modes_to_keep), self.res.N_phases))
        self.res.single2s = np.zeros((len(self.res.modes_to_keep), self.res.N_phases))
        self.res.coin_window = coin_window
        self.initial_phases = intial_phases if intial_phases is not None else np.zeros(N_SPOTS*2)
        self.res.phases = self.initial_phases
        self._get_hardware(remote_tagger=remote_tagger)

    def _get_hardware(self, remote_tagger=True):
        self.zaber_ms = ZaberMotors()
        self.m_sig_x = self.zaber_ms.motors[1]
        self.m_sig_y = self.zaber_ms.motors[0]
        print("Got Zaber motors!")

        self.m_idl_x = ThorlabsKcubeDC(thorlabs_x_serial)
        self.m_idl_y = ThorlabsKcubeStepper(thorlabs_y_serial)
        print("Got Thorlabs motors!")

        self.time_tagger = QPTimeTagger(integration_time=self.res.integration_time, remote=remote_tagger,
                                        coin_window=self.res.coin_window)
        print("Got TimeTagger!")

    def find_phases(self):
        for i, mode_no in enumerate(self.res.modes_to_keep):
            for j, phase in enumerate(self.res.phase_vec):
                # Python 0-based, and modes begin at 1
                self.res.phases[mode_no-1] = phase
                masks = add_phase_input_spots(self.orig_masks, self.res.phases)
                self.mplc.load_masks(masks)
                time.sleep(0.1)

                s1, s2, c = self.time_tagger.read_interesting()
                self.res.single1s[i, j] = s1
                self.res.single2s[i, j] = s2
                self.res.coincidences[i, j] = c
                print(f'{i},{j}')

            CC = (self.res.coincidences[i, :] * np.exp(1j * self.res.phase_vec)).sum()
            phi_best = np.mod(np.angle(CC)+2*np.pi, 2*np.pi)
            self.res.phases[mode_no-1] = phi_best
            self.res.saveto(self.res.path)

    def close(self):
        self.zaber_ms.close()
        self.time_tagger.close()
        self.m_sig_x.close()
        self.m_sig_y.close()


def QKD_row_3_3():
    m = MPLCDevice()

    wfm_masks_path = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\Masks_31_10_23_QKD5d_MUB2_mm_33_3_conjbases.mat"
    phases_path = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\phase_align_QKD5d_10_11_23_3.mat"
    modes_to_keep = np.array([3, 8, 13, 18, 23, 28, 33, 38, 43, 48])
    masks = get_masks_matlab(wfm_masks_path=wfm_masks_path)
    masks = remove_input_modes(masks, modes_to_keep=modes_to_keep)
    phases = np.squeeze(scipy.io.loadmat(phases_path)['phases'])
    # phases = np.zeros(N_SPOTS*2)
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
    pf = PhaseFinder(mplc=m, integration_time=25, remote_tagger=True, run_name='QKD_row3_phases',
                     modes_to_keep=modes_to_keep, intial_phases=phases, coin_window=2e-9)
    pf.m_idl_x.move_absolute(locs_idler[i, 0])
    pf.m_idl_y.move_absolute(locs_idler[j, 1])
    pf.m_sig_x.move_absolute(locs_signal[i, 0])
    pf.m_sig_y.move_absolute(locs_signal[j, 1])
    time.sleep(2)

    pf.find_phases()

    pf.close()
    m.close()


if __name__ == "__main__":
    QKD_row_3_3()
