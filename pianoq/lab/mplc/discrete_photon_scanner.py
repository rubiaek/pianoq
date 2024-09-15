import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from pianoq.lab.thorlabs_motor import ThorlabsKcubeDC, ThorlabsKcubeStepper
from pianoq.lab.zaber_motor import ZaberMotors
from pianoq.lab.time_tagger import QPTimeTagger
from pianoq.lab.mplc.consts import thorlabs_x_serial, thorlabs_y_serial, TIMETAGGER_DELAYS, TIMETAGGER_COIN_WINDOW
from pianoq.lab.mplc.discrete_scan_result import DiscreetScanResult
from pianoq.lab.mplc.mplc_device import MPLCDevice
from pianoq.lab.mplc.mask_utils import remove_input_modes, add_phase_input_spots, get_masks_matlab
from pianoq.lab.mplc.phase_finder_result import PhaseFinderResult

LOGS_DIR = r"G:\My Drive\Projects\MPLC\results\lab\temp"


class DiscretePhotonScanner:
    def __init__(self, locs_signal, locs_idler, integration_time=1, coin_window=1e-9, remote_tagger=True,
                 saveto_path='', run_name='', backlash=0., wait_after_move=0.1, no_hw_mode=False):
        self.res = DiscreetScanResult()
        self.res.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.res.path = saveto_path or f"{LOGS_DIR}\\{self.res.timestamp}_{run_name}.dscan"

        self.res.locs_signal = locs_signal
        self.res.locs_idler = locs_idler
        self.res.integration_time = integration_time
        self.res.coin_window = coin_window
        self.res.wait_after_move = wait_after_move
        self.res.backlash = backlash

        self.zaber_ms = None
        self.m_sig_x = None
        self.m_sig_y = None
        # idler with lower y values
        self.m_idl_x = None
        self.m_idl_y = None
        self.time_tagger = None
        if not no_hw_mode:
            self._get_hardware(remote_tagger=remote_tagger)

    def _get_hardware(self, remote_tagger=True):
        self.zaber_ms = ZaberMotors(backlash=self.res.backlash, wait_after_move=self.res.wait_after_move)
        self.m_sig_x = self.zaber_ms.motors[1]
        self.m_sig_y = self.zaber_ms.motors[0]
        print("Got Zaber motors!")

        self.m_idl_x = ThorlabsKcubeDC(thorlabs_x_serial,
                                       backlash=self.res.backlash, wait_after_move=self.res.wait_after_move)
        self.m_idl_y = ThorlabsKcubeStepper(thorlabs_y_serial,
                                            backlash=self.res.backlash, wait_after_move=self.res.wait_after_move)
        print("Got Thorlabs motors!")

        self.time_tagger = QPTimeTagger(integration_time=self.res.integration_time, remote=remote_tagger,
                                        single_channel_delays=TIMETAGGER_DELAYS, coin_window=self.res.coin_window)
        print("Got TimeTagger!")

    def scan(self):
        print('beginning scan')
        self.res.single1s = np.zeros((len(self.res.locs_signal), len(self.res.locs_idler)))
        self.res.single2s = np.zeros_like(self.res.single1s)
        self.res.coincidences = np.zeros_like(self.res.single1s)

        for i, loc_idl in enumerate(self.res.locs_idler):
            self.m_idl_x.move_absolute(loc_idl[0])
            self.m_idl_y.move_absolute(loc_idl[1])

            for j, loc_sig in enumerate(self.res.locs_signal):
                self.m_sig_x.move_absolute(loc_sig[0])
                self.m_sig_y.move_absolute(loc_sig[1])

                s1, s2, c12 = self.time_tagger.read_interesting()
                self.res.single1s[i, j] = s1
                self.res.single2s[i, j] = s2
                self.res.coincidences[i, j] = c12
                print(rf'{i}, {j}: {s1:.2f}, {s2:.2f}, {c12:.2f}')

            self.res.saveto(self.res.path)

    def close(self):
        self.zaber_ms.close()
        self.time_tagger.close()
        self.m_sig_x.close()
        self.m_sig_y.close()
        self.m_idl_x.close()
        self.m_idl_y.close()


def run_QKD_row_3_3():
    coin_window = 0.4e-9
    integration_time = 4

    # Full python impl.
    if True:
        mplc = MPLCDevice()
        path = r"G:\My Drive\Projects\MPLC\results\lab\temp\rss_wfm1.masks"
        masks = np.load(path)['masks']
        modes_to_keep = np.array([3, 8, 13, 18, 23, 28, 33, 38, 43, 48])
        masks = remove_input_modes(masks, modes_to_keep=modes_to_keep)
        phases_result = PhaseFinderResult()
        phases_result.loadfrom(r"G:\My Drive\Projects\MPLC\results\lab\temp\2024_08_11_10_26_11_QKD_row3_phases.phases")
        masks = add_phase_input_spots(masks, phases_result.phases)
        mplc.load_masks(masks, linear_tilts=True)

    # Matlab WFM - python lab
    if False:
        mplc = MPLCDevice()
        # phases
        phases_path = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\phase_align_QKD5d_10_11_23_3.mat"
        phases = np.squeeze(scipy.io.loadmat(phases_path)['phases'])
        # phases_result = PhaseFinderResult()
        # phases_result.loadfrom(r"G:\My Drive\People\Ronen\PHD\MPLC\results\2024_08_05_14_42_39_QKD_row3_phases.phases")
        # phases = phases_result.phases

        wfm_masks_path = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\Masks_31_10_23_QKD5d_MUB2_mm_33_3_conjbases.mat"
        masks = get_masks_matlab(wfm_masks_path=wfm_masks_path)

        modes_to_keep = np.array([3, 8, 13, 18, 23, 28, 33, 38, 43, 48])
        masks = remove_input_modes(masks, modes_to_keep=modes_to_keep)
        # masks = add_phase_input_spots(masks, phases)
        run_name = f'QKD_row_3_3_matlab_masks_python_code_zero_phases_{integration_time}s_coin_400ps'
        mplc.load_masks(masks, linear_tilts=True)

    # Matlab slm_mask calc
    if False:
        mplc = MPLCDevice()
        total_mask_path = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\total_phase_mask.mat"
        mplc.load_slm_mask(total_mask_path)

    locs_idler = np.array(
        [(9.078127869934999, 3.1259338321488515),
         (9.052534495520005, 2.7471518908069443),
         (9.011585096456015, 2.383725974114033),
         (8.978313709716524, 2.0049440327721255),
         (8.95272033530153, 1.6287214288717176)]
    )
    # locs_idler = np.array(
    #     [(9.039298460536191, 3.1122034726451306),
    #      (9.01663845968644, 2.738313458624255),
    #      (8.988313458624253, 2.3870834454531296),
    #      (8.94865845713719, 1.990533430582504),
    #      (8.942993456924754, 1.6336384171989407)]
    # )

    locs_signal = np.array(
        [(11.528657427216665, 8.774409886706874),
         (11.56436061831159, 9.141642709397514),
         (11.587312669729753, 9.508875532088153),
         (11.62811631669538, 9.878658582714145),
         (11.656168823984249, 10.245891405404784)]
    )
    # locs_signal = np.array(
    #     [(11.499064980955438, 8.758022042739357),
    #      (11.541896119192938, 9.137383552842916),
    #      (11.548014853226867, 9.50450759487862),
    #      (11.596964725498292, 9.889987839016108),
    #      (11.645914597769721, 10.250993147017882)]
    # )

    backlash = 0.0
    wait_after_move = 0.3

    dps = DiscretePhotonScanner(locs_signal, locs_idler, integration_time=integration_time, remote_tagger=True, run_name='QKD_row3',
                                backlash=backlash, wait_after_move=wait_after_move, coin_window=coin_window)
    dps.scan()
    dps.close()
    mplc.close()


    dps.res.show()
    dps.res.show_singles()


if __name__ == '__main__':
    run_QKD_row_3_3()
    plt.show()

