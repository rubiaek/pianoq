# Motors
from pianoq.lab.mplc.consts import thorlabs_x_serial, thorlabs_y_serial
from pianoq.lab.thorlabs_motor import ThorlabsKcubeDC, ThorlabsKcubeStepper
from pianoq.lab.zaber_motor import ZaberMotors

backlash = 0.0
wait_after_move = 0.0

mxi = ThorlabsKcubeDC(thorlabs_x_serial, backlash=backlash, wait_after_move=wait_after_move)
myi = ThorlabsKcubeStepper(thorlabs_y_serial, backlash=backlash, wait_after_move=wait_after_move)

zaber_ms = ZaberMotors(backlash=backlash, wait_after_move=wait_after_move)
mxs = zaber_ms.motors[1]
mys = zaber_ms.motors[0]


def move_i(x, y):
    mxi.move_absolute(x)
    myi.move_absolute(y)


def move_s(x, y):
    mxs.move_absolute(x)
    mys.move_absolute(y)

mxi.close()
myi.close()
mxs.close()
mys.close()


# Timetagger
from pianoq.lab.time_tagger import QPTimeTagger
integration_time = 1
tt = QPTimeTagger(integration_time=integration_time, remote=True)

# MPLC
from pianoq.lab.mplc.mplc_device import MPLCDevice
from pianoq.lab.mplc.phase_finder_result import PhaseFinderResult
from pianoq.lab.mplc.mask_utils import get_masks_matlab, remove_input_modes, add_phase_input_spots
import numpy as np

mplc = MPLCDevice()
phases_result = PhaseFinderResult()
phases_result.loadfrom(r"G:\My Drive\People\Ronen\PHD\MPLC\results\2024_08_05_14_42_39_QKD_row3_phases.phases")

wfm_masks_path = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\Masks_31_10_23_QKD5d_MUB2_mm_33_3_conjbases.mat"
masks = get_masks_matlab(wfm_masks_path=wfm_masks_path)

modes_to_keep = np.array([3, 8, 13, 18, 23, 28, 33, 38, 43, 48])
masks = remove_input_modes(masks, modes_to_keep=modes_to_keep)
masks = add_phase_input_spots(masks, phases_result.phases)

mplc.load_masks(masks, linear_tilts=True)


# alignment
from pianoq.lab.mplc.mask_aligner import MPLCAligner
ml = MPLCAligner()
print(f'{ml.centers_x}\r\n{ml.centers_y}')

ml.update(imaging1='1to5w4f', imaging2='5to11w8', pi_steps_x=ml.centers_x[1])
ml.update(imaging1='1to5w4f', imaging2='5to11w8', pi_steps_y=ml.centers_y[1] + 12)
ml.update(imaging1='1to5w4f', imaging2='5to11w8', pi_steps_y=ml.centers_y[1] - 12)
ml.update_interactive(imaging1='1to5w4f', imaging2='5to11w8', pi_step_y=ml.centers_y[1] - 12)
