import numpy as np
import scipy.io

from pianoq.lab.mplc.utils import make_grid_phasmask, calc_pos_modes_in
from pianoq.lab.mplc.consts import PIXEL_SIZE, N_SPOTS, SPOT_WAIST_IN


ORIG_MASKS_PATH = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\Masks_31_10_23_QKD5d_MUB2_mm_33_3_conjbases.mat"
ORIG_PHASES_PATH = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\phase_align_QKD5d_10_11_23_3.mat"

"""
modes are ordered like this:
25 20 15 10 5 
24 19 14 9  4 
23 18 13 8  3 
22 17 12 7  2 
21 16 11 6  1 

26 31 36 41 46
27 32 37 42 47
28 33 35 43 48
29 34 39 44 49
30 35 40 45 50 
"""


def remove_input_modes(MASKS, modes_to_keep):
    masks = MASKS.copy()
    XX, YY = make_grid_phasmask()
    x_modes_in, y_modes_in = calc_pos_modes_in()

    lin_mask = np.exp(-2j*np.pi*XX/(8*PIXEL_SIZE))  # 2 pi within 8 pixels

    for k in range(2 * N_SPOTS):
        if k + 1 not in modes_to_keep:  # Python uses 0-based indexing, so we add 1 to match MATLAB's 1-based indexing
            condition = np.sqrt((XX - x_modes_in[k])**2 + (YY - y_modes_in[k])**2) < (2 * SPOT_WAIST_IN)
            masks[0][condition] = lin_mask[condition]

    return masks


def add_phase_input_spots(MASKS, phases_path):
    masks = MASKS.copy()
    XX, YY = make_grid_phasmask()
    x_modes_in, y_modes_in = calc_pos_modes_in()
    phases = np.squeeze(scipy.io.loadmat(phases_path)['phases'])
    for k in range(2 * N_SPOTS):
        condition = np.sqrt((XX - x_modes_in[k])**2 + (YY - y_modes_in[k])**2) < (2 * SPOT_WAIST_IN)
        masks[0][condition] = masks[0][condition] * np.exp(1j*phases[k])

    return masks


def matlab_WFM_masks_to_masks(out_path=None, wfm_masks_path=ORIG_MASKS_PATH, phases_path=ORIG_PHASES_PATH):
    MASKS = scipy.io.loadmat(wfm_masks_path)['MASKS'].astype(complex)
    # This is the size_factor which is basically always 3 (3*MASK_DIMS)
    height, width = (1080, 420)
    assert MASKS[0].shape == (height, width)

    # Matlab "combine_masks" gets two mask sets, and takes the upper part of the first set
    # and the lower part of the second set
    # I assume that this concatenation will happen before, because this is just weird

    # Matlab "cut_masks" - carves out the middle, and takes only first 10 masks
    MASKS = MASKS[:10, 360:720, 140:280]

    modes_to_keep = np.array([1, 6, 11, 16, 21, 26, 31, 36, 41, 46])
    row_num_signal = 3
    row_num_idler = 3
    modes_to_keep[:5] += (row_num_signal - 1)
    modes_to_keep[5:] += (row_num_idler - 1)

    MASKS = remove_input_modes(MASKS, modes_to_keep=modes_to_keep)
    MASKS = add_phase_input_spots(MASKS, phases_path)

    MASKS = np.angle(MASKS).astype(float)
    if out_path is not None:
        f = open(out_path, 'wb')
        np.savez(f, masks=MASKS)
        f.close()

    return MASKS


"""
TEST: 
    from pianoq.lab.mplc.wfm_res_to_masks import matlab_WFM_masks_to_masks
    from pianoq.lab.mplc.mplc_device import MPLCDevice
    plt.close('all')
    matlab_WFM_masks_to_masks('C:\\temp\\r1.masks')
    m = MPLCDevice()
    m.load_slm_mask(r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\total_phase_mask.mat")
    Q_m = m.convert_to_uint8(m.slm_mask)
    mimshow(Q_m, cmap='gray')
    
    m.load_masks("C:\\temp\\r1.masks")
    Q_p = m.convert_to_uint8(m.slm_mask)
    mimshow(Q_p, cmap='gray')
    mimshow(Q_p - Q_m, cmap='gray')
"""
