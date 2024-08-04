import numpy as np
import scipy.io

from pianoq.lab.mplc.utils import make_grid_phasmask, calc_pos_modes_in
from pianoq.lab.mplc.consts import PIXEL_SIZE, N_SPOTS, SPOT_WAIST_IN


ORIG_MASKS_PATH = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\Masks_31_10_23_QKD5d_MUB2_mm_33_3_conjbases.mat"
ORIG_PHASES_PATH = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\phase_align_QKD5d_10_11_23_3.mat"


def matlab_WFM_masks_to_ronen_masks(out_path, orig_masks_path=ORIG_MASKS_PATH, orig_phases_path=ORIG_PHASES_PATH):

    MASKS = scipy.io.loadmat(orig_masks_path)['MASKS'].astype(complex)
    # This is the size_factor which is basically always 3 (3*MASK_DIMS)
    height, width = (1080, 420)
    assert MASKS[0].shape == (height, width)

    # Matlab "combine_masks" gets two mask sets, and takes the upper part of the first set
    # and the lower part of the second set
    # I assume that this concatenation will happen before, because this is just weird

    # Matlab "cut_masks" - carves out the middle, and takes only first 10 masks
    MASKS = MASKS[:10, 360:720, 140:280]
    MASKS = np.angle(MASKS).astype(float)

    # Matlab `remove_input_modes`
    XX, YY = make_grid_phasmask()
    x_modes_in, y_modes_in = calc_pos_modes_in()

    modes_to_keep = np.array([1, 6, 11, 16, 21, 26, 31, 36, 41, 46])
    row_num_signal = 3
    row_num_idler = 3
    modes_to_keep[:5] += (row_num_signal - 1)
    modes_to_keep[5:] += (row_num_idler - 1)

    lin_mask = -2*np.pi*XX/(8*PIXEL_SIZE)  # 2 pi within 8 pixels

    for k in range(2 * N_SPOTS):
        if k + 1 not in modes_to_keep:  # Python uses 0-based indexing, so we add 1 to match MATLAB's 1-based indexing
            condition = np.sqrt((XX - x_modes_in[k])**2 + (YY - y_modes_in[k])**2) < (2 * SPOT_WAIST_IN)
            MASKS[0][condition] = lin_mask[condition]

    # Matlab `add_phase_input_spots`
    # phases = np.zeros(50)
    phases = np.squeeze(scipy.io.loadmat(orig_phases_path)['phases'])
    for k in range(2 * N_SPOTS):
        condition = np.sqrt((XX - x_modes_in[k])**2 + (YY - y_modes_in[k])**2) < (2 * SPOT_WAIST_IN)
        MASKS[0][condition] = MASKS[0][condition] + phases[k]

    f = open(out_path, 'wb')
    np.savez(f, masks=MASKS)
    f.close()


"""
TEST: 
    from pianoq.lab.mplc.ohad_masks_to_my_masks import matlab_WFM_masks_to_ronen_masks
    from pianoq.lab.mplc.mplc_device import MPLCDevice
    plt.close('all')
    matlab_WFM_masks_to_ronen_masks('C:\\temp\\r1.masks')
    m = MPLCDevice()
    m.load_slm_mask(r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\total_phase_mask.mat")
    Q_m = m.convert_to_uint8(m.slm_mask)
    mimshow(Q_m, cmap='gray')
    
    m.load_masks("C:\\temp\\r1.masks")
    Q_p = m.convert_to_uint8(m.slm_mask)
    mimshow(Q_p, cmap='gray')
    mimshow(Q_p - Q_m, cmap='gray')
"""
