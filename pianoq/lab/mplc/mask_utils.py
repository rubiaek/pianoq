import numpy as np
from pianoq.lab.mplc.consts import (SLM_DIMS, MASK_DIMS, PIXEL_SIZE, N_SPOTS, D_BETWEEN_SPOTS_INPUT, SPOT_WAIST_IN, K,
                                    D_BETWEEN_PLANES, D1, D2)
import scipy.io

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


def mask_centers_to_mask_slices(mask_centers):
    mask_slices = []
    for mask_center in mask_centers:
        x_c, y_c = mask_center
        sl = np.index_exp[y_c - MASK_DIMS[0]//2: y_c + MASK_DIMS[0]//2,
                          x_c - MASK_DIMS[1]//2: x_c + MASK_DIMS[1]//2]
        mask_slices.append(sl)
    return mask_slices


def make_grid_phasmask():
    # Like Ohad did in the MPLC class, centered around 0
    Nx = MASK_DIMS[1]
    Ny = MASK_DIMS[0]
    dx = PIXEL_SIZE
    dy = PIXEL_SIZE
    X = (np.arange(Nx) - (Nx//2 - 0.5)) * dx  # 0.5 for symmetry
    Y = (np.arange(Ny) - (Ny//2 - 0.5)) * dy  # 0.5 for symmetry
    XX, YY = np.meshgrid(X, Y)
    return XX, YY


def calc_pos_modes_in():
    n_steps_x = []
    n_steps_y = []
    dim = int(np.sqrt(N_SPOTS))

    # this loop results in n_steps_x a len 25 array, of step sizes from middle,
    # according to the numbering from the middle right, upwards, then middle second from right, etc.
    # for say 3 it will be: 1,1,1,0,0,0,-1,-1,-1
    # because of the numbering convention, in y it will be 0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5
    for l in range(dim):
        if dim % 2:
            n_steps_x.extend([(dim - 1) / 2 - l] * dim)
        else:
            n_steps_x.extend([0.5 + (dim / 2 - 1 - l)] * dim)
        n_steps_y.extend([0.5 + i for i in range(dim)])

    # adding the other 25 modes
    n_steps_x = n_steps_x + n_steps_x[::-1]
    n_steps_y = n_steps_y + [-y for y in n_steps_y]

    # These are lists of the middle of all 50 modes
    x_modes_in = D_BETWEEN_SPOTS_INPUT * np.array(n_steps_x)
    y_modes_in = D_BETWEEN_SPOTS_INPUT * np.array(n_steps_y)
    return x_modes_in, y_modes_in


def add_phase_input_spots(masks, phases):
    masks = masks.copy()
    XX, YY = make_grid_phasmask()
    x_modes_in, y_modes_in = calc_pos_modes_in()

    for k in range(2 * N_SPOTS):
        condition = np.sqrt((XX - x_modes_in[k])**2 + (YY - y_modes_in[k])**2) < (2 * SPOT_WAIST_IN)
        masks[0][condition] = masks[0][condition] * np.exp(1j*phases[k])

    return masks


def remove_input_modes(masks, modes_to_keep):
    masks = masks.copy()
    XX, YY = make_grid_phasmask()
    x_modes_in, y_modes_in = calc_pos_modes_in()

    lin_mask = np.exp(-2j*np.pi*XX/(8*PIXEL_SIZE))  # 2 pi within 8 pixels

    for k in range(2 * N_SPOTS):
        if k + 1 not in modes_to_keep:  # Python uses 0-based indexing, so we add 1 to match MATLAB's 1-based indexing
            condition = np.sqrt((XX - x_modes_in[k])**2 + (YY - y_modes_in[k])**2) < (2 * SPOT_WAIST_IN)
            masks[0][condition] = lin_mask[condition]

    return masks


def get_lens(f):
    """ f in units of D_BETWEEN_PLANES. return e^i*phase """
    XX, YY = make_grid_phasmask()
    phase_lens = -K * (XX ** 2 + YY ** 2) / (2 * f * D_BETWEEN_PLANES)
    phase_lens = phase_lens - np.min(phase_lens) + 0.01
    return np.exp(1j*phase_lens)

def get_imaging_masks():
    """ image plane 1 to plane 11 of detectors """
    masks = np.ones((10, MASK_DIMS[0], MASK_DIMS[1]), dtype=complex)
    # 4f between 1 and 5 with 2 and 4
    masks[1] = get_lens(f=1)
    masks[3] = get_lens(f=1)
    masks[7] = get_lens(3 * (2 * D1 + D2) / (5 * D1 + D2))
    return masks

def get_masks_matlab(wfm_masks_path):
    masks = scipy.io.loadmat(wfm_masks_path)['MASKS'].astype(complex)
    # This is the size_factor which is basically always 3 (3*MASK_DIMS)
    height, width = (1080, 420)
    assert masks[0].shape == (height, width)
    # Matlab "cut_masks" - carves out the middle, and takes only first 10 masks
    masks = masks[:10, 360:720, 140:280]
    # Matlab "combine_masks" gets two mask sets, and takes the upper part of the first set
    # and the lower part of the second set
    # I assume that this concatenation will happen before, because this is just weird
    return masks


""" 
TEST:
    from pianoq.lab.mplc.mask_utils import get_masks_matlab, remove_input_modes, add_phase_input_spots
    from pianoq.lab.mplc.mplc_device import MPLCDevice
    import matplotlib.pyplot as plt
    from pianoq.misc.mplt import *
    plt.close('all')
    wfm_masks_path = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\Masks_31_10_23_QKD5d_MUB2_mm_33_3_conjbases.mat"
    phases_path = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\phase_align_QKD5d_10_11_23_3.mat"
    modes_to_keep = np.array([3, 8, 13, 18, 23, 28, 33, 38, 43, 48])
    masks = get_masks_matlab(wfm_masks_path=wfm_masks_path)
    masks = remove_input_modes(masks, modes_to_keep=modes_to_keep)
    phases = np.squeeze(scipy.io.loadmat(phases_path)['phases'])
    masks = add_phase_input_spots(masks, phases)
    
    m = MPLCDevice()
    m.load_masks(masks, linear_tilts=True)
    Q_p = m.convert_to_uint8(m.slm_mask)
    mimshow(Q_p, cmap='gray')
    full_mask_path = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\total_phase_mask.mat"
    m.load_slm_mask(full_mask_path)
    Q_m = m.convert_to_uint8(m.slm_mask)
    mimshow(Q_m, cmap='gray')
    
    mimshow(Q_p - Q_m, cmap='gray')

"""
