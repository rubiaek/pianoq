import numpy as np
from pianoq.lab.mplc.consts import SLM_DIMS, MASK_DIMS, PIXEL_SIZE, N_SPOTS, D_BETWEEN_SPOTS_INPUT, SPOT_WAIST_IN


def mask_centers_to_mask_slices(mask_centers):
    mask_slices = []
    for mask_center in mask_centers:
        x_c, y_c = mask_center
        sl = np.index_exp[y_c - MASK_DIMS[0]//2: y_c + MASK_DIMS[0]//2,
                          x_c - MASK_DIMS[1]//2: x_c + MASK_DIMS[1]//2]
        mask_slices.append(sl)
    return mask_slices


def make_grid_phasmask():
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
