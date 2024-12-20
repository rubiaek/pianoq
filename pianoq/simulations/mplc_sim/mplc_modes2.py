"""
After implementing things in mplc_modes.py I wanted a version exactly like Ohad's code, so things are very similar,
just making sure it will be one to one
"""

import numpy as np

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


def gen_input_spots_array(waist, D_between_modes, XX, YY, dim, deltax_in=0, deltay_in=0, dead_middle_zone=0):
    # generates the 50 spots
    n_steps_x = []
    n_steps_y = []

    for l in range(dim):
        if dim % 2:
            n_steps_x.extend([(dim - 1) / 2 - l] * dim)
        else:
            n_steps_x.extend([0.5 + (dim / 2 - 1 - l)] * dim)
        n_steps_y.extend([0.5 + i for i in range(dim)])

    n_steps_x = n_steps_x + n_steps_x[::-1]
    n_steps_y = n_steps_y + [-i for i in n_steps_y]

    x_modes_in = D_between_modes * np.array(n_steps_x) + deltax_in
    y_modes_in = D_between_modes * np.array(n_steps_y) + deltay_in

    SPOTS = np.zeros((2 * dim ** 2, XX.shape[0], XX.shape[1]), dtype=complex)

    if dead_middle_zone != 0:
        y_modes_in[y_modes_in < 0] -= dead_middle_zone
        y_modes_in[y_modes_in > 0] += dead_middle_zone

    for i in range(2 * dim ** 2):
        spot = np.exp(-((XX - x_modes_in[i]) ** 2 + (YY - y_modes_in[i]) ** 2) / waist ** 2)  # assuming a Gaussian
        spot = spot / np.sqrt(np.sum(np.abs(spot) ** 2))  # normalization
        SPOTS[i, :, :] = spot

    return SPOTS, x_modes_in, y_modes_in


def gen_output_modes_Unitary(waist_out, D_between_modes, XX, YY, Matrix_trans, dim, which_modes,
                             deltax_out=0, deltay_out=0, dead_middle_zone=0):
    SPOTS_OUT, x_modes_in, y_modes_in, = gen_input_spots_array(waist_out, D_between_modes, XX, YY, dim,
                                                                deltax_in=deltax_out, deltay_in=deltay_out,
                                                                dead_middle_zone=dead_middle_zone)

    SPOTS_OUT = SPOTS_OUT[which_modes]
    x_modes_in = x_modes_in[which_modes]
    y_modes_in = y_modes_in[which_modes]

    MODES = np.zeros((len(which_modes), XX.shape[0], XX.shape[1]), dtype=complex)
    phase_pos_x = np.zeros(len(which_modes))
    phase_pos_y = np.zeros(len(which_modes))

    for j in range(len(which_modes)):
        # Fancy matrix multiplication, to take the values from the transformation and multiply the spot modes
        # which are 2d complex pictures
        mode = np.sum(Matrix_trans[:, j][:, np.newaxis, np.newaxis] * SPOTS_OUT, axis=0)
        mode /= np.sqrt(np.sum(np.abs(mode) ** 2))  # normalization
        MODES[j] = mode
        # phase_pos_x(y) was for some old idea, that the phase here needs to be zero, and like this find the
        # global required phase shifts of input modes
        phase_pos_x[j] = x_modes_in[j]
        phase_pos_y[j] = y_modes_in[j]

    return MODES, phase_pos_x, phase_pos_y
