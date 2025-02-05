import copy
import datetime
import numpy as np
from pianoq.simulations.mplc_sim.mplc_sim import MPLCSim
from pianoq.simulations.mplc_sim.mplc_modes2 import gen_input_spots_array, gen_output_modes_Unitary
from pianoq.simulations.mplc_sim.consts import default_wfm_conf
from pianoq.simulations.mplc_sim.mplc_modes import get_speckle_modes_conf, get_spots_modes_conf
from pianoq.simulations.mplc_sim.mplc_utils import get_lens_mask_conf, get_lens_mask


DEFAULT_DIR = r"G:\My Drive\Projects\MPLC\results\simulations\masks"


def create_WFM_diffuser_masks(same_diffuser=False, out_path=None, name=None, N_iterations=None):
    # All in m

    # MPLC WFM conf #
    # Lense in plane 9 between 7 and 11. Allow phases freedom in plane 11 since I measure intensity
    active_planes = np.array([True] * 11)
    active_planes[7:10] = False  # [7,8,9] which is planes 8,9,10 (one before and after lens)
    N_modes = 16  # for photons 1 and 2, 4 spots to speckle and 4 speckles to spots (4+4)*2 = 16

    conf = copy.deepcopy(default_wfm_conf)
    conf['active_planes'] = active_planes
    conf['N_modes'] = N_modes
    conf['symmetric_masks'] = same_diffuser
    if N_iterations is not None:
        conf['N_iterations'] = N_iterations

    mplc = MPLCSim(conf=conf)

    lens_mask = get_lens_mask(Nx=conf['Nx'], Ny=conf['Ny'] // 2, dx=conf['dx'], dy=conf['dy'], wl=mplc.wavelength,
                              f=2 * 87e-3)
    mplc.res.masks[8][mplc.res.active_slice] = np.vstack((lens_mask, lens_mask))

    # input output modes #
    waist_in = 80e-6
    waist_out = 80e-6
    D_between_modes_in = 300e-6
    D_between_modes_out = 300e-6
    dim = 5  # to create the spot array similar to the spots Cr masks
    # modes are ordered like this: lab/mplc/mask_utils.py
    which_modes = np.array([7, 17, 9, 19,
                            32, 42, 34, 44])

    input_spots, _, _ = gen_input_spots_array(waist=waist_in, D_between_modes=D_between_modes_in, XX=mplc.XX,
                                              YY=mplc.YY, dim=dim)
    input_spots = input_spots[which_modes]
    output_spots, _, _ = gen_input_spots_array(waist=waist_out, D_between_modes=D_between_modes_out, XX=mplc.XX,
                                               YY=mplc.YY, dim=dim)
    output_spots = output_spots[which_modes]

    upper_active_slice = np.index_exp[360 + 30:360 + 180 - 15, 160:260]
    upper_active_slice = None
    upper_speckles = get_speckle_modes_conf(conf, N_modes=8, sig=0.48e-3, diffuser_pix_size=0.15e-3,
                                            active_slice=upper_active_slice, y_displace=72)
    lower_active_slice = np.index_exp[360 + 180 + 15:360 + 180 + 150, 160:260]
    lower_active_slice = None  # TODO: maybe I do want this?
    lower_speckles = get_speckle_modes_conf(conf, N_modes=8, sig=0.48e-3, diffuser_pix_size=0.15e-3,
                                            active_slice=lower_active_slice, y_displace=-72)
    if not same_diffuser:
        # lower number spots are for lower half of SLM speckles
        input_modes = np.concatenate([input_spots, lower_speckles[:4], upper_speckles[:4]])
        output_modes = np.concatenate([lower_speckles[4:], upper_speckles[4:],  output_spots])
    else:
        # the same speckles (+ correct flips) between upper and lower corresponding spots,
        # the same things happen to corresponding modes of upper and lower photons
        input_modes = np.concatenate([input_spots, lower_speckles[:4], np.fliplr(np.flipud(lower_speckles[:4]))])
        # Using input_spots twice, because the whole point is for them to be the same
        output_modes = np.concatenate([lower_speckles[4:], np.fliplr(np.flipud(lower_speckles[4:])),  input_spots])
        # TODO: check! before you uncomment (just look at the modes and make sure they are symmetric,
        #  and also run WFM once and see it works reasonably well)
        raise NotImplemented()

    # Run #
    mplc.set_modes(input_modes, output_modes)
    mplc.find_phases(fix_initial_phases=False)  # ovelaping speckles, can't use fix_initial_phases

    # save #
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    name = name or 'unitary'
    path = out_path or fr'{DEFAULT_DIR}\{timestamp}_{name}.masks'
    mplc.res.save_masks(path)

    return mplc


def create_WFM_unitary_masks(U1, U2=None, out_path=None, name=None, N_iterations=None,
                             dead_middle_zone=0, last_plane_extra_dist=8.4e-3, col_to_row=False):
    """
    Assuming U is a 5X5 unitary, and using the top and bottom third rows of spots for input, and 3rd columns for output
    if U2 is None, so U2 = conj(U)
    """
    # MPLC WFM conf #
    active_planes = np.array([True] * 11)
    N_modes = 10
    conf = copy.deepcopy(default_wfm_conf)
    conf['active_planes'] = active_planes
    conf['N_modes'] = N_modes
    if N_iterations is not None:
        conf['N_iterations'] = N_iterations

    conf['dist_after_plane'][9] = 87e-3 + last_plane_extra_dist
    print(conf)
    mplc = MPLCSim(conf=conf)

    # Transformation #
    zeros_mat = np.zeros((5, 5))
    # If U2=conj(U1) the correlations will be on the identity, since Klyshko provides the transpose,
    # and together with the conj we get U^-1 in the Klyshko return
    U2 = U2 or np.conj(U1)
    full_transformation = np.block([[U1, zeros_mat],
                                    [zeros_mat, U2]])

    # input output modes #
    waist_in = 80e-6
    waist_out = 45e-6
    D_between_modes_in = 300e-6
    D_between_modes_out = 330e-6
    dim = 5

    input_spots, x_modes_in, y_modes_in = gen_input_spots_array(waist=waist_in, D_between_modes=D_between_modes_in,
                                                                XX=mplc.XX, YY=mplc.YY, dim=dim)

    """
    modes are ordered like this when one indexed:
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
    # these are zero indexed
    which_modes_in = np.array([2, 7, 12, 17, 22,
                               27, 32, 37, 42, 47])

    which_modes_out = np.array([10, 11, 12, 13, 14,
                                35, 36, 37, 38, 39])

    if col_to_row:
        which_modes_in, which_modes_out = which_modes_out, which_modes_in
        D_between_modes_out = 310e-6

    input_modes = input_spots[which_modes_in]
    output_modes, phase_pos_x, phase_pos_y = gen_output_modes_Unitary(waist_out, D_between_modes_out, mplc.XX, mplc.YY,
                                                                      full_transformation, dim, which_modes_out,
                                                                      dead_middle_zone=dead_middle_zone)

    # run #
    mplc.set_modes(input_modes, output_modes)
    # important to fix initial phases in this spots configuration
    mplc.find_phases(fix_initial_phases=True)

    # save #
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    name = name or 'unitary'
    path = out_path or fr'{DEFAULT_DIR}\{timestamp}_{name}.masks'
    mplc.res.save_masks(path)

    return mplc


def create_WFM_QKD_masks(out_path=None, name='MUB2_QKD', N_iterations=None,
                             dead_middle_zone=0, last_plane_extra_dist=8.4e-3, col_to_row=False):
    #  Based on "All Mutually Unbiased Bases in Dimensions Two to Five" (2018)
    #  The columns in the matrix are the basis elements
    q = np.exp(2j * np.pi / 5)  # Complex fifth root of unity
    MUB = np.array([
        [1, 1, 1, 1, 1],
        [1, q, q ** 2, q ** 3, q ** 4],
        [1, q ** 2, q ** 4, q, q ** 3],
        [1, q ** 3, q, q ** 4, q ** 2],
        [1, q ** 4, q ** 3, q ** 2, q]
    ]) / np.sqrt(5)  # eq. 33

    U = MUB.conj().T  # To measure in X basis, we need to act with X^dag on the state
    return create_WFM_unitary_masks(U1=U, out_path=out_path, name=name, N_iterations=N_iterations,
                                    dead_middle_zone=dead_middle_zone, last_plane_extra_dist=last_plane_extra_dist,
                                    col_to_row=col_to_row)
