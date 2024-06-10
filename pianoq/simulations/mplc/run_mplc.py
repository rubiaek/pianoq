import numpy as np
import matplotlib.pyplot as plt
from pianoq.simulations.mplc.mplc_modes import get_speckle_modes_conf, get_spots_modes_conf
from pianoq.simulations.mplc.mplc_utils import get_lens_mask_conf
from pianoq.simulations.mplc.mplc import MPLC


def real_with_lens():
    # prop_distances_after_plane[i] is distance between planes i and i+1 (counting from 0)
    dist_after_plane = 87*np.ones(10)
    dist_after_plane[4] = 138

    # Lense in plane 9 between 7 and 11. Allow phases freedom in plane 11 since I measure intensity
    active_planes = np.array([True] * 11)
    active_planes[7:10] = False

    N_N_modes = 3
    # All in mm
    conf = {'wavelength': 810e-6,  # mm
            'dist_after_plane': dist_after_plane,  # mm
            'active_planes': active_planes,  # bool
            'N_iterations': 30,
            'Nx': 140,  # Number of grid points x-axis
            'Ny': 180,  # Number of grid points y-axis
            'dx': 12.5e-3,  # mm - SLM pixel sizes
            'dy': 12.5e-3,  # mm
            'max_k_constraint': 0.15,  # Ohad: better than 0.1 or 0.2, but not very fine-tuned
            'N_modes': N_N_modes*N_N_modes,
            'min_log_level': 2,
            'size_factor': 3,  # assumed to be odd. Have physical larger grid than the actual SLM planes
            'use_mask_offset': True,
            }

    mplc = MPLC(conf=conf)

    input_modes = get_spots_modes_conf(conf, sig=0.1, N_rows=N_N_modes, N_cols=N_N_modes, spacing=0.6)
    output_modes = get_speckle_modes_conf(conf, N_modes=len(input_modes),
                                          sig=0.25, diffuser_pix_size=0.05, active_slice=mplc.res.active_slice)
    mplc.set_modes(input_modes, output_modes)

    mplc.res.masks[8] = get_lens_mask_conf(conf, f=2*87)

    mplc.initialize_fields()
    # mplc.find_phases()
    return mplc


def short():
    # prop_distances_after_plane[i] is distance between planes i and i+1 (counting from 0)
    dist_after_plane = 87*np.ones(6)
    dist_after_plane[4] = 138

    # Lense in plane 9 between 7 and 11. Allow phases freedom in plane 11 since I measure intensity
    active_planes = np.array([True] * 7)

    N_N_modes = 2
    # All in mm
    conf = {'wavelength': 810e-6,  # mm
            'dist_after_plane': dist_after_plane,  # mm
            'active_planes': active_planes,  # bool
            'N_iterations': 15,
            'Nx': 140,  # Number of grid points x-axis
            'Ny': 180,  # Number of grid points y-axis
            'dx': 12.5e-3,  # mm - SLM pixel sizes
            'dy': 12.5e-3,  # mm
            'max_k_constraint': 0.15,  # Ohad: better than 0.1 or 0.2, but not very fine-tuned
            'N_modes': N_N_modes*N_N_modes,
            'min_log_level': 2,
            'size_factor': 3,  # assumed to be odd. Have physical larger grid than the actual SLM planes
            'use_mask_offset': True,
            }

    mplc = MPLC(conf=conf)
    input_modes = get_spots_modes_conf(conf, sig=0.1, N_rows=N_N_modes, N_cols=N_N_modes, spacing=0.6)
    output_modes = get_speckle_modes_conf(conf, N_modes=len(input_modes),
                                          sig=0.25, diffuser_pix_size=0.05, active_slice=mplc.res.active_slice)
    mplc.set_modes(input_modes, output_modes)
    mplc.initialize_fields()
    mplc.find_phases()
    return mplc


# mplc = real_with_lens()
mplc = short()
plt.show()
