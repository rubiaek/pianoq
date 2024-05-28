import numpy as np
import matplotlib.pyplot as plt
from pianoq.simulations.mplc.mplc import MPLC

conf = {'wavelength': 810e-6,  # mm
        'plane_spacing': 87,  # mm
        'N_planes': 9,
        'N_iterations': 30,
        'Nx': 2 * 140,  # Number of grid points x-axis
        'Ny': 2 * 180,  # Number of grid points y-axis
        'dx': 12.5e-3,  # mm - SLM pixel sizes
        'dy': 12.5e-3,  # mm
        'k_space_filter': 0.15
        }


def test_freespace_prop():
    mplc = MPLC(conf=conf)
    sig = 0.1
    sig2 = 0.2
    E_gaus = np.exp(-(mplc.XX ** 2 + mplc.YY ** 2) / (2 * sig ** 2)).astype(complex)
    E_gaus2 = np.exp(-((mplc.XX - 4 * sig) ** 2 + mplc.YY ** 2) / (2 * sig ** 2))
    E_2g = E_gaus + E_gaus2
    E_sqr = (np.abs(mplc.XX) < sig2).astype(float) * (np.abs(mplc.YY) < sig2).astype(complex)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5), constrained_layout=True)
    fig.suptitle('Ronen code')
    zs = (0, 30, 100, 150, 350)
    for ax_no, z in enumerate(zs):
        axes[0, ax_no].set_title(f'z={z}')
        mplc.show(mplc.prop(E_gaus, z), ax=axes[0, ax_no])
        axes[1, ax_no].set_title(f'z={z}')
        mplc.show(mplc.prop(E_sqr, z), ax=axes[1, ax_no])
    fig.show()


test_freespace_prop()
plt.show()
