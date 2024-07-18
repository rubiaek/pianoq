from pianoq.simulations.mplc_sim.mplc_utils import get_lens_mask
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pianoq.simulations.mplc_sim.mplc_sim import MPLCSim


dist_after_plane = 87 * np.ones(6)
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
        'N_modes': N_N_modes * N_N_modes,
        'min_log_level': 2,
        'size_factor': 3,  # assumed to be odd. Have physical larger grid than the actual SLM planes
        'use_mask_offset': True,
        }


def test_freespace_prop():
    mplc = MPLCSim(conf=conf)
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


def test_k_filter():
    mplc = MPLCSim(conf=conf)
    # create some random phase mask
    diffuser_pix_size = 0.05
    N_pixs_x = int(mplc.Nx * mplc.dx / diffuser_pix_size)
    N_pixs_y = int(mplc.Ny * mplc.dy / diffuser_pix_size)
    A = 2 * np.pi * np.random.rand(N_pixs_y, N_pixs_x)
    # .shape conventions on numpy and cv2 is the opposite
    A2 = cv2.resize(A, mplc.XX.shape[::-1], interpolation=cv2.INTER_NEAREST)

    sig = 0.6
    E_gaus = np.exp(-(mplc.XX**2 + mplc.YY**2) / (sig ** 2)).astype(complex)
    E_gaus *= np.exp(1j * A2)
    speckles = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E_gaus)))

    mask = np.angle(speckles)

    # show it in regular space and in k-space
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    imm = axes[0].imshow(mask.real)
    fig.colorbar(imm, ax=axes[0])
    axes[0].set_title('Real space')

    mask_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mask)))
    imm = axes[1].imshow(mask_kspace.real)
    fig.colorbar(imm, ax=axes[1])
    axes[1].set_title('k-space')

    # show the filter
    fig, ax = plt.subplots()
    imm = ax.imshow(mplc.k_constraint)
    fig.colorbar(imm, ax=ax)
    ax.set_title('k-space filter')

    # fix mask
    mask_kspace_fixed = mask_kspace * mplc.k_constraint
    new_mask = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mask_kspace_fixed)))

    # show the mask after fixing in k_space
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    imm = axes[0].imshow(new_mask.real)
    fig.colorbar(imm, ax=axes[0])
    axes[0].set_title('Real space fixed')

    imm = axes[1].imshow(mask_kspace_fixed.real)
    fig.colorbar(imm, ax=axes[1])
    axes[1].set_title('k-space fixed')


def test_lens(lens=True):
    mplc = MPLCSim(conf=conf)
    sig = 0.5
    E_gaus = np.exp(-(mplc.XX ** 2 + mplc.YY ** 2) / (2 * sig ** 2)).astype(complex)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].set_title(f'z=0')
    axes[0].imshow(np.abs(E_gaus)**2)

    E_gaus = mplc.prop(E_gaus, 100)
    axes[1].set_title(f'z=100')
    axes[1].imshow(np.abs(E_gaus)**2)

    if lens:
        mask = get_lens_mask(mplc.Nx, mplc.Ny, mplc.dx, mplc.dy, mplc.wl, f=100)
        E_gaus = E_gaus * mask
    E_gaus = mplc.prop(E_gaus, 100)
    axes[2].set_title(f'z=200')
    axes[2].imshow(np.abs(E_gaus)**2)

    fig.suptitle(f'{lens=}')
    fig.show()


# test_k_filter()
# test_freespace_prop()
test_lens(True)
test_lens(False)

# plt.show()
