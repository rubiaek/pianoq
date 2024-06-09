import numpy as np
import cv2
import matplotlib.pyplot as plt
from pianoq.simulations.mplc.mplc import MPLC

N_N_modes = 3
conf = {'wavelength': 810e-6,  # mm
        'plane_spacing': 87,  # mm
        'N_planes': 8,
        'N_iterations': 30,
        'Nx': 140,  # Number of grid points x-axis
        'Ny': 180,  # Number of grid points y-axis
        'dx': 12.5e-3,  # mm - SLM pixel sizes
        'dy': 12.5e-3,  # mm
        'max_k_constraint': 0.15,
        'N_modes': N_N_modes*N_N_modes,
        'min_log_level': 2,
        'size_factor': 3,  # assumed to be odd. Have physical larger grid than the actual SLM planes
        'use_mask_offset': True,
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


def test_k_filter():
    mplc = MPLC(conf=conf)
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


test_k_filter()
# test_freespace_prop()

plt.show()
