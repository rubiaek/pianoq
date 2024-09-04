import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def get_lens_mask_conf(conf, f):
    return get_lens_mask(Nx=conf['Nx'] * conf['size_factor'], Ny=conf['Ny'] * conf['size_factor'],
                         dx=conf['dx'], dy=conf['dy'], wl=conf['wavelength'], f=f)


def get_lens_mask(Nx, Ny, dx, dy, wl, f):
    X = np.arange(-Nx / 2, Nx / 2) * dx
    Y = np.arange(-Ny / 2, Ny / 2) * dy
    XX, YY = np.meshgrid(X, Y)
    k = 2 * np.pi / wl
    # important -i, assuming freespace is with +i
    mask = np.exp(-1j * (XX**2 + YY**2) * k / (2 * f))

    return mask


def show_field(E, figshow=True, active_slice=None, title=''):
    fig, ax = plt.subplots()
    imm = ax.imshow(np.abs(E)**2)
    ax.set_title(title)

    if active_slice:
        rows = active_slice[0]
        cols = active_slice[1]
        ax.add_patch(Rectangle((cols.start, rows.start), cols.stop - cols.start, rows.stop-rows.start,
                               facecolor="none", ec='k', lw=2))

    fig.colorbar(imm, ax=ax)
    if figshow:
        fig.show()


def show_mask(mask, figshow=True, active_slice=None, title=''):
    fig, ax = plt.subplots()
    imm = ax.imshow(mask, cmap='gray')
    ax.set_title(title)

    if active_slice:
        rows = active_slice[0]
        cols = active_slice[1]
        ax.add_patch(Rectangle((cols.start, rows.start), cols.stop - cols.start, rows.stop-rows.start,
                               facecolor="none", ec='k', lw=2))

    fig.colorbar(imm, ax=ax)
    if figshow:
        fig.show()


def downsample_phase(A, L, weighted=True):
    """ A is a complex field.
        Returns the down-sampled phase of A (in an exponent)"""
    N, M = A.shape
    phase = np.angle(A)
    amplitude = np.abs(A)

    # Pad arrays if necessary
    pad_n, pad_m = (L - N % L) % L, (L - M % L) % L
    phase_padded = np.pad(phase, ((0, pad_n), (0, pad_m)), mode='constant')  # default constant_values is 0
    amplitude_padded = np.pad(amplitude, ((0, pad_n), (0, pad_m)), mode='constant')  # the zero amplitude makes sure it won't affect the weighted mean

    # Reshape and compute weighted mean
    phase_reshaped = phase_padded.reshape(N // L, L, M // L, L)
    amplitude_reshaped = amplitude_padded.reshape(N // L, L, M // L, L)

    weights = amplitude_reshaped if weighted else None
    weighted_phase = np.average(phase_reshaped, weights=weights, axis=(1, 3))

    # Upsample the result back to NxM
    result = np.repeat(np.repeat(weighted_phase, L, axis=0), L, axis=1)

    # Trim to original size
    return np.exp(1j*result[:N, :M])

def corr(A, B):
    return np.abs((A.conj() * B).sum())
