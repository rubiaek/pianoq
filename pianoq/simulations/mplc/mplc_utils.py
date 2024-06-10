import numpy as np


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
