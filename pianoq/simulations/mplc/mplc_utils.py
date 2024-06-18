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
