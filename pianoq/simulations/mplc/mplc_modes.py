import numpy as np
import cv2


def get_spots_modes_conf(conf, N_rows=3, N_cols=3, sig=0.1, spacing=0.6):
    get_spots_modes(Nx=conf['Nx'] * conf['size_factor'], Ny=conf['Ny'] * conf['size_factor'],
                    dx=conf['dx'], dy=conf['dy'],
                    sig=sig, N_rows=N_rows, N_cols=N_cols, spacing=spacing)


def get_spots_modes(Nx, Ny, dx, dy, sig=0.1, N_rows=4, N_cols=4, spacing=0.6):
    # Ohad:
    # waist_in = 80e-6  # (play between 50 to 120)
    # waist_out = 65e-6  #  (play between 40 to 90)
    X = np.arange(-Nx / 2, Nx / 2) * dx
    Y = np.arange(-Ny / 2, Ny / 2) * dy
    XX, YY = np.meshgrid(X, Y)

    modes = []
    for i in range(N_rows):
        for j in range(N_cols):
            X0 = spacing * (j - (N_cols / 2.0) + 0.5)
            Y0 = spacing * (i - (N_rows / 2.0) + 0.5)
            E_gaus = np.exp(-((XX - X0) ** 2 + (YY - Y0) ** 2) / (sig ** 2)).astype(complex)
            E_gaus /= np.sqrt(((np.abs(E_gaus)) ** 2).sum())
            modes.append(E_gaus)

    return np.array(modes)


def get_speckle_modes_conf(conf, N_modes, sig=0.05, diffuser_pix_size=0.025, active_slice=None):
    return get_speckle_modes(Nx=conf['Nx'] * conf['size_factor'], Ny=conf['Ny'] * conf['size_factor'],
                             dx=conf['dx'], dy=conf['dy'],
                             N_modes=N_modes, sig=sig, diffuser_pix_size=diffuser_pix_size, active_slice=active_slice)


def get_speckle_modes(Nx, Ny, dx, dy, N_modes, sig=0.05, diffuser_pix_size=0.025, active_slice=None):
    modes = []
    X = np.arange(-Nx / 2, Nx / 2) * dx
    Y = np.arange(-Ny / 2, Ny / 2) * dy
    XX, YY = np.meshgrid(X, Y)

    for mode_no in range(N_modes):
        E_gaus = np.exp(-(XX**2 + YY**2) / (sig**2)).astype(complex)

        N_pixs_x = int(Nx*dx / diffuser_pix_size)
        N_pixs_y = int(Ny*dy / diffuser_pix_size)
        A = 2 * np.pi * np.random.rand(N_pixs_y, N_pixs_x)
        # .shape conventions on numpy and cv2 is the opposite
        A2 = cv2.resize(A, XX.shape[::-1], interpolation=cv2.INTER_NEAREST)
        E_gaus *= np.exp(1j*A2)

        speckles = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E_gaus)))

        if active_slice:
            # cut around the middle "real" mask
            filtered_speckles = np.zeros_like(speckles)
            filtered_speckles[active_slice] = speckles[active_slice]

            norm_factor = ((np.abs(filtered_speckles))**2).sum()
            filtered_speckles /= np.sqrt(norm_factor)
            modes.append(filtered_speckles)
        else:
            modes.append(speckles)

    return np.array(modes)
