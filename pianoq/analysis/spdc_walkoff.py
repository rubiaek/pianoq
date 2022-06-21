import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

PATH = r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\Walkoff\Preview_20220621_110519_0.5sec_Bin4_34" \
       r".8C_gain100.fit "

PIXEL_SIZE = 3.76e-6  # m
MAG_FACTOR = 100/150  # # from crystal to cam there is a 4f with 100mm than 150mm
N = 1.55
LAMBDA = 404e-9
ALPHA_WALKOFF_RAD = 3 * (2*np.pi)/360  # from https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10224/102242N/Broadband-biphotons-in-the-single-spatial-mode-through-high-pump/10.1117/12.2266937.full?SSO=1
                                       # In the section of Calculation results


def get_w_in_pixels(V):
    def gaussian(x, A, sigma, x0, C):
        # convention from edmund optics Gaussian beam propagation eq (1):
        # https://www.edmundoptics.in/knowledge-center/application-notes/lasers/gaussian-beam-propagation/
        return A*np.exp(-(2*(x-x0)**2)/(sigma**2)) + C

    xdata = np.arange(len(V))
    popt, pcov = curve_fit(gaussian, xdata, V, bounds=(0, [5e4, 100, 100, 1e3]))
    perrs = np.sqrt(np.diag(pcov))
    w = popt[1]
    w_err = perrs[1]
    assert w_err / w < 0.2, "Uncertainty in w larger that 20%!"
    return w, w_err


def plot_rayleigh(x_range, ws, w_errs, w0, z0, alpha):
    fig, ax = plt.subplots()
    x_range = x_range*1e3
    ws = ws*1e3
    w_errs = w_errs*1e3
    ax.errorbar(x_range, ws, yerr=w_errs, fmt='*', label='data')
    dummy_z = np.linspace(x_range[0], x_range[-1])
    fit_y = rayleigh(dummy_z, w0, z0, alpha)
    ax.plot(dummy_z, fit_y, '--', label='fit to rayleigh')
    ax.set_xlabel('z(mm)')
    ax.set_ylabel('w(mm)')
    ax.legend()
    fig.show()


def get_ws(im, x_range):
    ws = []
    w_errs = []

    for col in x_range:
        V = im[:, col]
        w, w_err = get_w_in_pixels(V)
        w_mm = w*PIXEL_SIZE
        w_err_mm = w_err*PIXEL_SIZE
        w_real = w_mm * MAG_FACTOR
        w_err_real = w_err_mm * MAG_FACTOR

        ws.append(w_real)
        w_errs.append(w_err_real)

    return np.array(ws), np.array(w_errs)


def rayleigh(z, w0, z0, alpha):
    z_r = (np.pi*(w0**2)*N)/LAMBDA
    z_r_eff = z_r*alpha#ALPHA_WALKOFF_RAD
    return w0 * np.sqrt(1 + ((z-z0) / z_r_eff) ** 2)


def get_w0(x_range_m, ws):
    popt, pcov = curve_fit(rayleigh, x_range_m, ws, bounds=([3e-6, 0, 0], [10e-6, 3e-5, 10/60]))
    perrs = np.sqrt(np.diag(pcov))
    w0, z0, alpha = popt
    w0_err, z0_err, alpha_err = perrs

    return w0, w0_err, z0, z0_err, alpha, alpha_err


if __name__ == "__main__":
    img = fits.open(PATH)[0]
    im = img.data[750:860, 760:910]  # TODO: make more generic
    x_range_pix = np.arange(45, 101)  # TODO: make more generic

    ws, w_errs = get_ws(im, x_range_pix)

    x_range_m = x_range_pix - x_range_pix.mean()
    x_range_m = x_range_m * PIXEL_SIZE
    x_range_m = x_range_m * MAG_FACTOR
    w0, w0_err, z0, z0_err, alpha, alpha_err = get_w0(x_range_m[21:44], ws[21:44])  # TODO: make more generic

    plot_rayleigh(x_range_m, ws, w_errs, w0, z0, alpha)
