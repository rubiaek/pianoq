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

ALPHA_WALKOFF_DEG = 4.0  # from https://www.spiedigitallibrary.org/conference-proceedings-of-spie/10224/102242N/Broadband-biphotons-in-the-single-spatial-mode-through-high-pump/10.1117/12.2266937.full?SSO=1
# In the section of Calculation results
DEG2RAD = (2*np.pi)/360


class BowtieImage(object):
    def __init__(self, path):
        self.path = path
        self.img = fits.open(path)


def get_w_in_pixels(V):
    def gaussian(x, A, sigma, x0, C):
        # convention from edmund optics Gaussian beam propagation eq (1):
        # https://www.edmundoptics.in/knowledge-center/application-notes/lasers/gaussian-beam-propagation/
        return A*np.exp(-(2*(x-x0)**2)/(sigma**2)) + C

    xdata = np.arange(len(V))
    popt, pcov = curve_fit(gaussian, xdata, V, bounds=(0, [5e13, 100, 100, 1e3]))
    perrs = np.sqrt(np.diag(pcov))
    w = popt[1]
    w_err = perrs[1]
    # assert w_err / w < 0.2, "Uncertainty in w larger that 20%!"
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
    z_r_eff = z_r*alpha  # ALPHA_WALKOFF_DEG*DEG2RAD
    return w0 * np.sqrt(1 + ((z-z0) / z_r_eff) ** 2)


def get_w0(x_range_m, ws):
    popt, pcov = curve_fit(rayleigh, x_range_m, ws, bounds=([3e-6, 0, 0], [10e-6, 3e-5, 10/60]))
    perrs = np.sqrt(np.diag(pcov))
    w0, z0, alpha = popt
    w0_err, z0_err, alpha_err = perrs

    return w0, w0_err, z0, z0_err, alpha, alpha_err


def w_of_z(w0, z):
    z_r = (np.pi*(w0**2)*N)/LAMBDA
    return w0 * np.sqrt(1 + (z / z_r) ** 2)


def calc_expected_2d(w0=7.6e-6, alpha=ALPHA_WALKOFF_DEG):
    # since the walkoff isn't of a delta in y, but rather of a gaussian in y, the picture in the walkoff isn't expected
    # to be exactly rayleigh, but rather with some convolution with the gaussian width in the y direction
    # Define transvers plane
    X = np.linspace(-300e-6, 300e-6, 800)  # m
    Y = np.linspace(-300e-6, 300e-6, 800)  # m
    dx = X[1] - X[0]
    dy = Y[1] - Y[0]
    XX, YY = np.meshgrid(X, Y)

    # define optical axis
    Z = np.linspace(-2.5e-3, 2.5e-3, 100)
    w = w_of_z(w0, Z)
    x = alpha*DEG2RAD*Z
    gs = []
    for i in range(len(Z)):
        # Normalization is important here, since we sum different Guassians. We normalize to total power of 1
        C = 2/(np.pi*w[i]**2)  # with w^2 because a regular gaussian shuld have sqrt(w) and this is in 2D.
        w1 = w[i]
        x0 = x[i]
        g = C*np.exp(-2*((XX-x0)**2 + YY**2)/w1**2)
        gs.append(g)

    im = sum(gs)

    fig, ax = plt.subplots()
    (left, right, bottom, top) = (X[0], X[-1], Y[0], Y[-1])
    (left, right, bottom, top) = (left*1e6, right*1e6, bottom*1e6, top*1e6)
    extent1 = (left, right, bottom, top)
    imm = ax.imshow(im, extent=extent1, aspect='auto')
    fig.colorbar(imm, ax=ax)
    ax.set_title(f'sum of Gaussians walking off, alpha={alpha} deg, w0={w0*1e6}um')
    ax.set_xlabel('x(um)')
    ax.set_ylabel('y(um)')

    fig.show()

    return im


def show_meas(path=PATH):
    img = fits.open(path)[0]
    DC = img.data.mean() + 5
    im = img.data[790:830, 805:870] - DC

    y_pixs, x_pixs = im.shape
    (left, right, bottom, top) = (0, x_pixs*PIXEL_SIZE*MAG_FACTOR, 0, y_pixs*PIXEL_SIZE*MAG_FACTOR)  #pix -> m
    (left, right, bottom, top) = (left*1e6, right*1e6, bottom*1e6, top*1e6)  # m -> um
    extent = ((left-right)/2, (right-left)/2, (bottom-top)/2, (top-bottom)/2)  # center

    fig, ax = plt.subplots()
    imm = ax.imshow(im, extent=extent, aspect='auto')
    fig.colorbar(imm, ax=ax)
    ax.set_title('measurement')
    ax.set_xlabel('x(um)')
    ax.set_ylabel('y(um)')
    fig.show()


def main():
    # TODO: Automatically find the regoin of interest using center of mass etc.
    # TODO: normalize images intensity to 1 so fitting Gaussians will be easier
    # when using this set 5e4 in fit bounds
    img = fits.open(PATH)[0]
    # im = img.data[750:860, 760:910]  # TODO: make more generic?
    # x_range_pix = np.arange(45, 101)  # TODO: make more generic?
    # when using this set 5e13 in fit bounds
    im = calc_expected_2d(w0=6e-6, alpha=1.5)[350:450, 200:600]
    x_range_pix = np.arange(1, 400)

    ws, w_errs = get_ws(im, x_range_pix)

    x_range_m = x_range_pix - x_range_pix.mean()
    x_range_m = x_range_m * PIXEL_SIZE
    x_range_m = x_range_m * MAG_FACTOR
    w0, w0_err, z0, z0_err, alpha, alpha_err = get_w0(x_range_m[21:44], ws[21:44])  # TODO: make more generic?

    plot_rayleigh(x_range_m, ws, w_errs, w0, z0, alpha)

    fig, ax = plt.subplots()
    ax.plot(x_range_m, ws, '*')
    alpha = 0.9
    ax.set_title(f'alpha = {alpha} deg')
    ys = rayleigh(x_range_m, 7.6e-6, 11.2e-6, alpha/60)
    ax.plot(x_range_m, ys)
    fig.show()
