import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

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
    BT_HEIGHT = 25  # TODO: more generic?
    BT_WIDTH = 30  # TODO: more generic?

    def __init__(self, fits_path):
        self.path = fits_path
        self.img = fits.open(fits_path)[0]
        # Binning of 4 means pixel is 4 times larger
        self.pixel_size = PIXEL_SIZE * self.img.header['XBINNING']

        self.im = self._fix_image(self.img.data)
        self.ws, self.w_errs = self.get_ws()
        self.x_for_ws = self.get_xrange()
        self.ys = self.get_y_range()
        self.dummy_x = np.linspace(self.x_for_ws[0], self.x_for_ws[-1], 200)

    def _fix_image(self, img):
        DC = np.mean([img[0, :].mean(), img[-1, :].mean(), img[:, 0].mean(), img[:, -1].mean()])
        img = img - 0.95*DC  # keep 5% of DC to avoid negative values
        im = img / img.max()

        a = gaussian_filter(im, 10)  # In case I have noisy bright pixels
        ind_row, ind_col = np.unravel_index(np.argmax(a, axis=None), a.shape)  # returns a tuple
        im = im[ind_row-self.BT_HEIGHT:ind_row+self.BT_HEIGHT, ind_col-self.BT_WIDTH:ind_col+self.BT_WIDTH]
        return im

    def show_both(self, ax):
        if ax is None:
            fig, ax = plt.subplots()
        self.plot_ws(ax)
        popt, _ = self.fit_to_rayleigh()
        self.plot_rayleigh(ax, *popt)

    def get_xrange(self):
        x_range = np.arange(len(self.ws)) * self.pixel_size * MAG_FACTOR
        x_range = x_range - x_range.mean()
        return x_range

    def get_y_range(self):
        Y = np.arange(self.im.shape[0]) * self.pixel_size * MAG_FACTOR
        Y = Y - Y.mean()
        return Y

    def get_ws(self):
        ws = []
        w_errs = []

        for col in range(self.im.shape[1]):
            V = self.im[:, col]
            popt, perrs = self.get_w_in_pixels(V)

            w = popt[1]
            w_err = perrs[1]

            if w_err / w > 0.2:
                # These really don't look like Gaussians...
                # print(f'old w: {w}, old w_err: {w_err}')
                # w, w_err = self.get_w_in_pixels2(V)
                # print(f'new w: {w}, new w_err: {w_err}')
                print(f"Uncertainty in w larger that 20%! in col: {col}")

            w_m = w * self.pixel_size * MAG_FACTOR
            w_err_m = w_err * self.pixel_size * MAG_FACTOR

            ws.append(w_m)
            w_errs.append(w_err_m)

        return np.array(ws), np.array(w_errs)

    def show_gaussian_fit(self, index):
        V = self.im[:, index]
        X = np.arange(len(V))

        popt, perrs = self.get_w_in_pixels(V)

        fig, ax = plt.subplots()
        ax.plot(X, V, '*', label='data')
        ax.plot(X, self.gaussian(X, *popt), '--', label='fit')
        ax.legend()
        fig.show()

    @staticmethod
    def gaussian(x, A, sigma, x0, C):
        # convention from edmund optics Gaussian beam propagation eq (1):
        # https://www.edmundoptics.in/knowledge-center/application-notes/lasers/gaussian-beam-propagation/
        return A*np.exp(-(2*(x-x0)**2)/(sigma**2)) + C

    @staticmethod
    def get_w_in_pixels(V):
        xdata = np.arange(len(V))
        # image assumed to be normalized, so A shouldn't be more than 1, and the offset definitely shouldn't
        popt, pcov = curve_fit(BowtieImage.gaussian, xdata, V, bounds=(0, [2*V.max(), len(V), len(V), V.max()/3]))
        perrs = np.sqrt(np.diag(pcov))
        return popt, perrs

    def plot_ws(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.errorbar(self.x_for_ws, self.ws, yerr=self.w_errs, fmt='*', label='data')
        ax.figure.show()
        ax.set_xlabel('z(m)')
        ax.set_ylabel('w(m)')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-6, -6))
        ax.legend()
        return ax

    def plot_rayleigh(self, ax=None, *popt):
        if ax is None:
            fig, ax = plt.subplots()

        w0, _, alpha = popt

        fit_y = self.rayleigh(self.dummy_x, *popt)
        ax.plot(self.dummy_x, fit_y, '--', label=fr'fit rayleigh, $ w_0 $={w0:2f}, $ \alpha $={alpha:2f}')
        ax.legend()
        ax.figure.show()

    def show_im(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        extent = (self.x_for_ws[0], self.x_for_ws[-1], self.ys[0], self.ys[-1])  # (left, right, bottom, top)

        imm = ax.imshow(self.im, extent=extent, aspect='auto')
        ax.figure.colorbar(imm, ax=ax)
        ax.set_title('measurement')
        ax.set_xlabel('x(m)')
        ax.set_ylabel('y(m)')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(-6, -6))
        ax.figure.show()

    @staticmethod
    def rayleigh(z, w0, z0, alpha):
        z_r = (np.pi*(w0**2)*N)/LAMBDA
        z_r_eff = z_r*alpha
        return w0 * np.sqrt(1 + ((z-z0) / z_r_eff) ** 2)

    def fit_to_rayleigh(self):
        popt, pcov = curve_fit(self.rayleigh, self.x_for_ws, self.ws,
                               bounds=([np.min(self.ws)/3, self.x_for_ws[0], 0],
                                       [np.min(self.ws)*3, self.x_for_ws[-1], 10/60]))

        perrs = np.sqrt(np.diag(pcov))
        return popt, perrs


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


def different_thetas():
    DIR_PATH = r'G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\Walkoff\Change theta'
    paths = glob.glob(DIR_PATH + '\\*.fit')
    fig, ax = plt.subplots()
    for path in paths:
        bi = BowtieImage(path)
        bi.show_both(ax)


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
