import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import signal
from scipy.ndimage import gaussian_filter


class Speckle1D(object):
    def __init__(self, path, x_min, x_max, should_remove_freqs=False):
        self.f = fits.open(path)
        self.brightest = None
        self.should_remove_freqs = should_remove_freqs
        self.img = self._fix_image(self.f[0].data)
        self.x_min = x_min
        self.x_max = x_max

    def show(self, title=None, **args):
        fig, ax = plt.subplots()
        imm = ax.imshow(self.img, vmax=1.5, aspect='auto', **args)
        fig.colorbar(imm, ax=ax)
        if title:
            ax.set_title(title)
        fig.show()
        return fig, ax

    def _fix_image(self, img):
        img = img.astype(float)
        DC = img[:, 0:10].mean()
        img = img - DC

        a = gaussian_filter(img, 5)  # In case I have noisy bright pixels
        ind_row, ind_col = np.unravel_index(np.argmax(a, axis=None), a.shape)  # returns a tuple
        self.brightest = img[ind_row, ind_col]

        img = img / self.brightest

        if self.should_remove_freqs:
            img = self.remove_freqs(img)
        return img

    def remove_freqs(self, img, percent=0.5):
        img_k = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img)))
        # percent = self.should_remove_freqs if type(self.should_remove_freqs) == float else 0.5
        new_img_k = np.zeros_like(img)

        y, x = img.shape
        cropx = int(x * percent)
        cropy = int(y * percent)
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        new_img_k[starty:starty + cropy, startx:startx + cropx] = img_k[starty:starty + cropy, startx:startx + cropx]

        new_img = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(new_img_k))).real
        return new_img

    def _autocorrelation(self, V):
        # V = self.img[:, 1150]
        Y = np.correlate(V-V.mean(), V-V.mean(), mode='full')
        Y /= Y.max()
        X = signal.correlation_lags(len(V), len(V), 'full')
        # https://dsp.stackexchange.com/questions/78959/how-to-interpret-values-of-the-autocorrelation-sequence
        # https://www.dsprelated.com/freebooks/sasp/Biased_Sample_Autocorrelation.html
        Y = np.divide(Y, np.bartlett(len(Y)))

        return X, Y

    def plot_auto_corr(self, V, ax=None, label=None):
        if ax == None:
            _, ax = plt.subplots()
        lags, corr = self._autocorrelation(V)
        ax.plot(lags, corr, label=label)
        ax.figure.show()
        return ax

    def plot_few_autocorrs(self, N=2, title=None):
        fig, ax = plt.subplots()
        for col in np.arange(self.x_min, self.x_max)[::N]:
            self.plot_auto_corr(self.img[2500:3500, col], ax=ax, label=f'x={col}')
            # self.plot_auto_corr(self.img[:, col], ax=ax, label=f'x={col}')
        ax.legend()
        ax.set_title(title)
        ax.figure.show()

    def _contrast(self, V):
        Q = (V**2).mean() / ((V.mean())**2)
        return np.sqrt(Q-1)

    def contrast_R2L(self, row, window=100, ax=None, label=None):
        X = np.arange(self.x_min, self.x_max)
        Y = []
        for x in X:
            V = self.img[row:row+window, x]
            Y.append(self._contrast(V))

        if not ax:
            fig, ax = plt.subplots()
        ax.plot(X, Y, label=label)
        ax.set_title(f'contrast at row {row}')
        ax.set_xlabel('columns')
        ax.set_ylabel('contrast')
        ax.figure.show()

    def contrast_U2D(self, col=1150, window=70, ax=None, label=None):
        X = np.arange(0, self.img.shape[0])
        Y = []
        for x in X:
            V = self.img[x:x+window, col]
            Y.append(self._contrast(V))

        if not ax:
            fig, ax = plt.subplots()
        ax.plot(X, Y, label=label)
        ax.set_title(f'contrast at col {col} with window {window}')
        ax.set_xlabel('rows')
        ax.set_ylabel('contrast')
        ax.figure.show()

    def close(self):
        self.f.close()


def different_filters(col=3105, row=3200, window=100, yes_telescope=True):
    yes_telescope_paths = [
        r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\New_Setup_07-2022\2022-08-02\Nearfield\Preview_20220802_145013_0.5sec_Bin1_33.0C_gain300_yes_telescope_filter_3nm.fit",
        r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\New_Setup_07-2022\2022-08-02\Nearfield\Preview_20220802_142526_0.5sec_Bin1_32.8C_gain300_yes_telescope_filter_10nm.fit",
        r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\New_Setup_07-2022\2022-08-02\Nearfield\Preview_20220802_142401_0.5sec_Bin1_32.8C_gain300_yes_telescope_filter_80nm.fit",
    ]

    no_telescope_paths = [
        r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\New_Setup_07-2022\2022-08-02\Nearfield\Preview_20220802_155013_0.5sec_Bin1_33.5C_gain300_no_telescope_filter_3nm.fit",
        r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\New_Setup_07-2022\2022-08-02\Nearfield\Preview_20220802_154746_0.5sec_Bin1_33.5C_gain300_no_telescope_filter_10nm.fit",
        r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\New_Setup_07-2022\2022-08-02\Nearfield\Preview_20220802_155421_0.5sec_Bin1_31.8C_gain300_no_telescope_filter_80nm.fit",
    ]

    paths = yes_telescope_paths if yes_telescope else no_telescope_paths

    filters = [3, 10, 80]

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    for i in range(3):
        if yes_telescope:
            s = Speckle1D(paths[i], 3040, 3180)
        else:
            s = Speckle1D(paths[i], 3110, 3240)
        s.contrast_U2D(col=col, window=window, ax=ax, label=f'filter {filters[i]}nm')
        s.contrast_R2L(row=row, window=window, ax=ax2, label=f'filter {filters[i]}nm')
        s.close()

    ax.legend()
    ax.set_ylim((-0.25, 1.5))
    ax.set_xlim(left=1200)
    ax.figure.show()

    ax2.legend()
    ax2.figure.show()


if __name__ == "__main__":
    path = r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\New_Setup_07-2022\Preview_20220727_115409_1sec_Bin1_32.3C_gain100_1D_diffuser.fit"
    s = Speckle1D(path, 1115, 1185)
