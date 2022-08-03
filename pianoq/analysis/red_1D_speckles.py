import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import signal


class Speckle1D(object):
    def __init__(self, path, x_min, x_max):
        self.f = fits.open(path)
        self.img = self._fix_image(self.f[0].data)
        self.x_min = x_min
        self.x_max = x_max

    def _fix_image(self, im):
        img = im.astype(float)
        DC = img[:, 0:10].mean()
        img = img - DC
        im = img / img.max()  # TODO: take actual max and not some random burnt pixel
        return im

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

    def plot_few_autocorrs(self, N=2):
        fig, ax = plt.subplots()
        for col in np.arange(self.x_min, self.x_max)[::N]:
            # self.plot_auto_corr(self.img[950:1050, col], ax=ax, label=f'x={col}')
            self.plot_auto_corr(self.img[:, col], ax=ax, label=f'x={col}')
        ax.legend()

    def _contrast(self, V):
        Q = (V**2).mean() / ((V.mean())**2)
        return np.sqrt(Q-1)

    def contrast_R2L(self, row, window=40, ax=None):
        X = np.arange(self.x_min, self.x_max)
        Y = []
        for x in X:
            V = self.img[row-window:row+window, x]
            Y.append(self._contrast(V))

        if not ax:
            fig, ax = plt.subplots()
        ax.plot(X, Y)
        ax.set_title(f'contrast at row {row}')
        ax.set_xlabel('columns')
        ax.set_ylabel('contrast')
        ax.figure.show()

    def contrast_U2D(self, col=1150, window=70, ax=None):
        X = np.arange(0, self.img.shape[0])
        Y = []
        for x in X:
            V = self.img[x:x+window, col]
            Y.append(self._contrast(V))

        if not ax:
            fig, ax = plt.subplots()
        ax.plot(X, Y)
        ax.set_title(f'contrast at col {col}')
        ax.set_xlabel('rows')
        ax.set_ylabel('contrast')
        ax.figure.show()

    def close(self):
        self.f.close()


def different_filters():
    paths = [
        r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\New_Setup_07-2022\2022-08-02\Nearfield\Preview_20220802_142401_0.5sec_Bin1_32.8C_gain300_yes_telescope_filter_80nm.fit",
        r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\New_Setup_07-2022\2022-08-02\Nearfield\Preview_20220802_142526_0.5sec_Bin1_32.8C_gain300_yes_telescope_filter_10nm.fit",
        r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\New_Setup_07-2022\2022-08-02\Nearfield\Preview_20220802_145013_0.5sec_Bin1_33.0C_gain300_yes_telescope_filter_3nm.fit"
    ]
    fig, ax = plt.subplots()


if __name__ == "__main__":
    path = r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\New_Setup_07-2022\Preview_20220727_115409_1sec_Bin1_32.3C_gain100_1D_diffuser.fit"
    s = Speckle1D(path, 1115, 1185)
