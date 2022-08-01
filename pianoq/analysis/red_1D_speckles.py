import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


class Speckle1D(object):
    def __init__(self, path, x_min, x_max):
        self.f = fits.open(path)
        self.img = self._fix_image(self.f[0].data)
        self.x_min = x_min
        self.x_max = x_max

    def _fix_image(self, img):
        # DC = np.mean([img[0, :].mean(), img[-1, :].mean(), img[:, 0].mean(), img[:, -1].mean()])
        # img = img - 0.94*DC  # keep 5% of DC to avoid negative values
        img = img - img.min()
        im = img / img.max()
        return im

    def _autocorrelation(self, V):
        # V = self.img[:, 1150]
        Y = np.correlate(V-V.mean(), V-V.mean(), mode='full')
        Y /= Y.max()
        X = signal.correlation_lags(len(V), len(V), 'full')
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
            self.plot_auto_corr(self.img[950:1050, col], ax=ax, label=f'x={col}')
        ax.legend()

    def _contrast(self, V):
        Q = (V**2).mean() / ((V.mean())**2)
        return np.sqrt(Q-1)

    def contrast_R2L(self, row, window=40):
        X = np.arange(self.x_min, self.x_max)
        Y = []
        for x in X:
            V = self.img[row-window:row+window, x]
            Y.append(self._contrast(V))

        fig, ax = plt.subplots()
        ax.plot(X, Y)
        ax.set_title(f'contrast at row {row}')
        ax.set_xlabel('columns')
        ax.set_ylabel('contrast')
        fig.show()

    def contrast_U2D(self, col=1150, window=70):
        X = np.arange(0, self.img.shape[0])
        Y = []
        for x in X:
            V = self.img[x:x+window, col]
            Y.append(self._contrast(V))

        fig, ax = plt.subplots()
        ax.plot(X, Y)
        ax.set_title(f'contrast at col {col}')
        ax.set_xlabel('rows')
        ax.set_ylabel('contrast')
        fig.show()


    def close(self):
        self.f.close()



if __name__ == "__main__":
    path = r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\New_Setup_07-2022\Preview_20220727_115409_1sec_Bin1_32.3C_gain100_1D_diffuser.fit"
    s = Speckle1D(path, 1115, 1185)