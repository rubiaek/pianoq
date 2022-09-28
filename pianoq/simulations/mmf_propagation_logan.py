import re
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import glob
N_POINTS_MODES = 2**8  # resolution of the window
FOLDER = r'G:\My Drive\Lab Wiki\Optical fibers\Nonlinear-Logan simulation\Logan Simulation\GMMNLSE-Solver-FINAL-master\Fibers\GRIN_ronen2'


class MMFiber(object):
    def __init__(self, folder, wavelength):
        files = glob.glob(folder + f'\\*wavelength{wavelength}.mat')

        d = {}
        for file in files:
            mode_num = re.findall('fieldscalarmode(\d+)', file)[0]
            Q = loadmat(file)
            d[int(mode_num) - 1] = {'profile': Q['phi'],
                           'neff': Q['neff'][0][0]}

        self.wavelength = wavelength
        self.V = np.ones(len(d)).astype('complex128')
        self.d = d

    @property
    def k(self):
        return 2*np.pi / self.wavelength

    def propagate_length(self, length):
        for i in range(len(self.V)):
            self.V[i] *= np.exp(1j * self.k * self.d[i]['neff'] * length)

    def show_field(self, V=None):
        V = V or self.V
        field = np.zeros(self.d[0]['profile'].shape).astype('complex128')
        for i in range(len(V)):
            f_temp = V[i] * self.d[i]['profile']
            field += f_temp

        fig, ax = plt.subplots()
        ax.imshow(np.abs(field))
        fig.show()
        return field


if __name__ == "__main__":
    pass
