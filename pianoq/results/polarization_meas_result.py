import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates.funcs as coord
from colorsys import hls_to_rgb

import traceback


class PolarizationMeasResult(object):
    def __init__(self):
        self.roi = None
        self.exposure_time = None

        self.meas1 = None  # qwp.angle = 0,  hwp.angle = 0
        self.meas2 = None  # qwp.angle = 45, hwp.angle = 22.5
        self.meas3 = None  # qwp.angle = 0,  hwp.angle = 22.5

        self.mask_of_interest = None


    def plot_polarization_speckle(self):
        S0, S1, S2, S3 = self.get_stokes()
        r, phi, theta = coord.cartesian_to_spherical(S1, S2, S3)
        r, phi, theta = r.value, phi.value, theta.value

        # TODO: turn phi and theta to R and phi of elipse polarity and eccrenticity?
        img = colorize(phi, theta)

        fig, ax = plt.subplots()
        ax.imshow(img)
        fig.show()


    def plot_poincare(self):

        S0, S1, S2, S3 = self.get_stokes()
        S1, S2, S3 = S1 / S0, S2 / S0, S3 / S0

        S1, S2, S3 = S1.flatten(), S2.flatten(), S3.flatten()

        b = qutip.Bloch()
        b.add_points([S1, S2, S3])  # TODO: see if this works with "points" convention...
        b.show()
        plt.show(block=False)

    def get_stokes(self):
        # don't think about noise
        self.meas1[np.invert(self.mask_of_interest)] = 0
        self.meas2[np.invert(self.mask_of_interest)] = 0
        self.meas3[np.invert(self.mask_of_interest)] = 0

        # S0, S1
        part1, part2 = self._get_parts(self.meas1)

        S0 = (np.abs(part1) ** 2) + (np.abs(part2) ** 2)
        S1 = (np.abs(part1) ** 2) - (np.abs(part2) ** 2)

        # S2
        part1, part2 = self._get_parts(self.meas2)
        S2 = (np.abs(part1) ** 2) - (np.abs(part2) ** 2)

        # S3
        part1, part2 = self._get_parts(self.meas2)
        S3 = (np.abs(part1) ** 2) - (np.abs(part2) ** 2)

        return S0, S1, S2, S3


    def _get_parts(self, meas):
        cm_row, cm_col = ndimage.measurements.center_of_mass(im)
        cm_row, cm_col = int(cm_row), int(cm_col)

        start_first = np.where(self.mask_of_interest[cm_row, :cm_col])[0][0]
        start_second = np.where(self.mask_of_interest[cm_row, cm_col:])[0][0]
        dist = start_second - start_first

        part1 = meas[:, start_first-5:cm_col]
        part2 = meas[:, start_first-5+dist:cm_col+dist]

        return part1, part2


    def saveto(self, path):
        try:
            f = open(path, 'wb')
            np.savez(f,
                     roi=self.roi,
                     exposure_time=self.exposure_time,
                     meas1=self.meas1,
                     meas2=self.meas2,
                     meas3=self.meas3,
                     mask_of_interest=self.mask_of_interest,
                     )
            f.close()
        except Exception as e:
            print("ERROR!!")
            print(e)
            traceback.print_exc()

    def loadfrom(self, path):
        # path = path or self.DEFAULT_PATH
        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        self.roi = data.get('roi', None)
        self.exposure_time = data.get('exposure_time', None)

        self.meas1 = data.get('meas1', None)
        self.meas2 = data.get('meas2', None)
        self.meas3 = data.get('meas3', None)

        self.mask_of_interest = data.get('mask_of_interest', None)


def colorize(r, arg):
    h = (arg + pi)  / (2 * pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    return c