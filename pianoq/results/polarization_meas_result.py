import cv2
import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates.funcs as coord
from colorsys import hls_to_rgb
from scipy import ndimage
import qutip

import traceback


class PolarizationMeasResult(object):
    def __init__(self):
        self.roi = None
        self.exposure_time = None

        self.meas1 = None  # qwp.angle = 0,  hwp.angle = 0
        self.meas2 = None  # qwp.angle = 45, hwp.angle = 22.5
        self.meas3 = None  # qwp.angle = 0,  hwp.angle = 22.5

        self.mask_of_interest = None

        self.dac_amplitudes = None

    def plot_polarization_speckle(self):
        S0, S1, S2, S3 = self.get_stokes()
        r, phi, theta = coord.cartesian_to_spherical(S1, S2, S3)
        r, phi, theta = r.value, phi.value, theta.value

        # TODO: turn phi and theta to R and phi of elipse polarity and eccrenticity?
        fig, ax = plt.subplots()
        img = colorize(phi, theta)
        ax.imshow(img)
        ax.set_title('colorization of phi and theta')
        fig.show()

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(phi, aspect='auto')
        axes[0].set_title('phi angle')
        axes[1].imshow(theta, aspect='auto')
        axes[1].set_title('theta angle')
        fig.show()

    def plot_poincare(self):

        S0, S1, S2, S3 = self.get_stokes()
        S1, S2, S3 = S1 / S0, S2 / S0, S3 / S0
        cl = 4
        S1, S2, S3 = np.clip(S1, -cl, cl), np.clip(S2, -cl, cl), np.clip(S3, -cl, cl)

        N = 35
        S1 = cv2.resize(S1, (N, N), interpolation=cv2.INTER_AREA)
        S2 = cv2.resize(S2, (N, N), interpolation=cv2.INTER_AREA)
        S3 = cv2.resize(S3, (N, N), interpolation=cv2.INTER_AREA)
        # area = np.index_exp[120:130, 100:110]
        # S1, S2, S3 = S1[area], S2[area], S3[area]

        S1, S2, S3 = S1.flatten(), S2.flatten(), S3.flatten()

        b = qutip.Bloch()
        b.add_points([S1, S2, S3])
        b.show()
        plt.show(block=False)

    def get_stokes(self):
        # don't think about noise
        self.meas1[np.invert(self.mask_of_interest)] = 0
        self.meas2[np.invert(self.mask_of_interest)] = 0
        self.meas3[np.invert(self.mask_of_interest)] = 0

        # S0, S1
        part1, part2 = self._get_parts(self.meas1)

        # The physical camera does the abs(*)**2
        # Adding 0.1 so we won't divide by zero...
        S0 = part1 + part2 + 0.1
        S1 = part1 - part2

        # S2
        part1, part2 = self._get_parts(self.meas2)
        S2 = part1 - part2

        # S3
        part1, part2 = self._get_parts(self.meas3)
        S3 = part1 - part2

        return S0, S1, S2, S3

    def _get_parts(self, meas):
        cm_row, cm_col = ndimage.measurements.center_of_mass(self.mask_of_interest)
        cm_row, cm_col = int(cm_row), int(cm_col)

        start_first = np.where(self.mask_of_interest[cm_row, :cm_col])[0][0]
        start_second = np.where(self.mask_of_interest[cm_row, cm_col:])[0][0]
        start_second += cm_col
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
                     dac_amplitudes=self.dac_amplitudes,
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
        self.dac_amplitudes = data.get('dac_amplitudes', None)


def colorize(r, arg):
    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    return c


if __name__ == "__main__":
    pom = PolarizationMeasResult()
    path = r"C:\temp\2021_08_30_10_50_47.polm"
    path = r"D:\Google Drive\Projects\Quantum Piano\Results\PolarizationMeasurements\2021_08_30_14_37_21.polm"
    path = r"D:\Google Drive\Projects\Quantum Piano\Results\PolarizationMeasurements\2021_08_30_14_49_48.polm"
    pom.loadfrom(path)
    pom.plot_poincare()
    pom.plot_polarization_speckle()
    plt.show()
