import cv2
import random
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

        # H and V come in. QWP nothing. HWP nothing.
        # Hence we measure H/V.
        self.meas1 = None  # qwp.angle = 0,  hwp.angle = 0;

        # + and - come in. QWP nothing. HWP turns +/- to H/V. Wallaston prism separates H/V.
        # # Hence we measure +/-.
        self.meas2 = None  # qwp.angle = 45, hwp.angle = 22.5

        # R and L come in. QWP turns R/L to +/-. HWP turns +/- to H/V. Wallaston prism separates H/V.
        # # Hence we measure R/L.
        self.meas3 = None  # qwp.angle = 0,  hwp.angle = 22.5

        self.mask_of_interest = None  # Mask where speckles get, to zero all the around

        self.dac_amplitudes = None
        self.version = 1
        self.start_first = None
        self.end_first = None
        self.dist_x = None
        self.dist_y = None

    def crop_up_down(self, remove_from_up=5, remove_from_down=5):
        self.meas1 = self.meas1[remove_from_up:-remove_from_down, :]
        self.meas2 = self.meas2[remove_from_up:-remove_from_down, :]
        self.meas3 = self.meas3[remove_from_up:-remove_from_down, :]

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

    def plot_stokes_params(self):
        S0, S1, S2, S3 = self.get_stokes(normalized=False)
        fig, axes = plt.subplots(2, 2)
        im = axes[0, 0].imshow(S0)
        axes[0, 0].set_title('S0')
        fig.colorbar(im, ax=axes[0, 0])

        im = axes[0, 1].imshow(S1)
        axes[0, 1].set_title('S1')
        fig.colorbar(im, ax=axes[0, 1])

        im = axes[1, 0].imshow(S2)
        axes[1, 0].set_title('S2')
        fig.colorbar(im, ax=axes[1, 0])

        im = axes[1, 1].imshow(S3)
        axes[1, 1].set_title('S3')
        fig.colorbar(im, ax=axes[1, 1])

        fig.show()

    def plot_poincare(self, points=1000, stds=2, point_size=2):

        S0, S1, S2, S3 = self.get_stokes(normalized=True)

        good_S0_indexes = np.where(S0 > (S0.mean() + stds*S0.std()))
        S1, S2, S3 = S1[good_S0_indexes], S2[good_S0_indexes], S3[good_S0_indexes]  # 2D to 1D
        print(f'len(S1): {len(S1)}')

        if points < len(S1):
            chosen_indexes = random.sample(range(len(S1)), k=points)
            S1, S2, S3 = S1[chosen_indexes], S2[chosen_indexes], S3[chosen_indexes]  # 1D to shorter 1D

        b = qutip.Bloch()
        b.point_size = [point_size]
        b.add_points([S1, S2, S3])
        b.show()
        plt.show(block=False)

    def get_degree_of_polarization(self, only_good_points=False, stds=2):
        S0, S1, S2, S3 = self.get_stokes(normalized=False)

        if only_good_points:
            good_S0_indexes = np.where(S0 > (S0.mean() + stds*S0.std()))
            S1, S2, S3 = S1[good_S0_indexes], S2[good_S0_indexes], S3[good_S0_indexes]  # 2D to 1D
            S0 = S0[good_S0_indexes]

        return np.sqrt(S1.sum() ** 2 + S2.sum() ** 2 + S3.sum() ** 2) / S0.sum()

    def get_stokes(self, normalized=True):
        # # don't think about noise
        # self.meas1[np.invert(self.mask_of_interest)] = 0
        # self.meas2[np.invert(self.mask_of_interest)] = 0
        # self.meas3[np.invert(self.mask_of_interest)] = 0

        # S0, S1
        part1, part2 = self._get_parts(self.meas1)

        # The physical camera does the abs(*)**2
        # Adding 0.1 so we won't divide by zero by mistake...
        S01 = part1 + part2
        S1 = part1 - part2
        if normalized:
            S1 = (part1 - part2) / (part1 + part2 + 0.1)

        # S2
        part1, part2 = self._get_parts(self.meas2)
        S02 = part1 + part2
        S2 = part1 - part2
        if normalized:
            S2 = (part1 - part2) / (part1 + part2 + 0.1)

        # S3
        part1, part2 = self._get_parts(self.meas3)
        S03 = part1 + part2
        S3 = part1 - part2
        if normalized:
            S3 = (part1 - part2) / (part1 + part2 + 0.1)

        S0 = ((S01 + S02 + S03) / 3) + 0.1

        return S0, S1, S2, S3

    def _get_parts(self, meas):
        if self.version == 1:
            return self._get_parts_v1(meas)
        elif self.version == 2:
            return self._get_parts_v2(meas)

    def _get_parts_v1(self, meas):
        cm_row, cm_col = ndimage.measurements.center_of_mass(self.mask_of_interest)
        cm_row, cm_col = int(cm_row), int(cm_col)

        start_first = np.where(self.mask_of_interest[cm_row, :cm_col])[0][0]
        start_second = np.where(self.mask_of_interest[cm_row, cm_col:])[0][0]
        start_second += cm_col
        # dist = start_second - start_first
        dist = 268  # See misc\calibrate_two_pols_spots.py

        part1 = meas[:, start_first-5:cm_col]
        part2 = meas[:, start_first-5+dist:cm_col+dist]

        return part1, part2

    def _get_parts_v2(self, meas):

        if self.dist_y > 0:
            part1 = meas[self.dist_y:, self.start_first:self.end_first]
            part2 = meas[:-self.dist_y, self.start_first + self.dist_x:self.end_first + self.dist_x]
        elif self.dist_y < 0:
            part1 = meas[:self.dist_y, self.start_first:self.end_first]
            part2 = meas[-self.dist_y:, self.start_first + self.dist_x:self.end_first + self.dist_x]
        else:
            part1 = meas[:, self.start_first:self.end_first]
            part2 = meas[:, self.start_first + self.dist_x:self.end_first + self.dist_x]

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
                     version=self.version,
                     start_first=self.start_first,
                     end_first=self.end_first,
                     dist_x=self.dist_x,
                     dist_y=self.dist_y,
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

        self.version = data.get('version', None)
        self.start_first = data.get('start_first', None)
        self.end_first = data.get('end_first', None)
        self.dist_x = data.get('dist_x', None)
        self.dist_y = data.get('dist_y', None)


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
    path = r"G:\My Drive\Projects\Quantum Piano\Results\PolarizationMeasurements\2021_08_30_14_49_48.polm"
    path = r"G:\My Drive\Projects\Quantum Piano\Results\PolarizationMeasurements\2021_08_30_14_37_21.polm"
    pom.loadfrom(path)
    pom.plot_stokes_params()
    pom.plot_poincare()
    pom.plot_polarization_speckle()
    plt.show()
