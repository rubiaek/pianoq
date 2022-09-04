import re
import glob
import numpy as np
from astropy.io import fits
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


PATH = r'G:\My Drive\Projects\Quantum Piano\Results\Calibrations\Red Speckles After Fiber'
PATH_NO_TELESCOPE = r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\Red Speckles After Fiber\No telescope"


class RedMMFSpeckles(object):
    def __init__(self, path, roi=None, dark_columns=10, gaussian_filter_N=5):
        self.path = path
        self.roi = roi
        self.dark_columns = dark_columns
        self.gaussian_filter_N = gaussian_filter_N
        self.f = fits.open(path)
        self.img = self.f[0].data
        self.im = self.fix_image(self.img)
        self.f.close()
        self.filter_title = re.findall('.*filter_(.*).*.fit', path)[0]

    @property
    def contrast(self):
        im = self.im[self.roi]
        Q = (im ** 2).mean() / ((im.mean()) ** 2)
        return np.sqrt(Q - 1)

    @property
    def contrast2(self):
        # This is completely equivalent
        im = self.im[self.roi]
        return im.std() / im.mean()

    def fix_image(self, img):
        img = img.astype(float)
        DC = img[:, 0:self.dark_columns].mean()
        img = img - DC

        a = gaussian_filter(img, self.gaussian_filter_N)  # In case I have noisy bright pixels
        ind_row, ind_col = np.unravel_index(np.argmax(a, axis=None), a.shape)  # returns a tuple
        brightest = img[ind_row, ind_col]

        img = img / brightest

        return img

    def print_contrast(self):
        print(f'{self.filter_title} \t\t::\t {self.contrast:.3f}')

    def print_contrast2(self):
        print(f'{self.filter_title.ljust(12)} \t::\t {self.contrast2:.3f}')

    def show_roi(self, N_show_around=200):
        fig, ax = plt.subplots()
        ax.imshow(self.im, vmax=1.1)
        rows = self.roi[0]
        cols = self.roi[1]
        ax.add_patch(Rectangle((cols.start, rows.start), cols.stop - cols.start, rows.stop-rows.start, facecolor="grey", ec='k', lw=2))
        ax.set_xlim(cols.start - N_show_around, cols.stop + N_show_around)
        ax.set_ylim(rows.start - N_show_around, rows.stop + N_show_around)
        ax.set_title(f'{self.filter_title}')
        fig.show()


if __name__ == "__main__":
    yes_telescope = False
    if yes_telescope:
        paths = glob.glob(PATH + r'\*.fit')
        rs = [RedMMFSpeckles(path) for path in paths]
        rs[0].roi = np.index_exp[2120: 2220, 3655: 3760]
        rs[1].roi = np.index_exp[2120: 2220, 3655: 3760]
        rs[2].roi = np.index_exp[1910: 2010, 3680: 3780]
        rs[3].roi = np.index_exp[2130: 2210, 3670: 3750]
        rs[4].roi = np.index_exp[2040: 2120, 3680: 3775]
        rs[5].roi = np.index_exp[2120: 2220, 3680: 3780]
        rs[6].roi = np.index_exp[2080: 2160, 3710: 3800]
        rs[7].roi = np.index_exp[2120: 2220, 3665: 3760]
        rs[8].roi = np.index_exp[2100: 2200, 3700: 3785]
        rs[9].roi = np.index_exp[1780: 1890, 2760: 2860]
        rs[10].roi = np.index_exp[1770: 1880, 2780: 2880]
        rs[11].roi = np.index_exp[2510: 2620, 3320: 3430]
        rs[12].roi = np.index_exp[2380: 2480, 3350: 3440]

    else:
        paths = glob.glob(PATH_NO_TELESCOPE + r'\*.fit')
        rs = [RedMMFSpeckles(path, np.index_exp[2360: 2460, 3345: 3460]) for path in paths]

    for r in rs:
        r.print_contrast2()
        r.show_roi()
