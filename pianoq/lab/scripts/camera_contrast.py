import cv2
import matplotlib.pyplot as plt
import numpy as np
import datetime

from pianoq.lab.time_tagger import QPTimeTagger
from pianoq.misc.borders import Borders
from pianoq.lab import Edac40, ASICam
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq_results.QPPickleResult import QPPickleResult
from pianoq.analysis.contrast_lib import calc_contrast, contrast_to_N_modes
from matplotlib.patches import Rectangle
from uncertainties import ufloat

PATH = r'G:\My Drive\Projects\Dispersion Cancelation\Results\Classical'


class ImageList(QPPickleResult):
    def __init__(self, path=None, exposure_time=None, timestamp=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.exposure_time = exposure_time
        self.timestamp = timestamp
        self.images = []

    def contrast_per_pixel(self, vmin=0, vmax=1.5, binning=1, remove_background=True, N=None,
                           show_N_modes=False, N_min=0.5, N_max=20):
        images = self.images

        if N is not None:
            images = images[:N]

        if binning != 1:
            rows, cols = images[0].shape
            images = np.array([cv2.resize(im, (cols//binning, rows//binning), interpolation=cv2.INTER_AREA) for im in images])

        if remove_background:
            V = il.images[:, 0, 0]
            bg = V.mean()
            images = images.astype(float) - bg

        fig, ax = plt.subplots()
        mean = images.mean(axis=0)
        std = images.std(axis=0)
        contrast = std/mean
        imm = ax.imshow(contrast, vmin=vmin, vmax=vmax)

        X = 50 // binning
        L = 100 // binning
        ax.add_patch(Rectangle((X, X), L, L, facecolor="none", ec='k', lw=0.5))
        mean_contrast = contrast[X:X+L, X:X+L].mean()
        std_contrast = contrast[X:X+L, X:X+L].std()
        c = ufloat(mean_contrast, std_contrast)

        ax.set_title(f'mean contrast: {c:.3f}. Mode num: {1/c**2:.2f}')
        fig.colorbar(imm, ax=ax)
        fig.show()

        if show_N_modes:
            fig, ax = plt.subplots()
            imm = ax.imshow(1/contrast**2, vmin=N_min, vmax=N_max)
            ax.set_title(f'mean contrast: {c:.3f}. Mode num: {1/c**2:.2f}')
            fig.colorbar(imm, ax=ax)
            fig.show()

        return contrast

    def loadfrom(self, path):
        super().loadfrom(path)
        self.images = np.array(self.images)


class SingleCountsList(QPPickleResult):
    def __init__(self, path=None, exposure_time=None, timestamp=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.exposure_time = exposure_time
        self.timestamp = timestamp
        self.single_counts = []

    def loadfrom(self, path):
        super().loadfrom(path)
        self.single_counts = np.array(self.single_counts)


def main(cam_type='vimba'):
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if cam_type == 'vimba':
        exposure_time = 2000
        N = 300

        cam = VimbaCamera(0)
        cam.set_exposure_time(exposure_time)
        cam.set_borders(Borders(490, 380, 700, 610))
    elif cam_type == 'ASI':
        exposure_time = 0.1
        N = 1000

        cam = ASICam(exposure=5, binning=2, image_bits=16, roi=(None, None, None, None))
        cam.set_roi(1100, 856, 200, 200)  # With Wollaston prism
        # cam.set_roi(1070, 790, 200, 200)
        cam.set_exposure(exposure_time)
    elif cam_type == 'SPCM':
        exposure_time = 1
        N = 1000
        cam = QPTimeTagger(integration_time=exposure_time, coin_window=1e-9, single_channels=[2], coin_channels=[])
    else:
        raise Exception('which cam type?')

    dac = Edac40(max_piezo_voltage=120)
    dac.SLEEP_AFTER_SEND = 0.5

    if cam_type == 'SPCM':
        res = SingleCountsList()
    else:
        res = ImageList()
    res.exposure_time = exposure_time
    res.timestamp = timestamp

    saveto_path = f"{PATH}\\{timestamp}_toptica_SPCM_PBS_1s_N=1000.lcam"

    try:
        for i in range(N):
            amps = np.random.rand(40)
            dac.set_amplitudes(amps)
            if cam_type in ['vimba', 'PCO']:
                im = cam.get_image()
                res.images.append(im)
            elif cam_type == 'SPCM':
                s = cam.read_interesting()[0]
                res.single_counts.append(s)
            if i % 20 == 19:
                res.saveto(saveto_path)
                print(f'{i}, ',end='')
    except Exception:
        res.saveto(saveto_path)

    res.saveto(saveto_path)
    cam.close()
    dac.close()


if __name__ == "__main__":
    main(cam_type='SPCM')
