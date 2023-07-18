import cv2
import matplotlib.pyplot as plt
import numpy as np
import datetime

from pianoq.misc.borders import Borders
from pianoq.lab import Edac40, ASICam
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.lab.time_tagger import QPTimeTagger
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

    def contrast_per_pixel(self, vmin=0, vmax=1.5, binning=1, remove_background=True, N_images=None,
                           show_N_modes=False, N_min=0.5, N_max=20, ax=None, title=''):
        images = self.images

        if N_images is not None:
            images = images[:N_images]

        if binning != 1:
            rows, cols = images[0].shape
            images = np.array([cv2.resize(im, (cols//binning, rows//binning), interpolation=cv2.INTER_AREA) for im in images])

        if remove_background:
            V = images[:, 0, 0]
            bg = V.mean()
            images = images.astype(float) - bg

        if ax is None:
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
        print(f'mean_contrast: \t\t\t{c}')
        cut_ims = images[:, X:X+L, X:X+L]
        tot_im = cut_ims.sum(axis=0)
        cut_ims = np.array([cut_im / tot_im for cut_im in cut_ims])

        V = cut_ims.ravel()
        print(f'contrast from all pixels:\t {calc_contrast(V)}')

        ax.set_title(f'{title}contrast: {c:.3f}. N: {1/c**2:.2f}')
        ax.figure.colorbar(imm, ax=ax)
        ax.figure.show()

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


def show_BS_PBS_Cobolt_Toptica(binning=1, suptitle='binning=2'):
    res_toptica_PBS = ImageList()
    res_toptica_PBS.loadfrom('G:\\My Drive\\Projects\\Dispersion Cancelation\\Results\\Classical\\2023_06_28_09_09_10_toptica_ASI_PBS_50ms_N=1000.lcam')
    res_toptica_no_PBS = ImageList()
    res_toptica_no_PBS.loadfrom('G:\\My Drive\\Projects\\Dispersion Cancelation\\Results\\Classical\\2023_06_28_09_43_34_toptica_ASI_no_PBS_50ms_N=1000_charmed.lcam')
    res_cobolt_no_PBS = ImageList()
    res_cobolt_no_PBS.loadfrom('G:\\My Drive\\Projects\\Dispersion Cancelation\\Results\\Classical\\2023_06_28_10_05_25_cobolt_ASI_no_PBS_50ms_N=1000.lcam')
    res_cobolt_PBS = ImageList()
    res_cobolt_PBS.loadfrom('G:\\My Drive\\Projects\\Dispersion Cancelation\\Results\\Classical\\2023_06_28_10_22_01_cobolt_ASI_PBS_100ms_N=1000.lcam')

    fig, axes = plt.subplots(2, 2)
    res_toptica_PBS.contrast_per_pixel(binning=binning, ax=axes[0, 0], title='Toptica PBS: ')
    res_toptica_no_PBS.contrast_per_pixel(binning=binning, ax=axes[0, 1], title='Toptica no PBS: ')
    res_cobolt_PBS.contrast_per_pixel(binning=binning, ax=axes[1, 0], title='Cobolt PBS: ')
    res_cobolt_no_PBS.contrast_per_pixel(binning=binning, ax=axes[1, 1], title='Cobolt no PBS: ')
    fig.suptitle(suptitle)
    fig.show()

def print_classical_SPCM_results():
    print()
    res_toptica_PBS = SingleCountsList()
    res_toptica_PBS.loadfrom(r'G:\\My Drive\\Projects\\Dispersion Cancelation\\Results\\Classical\\2023_06_29_15_44_23_toptica_SPCM_PBS_1s_N=1000.lcam')
    c = calc_contrast(res_toptica_PBS.single_counts)
    print(f'SPCM Toptica PBS ::\t C={c:.3f}, N={1/c**2:.3f}')
    res_toptica_no_PBS = SingleCountsList()
    res_toptica_no_PBS.loadfrom(r'G:\\My Drive\\Projects\\Dispersion Cancelation\\Results\\Classical\\2023_06_29_14_49_02_toptica_SPCM_BS_1s_N=1000.lcam')
    c = calc_contrast(res_toptica_no_PBS.single_counts)
    print(f'SPCM Toptica no PBS ::\t C={c:.3f}, N={1/c**2:.3f}')
    res_cobolt_PBS = SingleCountsList()
    res_cobolt_PBS.loadfrom(r'G:\\My Drive\\Projects\\Dispersion Cancelation\\Results\\Classical\\2023_07_02_13_03_40_Cobolt_SPCM_PBS_1s_N=1000.lcam')
    c = calc_contrast(res_cobolt_PBS.single_counts)
    print(f'SPCM Cobolt PBS ::\t C={c:.3f}, N={1/c**2:.3f}')
    res_cobolt_no_PBS = SingleCountsList()
    res_cobolt_no_PBS.loadfrom(r"G:\My Drive\Projects\Dispersion Cancelation\Results\Classical\2023_07_02_13_36_27_Cobolt_SPCM_no_PBS_1s_N=1000.lcam")
    c = calc_contrast(res_cobolt_no_PBS.single_counts)
    print(f'SPCM Cobolt no PBS ::\t C={c:.3f}, N={1/c**2:.3f}')


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

    saveto_path = f"{PATH}\\{timestamp}_Cobolt_SPCM_no_PBS_1s_N=1000.lcam"

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


# if __name__ == "__main__":
#     main(cam_type='SPCM')
