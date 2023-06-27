import time
import matplotlib.pyplot as plt
import numpy as np
import datetime
from pianoq.misc.borders import Borders
from pianoq.lab import Edac40, ASICam
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq_results.QPPickleResult import QPPickleResult
from pianoq.analysis.contrast_lib import calc_contrast, contrast_to_N_modes
from matplotlib.patches import Rectangle

PATH = r'G:\My Drive\Projects\Dispersion Cancelation\Results\Classical'


class ImageList(QPPickleResult):
    def __init__(self, path=None, exposure_time=None, timestamp=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.exposure_time = exposure_time
        self.timestamp = timestamp
        self.images = []

    def contrast_from_roi_size(self, N_pixels=4):
        im = self.images[0]
        fig, ax = plt.subplots()
        ax.imshow(im)
        fig.colorbar(im, ax=ax)
        mid_x, mid_y = im.shape[1] // 2, im.shape[0] // 2
        ax.add_patch(Rectangle((mid_x-N_pixels, mid_y-N_pixels), 2*N_pixels, 2*N_pixels, facecolor="grey", ec='k', lw=2))

        contrast = 0 # TODO: calc
        ax.set_title(f'contrast on this area: {contrast}')
        fig.show()

    def contrast_one_pixel(self, vmin=0, vmax=1.5):
        fig, ax = plt.subplots()
        mean = self.images.mean(axis=0)
        std = self.images.std(axis=0)
        contrast = std/mean
        imm = ax.imshow(contrast, vmin=vmin, vmax=vmax)

        im = self.images[0]
        mid_x, mid_y = im.shape[1] // 2, im.shape[0] // 2
        V = [im[mid_y, mid_x] for im in self.images]
        V = np.array(V)
        sample_contrast = calc_contrast(V)
        ax.set_title(f'sample contrast: {sample_contrast}')
        fig.colorbar(imm, ax=ax)
        # print(f'contrast according to single pixel: {calc_contrast(V)}')
        fig.show()

    def loadfrom(self, path):
        super().loadfrom(path)
        self.images = np.array(self.images)


def main(cam_type='vimba'):
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if cam_type == 'vimba':
        exposure_time = 2000
        N = 300

        cam = VimbaCamera(0)
        cam.set_exposure_time(exposure_time)
        cam.set_borders(Borders(490, 380, 700, 610))
    elif cam_type == 'ASI':
        exposure_time = 5
        N = 300

        cam = ASICam(exposure=5, binning=2, image_bits=16, roi=(None, None, None, None))
        cam.set_roi(1100, 830, 200, 400)
        cam.set_exposure(exposure_time)
    else:
        raise Exception('which cam type?')

    dac = Edac40(max_piezo_voltage=120)
    dac.SLEEP_AFTER_SEND = 0.5

    res = ImageList()
    res.exposure_time = exposure_time
    res.timestamp = timestamp

    saveto_path = f"{PATH}\\{timestamp}_toptica_ASI_PBS_5s.lcam"

    try:
        for i in range(N):
            amps = np.random.rand(40)
            dac.set_amplitudes(amps)
            im = cam.get_image()
            res.images.append(im)
            res.saveto(saveto_path)
            print(f'{i}, ',end='')
    except Exception:
        res.saveto(saveto_path)

    cam.close()
    dac.close()


if __name__ == "__main__":
    main(cam_type='ASI')
