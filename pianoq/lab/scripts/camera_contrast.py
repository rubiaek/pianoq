import time
import matplotlib.pyplot as plt
import numpy as np
import datetime
from pianoq.misc.borders import Borders
from pianoq.lab import Edac40
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq_results.QPPickleResult import QPPickleResult
from pianoq.analysis.contrast_lib import calc_contrast, contrast_to_N_modes
from matplotlib.patches import Rectangle

PATH = r'G:\My Drive\Projects\Dispersion Cancelation\Results\Classical'

class ImageList(QPPickleResult):
    def __init__(self, exposure_time=None, timestamp=None):
        self.exposure_time = exposure_time
        self.timestamp = timestamp
        self.images = []

    def contrast_from_roi_size(self, N_pixels=4):
        im = self.images[0]
        fig, ax = plt.supblots()
        ax.imshow(im)
        fig.colorbar(im, ax=axes)
        mid_x, mid_y = im.shape[1] // 2, im.shape[0] // 2
        ax.add_patch(Rectangle((mid_X-N_pixels, mid_y-N_pixels), 2*N_pixels, 2*N_pixels, facecolor="grey", ec='k', lw=2))

        ax.set_title(f'contrast on this area: {contrast}')
        fig.show()

    def contrast_one_pixel(self):
        im = self.images[0]
        mid_x, mid_y = im.shape[1] // 2, im.shape[0] // 2
        V = [im[mid_y, mid_x] for im in self.images]
        V = np.array(V)
        print(f'contrast according to single pixel: {calc_contrast(V)}')

    def loadfrom(self):
        super().loadfrom()
        self.images = np.array(self.images)



def main():
    exposure_time = 2000
    N = 300
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    cam = VimbaCamera(0)
    cam.set_exposure_time(exposure_time)
    cam.set_borders(Borders(490, 380, 700, 610))

    dac = Edac40(max_piezo_voltage=120)
    dac.SLEEP_AFTER_SEND = 0.5

    res = ImageList()
    res.exposure_time = exposure_time
    res.timestamp = timestamp

    saveto_path = f"{PATH}\\{timestamp}_toptica_no_PBS.lcam"

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
    main()
