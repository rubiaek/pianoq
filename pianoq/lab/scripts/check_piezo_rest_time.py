import numpy as np

from pianoq.lab.Edac40 import Edac40
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.misc.borders import Borders


def get_correlation(im1, im2):
    # distance normalized cross correlation. 1 in perfect. 0 is bad.
    dist_ncc = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2))) / ((im1.size - 1) * np.std(im1) * np.std(im2))
    return dist_ncc


def check_rest_time():
    e = Edac40(max_piezo_voltage=30, ip=Edac40.DEFAULT_IP)
    cam = VimbaCamera(2, exposure_time=800)
    borders = Borders(300, 420, 900, 780)
    cam.set_borders(borders)
    e.SLEEP_AFTER_SEND = 0.0001

    times = []
    correlations = []
    print('times\t\tcorrelations')
    print('---------------------------------')

    amps = np.random.uniform(0, 1, 40)
    e.set_amplitudes(amps)

    import time
    start = time.time()
    old_im = cam.get_image()

    for i in range(20):
        im = cam.get_image()
        now = time.time()
        times.append(now-start)
        corr = get_correlation(old_im, im)  # Doing this later will probably give me better resollution
        correlations.append(corr)
        old_im = im

    for time, corr in zip(times, correlations):
        print(f'{time:.3f}\t\t{corr}')

    cam.close()
    e.close()


if __name__ == "__main__":
    check_rest_time()
