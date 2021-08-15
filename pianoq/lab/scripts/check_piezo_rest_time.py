import numpy as np

from pianoq.lab.Edac40 import Edac40
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.misc.calc_correlation import get_correlation
from pianoq.misc.consts import DEFAULT_BORDERS


def check_rest_time():
    e = Edac40(max_piezo_voltage=30, ip=Edac40.DEFAULT_IP)
    cam = VimbaCamera(2, exposure_time=800)
    borders = DEFAULT_BORDERS
    cam.set_borders(borders)
    e.SLEEP_AFTER_SEND = 0.0001

    times = []
    ims = []
    correlations = []
    print('times\t\tcorrelations')
    print('---------------------------------')

    amps = np.random.uniform(0, 1, 40)
    e.set_amplitudes(amps)

    import time
    start = time.time()
    times.append(0)
    im = cam.get_image()
    ims.append(im)

    for i in range(10):
        im = cam.get_image()
        ims.append(im)
        now = time.time()
        times.append(now-start)

    for i in range(10):
        corr = get_correlation(ims[i+1], ims[i])
        correlations.append(corr)

        print(f'{times[i+1]:.3f}\t\t{corr}')

    cam.close()
    e.close()


if __name__ == "__main__":
    check_rest_time()
