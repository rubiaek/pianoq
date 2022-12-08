import time
import glob
import os

import matplotlib.pyplot as plt

from pianoq.misc.calc_correlation import get_correlation
from pianoq.misc.borders import Borders
from pianoq.lab import VimbaCamera
from pianoq_results.image_result import load_image


def main(interval=60):
    cam = VimbaCamera(0)
    cam.set_exposure_time(250)
    cam.set_borders(Borders(400, 170, 950, 450))
    start_time = time.time()
    i = 1
    while True:
        path = rf'G:\My Drive\Projects\Quantum Piano\Results\Calibrations\Stability\{(time.time()-start_time):2f}.cam'
        cam.save_image(path)
        time.sleep(interval)
        i += 1
        print(f'i: {(time.time()-start_time):2f}')


def analyse(dir_path=rf'G:\My Drive\Projects\Quantum Piano\Results\Calibrations\Stability\3\*.cam', i0=0, skip_every=1):
    paths = glob.glob(dir_path)

    im01 = load_image(paths[i0])[100:150, 370:420]
    im02 = load_image(paths[i0])[100:160, 95:155]
    t0 = int(float(os.path.basename(paths[i0])[:-4])) / 60

    corrs1 = []
    corrs2 = []
    times = []

    for path in paths[i0::skip_every]:
        im1 = load_image(path)[100:150, 370:420]
        im2 = load_image(path)[100:160, 95:155]
        corr1 = get_correlation(im01, im1)
        corr2 = get_correlation(im02, im2)
        t = float(os.path.basename(path)[:-4]) / 60
        corrs1.append(corr1)
        corrs2.append(corr2)
        times.append(t-t0)

    fig, ax = plt.subplots()
    ax.plot(times, corrs1, label='H pol')
    ax.plot(times, corrs2, label='V pol')
    ax.set_xlabel('time (min)')
    ax.set_ylabel('PCC')
    ax.legend()
    fig.show()

    # return corrs1, corrs2, times


if __name__ == "__main__":
    # main(interval=60)
    pass
