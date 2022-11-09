import time
import glob
import os

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


def analyse(dir_path=rf'G:\My Drive\Projects\Quantum Piano\Results\Calibrations\Stability\*.cam'):
    paths = glob.glob(dir_path)
    im0 = load_image(paths[0])
    corrs = []
    times = []
    for path in paths:
        im = load_image(path)
        corr = get_correlation(im0, im)
        time = os.path.basename(path)
        corrs.append(corr)
        times.append(time)

    return corrs, times


if __name__ == "__main__":
    # main(interval=60)
    pass
