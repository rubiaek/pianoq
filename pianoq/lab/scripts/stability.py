import time

from pianoq.misc.borders import Borders
from pianoq.lab import VimbaCamera


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


if __name__ == "__main__":
    main(interval=60)
