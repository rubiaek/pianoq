import numpy as np
import time
from pianoq.misc.borders import Borders
from pianoq.lab.Edac40 import Edac40
from pianoq.lab.VimbaCamera import VimbaCamera


def get_correlation(im1, im2):
    # distance normalized cross correlation. 1 in perfect. 0 is bad.
    dist_ncc = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2))) / ((im1.size - 1) * np.std(im1) * np.std(im2))
    return dist_ncc


def check_piezo(e: Edac40, cam: VimbaCamera, piezo_num):
    amps = np.zeros(e.NUM_OF_PIEZOS)

    e.set_amplitudes(amps)
    im1 = cam.get_image()[470:720, 350:600]

    amps[piezo_num] = 1
    e.set_amplitudes(amps)
    im2 = cam.get_image()[470:720, 350:600]

    correlation = get_correlation(im1, im2)
    print(f'{piezo_num} \t\t {correlation:.3f}')

    """
    amps[piezo_num] = 0
    e.set_amplitudes(amps)
    im2 = cam.get_image()[470:720, 350:600]
    correlation = get_correlation(im1, im2)

    print(f'correlation with no difference: {correlation}')
    """


def check_all_piezos():
    """
    print for each piezo index how much moving it decorrelates the picture,
    so indexes with correlaction > 0.98 probably don't work
    """
    e = Edac40(max_piezo_voltage=30, ip=Edac40.DEFAULT_IP)
    cam = VimbaCamera(2, exposure_time=800)
    borders = Borders(0, 0, 1280, 1024)
    cam.set_borders(borders)

    print("Piezo num\t correlation")
    print("----------------------")
    # good_piezo_indexes = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    for i in range(40):
        check_piezo(e, cam, i)

    cam.close()
    e.close()


def check_piezo_movement(piezo_num, sleep_duration=0.5):
    e = Edac40(max_piezo_voltage=30, ip=Edac40.DEFAULT_IP)
    amps = np.zeros(40)
    amps2 = np.zeros(40)
    amps2[piezo_num] = 1

    try:

        while True:
            print(time.time())
            e.set_amplitudes(amps)
            time.sleep(sleep_duration)
            e.set_amplitudes(amps2)
            time.sleep(sleep_duration)
    except KeyboardInterrupt:
        e.close()


if __name__ == "__main__":
    check_all_piezos()
    # check_piezo_movement(2)