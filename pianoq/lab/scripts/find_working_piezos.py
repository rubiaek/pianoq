import numpy as np
import time

from pianoq import Borders
from pianoq.lab.Edac40 import Edac40
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.misc.calc_correlation import get_correlation
from pianoq.misc.consts import DEFAULT_BORDERS, DEFAULT_CAM_NO, DEFAULT_BORDERS2


def crop_image(im):
    return im
    # part1 = im[:, 60:140]
    # part2 = im[:, 330:410]
    # im = np.concatenate((part1, part2), axis=1)
    # return im

def check_piezo(e: Edac40, cam: VimbaCamera, piezo_num):
    amps = np.zeros(e.NUM_OF_PIEZOS)

    e.set_amplitudes(amps)
    im1 = cam.get_image()

    amps[piezo_num] = 1
    e.set_amplitudes(amps)
    im2 = cam.get_image()

    # Check Cropping manually!
    correlation = get_correlation(crop_image(im1), crop_image(im2), use_mask=False)
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
    e = Edac40(max_piezo_voltage=70, ip=Edac40.DEFAULT_IP)
    cam = VimbaCamera(DEFAULT_CAM_NO, exposure_time=5.5e3)
    # cam.set_borders(Borders(330, 550, 800, 640))
    cam.set_borders(DEFAULT_BORDERS)

    print("Piezo num\t correlation")
    print("----------------------")
    # good_piezo_indexes = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    for i in range(10,19):
        check_piezo(e, cam, i)

    cam.close()
    e.close()


def check_piezo_movement(piezo_num, sleep_duration=0.5):
    e = Edac40(max_piezo_voltage=70, ip=Edac40.DEFAULT_IP)
    amps = np.zeros(40)
    amps[9] = 0
    amps2 = np.zeros(40)
    amps2[9] = 1

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
