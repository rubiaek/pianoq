import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    return piezo_num, correlation

    """
    amps[piezo_num] = 0
    e.set_amplitudes(amps)
    im2 = cam.get_image()[470:720, 350:600]
    correlation = get_correlation(im1, im2)

    print(f'correlation with no difference: {correlation}')
    """


def check_all_piezos(max_piezo_voltage=100):
    """
    print for each piezo index how much moving it decorrelates the picture,
    so indexes with correlaction > 0.98 probably don't work
    """
    e = Edac40(max_piezo_voltage=max_piezo_voltage, ip=Edac40.DEFAULT_IP)
    cam = VimbaCamera(DEFAULT_CAM_NO, exposure_time=450)
    # cam.set_borders(Borders(330, 550, 800, 640))
    # cam.set_borders(DEFAULT_BORDERS)

    print("Piezo num\t correlation")
    print("----------------------")
    # good_piezo_indexes = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    res = []
    for i in range(40):
        piezo_num, correlation = check_piezo(e, cam, i)
        res.append(list((piezo_num, correlation)))

    cam.close()
    e.close()

    res = np.array(res)

    fig, ax = plt.subplots()
    ax.plot(res[:, 0], res[:, 1])
    fig.show()

    return res, ax


def check_piezo_movement(piezo_num, sleep_duration=0.5):
    e = Edac40(max_piezo_voltage=70, ip=Edac40.DEFAULT_IP)
    amps = np.zeros(40)
    amps[piezo_num] = 0
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


global ani
global flag


def live_piezo_diff(dac, cam, piezzo_num, close_at_end=False):
    global flag
    flag = True
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    global im0
    global im1
    im0 = cam.get_image()
    imm0 = axes[0].imshow(im0, animated=True)
    im1 = cam.get_image()
    imm1 = axes[1].imshow(im1, animated=True)
    title = fig.suptitle('foo', fontsize=36)
    axes[0].set_title('piezzo 0 press')
    axes[1].set_title('piezzo 1 press')
    fig.colorbar(imm0, ax=axes[0])
    fig.colorbar(imm1, ax=axes[1])

    def update(i):
        global flag
        global im0
        global im1
        if flag:
            amps = np.zeros(40); dac.set_amplitudes(amps)
            im0 = cam.get_image()
            imm0.set_data(im0)
        else:
            amps = np.zeros(40); amps[piezzo_num] = 1; dac.set_amplitudes(amps)
            im1 = cam.get_image()
            imm1.set_data(im1)
        flag = not flag

        corr = get_correlation(im0, im1, False)
        title.set_text(f'PCC: {corr:.3f}')

        # ax.set_title('%03d' % i)

    global ani
    ani = FuncAnimation(fig, update)

    def close(event):
        if event.key == 'q':
            plt.close(event.canvas.figure)
            if close_at_end:
                cam.close()

    cid = fig.canvas.mpl_connect("key_press_event", close)

    plt.show(block=False)


if __name__ == "__main__":
    # res = check_all_piezos()
    # res = np.array(res)
    # Q = res[:, 1]
    # check_piezo_movement(2)
    pass
