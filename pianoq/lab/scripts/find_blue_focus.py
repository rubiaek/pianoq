import time
import matplotlib.pyplot as plt
from pianoq.lab.VimbaCamera import VimbaCamera


def main():
    """
    We run this script and move the camera back and forth,
    and try to find z-location so the 255 bar in histogram is largest.
    """
    cam = VimbaCamera(0, 250)  # exposure time close to saturation

    fig_hist, ax_hist = plt.subplots()
    fig_cam, ax_cam = plt.subplots()

    flag = True
    while flag:
        im = cam.get_image()
        ax_cam.imshow(im)
        fig_cam.show()
        ax_hist.clear()
        ax_hist.hist(im.ravel(), bins=56, range=(200, 256))
        ax_hist.set_ylim(0, 8)
        fig_hist.show()
        plt.pause(0.01)
        time.sleep(0.01)

    cam.close()


if __name__ == '__main__':
    main()
