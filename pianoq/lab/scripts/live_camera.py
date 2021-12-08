import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.misc.consts import DEFAULT_CAM_NO


global ani


def main():
    cam = VimbaCamera(DEFAULT_CAM_NO, exposure_time=200)
    live_cam(cam)


def live_cam(cam, interval=100, close_at_end=False):
    fig, ax = plt.subplots()
    im = ax.imshow(cam.get_image())
    fig.colorbar(im, ax=ax)

    def update(i):
        im.set_data(cam.get_image())
        # ax.set_title('%03d' % i)

    global ani
    ani = FuncAnimation(fig, update, interval=interval)  # in ms

    def close(event):
        if event.key == 'q':
            plt.close(event.canvas.figure)
            if close_at_end:
                cam.close()

    cid = fig.canvas.mpl_connect("key_press_event", close)

    plt.show(block=False)


if __name__ == "__main__":
    main()
    plt.show()
