import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.misc.consts import DEFAULT_CAM_NO


global ani


def main():
    cam = VimbaCamera(DEFAULT_CAM_NO, exposure_time=200)
    live_cam(cam)


def live_cam(cam, interval=100, close_at_end=False, remove_min=True, cut_line=None, show_max=True, **kwargs):
    if cut_line is None:
        fig, ax = plt.subplots()
    else:
        fig, axes = plt.subplots(2)
        ax = axes[0]
        ax_line = axes[1]
    imm = cam.get_image()
    if remove_min:
        imm -= imm.min()
    im = ax.imshow(imm, **kwargs)
    title = fig.suptitle('foo', fontsize=36)
    fig.colorbar(im, ax=ax)
    if cut_line is not None:
        line = ax_line.plot(imm[cut_line, :])
        line = line[0]

    def update(i):
        imm = cam.get_image()
        if remove_min:
            imm -= imm.min()
        im.set_data(imm)
        if show_max:
            title.set_text(f'Max pixel: {imm.max():.3f}')
        else:
            title.set_text(f'Total power: {imm.sum():.3f}')
        # ax.set_title('%03d' % i)
        if cut_line is not None:
            line.set_ydata(imm[cut_line, :])

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
