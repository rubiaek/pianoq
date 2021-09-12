import traceback
import numpy as np
import matplotlib.pyplot as plt


class VimbaImage(object):
    def __init__(self, path=None):
        self.image = None
        self.path = path
        self.exposure_time = None
        self.timestamp = None

        if path:
            self.loadfrom(path)

    def saveto(self, path):
        try:
            f = open(path, 'wb')
            np.savez(f,
                     image=self.image,
                     path=self.path,
                     exposure_time=self.exposure_time,
                     timestamp=self.timestamp
                     )
            f.close()
        except Exception as e:
            print("ERROR!!")
            print(e)
            traceback.print_exc()

    def loadfrom(self, path):
        # path = path or self.DEFAULT_PATH
        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        self.image = data.get('image', None)
        self.path = data.get('path', None)
        self.exposure_time = data.get('exposure_time', None)
        self.timestamp = data.get('timestamp', None)

    def show_image(self, aspect=None):
        fig, axes = plt.subplots()
        im = axes.imshow(self.image, aspect=aspect)
        fig.colorbar(im, ax=axes)
        fig.show()
        return fig, axes


def show_image(path):
    vim = VimbaImage(path)
    return vim.image, vim.show_image()

def load_image(path):
    vim = VimbaImage(path)
    return vim.image
