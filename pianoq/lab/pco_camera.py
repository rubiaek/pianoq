try:
    import pco
except ImportError:
    print("can't use pco camera")
import matplotlib.pyplot as plt


class PCOCamera:
    def __init__(self, exposure_time=None):
        """
        :param borders: (1-2054, 1-2054)
        :param exposure_time: in seconds
        """
        self._cam = pco.Camera()

        if exposure_time is not None:
            self.set_exposure_time(exposure_time)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_image(self, roi=None):
        """ roi is a tuple (min_x, min_y, max_x, max_y)"""
        self._cam.record()
        im, meta = self._cam.image(roi=roi)
        return im

    def show_image(self, im=None):
        if im is None:
            im = self.get_image()
        fig, axes = plt.subplots()
        axes.imshow(im)
        fig.show()
        return fig, axes

    def save_image(self, path):
        im = self.get_image()
        f = open(path, 'wb')
        np.savez(f, image=im)
        f.close()

    def load_image(self, path, show=True):
        im = np.load(path)['image']
        if show:
            self.show_image(im)
        return im

    def set_exposure_time(self, exposure_time):
        """ Set exposure_time in seconds"""
        return self._cam.set_exposure_time(exposure_time)

    def get_exposure_time(self):
        """ Get exposure_time in seconds"""
        d = self._cam.sdk.get_delay_exposure_time()
        exposure = d['exposure']
        timebase = d['exposure timebase']
        if timebase == 'ms':
            exposure *= 1e-3
        elif timebase == 'us':
            exposure *= 1e-6
        elif timebase == 'ns':
            exposure *= 1e-9
        return exposure

    def close(self):
        self._cam.close()
