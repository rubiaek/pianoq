try:
    import pco
except ImportError:
    print("can't use pco camera")
import matplotlib.pyplot as plt
import os, datetime
from astropy.io import fits

class PCOCamera:

    pixel_size = 6.5e-6

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

    def show_image(self, im=None, roi=None):
        if im is None:
            im = self.get_image(roi=roi)
        fig, axes = plt.subplots()
        imm = axes.imshow(im)
        fig.colorbar(imm, ax=axes)
        fig.show()
        return fig, axes

    def save_image(self, path, im=None, comment='', add_timestamp_to_name=True, roi=None):
        if im is None:
            im = self.get_image(roi=roi)

        if add_timestamp_to_name:
            path = os.path.join(os.path.dirname(path),
                                f'{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_{os.path.basename(path)}')

        # Following https://docs.astropy.org/en/stable/io/fits/index.html
        hdu = fits.PrimaryHDU(im)
        # hdul = fits.HDUList([hdu])
        # hdul.writeto(path)
        hdu.header['CAM_TYPE'] = 'PCO'
        hdu.header['EXPOINUS'] = self.get_exposure_time() * 1e6
        hdu.header['XPIXSZ'] = self.pixel_size
        hdu.header['YPIXSZ'] = self.pixel_size
        hdu.header['EXPTIME'] = hdu.header['EXPOSURE'] = self.get_exposure_time()  # In seconds
        hdu.header['COMMENT'] = comment
        hdu.header['ROI'] = str(roi)

        hdu.writeto(path)

    def load_image(self, path, show=True):
        im = np.load(path)['image']
        if show:
            self.show_image(im)
        return im

    def set_exposure_time(self, exposure_time):
        """ Set exposure_time in seconds"""
        self._cam.exposure_time = exposure_time

    def get_exposure_time(self):
        """ Get exposure_time in seconds
        d = self._cam.sdk.get_delay_exposure_time()
        exposure = d['exposure']
        timebase = d['exposure timebase']
        if timebase == 'ms':
            exposure *= 1e-3
        elif timebase == 'us':
            exposure *= 1e-6
        elif timebase == 'ns':
            exposure *= 1e-9"""
        return self._cam.exposure_time
        return exposure

    def close(self):
        self._cam.close()
