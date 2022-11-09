import zwoasi as asi
import matplotlib.pyplot as plt
import datetime
from astropy.io import fits
import astropy.time

# asi.init('C:\\code\\ASI_Windows_SDK_V1.22\\ASI SDK\\lib\\x64\\ASICamera2.dll')


class ASICam(object):
    def __init__(self, exposure=1.5, binning=2, image_bits=16, roi=(1500, 950, 200, 200)):
        self._cam = asi.Camera(0)  # Assume we will always have only 1 camera
        self.set_roi = self._cam.set_roi
        self.get_roi = self._cam.get_roi
        self.get_image = self._cam.capture
        self.image_bits = self.set_image_bits(8)
        self._cam.disable_dark_subtract()
        self.pixel_size = self._cam.get_camera_property()['PixelSize']
        self.set_binning(binning)
        self.set_exposure(exposure)
        self.set_image_bits(image_bits)
        self.set_roi(*roi)

    def show_image(self, im=None, title=None, **kwargs):
        if im is None:
            im = self.get_image()
        fig, ax = plt.subplots()
        imm = ax.imshow(im, **kwargs)
        fig.colorbar(imm, ax=ax)
        if title:
            ax.set_title(title)
        fig.show()
        return im, ax

    def save_image(self, path, im=None, comment=''):
        if im is None:
            im = self.get_image()

        # Following https://docs.astropy.org/en/stable/io/fits/index.html
        hdu = fits.PrimaryHDU(im)
        # hdul = fits.HDUList([hdu])
        # hdul.writeto(path)
        hdu.header['XBINNING'] = self.get_binning()
        hdu.header['YBINNING'] = self.get_binning()
        hdu.header['EXPOINUS'] = self.get_exposure()
        hdu.header['GAIN'] = self.get_gain()
        hdu.header['OFFSET'] = hdu.header['BRIGHTNS'] = self.get_brightness()
        hdu.header['DATE-OBS'] = astropy.time.Time(datetime.datetime.now()).fits
        hdu.header['COLORTYP'] = 'RAW8' if self.image_bits == 8 else 'RAW16'
        hdu.header['INPUTFMT'] = 'FITS'
        hdu.header['XPIXSZ'] = self.pixel_size
        hdu.header['YPIXSZ'] = self.pixel_size
        hdu.header['EXPTIME'] = hdu.header['EXPOSURE'] = self.get_exposure() * 1e6  # In seconds
        hdu.header['CCD-TEMP'] = self.get_temperature()
        hdu.header['COMMENT'] = comment

        hdu.writeto(path)

    def set_exposure(self, exposure):
        """ in seconds """
        self._cam.set_control_value(asi.ASI_EXPOSURE, int(exposure * 1e6))

    def set_gain(self, gain):
        self._cam.set_control_value(asi.ASI_GAIN, gain)

    def set_binning(self, bins):
        """ bins shuold be 1, 2, 3 or 4"""
        # TODO: this resets the roi
        self.set_roi(bins=bins)

    def set_brightness(self, brightness=50):
        self._cam.set_control_value(asi.ASI_BRIGHTNESS, brightness)

    def set_gamma(self, gamma=50):
        self._cam.set_control_value(asi.ASI_GAMMA, gamma)

    def set_image_bits(self, bits=16):
        if bits == 16:
            self._cam.set_image_type(asi.ASI_IMG_RAW16)
        elif bits == 8:
            self._cam.set_image_type(asi.ASI_IMG_RAW8)
        else:
            raise Exception("You must choose either 8 bits or 16 bits")

        self.image_bits = bits

    def get_gain(self):
        return self._cam.get_control_value(asi.ASI_GAIN)[0]

    def get_exposure(self):
        """ in seconds """
        return self._cam.get_control_value(asi.ASI_EXPOSURE)[0] * 1e-6

    def get_exposure_time(self):
        return self.get_exposure()

    def get_binning(self):
        return self._cam.get_roi_format()[2]

    def get_brightness(self):
        return self._cam.get_control_value(asi.ASI_BRIGHTNESS)[0]

    def get_gamma(self):
        return self._cam.get_control_value(asi.ASI_GAMMA)[0]

    def get_temperature(self):
        return self._cam.get_control_value(asi.ASI_TEMPERATURE)[0] / 10

    def close(self):
        self._cam.close()


"""
Useful:
    https://pocs.readthedocs.io/en/latest/api/panoptes.pocs.camera.html#panoptes.pocs.camera.libasi.ASIDriver.enable_dark_subtract
    https://pocs.readthedocs.io/en/latest/_modules/panoptes/pocs/camera/libasi.html
    https://zwoasi.readthedocs.io/en/latest/index.html
    https://github.com/python-zwoasi/python-zwoasi
"""
