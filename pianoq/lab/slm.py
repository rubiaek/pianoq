import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import cv2


configs = {
    0: {  # SLM no. 02 from Ori lab with 808
        'correction_path': r"G:\My Drive\Projects\Klyshko Optimization\Equipment\02 Ori Katz\deformation_correction_pattern\CAL_LSH0801927_810nm.bmp",
        'alpha': 213,
        'geometry': '1272x1024+1913+-30',
        'monitor': 0,  # If using slmpy
        # 'active_mask_slice': np.index_exp[430:630, 380:580]
        'active_mask_slice': np.index_exp[380:680, 350:650]
    },
    1: {  # SLM no. 1 with 404nm
        'correction_path': r'F:\SLM-x13138-05\deformation_correction_pattern\CAL_LSH0801946_400nm.bmp',
        'alpha': 215,  # for 404nm, taken from the data sheet
        'geometry': '1272x1024+1913+-30',
        'monitor': 0,  # If using slmpy
        'active_mask_slice': np.index_exp[240:510, 630:780]
    },
    2: {  # SLM no. 2 with 404nm
        'correction_path': r'F:\SLM-x13138-01\deformation_correction_pattern\CAL_LSH0801676_400nm.bmp',
        'alpha': 95,  # for 404nm, taken from the data sheet
        'geometry': '1272x1024+3193+-30',
        'monitor': 2,  # If using slmpy
        'active_mask_slice': np.index_exp[312:582, 323:473]
    },
    3: {  # SLM no. 2 with 808nm
        'correction_path': r'F:\SLM-x13138-01\deformation_correction_pattern\CAL_LSH0801676_700nm.bmp',
        'alpha': 270,  # for 808nm, extrapolated from the data sheet.
        'geometry': '1272x1024+3193+-30',
        'monitor': 2,  # If using slmpy
        'active_mask_slice': np.index_exp[312:582, 323:473]
    },
    10: {  # SLM for playing at home
        'correction_path': r'C:\Temp\2020.09.21\fix_test_slm.bmp',
        'alpha': 270,  # for 808nm, extrapolated from the data sheet.
        'geometry': '636x512+0+0',
        'monitor': 2,  # If using slmpy
        'active_mask_slice': np.index_exp[0:1024, 0:1272]
        # 'active_mask_slice': np.index_exp[312:582, 323:473]
    }
}


# try:
#     import opticalsimulator.external.slmpy as slmpy
# except TypeError:
#     print("Can't use slmpy")


class SLMDevice(object):
    pixel_size_x = 12.5e-6
    pixel_size_y = 12.5e-6

    def __init__(self, config_num=1, use_mirror=False, use_slmpy=False, alpha=None):
        # I thought using slmpy would be faster than matplotlib method - but i guess not...
        # You also shouldn't create 2 slms with slmpy - it doesn't work so well...
        self.config_num = config_num
        self.config = configs[config_num]
        self.alpha = alpha or self.config['alpha']
        self.active_mask_slice = self.config['active_mask_slice']
        self.correction: np.ndarray = plt.imread(self.config['correction_path'])

        self.pixels_y, self.pixels_x = self.correction.shape

        self.phase_grid = np.zeros(self.correction.shape)
        x = np.arange(0, self.pixels_x)
        y = np.arange(0, self.pixels_y)
        self.x_pixel_indexes, self.y_pixel_indexes = np.meshgrid(x, y)
        self.use_mirror = use_mirror
        self.use_slmpy = use_slmpy

        self.is_pinhole = False
        self.center = None
        self.radius = None
        self.pinhole_type = 'rand'
        self.rand_phase = 2 * np.pi * np.random.rand(*self.correction.shape)

        if not self.use_slmpy:
            self.fig = plt.figure(f'SLM{self.config_num}-Figure', frameon=False)
            self.axes = self.fig.add_axes([0., 0., 1., 1., ])
            self.fig.canvas.toolbar.pack_forget()
            # This pause is necessary to make sure the location of the windows is actually changed when using TeamViewer
            plt.pause(0.1)
            self.fig.canvas.manager.window.geometry(self.config['geometry'])
            self.axes.set_axis_off()
            self.image = self.axes.imshow(self.phase_grid, cmap='gray', vmin=0, vmax=255)
            self.fig.canvas.draw()
            self.fig.show()

            self.background = self.fig.canvas.copy_from_bbox(self.axes.bbox)
        else:
            self.slm = slmpy.SLMdisplay(monitor=self.config['monitor'])

        self.normal()

        # I want to use the pinhole, and then the square macro pixels aren't the best...
        # Probably should have a slm.set_hexs that gets a vector as an input and the partitioning should just do that
        # It will beed a slight redesign also of slm_optimize.
        # For anything to work you also need to start with monkey patching np.int = int and np.float = float
        # hexs = SLMlayout.Hexagons(radius=150, cellSize=20, resolution=slm.correction.shape, center=(500, 500), method='equal')
        # patt = hexs.getImageFromVec(np.random.rand(hexs.nParts), dtype=float)  # -> phase mask with hexagons with different phases
        # mimshow(patt)
        # slm.update_phase(patt)
        # https://www.wavefrontshaping.net/post/id/24

    @property
    def active_x_pixels(self):
        return self.active_mask_slice[1].stop - self.active_mask_slice[1].start

    @property
    def active_y_pixels(self):
        return self.active_mask_slice[0].stop - self.active_mask_slice[0].start

    def restore_position(self):
        if matplotlib.get_backend() == 'TkAgg':
            self.fig.canvas.manager.window.geometry(self.config['geometry'])

    def save_phase(self, path):
        """ takes whole phase and saves it as is """
        f = open(path, 'wb')
        np.savez(f, phase_grid=self.phase_grid)
        f.close()

    def load_phase(self, path, should_update=True):
        """ takes whole phase and loads it as is """
        phase_grid = np.load(path)['phase_grid']
        if should_update:
            # Because the mirror came in already with
            self.update_phase(phase_grid, dont_use_mirror=True)
        return phase_grid

    def save_diffuser(self, A, path):
        """ takes diffuser matrix A and saves it, for later use with load_diffuser """
        f = open(path, 'wb')
        np.savez(f, diffuser=A)
        f.close()

    def load_diffuser(self, path, should_update=True):
        A = np.load(path)['diffuser']
        if should_update:
            # Because the mirror came in already with
            self.update_phase_in_active(A)
        return A

    def update_phase_in_active(self, phase, active_mask_slice=None):
        active_mask_slice = active_mask_slice or self.active_mask_slice

        active_x_pixels = active_mask_slice[1].stop - active_mask_slice[1].start
        active_y_pixels = active_mask_slice[0].stop - active_mask_slice[0].start

        phase_mask = cv2.resize(phase, (active_x_pixels, active_y_pixels), interpolation=cv2.INTER_AREA)
        final_mask = np.zeros(self.correction.shape)
        final_mask[active_mask_slice] = phase_mask
        self.update_phase(final_mask)

    def update_phase(self, phase, dont_use_mirror=False):
        """ Phase between 0-2*pi.
            Mirror is used usually to work away from the area where there is DC.
            There is a relatively large amount of DC because the SLM isn't meant for 808nm
        """
        if phase.shape != self.correction.shape:
            print(f"Tried to update phase on SLM with wrong shape of {phase.shape}. "
                  f"Reshaping to {self.correction.shape}")
            phase = cv2.resize(phase, (self.correction.shape[1], self.correction.shape[0]),
                               interpolation=cv2.INTER_AREA)

        self.phase_grid = phase
        self.update(dont_use_mirror=dont_use_mirror)

    def update(self, dont_use_mirror=False):

        phase = self.phase_grid.copy()
        if self.use_mirror and not dont_use_mirror:
            mirror_phase = self._get_mirror_phase(m=15, angle=np.pi/2)
            phase += mirror_phase

        if self.is_pinhole:
            phase += self._get_pinhole_phase(self.radius, self.center, self.pinhole_type)

        # Calculating image to send to the SLM (see data sheet for an example)
        phase = np.mod(phase * 255 / (2 * np.pi) + self.correction, 256)
        phase = phase * self.alpha / 255

        if self.alpha > 255:
            middle = (self.alpha + 255) / 2
            phase[np.logical_and((255 < phase), (phase < middle))] = 255
            phase[np.logical_and((middle < phase), (phase < self.alpha))] = 0

        phase = np.uint8(phase)

        if self.use_slmpy:
            self.slm.updateArray(phase)
        else:
            # restore background
            self.fig.canvas.restore_region(self.background)
            self.image.set_data(phase)

            # redraw just the points
            self.axes.draw_artist(self.image)

            # fill in the axes rectangle
            self.fig.canvas.blit(self.axes.bbox)

        plt.pause(0.001)

    def normal(self):
        phase = np.zeros(self.correction.shape)
        self.update_phase(phase)

    def pi_step(self, x, half_pi=False, another_x=None):
        half_factor = 0.5 if half_pi else 1
        phase = np.zeros(self.correction.shape)
        phase[:, x:] = np.pi * half_factor
        if another_x:
            phase[:, another_x:] = 0
        self.update_phase(phase)

    def pi_step_y(self, y, half_pi=False, another_y=None):
        half_factor = 0.5 if half_pi else 1
        phase = np.zeros(self.correction.shape)
        phase[y:, :] = np.pi * half_factor
        if another_y:
            phase[another_y:, :] = 0
        self.update_phase(phase)

    def pi_step_x_y(self, x, y, half_pi=False):
        half_factor = 0.5 if half_pi else 1
        phase = np.zeros(self.correction.shape)
        phase[y:, :] = np.pi * half_factor
        phase[:, x:] = np.pi * half_factor
        self.update_phase(phase)

    def rand_range(self, min_x=None, max_x=None, min_y=None, max_y=None):
        phase = np.zeros(self.correction.shape)
        min_x = min_x or 0
        min_y = min_y or 0
        max_x = max_x or self.correction.shape[1]
        max_y = max_y or self.correction.shape[0]
        phase[min_y:max_y, min_x:max_x] = 2 * np.pi * np.random.rand(max_y-min_y, max_x-min_x)
        self.update_phase(phase)

    def not_rand_range(self, min_x=None, max_x=None, min_y=None, max_y=None):
        phase = 2 * np.pi * np.random.rand(self.correction.shape[0], self.correction.shape[1])
        min_x = min_x or 0
        min_y = min_y or 0
        max_x = max_x or self.correction.shape[1]
        max_y = max_y or self.correction.shape[0]
        phase[min_y:max_y, min_x:max_x] = 0
        self.update_phase(phase)

    def set_diffuser(self, macro_pixels, active_mask_slice=None):
        active_mask_slice = active_mask_slice or self.active_mask_slice
        orig_mask = 2 * np.pi * np.random.rand(macro_pixels, macro_pixels)

        phase_mask = cv2.resize(orig_mask, (self.active_x_pixels, self.active_y_pixels), interpolation=cv2.INTER_AREA)
        final_mask = np.zeros(self.correction.shape)
        final_mask[active_mask_slice] = phase_mask
        self.update_phase(final_mask)
        return phase_mask

    def set_diffuser2(self, sigma):
        """
        Creates a smooth and realistic diffuser phase mask.
        This method is more realistic then the macro-pixel method since macro-pixels approach has
        non-physical discontinuities (jumps) in the phase mask.
        This approach uses something similar to Gerchberg-Saxton algorithm.
        It is base on a code Ohad Lib wrote.
        """
        N_x = self.pixels_x
        N_y = self.pixels_y
        dx = self.pixel_size_x
        dy = self.pixel_size_x
        # The spacing between neighboring cells in the frequency grid
        # df_x = 1/(N_x*dx)
        # df_y = 1/(N_y*dy)
        f_x = np.fft.fftshift(np.fft.fftfreq(N_x, dx))
        f_y = np.fft.fftshift(np.fft.fftfreq(N_y, dy))
        f_Xs, f_Ys = np.meshgrid(f_x, f_y)

        # Power spectrum (von Karman) (What?)
        PSD = np.exp(-(f_Xs ** 2 + f_Ys ** 2) / (2 * sigma ** 2))
        PSD[int(N_y / 2), int(N_x / 2)] = 0  # removing the zero freq term
        # print(f_Xs[int(N_y / 2), int(N_x / 2)], f_Ys[int(N_y / 2), int(N_x / 2)])
        rand_spectrum = (np.random.randn(N_y, N_x) + 1j * np.random.randn(N_y, N_x)) * np.sqrt(PSD)
        ifft = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(rand_spectrum), norm='ortho'))
        self.update_phase(np.real(ifft))

    def set_kolmogorov(self, cn2=1e-16, L=100, l_o=10, l_i=0.5e-2, is_test=False):
        """
        is_test is for comparing phase generation to matlab. use also:
        phase = np.round(slm1.set_kolmogorov(cn2=1e-16, L=3800, is_test=True), 4)
        mat_phase = np.round(loadmat("C:\\Users\\Owner\\Google Drive\\People\\Ronen\\kolmogorov\\Kolmogorov_code\\phase.mat")['phase'], 4).T
        """
        k_spdc = 2*np.pi/808e-9
        # L = 100  # m
        # cn2 = 1e-16  # 1e-17, 1e-16, 1e-15 for easy, moderate and hard torbulence
        r0 = (0.4229 * (k_spdc**2) * L * cn2)**(-3/5)  # m, for the SPDC wavelength

        l_o *= 1e-3
        l_i *= 1e-3
        r0 *= 1e-3

        if is_test:
            N_x = 128
            N_y = 128
        else:
            N_x = self.active_x_pixels
            N_y = self.active_y_pixels

        dx = self.pixel_size_x
        dy = self.pixel_size_y

        # The spacing between neighboring cells in the frequency grid
        df_x = 1/(N_x*dx)
        df_y = 1/(N_y*dy)

        f_x = np.fft.fftshift(np.fft.fftfreq(N_x, dx))
        f_y = np.fft.fftshift(np.fft.fftfreq(N_y, dy))
        f_Xs, f_Ys = np.meshgrid(f_x, f_y)

        # Power spectrum (von Karman)
        fm = (5.92/l_i) / (2*np.pi) # inner scale frequency[1 / m]
        fo = 1 / l_o  # outer scale frequency[1 / m]
        PSD = 0.023*(r0**(-5/3)) * np.exp(-((f_Xs**2 + f_Ys**2) / fm**2)) * (f_Xs**2 + f_Ys**2 + fo**2)**(-11/6)
        PSD[N_y//2, N_x//2] = 0  # removing the zero freq term

        if is_test:
            np.random.seed(1)
            A = norm.ppf(np.random.rand(N_y, N_x))
            B = norm.ppf(np.random.rand(N_y, N_x))
        else:
            A = np.random.randn(N_y, N_x)
            B = np.random.randn(N_y, N_x)

        rand_spectrum = (A + 1j * B) * np.sqrt(PSD / (df_x*df_y))
        phase_screen = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(rand_spectrum))) * df_x*df_y * N_x*N_y
        phase_screen = self._add_subharms(phase_screen)

        phase_screen = np.real(phase_screen)

        self.update_phase_in_active(phase_screen)
        return phase_screen

    def _add_subharms(self, phase_screen):
        # TODO: this
        return phase_screen

    def set_mirror(self, m, angle=0):
        """
        :param m: amount of pi's in linear tilt
        :param angle: 0 is up, in radians
        :return:
        """
        phase = self._get_mirror_phase(m, angle)
        self.update_phase(phase)
        return phase

    def set_mirror_in_active(self, m=50, angle=0, active_mask_slice=None):
        active_mask_slice = active_mask_slice or self.config['active_mask_slice']
        phase = self._get_mirror_phase(m, angle)
        self.update_phase_in_active(phase, active_mask_slice)

    def _get_mirror_phase(self, m, angle=0):
        X = np.linspace(0, m * np.pi, self.correction.shape[1])
        Y = np.linspace(0, m * np.pi, self.correction.shape[0])
        Xs, Ys = np.meshgrid(X, Y)
        phase = np.sin(angle) * Xs + np.cos(angle) * Ys
        return phase

    def _get_out_of_disk_mask(self, radius, center=None, shape=None):
        if not shape:
            shape = self.correction.shape
        if not center:
            center = (shape[0]//2, shape[1]//2)
        X, Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        mask = (Y-center[0])**2 + (X-center[1])**2 > radius**2
        return mask.astype(int)

    def _get_pinhole_phase(self, radius, center, pinhole_type=None):
        pinhole_type = pinhole_type or self.pinhole_type
        mask = self._get_out_of_disk_mask(radius, center)
        if pinhole_type == 'rand':
            phase = self.rand_phase
        elif pinhole_type == 'mirror':
            phase = self._get_mirror_phase(m=100, angle=-np.pi/2)
        else:
            raise NotImplementedError()

        return mask * phase

    def set_pinhole(self, radius, center, pinhole_type='rand'):
        # slm.set_pinhole(150, (530, 500))
        self.is_pinhole = True
        self.center = center
        self.radius = radius
        self.pinhole_type = pinhole_type
        self.update()

    def set_not_pinhole(self):
        self.is_pinhole = False

    def close(self):
        if not self.use_slmpy:
            plt.close(self.fig)
