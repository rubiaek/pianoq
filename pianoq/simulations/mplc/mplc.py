import cv2
import numpy as np
import matplotlib.pyplot as plt


class MPLC:
    def __init__(self, conf):
        self.conf = conf
        self.wl = self.wavelength = conf['wavelength']
        self.k = 2 * np.pi / self.wl
        self.L = self.plane_spacing = conf['plane_spacing']
        self.N_planes = conf['N_planes']
        self.N_iterations = conf['N_iterations']
        self.Nx = conf['Nx']
        self.Ny = conf['Ny']
        self.dx = conf['dx']
        self.dy = conf['dy']
        self.k_space_filter = conf['k_space_filter']

        # TODO: have reality grid 3X larger than the mask size of SLM, so I won't reach the edges and have edge effects.
        #  In the propagation zero manully the phases outside SLM mask
        # TODO: have a finer dx for reality grid, with each SLM pixel being 2X2 reality pixels etc.
        self.X = np.arange(-self.Nx / 2, self.Nx / 2) * self.dx
        self.Y = np.arange(-self.Ny / 2, self.Ny / 2) * self.dy
        self.XX, self.YY = np.meshgrid(self.X, self.Y)
        self.TH, self.R = np.arctan2(self.YY, self.XX), np.sqrt(self.XX ** 2 + self.YY ** 2)
        self.maxR = np.max(self.R)
        self.k_z_mat = self._generate_kz_mat()

        self.masks = np.zeros((self.N_planes, self.Ny, self.Nx))

        self.show = self.show_field_intensity
        self.prop = self.propagate_freespace
        self.input_modes = []
        self.output_modes = []

    def set_input_spots_modes(self, sig=0.1, N_rows=4, N_cols=4, spacing=0.6):
        self.input_modes = []
        # TODO: take care of upper and lower halves of SLM for each photon
        # TODO: Gaussian normalization. The power currently does not sum to 1
        for i in range(N_rows):
            for j in range(N_cols):
                C = 1 / (sig ** 2 * 2 * np.pi)
                X0 = spacing * (j - (N_cols / 2.0) + 0.5)
                Y0 = spacing * (i - (N_rows / 2.0) + 0.5)
                E_gaus = C * np.exp(-((self.XX - X0) ** 2 + (self.YY - Y0) ** 2) / (4 * sig ** 2)).astype(complex)
                self.input_modes.append((E_gaus))

    def set_output_speckle_modes(self, sig=0.05, diffuser_pix_size=0.025):
        self.output_modes = []
        for i in range(len(self.input_modes)):
            self.output_modes.append(self._get_speckles(sig=sig, diffuser_pix_size=diffuser_pix_size))

    def _get_speckles(self, sig=0.4, diffuser_pix_size=0.02):
        C = 1 / (sig ** 2 * 2 * np.pi)
        X0 = 0
        Y0 = 0
        E_gaus = C * np.exp(-((self.XX - X0) ** 2 + (self.YY - Y0) ** 2) / (4 * sig ** 2)).astype(complex)

        N_pixs_x = int(self.Nx*self.dx / diffuser_pix_size)
        N_pixs_y = int(self.Ny*self.dy / diffuser_pix_size)
        A = 2 * np.pi * np.random.rand(N_pixs_y, N_pixs_x)
        # .shape conventions on numpy and cv2 is the opposite
        A2 = cv2.resize(A, self.XX.shape[::-1], interpolation=cv2.INTER_NEAREST)
        E_gaus *= np.exp(1j*A2)

        E_out = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E_gaus)))
        return E_out

    def find_phases(self):
        assert len(self.input_modes) > 0, 'Need input modes to perform wavefront matching'
        assert len(self.output_modes) > 0, 'Need output modes to perform wavefront matching'

    def _generate_kz_mat(self):
        freq_x = np.fft.fftshift(np.fft.fftfreq(self.Nx, d=self.dx))
        freq_y = np.fft.fftshift(np.fft.fftfreq(self.Ny, d=self.dy))
        freq_XXs, freq_YYs = np.meshgrid(freq_x, freq_y)
        light_k = 2 * np.pi / self.wl
        k_xx = freq_XXs * 2 * np.pi
        k_yy = freq_YYs * 2 * np.pi

        k_z_sqr = light_k ** 2 - (k_xx ** 2 + k_yy ** 2)
        # Remove all the negative component, as they represent evanescent waves, see Fourier Optics page 58
        np.maximum(k_z_sqr, 0, out=k_z_sqr)
        k_z = np.sqrt(k_z_sqr)

        # TODO: large angles filter
        # k_z = k_z * (self.R < self.maxR * self.kSpaceFilter)
        return k_z

    def propagate_freespace(self, E, L, backprop=False):
        assert E.shape == self.XX.shape, 'Bad shape for E'
        E2 = np.copy(E)

        sign = 1 if not backprop else -1

        E_K = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E2)))
        # Apply the transfer function of free-space, see Fourier Optics page 74
        E_K *= np.exp(-1j * sign * self.k_z_mat * L)
        E_out = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(E_K)))
        return E_out

    @staticmethod
    def show_field_intensity(E, ax=None):
        # TODO: colorize
        if not ax:
            fig, ax = plt.subplots()
        im0 = ax.imshow(np.abs(E) ** 2)
        ax.figure.colorbar(im0, ax=ax)
        ax.figure.show()


# All in mm
conf = {'wavelength': 810e-6,  # mm
        'plane_spacing': 87,  # mm
        'N_planes': 9,
        'N_iterations': 30,
        'Nx': 2 * 140,  # Number of grid points x-axis
        'Ny': 2 * 180,  # Number of grid points y-axis
        'dx': 12.5e-3,  # mm - SLM pixel sizes
        'dy': 12.5e-3,  # mm
        'k_space_filter': 0.15
        }

mplc = MPLC(conf=conf)
mplc.set_input_spots_modes(sig=0.1, N_rows=4, N_cols=4, spacing=0.6)
mplc.set_output_speckle_modes(sig=0.05, diffuser_pix_size=0.025)
