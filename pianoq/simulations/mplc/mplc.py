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
        self.N_modes = conf['N_modes']
        self._size_factor = conf['size_factor']
        self.active_Nx = conf['Nx']
        self.active_Ny = conf['Ny']
        self.Nx = self.active_Nx * self._size_factor
        self.Ny = self.active_Ny * self._size_factor
        self._pad_factor = (self._size_factor - 1) // 2  # assuming self.size_factor is odd

        self._active_y_start = self.active_Ny * self._pad_factor
        self._active_y_end = self.active_Ny * (self._size_factor - self._pad_factor)
        self._active_x_start = self.active_Nx * self._pad_factor
        self._active_x_end = self.active_Nx * (self._size_factor - self._pad_factor)
        self.active_slice = np.index_exp[self._active_y_start: self._active_y_end,
                                         self._active_x_start: self._active_x_end]

        self.dx = conf['dx']
        self.dy = conf['dy']
        self.min_log_level = conf['min_log_level']

        self.k_space_filter = conf['k_space_filter']
        self.use_mask_offset = conf['use_mask_offset']
        # TODO: this does not make sense, N_modes should be upstairs
        self.mask_offset = np.sqrt(1e-3/(self.Nx * self.Ny * self.N_modes))

        # TODO: have a finer dx for reality grid, with each SLM pixel being 2X2 reality pixels etc.?
        self.X = np.arange(-self.Nx / 2, self.Nx / 2) * self.dx
        self.Y = np.arange(-self.Ny / 2, self.Ny / 2) * self.dy
        self.XX, self.YY = np.meshgrid(self.X, self.Y)
        self.k_z_mat = self._generate_kz_mat()

        # masks are always exp(i*\phi(x, y))
        self.masks = np.exp(1j*np.zeros((self.N_planes, self.Ny, self.Nx), dtype=np.complex128))
        # forward_fields[0] will be the input spots
        self.forward_fields = np.zeros((self.N_planes, self.N_modes, self.Ny, self.Nx), dtype=np.complex128)
        # backward_fields[N_modes-1] will be the output speckles
        self.backward_fields = np.zeros((self.N_planes, self.N_modes, self.Ny, self.Nx), dtype=np.complex128)

        self.show = self.show_field_intensity
        self.prop = self.propagate_freespace

    def set_input_spots_modes(self, sig=0.1, N_rows=4, N_cols=4, spacing=0.6):
        # TODO: Gaussian normalization. The power currently does not sum to 1
        mode_no = 0
        for i in range(N_rows):
            for j in range(N_cols):
                C = 1 / (sig ** 2 * 2 * np.pi)
                X0 = spacing * (j - (N_cols / 2.0) + 0.5)
                Y0 = spacing * (i - (N_rows / 2.0) + 0.5)
                E_gaus = C * np.exp(-((self.XX - X0) ** 2 + (self.YY - Y0) ** 2) / (sig ** 2)).astype(complex)
                self.forward_fields[0, mode_no, :, :] = E_gaus
                mode_no += 1

    def set_output_speckle_modes(self, sig=0.05, diffuser_pix_size=0.025):
        for mode_no in range(self.N_modes):
            speckles = self.get_speckles(sig=sig, diffuser_pix_size=diffuser_pix_size)
            self.backward_fields[self.N_planes-1, mode_no, :, :] = speckles

    def get_speckles(self, sig=0.4, diffuser_pix_size=0.02):
        X0 = 0
        Y0 = 0
        E_gaus = np.exp(-((self.XX - X0) ** 2 + (self.YY - Y0) ** 2) / (sig ** 2)).astype(complex)

        N_pixs_x = int(self.Nx*self.dx / diffuser_pix_size)
        N_pixs_y = int(self.Ny*self.dy / diffuser_pix_size)
        A = 2 * np.pi * np.random.rand(N_pixs_y, N_pixs_x)
        # .shape conventions on numpy and cv2 is the opposite
        A2 = cv2.resize(A, self.XX.shape[::-1], interpolation=cv2.INTER_NEAREST)
        E_gaus *= np.exp(1j*A2)

        speckles = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E_gaus)))

        # cut around the middle "real" mask
        filtered_speckles = np.zeros_like(speckles)  # np.exp(1j*0) = 1
        filtered_speckles[self.active_slice] = speckles[self.active_slice]

        filtered_speckles /= ((np.abs(filtered_speckles))**2).sum()
        return filtered_speckles

    def find_phases(self):
        # Populate initial forward and backward fields in all planes.
        self.initialize_fields()

        for i in range(self.N_iterations):
            self.log(f'Iter num: {i}')

            # Given current fields, update mask 1. Then update forward field in mask 2, and update it, etc.
            # the backward fields don't need to be updated till we get to the last plane, N-1, where
            # we update field N, and the Mask N is updated in the first iteration of backwards loop
            # In reality we do not have an SLM at the measurement plane ("plane 11"), but since we don't care
            # about the phases there, we let the algorithm optimize the phases over there.
            for plane_no in range(self.N_planes - 1):
                self.update_mask(plane_no)
                # this should be done for all modes
                for mode_no in range(self.N_modes):
                    E = np.copy(self.forward_fields[plane_no, mode_no, :, :])
                    # regular prop is with + in the exponent
                    E *= np.exp(+1j*np.angle(self.masks[plane_no]))
                    self.forward_fields[plane_no+1, mode_no, :, :] = self.prop(E, self.L, backprop=False)

            # Same logic, but backwards. Last plane is 1, where field 0 is updated,
            # and mask 0 will be updated in next forward iteration.
            # starts from N-1 because that is the N'th element of the vector...
            for plane_no in range(self.N_planes - 1, 0, -1):
                self.update_mask(plane_no)
                for mode_no in range(self.N_modes):
                    E = np.copy(self.backward_fields[plane_no, mode_no, :, :])
                    # backward prop is with - in the exponent
                    E *= np.exp(-1j*np.angle(self.masks[plane_no]))
                    self.backward_fields[plane_no-1, mode_no, :, :] = self.prop(E, self.L, backprop=True)

    def initialize_fields(self):
        # propagate modes forwards and backwards, and record initial fields in each plane
        for plane_no in range(self.N_planes - 1):
            # Forward fields
            for mode_no in range(self.N_modes):
                self.forward_fields[plane_no + 1, mode_no, :, :] = self.prop(
                    self.forward_fields[plane_no, mode_no, :, :], self.L, backprop=False
                )

        # planes N-1->1
        for plane_no in range(self.N_planes-1, 0, -1):
            for mode_no in range(self.N_modes):
                self.backward_fields[plane_no - 1, mode_no, :, :] = self.prop(
                    self.backward_fields[plane_no, mode_no, :, :], self.L, backprop=True
                )

    def update_mask(self, plane_no):
        # note that we work vectorially here on fields in all modes
        # Focusing on this plane
        cur_mask = np.copy(np.exp(+1j*np.angle(self.masks[plane_no])))  # [N_x, N_y]
        self.log(f'{cur_mask.shape=}', 1)
        F_fields = np.copy(self.forward_fields[plane_no, :, :, :])  # [N_modes, N_x, N_y]
        B_fields = np.copy(self.backward_fields[plane_no, :, :, :])  # [N_modes, N_x, N_y]
        self.log(f'{F_fields.shape=}', 1)

        # normalize each mode to its total power, to give a fighting chance
        # Power may be lost in k-space filter
        F_powers = (np.abs(F_fields)**2).sum(axis=(1, 2))  # [N_modes]
        B_powers = (np.abs(F_fields)**2).sum(axis=(1, 2))  # [N_modes]
        F_powers = F_powers[:, np.newaxis, np.newaxis]  # for dividing (N_modes, Nx, Ny) by (N_modes)
        B_powers = B_powers[:, np.newaxis, np.newaxis]  # for dividing (N_modes, Nx, Ny) by (N_modes)
        self.log(f'{F_powers.shape=}', 1)
        F_fields /= np.sqrt(F_powers)
        B_powers /= np.sqrt(B_powers)
        self.log(f'{F_fields.shape=}', 1)

        # Again, per each normalized mode, "combine" the fields
        # the phase of this combined field (c_field) is the desired phase mask (per mode) on the SLM
        # (c_phase + F_phase = B_phase, as desired)
        # # amplitude of c_field (per mode) is the desired weight for each mode. When it is dark -
        # it is not so important for this mode what the phase will be.
        c_fields = np.conj(F_fields) * B_fields  # [N_modes, N_x, N_y]
        self.log(f'{c_fields.shape=}', 1)

        # To perform the weighted sum, we sum (pixel-wise) the complex numbers of all the modes. Thinking with phasors,
        # this is exactly the desired weighted sum.
        # however, we add some probably redundant dark magic: we try and make the change from the previous mask to be
        # small, so find a global phase to add for each of the phase masks of the different modes.
        d_phi = (c_fields * cur_mask).sum(axis=(1, 2))  # [N_modes]
        d_phi = d_phi[:, np.newaxis, np.newaxis]  # for dividing (N_modes, Nx, Ny) by (N_modes)
        self.log(f'{d_phi.shape=}', 1)
        # new_mask are now rather large complex numbers
        new_mask = (c_fields*np.exp(-1j*np.angle(d_phi))).sum(axis=0)  # [N_x, N_y]
        self.log(f'{new_mask.shape=}', 1)

        # A small offset that is added to the mask just before the phase-only is
        # taken. This discourages the simulation from phase-matching low-intensity
        # parts of the field, and encourages solutions which are higher-bandwidth,
        # smoother and with less scatter. The Nx.*Ny.*modeCount normalization tries
        # to keep the value consistent even if the resolution or number of modes is
        # changed.
        if self.use_mask_offset:
            new_mask += self.mask_offset

        # make phase mask really a phase mask with no varying amplitude
        new_mask = np.exp(+1j * np.angle(new_mask))

        fixed_mask = self.fix_mask(new_mask)

        # ensure phase mask really a phase mask with no varying amplitude
        self.masks[plane_no] = np.exp(+1j * np.angle(fixed_mask))

    def fix_mask(self, mask):
        new_mask = np.copy(mask)

        # TODO: after calculation of each mask - do the zeroing, and also filter in k-space the e^(i*phi)
        if self.k_space_filter:
            pass

        # 3) remove padding from size factor
        if self._size_factor != 1:
            filtered_mask = np.ones_like(new_mask)  # np.exp(1j*0) = 1
            filtered_mask[self.active_slice] = new_mask[self.active_slice]
            new_mask = filtered_mask

        return new_mask

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

        return k_z

    def propagate_freespace(self, E, L, backprop=False):
        assert E.shape == self.XX.shape, 'Bad shape for E'
        E2 = np.copy(E)

        sign = 1 if not backprop else -1

        E_K = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E2)))
        # Apply the transfer function of free-space, see Fourier Optics page 74
        # normal forward motion is with + in the exponent
        E_K *= np.exp(+1j * sign * self.k_z_mat * L)
        E_out = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(E_K)))
        return E_out

    @staticmethod
    def show_field_intensity(E, ax=None, fig_show=True):
        # TODO: colorize
        if not ax:
            fig, ax = plt.subplots()
        im0 = ax.imshow(np.abs(E) ** 2)
        ax.figure.colorbar(im0, ax=ax)
        if fig_show:
            ax.figure.show()

    def log(self, txt, level=3):
        if level >= self.min_log_level:
            print(txt)

    def show_all(self, mode_no=0):
        fig, axes = plt.subplots(3, self.N_planes)
        for plane_no in range(self.N_planes):
            im = axes[0, plane_no].imshow(np.angle(self.masks[plane_no]), cmap='gray')
            fig.colorbar(im, ax=axes[0, plane_no])

            im = axes[1, plane_no].imshow(np.abs(self.forward_fields[plane_no, mode_no, :, :])**2)
            fig.colorbar(im, ax=axes[1, plane_no])
            axes[1, plane_no].set_title(f'forward {plane_no=}')

            im = axes[2, plane_no].imshow(np.abs(self.backward_fields[plane_no, mode_no, :, :])**2)
            fig.colorbar(im, ax=axes[2, plane_no])
            axes[2, plane_no].set_title(f'backward {plane_no=}')

        fig.show()


N_N_modes = 1
# All in mm
conf = {'wavelength': 810e-6,  # mm
        'plane_spacing': 87,  # mm
        'N_planes': 4,
        'N_iterations': 5,
        'Nx': 140,  # Number of grid points x-axis
        'Ny': 180,  # Number of grid points y-axis
        'dx': 12.5e-3,  # mm - SLM pixel sizes
        'dy': 12.5e-3,  # mm
        'k_space_filter': 0.15,
        'N_modes': N_N_modes*N_N_modes,
        'min_log_level': 2,
        'size_factor': 3,  # assumed to be odd. Have physical larger grid than the actual SLM planes
        'use_mask_offset': True,
        }

mplc = MPLC(conf=conf)
mplc.set_input_spots_modes(sig=0.1, N_rows=N_N_modes, N_cols=N_N_modes, spacing=0.6)
mplc.set_output_speckle_modes(sig=0.25, diffuser_pix_size=0.05)
# mplc.show_field_intensity(mplc.get_speckles())
# mplc.initialize_fields()
# mplc.find_phases()
# mplc.show_all()
