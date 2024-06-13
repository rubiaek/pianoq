import numpy as np
from tqdm import tqdm
from pianoq.simulations.mplc.mplc_result import MPLCResult


class MPLC:
    """
        This class is mainly for performing the wavefront matching protocol, but I will also use it in general
        for propagating fields through many phase masks
    """
    def __init__(self, conf):
        self.res = MPLCResult(conf)

        self.wl = self.wavelength = conf['wavelength']
        self.k = 2 * np.pi / self.wl
        self.dist_after_plane = conf['dist_after_plane']
        self.active_planes = conf['active_planes']
        self.N_planes = len(self.dist_after_plane) + 1
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
        self.res.active_slice = np.index_exp[self._active_y_start: self._active_y_end,
                                             self._active_x_start: self._active_x_end]

        self.dx = conf['dx']
        self.dy = conf['dy']
        self.min_log_level = conf['min_log_level']

        self.max_k_constraint = conf['max_k_constraint']
        self.use_mask_offset = conf['use_mask_offset']
        # TODO: this does not make sense, N_modes has opposite effect than Nx, Ny
        self.mask_offset = np.sqrt(1e-3/(self.Nx * self.Ny * self.N_modes))

        self.X = np.arange(-self.Nx / 2, self.Nx / 2) * self.dx
        self.Y = np.arange(-self.Ny / 2, self.Ny / 2) * self.dy
        self.XX, self.YY = np.meshgrid(self.X, self.Y)
        self.freq_x = np.fft.fftshift(np.fft.fftfreq(self.Nx, d=self.dx))
        self.freq_y = np.fft.fftshift(np.fft.fftfreq(self.Ny, d=self.dy))
        self.freq_XXs, self.freq_YYs = np.meshgrid(self.freq_x, self.freq_y)
        self.light_k = 2 * np.pi / self.wl
        self.k_xx = self.freq_XXs * 2 * np.pi
        self.k_yy = self.freq_YYs * 2 * np.pi


        self.k_z_mat = self._generate_kz_mat()
        self.k_constraint = self._generate_k_constraint()

        # masks are always exp(i*\phi(x, y))
        self.res.masks = np.exp(1j*np.zeros((self.N_planes, self.Ny, self.Nx), dtype=np.complex128))
        # forward_fields[0] will be the input spots
        self.res.forward_fields = np.zeros((self.N_planes, self.N_modes, self.Ny, self.Nx), dtype=np.complex128)
        # backward_fields[N_modes-1] will be the output speckles
        self.res.backward_fields = np.zeros((self.N_planes, self.N_modes, self.Ny, self.Nx), dtype=np.complex128)

        self.prop = self.propagate_freespace

    def set_modes(self, input_modes, output_modes):
        self.res.forward_fields[0, :, :, :] = input_modes
        self.res.backward_fields[self.N_planes - 1, :, :, :] = output_modes

    def find_phases(self, iterations=None):
        # Running with iterations = 1 will result with only field initialization
        iterations = iterations or self.N_iterations
        for iter_no in tqdm(range(iterations)):
            # Given current fields, update mask 1. Then update forward field in mask 2, and update it, etc.
            # the backward fields don't need to be updated till we get to the last plane, N-1, where
            # we update field N, and the Mask N is updated in the first iteration of backwards loop
            # In reality we do not have an SLM at the measurement plane ("plane 11"), but since we don't care
            # about the phases there, we let the algorithm optimize the phases over there.
            for plane_no in range(self.N_planes - 1):
                # In first iteration we populate the initial fields for each mode in each plane
                if iter_no != 0:
                    self.update_mask(plane_no)
                # this should be done for all modes
                for mode_no in range(self.N_modes):
                    E = np.copy(self.res.forward_fields[plane_no, mode_no, :, :])
                    # regular prop is with + in the exponent
                    E *= np.exp(+1j*np.angle(self.res.masks[plane_no]))
                    self.res.forward_fields[plane_no+1, mode_no, :, :] = self.prop(E,
                                                                                   self.dist_after_plane[plane_no],
                                                                                   backprop=False)

            # Same logic, but backwards. Last plane is 1, where field 0 is updated,
            # and mask 0 will be updated in next forward iteration.
            # starts from N-1 because that is the N'th element of the vector...
            for plane_no in range(self.N_planes - 1, 0, -1):
                # In first iteration we populate the initial fields for each mode in each plane
                if iter_no != 0:
                    self.update_mask(plane_no)
                for mode_no in range(self.N_modes):
                    E = np.copy(self.res.backward_fields[plane_no, mode_no, :, :])
                    # backward prop is with - in the exponent
                    E *= np.exp(-1j*np.angle(self.res.masks[plane_no]))
                    self.res.backward_fields[plane_no-1, mode_no, :, :] = self.prop(E,
                                                                                    # when at plane i, want distance
                                                                                    # after plane i-1
                                                                                    self.dist_after_plane[plane_no-1],
                                                                                    backprop=True)

    def update_mask(self, plane_no):
        # some planes have constant phase mask (lens, next to lens, etc.)
        if not self.active_planes[plane_no]:
            return

        # note that we work vectorially here on fields in all modes
        # Focusing on this plane
        cur_mask = np.copy(np.exp(+1j*np.angle(self.res.masks[plane_no])))  # [N_x, N_y]
        self.log(f'{cur_mask.shape=}', 1)
        F_fields = np.copy(self.res.forward_fields[plane_no, :, :, :])  # [N_modes, N_x, N_y]
        B_fields = np.copy(self.res.backward_fields[plane_no, :, :, :])  # [N_modes, N_x, N_y]
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
        self.res.masks[plane_no] = np.exp(+1j * np.angle(fixed_mask))

    def fix_mask(self, mask):
        new_mask = np.copy(mask)

        if self.max_k_constraint:
            mask_kspace = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(new_mask)))
            mask_kspace = mask_kspace * self.k_constraint
            new_mask = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(mask_kspace)))

        # 3) remove padding from size factor
        if self._size_factor != 1:
            filtered_mask = np.ones_like(new_mask)  # np.exp(1j*0) = 1
            filtered_mask[self.res.active_slice] = new_mask[self.res.active_slice]
            new_mask = filtered_mask

        return new_mask

    def _generate_kz_mat(self):
        # Calculating once for efficiency
        k_z_sqr = self.light_k ** 2 - (self.k_xx ** 2 + self.k_yy ** 2)
        # Remove all the negative component, as they represent evanescent waves, see Fourier Optics page 58
        np.maximum(k_z_sqr, 0, out=k_z_sqr)
        k_z = np.sqrt(k_z_sqr)

        return k_z

    def _generate_k_constraint(self):
        # max freq is constant for constant pixel size, and max_k_constraint is determined by / related to this size,
        # so it makes sense to normalize by it, but it doesn't really matter, just changes the magic number
        k_constraint = np.sqrt(((self.freq_XXs**2 + self.freq_YYs**2) / (self.freq_XXs**2 + self.freq_YYs**2).max())) \
                       < self.max_k_constraint
        return k_constraint

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

    def log(self, txt, level=3):
        if level >= self.min_log_level:
            print(txt)


# TODO:
#  Further future:
#  * start actual scaling experiments with "SLMs" in planes 1, 7
#  * see under MPLCResult
