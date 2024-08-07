import traceback
import numpy as np
import matplotlib.pyplot as plt
import copy


class MPLCSimResult:
    def __init__(self, conf=None):
        self.conf = conf or {}
        self.masks = np.array([])
        self.forward_fields = np.array([], dtype=np.complex128)
        self.backward_fields = np.array([], dtype=np.complex128)
        self.active_slice = None
        self.N_modes = 0
        self.__dict__.update(self.conf)
        self.N_planes = len(self.conf.get('active_planes', ()))

        self.forward_fidelity = np.array([], dtype=np.complex128)
        self.backward_fidelity = np.array([], dtype=np.complex128)
        self.forward_losses = np.zeros(self.N_modes)
        self.backward_losses = np.zeros(self.N_modes)

    def _calc_fidelity(self):
        self.forward_fidelity = np.zeros((self.N_modes, self.N_modes), dtype=np.complex128)
        self.backward_fidelity = np.zeros((self.N_modes, self.N_modes), dtype=np.complex128)

        # TODO: have a finer dx for reality grid, with each SLM pixel being 2X2 reality pixels etc.?
        for in_mode in range(self.N_modes):
            for out_mode in range(self.N_modes):
                forward_fidelity = (self.forward_fields[-1, in_mode] *
                                    self.backward_fields[-1, out_mode].conj()
                                    ).sum()
                self.forward_fidelity[in_mode, out_mode] = forward_fidelity

                backward_fidelity = (self.backward_fields[0, in_mode] *
                                     self.forward_fields[0, out_mode].conj()
                                     ).sum()
                self.backward_fidelity[in_mode, out_mode] = backward_fidelity

    def _calc_loss(self):
        # TODO: check this
        forward_losses = np.zeros(self.N_modes)
        for mode_no in range(self.N_modes):
            forward_losses[mode_no] = (np.abs(self.forward_fields[-1, mode_no]) ** 2)[self.active_slice].sum()

        backward_losses = np.zeros(self.N_modes)
        for mode_no in range(self.N_modes):
            backward_losses[mode_no] = (np.abs(self.backward_fields[0, mode_no]) ** 2)[self.active_slice].sum()

    def show_fidelity(self):
        self._calc_fidelity()
        fig, axes = plt.subplots(1, 2, constrained_layout=True)
        imm = axes[0].imshow(np.abs(self.forward_fidelity))
        fig.colorbar(imm, ax=axes[0])
        axes[0].set_title('Forward fidelity')
        axes[0].set_ylabel('Input modes')
        axes[0].set_xlabel('Output modes')

        for i in range(self.forward_fidelity.shape[0]):
            for j in range(self.forward_fidelity.shape[1]):
                axes[0].text(j, i, f'{np.abs(self.forward_fidelity[i, j]):.2f}', ha='center', va='center', color='white')

        imm = axes[1].imshow(np.abs(self.backward_fidelity))
        fig.colorbar(imm, ax=axes[1])
        axes[1].set_title('Backward fidelity')
        axes[1].set_xlabel('Input modes')
        axes[1].set_ylabel('Output modes')
        for i in range(self.backward_fidelity.shape[0]):
            for j in range(self.backward_fidelity.shape[1]):
                axes[1].text(j, i, f'{np.abs(self.backward_fidelity[i, j]):.2f}', ha='center', va='center', color='white')

        fig.show()

    @staticmethod
    def show_field_intensity(E, ax=None, fig_show=True):
        # TODO: colorize
        if not ax:
            fig, ax = plt.subplots()
        im0 = ax.imshow(np.abs(E) ** 2)
        ax.figure.colorbar(im0, ax=ax)
        if fig_show:
            ax.figure.show()

    @property
    def input_modes(self):
        return self.forward_fields[0, :, :, :]

    @property
    def output_modes(self):
        return self.backward_fields[self.N_planes - 1, :, :, :]

    def show_mask(self, plane_no):
        fig, ax = plt.subplots()
        mask = self.masks[plane_no][self.active_slice]
        imm = ax.imshow(np.angle(mask), cmap='gray')
        fig.colorbar(imm, ax=ax)
        ax.tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False, labelbottom=False)

    def show_masks(self):
        fig, axes = plt.subplots(1, self.N_planes, figsize=(13, 8), constrained_layout=True)
        for plane_no in range(self.N_planes):
            mask = self.masks[plane_no][self.active_slice]
            imm = axes[plane_no].imshow(np.angle(mask), cmap='gray')
            fig.colorbar(imm, ax=axes[plane_no])
            axes[plane_no].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False,
                                       labelbottom=False)

    def show_all(self, mode_no=0, only_active_slice=True):
        fig, axes = plt.subplots(3, len(self.forward_fields), figsize=(13, 5), constrained_layout=True)
        for plane_no in range(len(self.forward_fields)):
            mask = self.masks[plane_no][self.active_slice] if only_active_slice else self.masks[plane_no]
            imm = axes[0, plane_no].imshow(np.angle(mask), cmap='gray')
            fig.colorbar(imm, ax=axes[0, plane_no])
            axes[0, plane_no].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False,
                                          labelbottom=False)

            im = self.forward_fields[plane_no, mode_no][self.active_slice] if only_active_slice else self.forward_fields[plane_no, mode_no]
            imm = axes[1, plane_no].imshow(np.abs(im)**2)
            fig.colorbar(imm, ax=axes[1, plane_no])
            axes[1, plane_no].set_title(f'forward {plane_no=}')
            axes[1, plane_no].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False,
                                          labelbottom=False)

            im = self.backward_fields[plane_no, mode_no][self.active_slice] if only_active_slice else self.backward_fields[plane_no, mode_no]
            imm = axes[2, plane_no].imshow(np.abs(im)**2)
            fig.colorbar(imm, ax=axes[2, plane_no])
            axes[2, plane_no].set_title(f'backward {plane_no=}')
            axes[2, plane_no].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False,
                                          labelbottom=False)

        fig.show()

    def loadfrom(self, path):
        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        for k, v in data.items():
            if v.shape == ():
                self.__dict__[k] = v.item()
            else:
                self.__dict__[k] = v
        self.active_slice = tuple(self.active_slice)
        if self.masks[0].shape != self.forward_fields.shape[-2:]:
            new_masks = np.zeros((len(self.masks), *self.forward_fields.shape[-2:]))
            m_shape = new_masks.shape
            new_masks[:, m_shape[1] // 3: 2*m_shape[1] // 3, m_shape[2] // 3: 2*m_shape[2] // 3] = self.masks
            self.masks = new_masks
        f.close()

    def saveto(self, path, smaller=True):
        try:
            f = open(path, 'wb')
            if smaller:
                # TODO: could reduce 9X the masks, since by definition only the middle is full, and the rest is zero
                d = copy.deepcopy(self.__dict__)
                ff = d.pop('forward_fields')
                bf = d.pop('backward_fields')
                d['forward_fields'] = [ff[0], ff[-1]]
                d['backward_fields'] = [bf[0], bf[-1]]

                m_shape = self.masks.shape
                new = self.masks[:, m_shape[1] // 3: 2 * m_shape[1] // 3, m_shape[2] // 3: 2 * m_shape[2] // 3]
                d['masks'] = new

                np.savez(f, **d)
                del d
            else:
               np.savez(f, **self.__dict__)
            f.close()
        except Exception as e:
            print(e)
            traceback.print_exc()
