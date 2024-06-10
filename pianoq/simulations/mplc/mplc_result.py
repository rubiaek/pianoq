import traceback
import numpy as np
import matplotlib.pyplot as plt


class MPLCResult:
    def __init__(self, conf=None):
        self.conf = conf or {}
        self.masks = np.array([])
        self.forward_fields = np.array([])
        self.backward_fields = np.array([])
        self.active_slice = None
        self.N_modes = 0
        self.__dict__.update(self.conf)
        self.N_planes = len(self.conf['active_planes'])

        self.forward_fidelity = np.zeros((self.N_modes, self.N_modes), dtype=np.complex128)
        self.backward_fidelity = np.zeros((self.N_modes, self.N_modes), dtype=np.complex128)
        self.forward_losses = np.zeros(self.N_modes)
        self.backward_losses = np.zeros(self.N_modes)

    def _calc_fidelity(self):
        # TODO: have a finer dx for reality grid, with each SLM pixel being 2X2 reality pixels etc.?
        for in_mode in range(self.N_modes):
            for out_mode in range(self.N_modes):
                forward_fidelity = (self.forward_fields[self.N_planes-1, in_mode] *
                                    self.backward_fields[self.N_planes-1, out_mode].conj()
                                    ).sum()
                self.forward_fidelity[in_mode, out_mode] = forward_fidelity

                backward_fidelity = (self.backward_fields[0, in_mode] *
                                     self.forward_fields[0, out_mode].conj()
                                     ).sum()
                self.backward_fidelity[in_mode, out_mode] = backward_fidelity

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

    def show_all(self, mode_no=0, only_active_slice=True):
        fig, axes = plt.subplots(3, self.N_planes)
        for plane_no in range(self.N_planes):
            mask = self.masks[plane_no][self.active_slice] if only_active_slice else self.masks[plane_no]
            imm = axes[0, plane_no].imshow(np.angle(mask), cmap='gray')
            fig.colorbar(imm, ax=axes[0, plane_no])

            im = self.forward_fields[plane_no, mode_no][self.active_slice] if only_active_slice else self.forward_fields[plane_no, mode_no]
            imm = axes[1, plane_no].imshow(np.abs(im)**2)
            fig.colorbar(imm, ax=axes[1, plane_no])
            axes[1, plane_no].set_title(f'forward {plane_no=}')

            im = self.backward_fields[plane_no, mode_no][self.active_slice] if only_active_slice else self.backward_fields[plane_no, mode_no]
            imm = axes[2, plane_no].imshow(np.abs(im)**2)
            fig.colorbar(imm, ax=axes[2, plane_no])
            axes[2, plane_no].set_title(f'backward {plane_no=}')

        fig.show()

    def loadfrom(self, path):
        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        for k, v in data.items():
            self.k = v
        f.close()
        self.__dict__ = data

    def saveto(self, path):
        try:
            f = open(path, 'wb')
            np.savez(f, **self.__dict__)
            f.close()
        except Exception as e:
            print(e)
            traceback.print_exc()
