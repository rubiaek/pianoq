import datetime
import traceback
import numpy as np
import matplotlib.pyplot as plt
import copy

from torch.jit.frontend import build_ignore_context_manager


class MPLCSimResult:
    def __init__(self, conf=None):
        self.conf = conf or {}
        self.masks = np.array([], dtype=np.complex64)
        self.forward_fields = np.array([], dtype=np.complex64)
        self.backward_fields = np.array([], dtype=np.complex64)
        self.active_slice = None
        self.N_modes = 0
        self.__dict__.update(self.conf)
        self.N_planes = len(self.conf.get('active_planes', ()))

        self.forward_overlap = np.array([], dtype=np.complex64)
        self.backward_overlap = np.array([], dtype=np.complex64)
        self.forward_losses = np.zeros(self.N_modes)
        self.backward_losses = np.zeros(self.N_modes)
        self.fidelity = 0

    def _calc_fidelity(self):
        self._calc_normalized_overlap()
        M = self.forward_overlap
        U = np.eye(self.N_modes)
        numerator = abs(np.trace(np.dot(U, np.conj(M).T)))
        denominator = np.sqrt(abs(np.trace(np.dot(U, np.conj(U).T)) * np.trace(np.dot(M, np.conj(M).T))))
        self.fidelity = numerator / denominator

    def _calc_normalized_overlap(self):
        self.forward_overlap = np.zeros((self.N_modes, self.N_modes), dtype=np.complex64)
        self.backward_overlap = np.zeros((self.N_modes, self.N_modes), dtype=np.complex64)

        # TODO: have a finer dx for reality grid, with each SLM pixel being 2X2 reality pixels etc.?
        for in_mode in range(self.N_modes):
            for out_mode in range(self.N_modes):
                desired_field = self.backward_fields[-1, out_mode]
                expected_field = self.forward_fields[-1, in_mode]

                # Normalize
                desired_field = desired_field / np.sqrt(np.sum(np.abs(desired_field) ** 2))
                expected_field = expected_field / np.sqrt(np.sum(np.abs(expected_field) ** 2))

                forward_overlap = (expected_field * desired_field.conj()).sum()
                self.forward_overlap[in_mode, out_mode] = forward_overlap

                desired_field = self.backward_fields[0, in_mode]
                expected_field = self.forward_fields[0, out_mode]

                # Normalize
                desired_field = desired_field / np.sqrt(np.sum(np.abs(desired_field) ** 2))
                expected_field = expected_field / np.sqrt(np.sum(np.abs(expected_field) ** 2))

                backward_overlap = (expected_field * desired_field.conj()).sum()
                self.backward_overlap[in_mode, out_mode] = backward_overlap


    def _calc_loss(self):
        # TODO: check this
        forward_losses = np.zeros(self.N_modes)
        for mode_no in range(self.N_modes):
            forward_losses[mode_no] = (np.abs(self.forward_fields[-1, mode_no]) ** 2)[self.active_slice].sum()

        backward_losses = np.zeros(self.N_modes)
        for mode_no in range(self.N_modes):
            backward_losses[mode_no] = (np.abs(self.backward_fields[0, mode_no]) ** 2)[self.active_slice].sum()

    def show_overlap(self):
        self._calc_normalized_overlap()
        fig, axes = plt.subplots(1, 2, constrained_layout=True)
        imm = axes[0].imshow(np.abs(self.forward_overlap))
        fig.colorbar(imm, ax=axes[0])
        axes[0].set_title('Forward overlap')
        axes[0].set_ylabel('Input modes')
        axes[0].set_xlabel('Output modes')

        for i in range(self.forward_overlap.shape[0]):
            for j in range(self.forward_overlap.shape[1]):
                axes[0].text(j, i, f'{np.abs(self.forward_overlap[i, j]):.2f}', ha='center', va='center', color='white')

        imm = axes[1].imshow(np.abs(self.backward_overlap))
        fig.colorbar(imm, ax=axes[1])
        axes[1].set_title('Backward overlap')
        axes[1].set_xlabel('Input modes')
        axes[1].set_ylabel('Output modes')
        for i in range(self.backward_overlap.shape[0]):
            for j in range(self.backward_overlap.shape[1]):
                axes[1].text(j, i, f'{np.abs(self.backward_overlap[i, j]):.2f}', ha='center', va='center', color='white')

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
        fig.show()

    def show_masks(self):
        fig, axes = plt.subplots(1, self.N_planes, figsize=(13, 8), constrained_layout=True)
        for plane_no in range(self.N_planes):
            mask = self.masks[plane_no][self.active_slice]
            imm = axes[plane_no].imshow(np.angle(mask), cmap='gray')
            fig.colorbar(imm, ax=axes[plane_no])
            axes[plane_no].tick_params(axis='both', which='both', left=False, bottom=False, labelleft=False,
                                       labelbottom=False)
        fig.show()

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
            new_masks = np.zeros((len(self.masks), *self.forward_fields.shape[-2:]), dtype=np.complex64)
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

    def save_masks(self, path):
        m = MPLCMasks()
        m.masks = self.masks[:10, self.active_slice[0], self.active_slice[1]]
        m.conf = self.conf
        m.active_slice = self.active_slice
        m.saveto(path)


class MPLCMasks:
    def __init__(self, masks=None, conf=None):
        self.conf = conf or None
        self.timestamp = None
        self.masks = masks or np.array([], dtype=np.complex64)
        self.active_slice = None

    def saveto(self, path):
        f = open(path, 'wb')
        # N_masks, Ny3, Nx3 = self.masks.shape
        np.savez(f,
                 masks=self.masks,
                 timestamp=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'),
                 conf=self.conf,
                 active_slice=self.active_slice)
        f.close()

    def loadfrom(self, path):
        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        self.masks = data['masks']
        self.timestamp = data['timestamp']
        self.conf = data['conf']
        self.active_slice = data['active_slice']
        f.close()

    @property
    def big_masks(self):
        m_shape = self.masks.shape
        new_masks = np.zeros((m_shape[0], m_shape[1]*3, m_shape[2]*3), dtype=np.complex64)
        new_masks[:, m_shape[1]: 2 * m_shape[1],
                     m_shape[2]: 2 * m_shape[2]] = self.masks

        return new_masks
