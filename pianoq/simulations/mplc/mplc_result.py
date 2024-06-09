import numpy as np
import matplotlib.pyplot as plt


class MPLCResult:
    def __init__(self, conf=None):
        self.conf = conf or {}
        self.masks = np.array([])
        self.forward_fields = np.array([])
        self.backward_fields = np.array([])
        self.active_slice = None
        self.__dict__.update(self.conf)
        self.N_planes = len(self.conf['active_planes'])

        # TODO: have a finer dx for reality grid, with each SLM pixel being 2X2 reality pixels etc.?
        # TODO: calculate forward & backward fidelity matrix

    @staticmethod
    def show_field_intensity(E, ax=None, fig_show=True):
        # TODO: colorize
        if not ax:
            fig, ax = plt.subplots()
        im0 = ax.imshow(np.abs(E) ** 2)
        ax.figure.colorbar(im0, ax=ax)
        if fig_show:
            ax.figure.show()

    def show_all(self, mode_no=0):
        fig, axes = plt.subplots(3, self.N_planes)
        for plane_no in range(self.N_planes):
            im = axes[0, plane_no].imshow(np.angle(self.masks[plane_no][self.active_slice]), cmap='gray')
            fig.colorbar(im, ax=axes[0, plane_no])

            im = axes[1, plane_no].imshow(np.abs(self.forward_fields[plane_no, mode_no][self.active_slice])**2)
            fig.colorbar(im, ax=axes[1, plane_no])
            axes[1, plane_no].set_title(f'forward {plane_no=}')

            im = axes[2, plane_no].imshow(np.abs(self.backward_fields[plane_no, mode_no][self.active_slice])**2)
            fig.colorbar(im, ax=axes[2, plane_no])
            axes[2, plane_no].set_title(f'backward {plane_no=}')

        fig.show()

    def loadfrom(self, path):
        # TODO
        pass

    def saveto(self, path):
        # TODO
        pass
