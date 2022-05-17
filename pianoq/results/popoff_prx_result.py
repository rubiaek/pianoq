import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from matplotlib.colors import LogNorm
from colorsys import hls_to_rgb
import traceback

cur_dir = os.path.dirname(os.path.abspath(__file__))


class PopoffPRXResult(object):
    """
    Built with data from github.com/wavefrontshaping/article_MMF_disorder
    And this PRX article https://arxiv.org/abs/2010.14813
    """
    DEFAULT_PATH = os.path.join(cur_dir, "../data/popoff_polarization_data.npz")
    DEFAULT_PATH2 = os.path.join(cur_dir, "../data/popoff_polarization_data_fmf2.npz")

    def __init__(self, TM_modes=None, dxs=None, index_dx0=None, modes_out=None, L=None, M=None, path=None):
        self.TM_modes = TM_modes  # Transmission matrices in the mode basis for different dx values
        self.dxs = dxs  # \mu m
        self.index_dx0 = index_dx0
        self.modes_out = modes_out
        self.modes_out_full = None
        self.L = L  # L value of the i'th mode
        self.M = M  # M value of the i'th mode
        self.path = path

        if self.path:
            self.loadfrom(path)

    def _initialize(self, method='TM'):
        # The modes of the fiber, converts from mode basis to pixel basis
        self.modes_out_full = np.kron(np.array([[1, 0], [0, 1]]), self.modes_out)
        self.n = int(np.sqrt(self.modes_out.shape[1]))
        self.Nmodes = self.modes_out.shape[0] * 2  # 110
        self.hNmodes = self.Nmodes // 2
        self.all_polarization_ratios = self.get_all_polarization_ratios(method=method)

    def _E_to_I(self, x):
        return np.abs(x) ** 2

    def show_mode(self, mode_num):
        mode = self.modes_out[mode_num]
        profile = mode.reshape([self.n] * 2)
        fig, ax = plt.subplots()

        # ax.imshow(_colorize(profile,'white'))
        ax.imshow(_colorize(profile))
        ax.axis('off')
        # ax.set_title(f'Mode {mode_num} (l={self.L[mode_num]}, m={self.M[mode_num]})')
        ax.set_title(f'Mode {mode_num}')
        fig.show()
        return fig

    def show_TM(self, TM):
        fig, ax = plt.subplots()
        ax.imshow(_colorize(TM, beta=1.8, max_threshold=0.8))
        fig.show()
        return fig

    def set_Nmodes(self, Nmodes=None):

        if Nmodes is None:
            return

        # measured TMs have 55 * 2 modes, but we might want to simulate fiber with less modes
        # Also, the higher modes are lossier and have pixelization issues
        # Degeneracy of modes goes like n so amount of modes should be for example 2*(1+2+3+4+5) etc.
        assert Nmodes in np.array([1, 3, 6, 10, 15, 21, 28, 36, 45, 55]) * 2

        hNmodes = Nmodes // 2  # Half Nmodes

        # remove 10 highest degenerate modes, since they are very lossy, and have pixelization effects
        mask = np.ones(self.TM_modes[0].shape, dtype=bool)
        mask[:, hNmodes:self.hNmodes] = False
        mask[hNmodes:self.hNmodes:, :] = False
        mask[hNmodes+self.hNmodes:, :] = False
        mask[:, hNmodes+self.hNmodes:] = False

        self.Nmodes = Nmodes
        self.hNmodes = hNmodes
        self.TM_modes = [TM[mask].reshape(Nmodes, Nmodes) for TM in self.TM_modes]

        # Shuold take care of this so pop.propagate() will work.
        self.modes_out = self.modes_out[:hNmodes, :]
        self.modes_out_full = np.kron(np.array([[1, 0], [0, 1]]), self.modes_out)

    def normalize_TMs(self, method='mean'):
        # In original the elements are ~10^-6, so after 30 TMS we get to really small...
        if method == 'mean':
            self.TM_modes = [T / np.sqrt(np.mean(np.sum(np.abs(T) ** 2, 0))) for T in self.TM_modes]
        elif method =='svd1':
            new_TMS = []
            for TM in self.TM_modes:
                u, s, v = scipy.linalg.svd(TM)
                new_TMS.append(u @ v)
            self.TM_modes = new_TMS
        else:
            raise NotImplementedError

    def show_all_polarizations_ratios(self):
        fig, axes = plt.subplots(2, figsize=(8, 7))
        p0_ratios, p1_ratios = np.split(self.all_polarization_ratios, 2)

        extent0 = (self.dxs[0], self.dxs[-1], self.Nmodes / 2, 1)
        im0 = axes[0].imshow(p0_ratios, norm=LogNorm(), extent=extent0)
        axes[0].set_xlabel(r"dx ($ \mu m $)")
        axes[0].set_ylabel("Modes")
        axes[0].set_aspect('auto')
        fig.colorbar(im0, ax=axes[0])

        extent1 = (self.dxs[0], self.dxs[-1], self.Nmodes, self.Nmodes / 2)
        im1 = axes[1].imshow(p1_ratios, norm=LogNorm(), extent=extent1)
        axes[1].set_xlabel(r"dx ($ \mu m $)")
        axes[1].set_ylabel("Modes")
        axes[1].set_aspect('auto')
        fig.colorbar(im1, ax=axes[1])

        fig.show()
        return fig

    def show_polarizations_ratios_per_mode(self, mode_indexes, method='TM', logscale=True, legend=False):
        fig, ax = plt.subplots(figsize=(8, 7))
        fig.suptitle(f"energy ratio pol. in / other pol.  method={method}")

        for i, index in enumerate(mode_indexes):
            ratios = self.all_polarization_ratios[index, :]
            label = fr'mode={index}'
            ax.plot(self.dxs, ratios, label=label)

        ax.axhline(y=2, linestyle='--')
        ax.axhline(y=0.5, linestyle='--')

        ax.set_xlabel("dx $ \mu m $")
        ax.set_ylabel("polarization ratio")
        if logscale:
            ax.set_yscale('log')
        if legend:
            ax.legend()
        fig.show()
        return fig

    def show_polarizations_ratios_bar_plots(self, dx_indexes, method='TM'):
        fig, axes = plt.subplots(2, figsize=(8, 7))
        fig.suptitle(f"energy ratio pol. in / other pol.  method={method}")

        tot_width = 0.8
        w = tot_width / len(dx_indexes)
        rel_w = 0.9

        for i, index in enumerate(dx_indexes):
            ratios = self.all_polarization_ratios[:, index]
            dx = self.dxs[index]
            label = fr'dx={dx:.2f} $ \mu m $'
            axes[0].bar(np.array(range(1, 55+1)) + w*i, ratios[:55], width=rel_w*w, label=label)
            axes[1].bar(np.array(range(56, 110+1)) + w*i, 1 / ratios[55:], width=rel_w*w, label=label)

        axes[0].axhline(y=1, linestyle='--')
        axes[1].axhline(y=1, linestyle='--')

        axes[1].set_xlabel("Modes")
        axes[0].set_ylabel("P1 in")
        axes[1].set_ylabel("P2 in")
        axes[0].set_ylim(0, 15)
        axes[1].set_ylim(0, 15)
        axes[0].legend()
        axes[1].legend()
        fig.show()
        return fig

    def show_mixing_of_mode(self, TM, mode_num):
        fig, ax = plt.subplots()
        ax.bar(range(1, TM.shape[0]+1), np.abs(TM[mode_num, :])**2)
        fig.show()
        return fig

    def propagate(self, in_modes, TM):
        '''
        in_modes is a vector of length Nmodes, with weights of incoming modes.
        It propagates the input field through the fiber, and returns the pixel map in both polarizations
        '''
        output_state = TM @ in_modes
        out_pix = self.modes_out_full.transpose() @ output_state
        out_pix_p1 = out_pix[:self.n ** 2].reshape([self.n] * 2)
        out_pix_p2 = out_pix[self.n ** 2:].reshape([self.n] * 2)

        return out_pix_p1, out_pix_p2

    def get_polarization_ratios(self, TM, method='TM'):
        """
        Method is either TM or pixel.
        TM is given transmission matrix
        returns array (110, ) of ratios for each mode
        """

        if method == 'TM':
            TM = TM.transpose()
            p1, p2 = np.hsplit(TM, 2)

            # To real numbers
            energy_p1 = np.abs(p1) ** 2
            energy_p2 = np.abs(p2) ** 2

            tot_energy_p1 = energy_p1.sum(axis=1)
            tot_energy_p2 = energy_p2.sum(axis=1)

            ratios = tot_energy_p1 / tot_energy_p2
            return ratios

        elif method == 'pixel':
            ratios = []
            for mode_on_index in range(self.Nmodes):
                state = np.zeros(self.Nmodes, dtype=complex)
                state[mode_on_index] = 1  # 0 or 55 for fundamental mode for instance

                out_pix_p1, out_pix_p2 = self.propagate(state, TM)
                E_to_I = lambda x: np.abs(x) ** 2
                out_pix_p1, out_pix_p2 = self._E_to_I(out_pix_p1), E_to_I(out_pix_p2)

                ratio = out_pix_p1.sum() / out_pix_p2.sum()

                ratios.append(ratio)
            return np.array(ratios)

    def get_all_polarization_ratios(self, method='TM'):
        """ method should be pixel or TM """
        A = np.zeros((self.Nmodes, len(self.dxs)))  # Each row for different mode. Columns propagate dxs
        for i, dx, TM in zip(range(len(self.dxs)), self.dxs, self.TM_modes):
            A[:, i] = self.get_polarization_ratios(TM, method=method)
            if i >= 54:
                A[:, i] = 1/A[:, i]

        return A

    def saveto(self, path):
        try:
            np.savez(path,
                     TM_modes=self.TM_modes,
                     dxs=self.dxs,
                     index_dx0=self.index_dx0,
                     modes_out=self.modes_out,
                     L=self.L,
                     M=self.M)
        except Exception as e:
            print("ERROR!!")
            print(e)
            traceback.print_exc()

    def loadfrom(self, path=None):
        # data = np.load(path, allow_pickle=True)
        path = path or self.DEFAULT_PATH
        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        self.TM_modes = data['TM_modes']
        self.dxs = data['dxs']
        self.index_dx0 = data['index_dx0']
        self.modes_out = data['modes_out']
        self.L = data['L']
        self.M = data['M']
        self.path = path

        # taking only a few TMs before starting the real perturbation, and from here dxs[i+1] - dxs[i] = const = 2\mum
        first_relevant_dx_index = self.index_dx0 - 5
        self.TM_modes = self.TM_modes[first_relevant_dx_index:, :, :]
        self.dxs = self.dxs[first_relevant_dx_index:]
        self.index_dx0 = self.index_dx0 - first_relevant_dx_index

        self._initialize()


def _colorize(z, theme='dark', saturation=1., beta=1.4, transparent=False, alpha=1., max_threshold=1.):
    r = np.abs(z)
    r /= max_threshold * np.max(np.abs(r))
    arg = np.angle(z)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1. / (1. + r ** beta) if theme == 'white' else 1. - 1. / (1. + r ** beta)
    s = saturation

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    if transparent:
        a = 1. - np.sum(c ** 2, axis=-1) / 3
        alpha_channel = a[..., None] ** alpha
        return np.concatenate([c, alpha_channel], axis=-1)
    else:
        return c


# pop = PopoffPolarizationRotationResult()
# pop.loadfrom("C:\\temp\\popoff_polarization_data4.npz")
# pop.show_TM(pop.TM_modes[pop.index_dx0])
# pop.show_polarizations_ratios_per_mode(range(0, 55, 2), logscale=True, legend=True)
# pop.show_polarizations_ratios_bar_plots([0, 20, 30, 40])
