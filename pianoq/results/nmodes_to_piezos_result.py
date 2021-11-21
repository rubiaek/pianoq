import pickle
import matplotlib.pyplot as plt
import numpy as np


class PiezosForSpecificModeResult(object):
    def __init__(self):
        self.Nmodes = None
        self.piezo_nums = []
        self.costs = []
        self.cost_stds = []
        self.example_befores = []
        self.example_afters = []

    def show_before_after(self, index):
        fig, axes = plt.subplots(2, 1, figsize=(5, 5.8), constrained_layout=True)

        piezo_num = self.piezo_nums[index]
        pix1_before, pix2_before = self.example_befores[index]
        pix1_after, pix2_after = self.example_afters[index]

        pixs_before = np.concatenate((pix1_before, pix2_before), axis=1)
        pixs_after = np.concatenate((pix1_after, pix2_after), axis=1)

        im0 = axes[0].imshow(np.abs(pixs_before)**2)
        im1 = axes[1].imshow(np.abs(pixs_after)**2)
        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
        axes[0].set_title('Before')
        axes[1].set_title('After')

        fig.suptitle(f'Nmodes={self.Nmodes}, piezo_num={piezo_num}')

        fig.show()


class NmodesToPiezosResult(object):
    def __init__(self):
        # list of PiezosForSpecificMode s
        self.different_modes = []
        self.version = 1.1
        self.timestamp = None
        self.cost_func = None
        self.normalize_TMs_method = None
        self.pso_n_pop = None
        self.pso_n_iterations = None
        self.pso_stop_after_n_const_iterations = None
        self.N_bends = None

    def __getattr__(self, attr: str):
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError

        value = self.__dict__.get(attr)
        if not value:
            print(f'{self.__class__.__name__}.{attr} is invalid in version {self.__dict__.get("version")}.')
            return None
        return value

    def saveto(self, path):
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()

    def loadfrom(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            self.__dict__ = obj.__dict__
            self.__class__ = obj.__class__

    def show_all_ratios(self):
        fig, ax = plt.subplots()
        ax.set_title(f'cost function: {self.cost_func}')
        ax.set_xlabel('piezo_num')
        if self.version == 1:
            ax.set_ylabel('percent in wanted polarization')
        elif self.version == 1.1:
            ax.set_ylabel('minus cost function')
        else:
            raise Exception()

        for r in self.different_modes:
            # For viewing while running when dimensions might not match
            l = len(r.costs)
            piezo_nums = r.piezo_nums[:l]

            ax.errorbar(piezo_nums, r.costs, yerr=r.cost_stds, fmt='.--', label=f'{r.Nmodes} modes')

        ax.legend()
        fig.show()
