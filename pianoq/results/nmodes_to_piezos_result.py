import pickle
import matplotlib.pyplot as plt
import numpy as np


class PiezosForSpecificModeResult(object):
    def __init__(self):
        self.Nmodes = None
        self.piezo_nums = []
        self.ratios = []
        self.ratio_stds = []
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
        self.version = 1
        self.timestamp = None
        self.cost_func = None
        self.normalize_TMs_method = None

    def saveto(self, path):
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()

    def loadfrom(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            self.__dict__ = obj.__dict__
            self.__class__ = obj.__class__

    def show_all(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('piezo_num')
        ax.set_ylabel('percent in wanted polarization')

        for r in self.different_modes:
            # For viewing while running when dimensions might not match
            l = len(r.ratios)
            piezo_nums = r.piezo_nums[:l]

            ax.errorbar(piezo_nums, r.ratios, yerr=r.ratio_stds, fmt='.--', label=f'{r.Nmodes} modes')

        ax.legend()
        fig.show()
