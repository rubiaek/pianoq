import pickle
import datetime

import numpy as np
import matplotlib.pyplot as plt


class WavePlateOptimizationResult(object):
    def __init__(self, path=None):
        self.H_angles = None
        self.Q_angles = None
        self.heatmap = None

        self.path = path
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        if path:
            self.loadfrom(path)

    def saveto(self, path=None):
        f = open(path or self.path, 'wb')
        pickle.dump(self, f)
        f.close()

    def loadfrom(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            self.__dict__ = obj.__dict__
            self.__class__ = obj.__class__

    def show_heatmap(self):
        # different rows are different HWP angles
        # zero is on the top left
        fig, ax = plt.subplots()
        ax.set_title("Energy in H polarization")
        im = ax.imshow(self.heatmap, extent=[self.Q_angles[0], self.Q_angles[-1], self.H_angles[-1], self.H_angles[0]])
        ax.set_xlabel(r'QWP angle')
        ax.set_ylabel(r'HWP angle')
        fig.colorbar(im, ax=ax)
        fig.show()

    def show_cross_section(self):
        fig, ax = plt.subplots()
        ax.set_title("Cross section with most visibility")
        I = self.heatmap.argmax()
        i, j = np.unravel_index(I, self.heatmap.shape)

        cross_section = self.heatmap[:, j]
        ax.plot(self.H_angles, cross_section)
        ax.set_xlabel(r'HWP angle')
        ax.set_ylabel(r'intensity in H')
        fig.show()

    def show_heatmap_with_cross_section(self):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
        ax = axes[0]
        ax.set_title("Energy in H polarization")
        im = ax.imshow(self.heatmap, extent=[self.Q_angles[0], self.Q_angles[-1], self.H_angles[-1], self.H_angles[0]])
        ax.set_xlabel(r'QWP angle')
        ax.set_ylabel(r'HWP angle')
        fig.colorbar(im, ax=ax)

        ax = axes[1]
        ax.set_title("Cross section with most visibility")
        I = self.heatmap.argmax()
        i, j = np.unravel_index(I, self.heatmap.shape)

        cross_section = self.heatmap[:, j]
        ax.plot(self.H_angles, cross_section)
        ax.set_xlabel(r'HWP angle')
        ax.set_ylabel(r'intensity in H')
        fig.show()

