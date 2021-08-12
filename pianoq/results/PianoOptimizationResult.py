import numpy as np
import matplotlib.pyplot as plt

import traceback


class PianoPSOOptimizationResult(object):
    def __init__(self):
        # TODO: add active piezos, active piezo amounts, piezo range, etc...

        self.costs = []
        self.amplitudes = []
        self.images = []
        self.exposure_times = []
        self.timestamps = []

        self.normalized_images = []
        self.normaloztion_to_one = None
        self.random_average_cost = None

        self.best_cost = None
        self.best_amps = None
        self.best_image = None

    def _get_normalized_images(self):
        norm_ims = []
        for i, im in enumerate(self.images):
            norm_im = im / self.exposure_times[i]
            norm_im = norm_im / self.normaloztion_to_one
            norm_ims.append(norm_im)
        return norm_ims

    def show_image(self, im, title=None):
        fig, ax = plt.subplots()
        im = ax.imshow(im)
        fig.colorbar(im, ax=ax)
        if title:
            ax.set_title(title)
        fig.show()
        return fig, ax

    def show_result(self):
        fig, axes = plt.subplots(2, 1, figsize=(5, 5.8), constrained_layout=True)
        im0 = axes[0].imshow(self.normalized_images[0])
        im1 = axes[1].imshow(self.normalized_images[-1])
        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
        axes[0].set_title('Before')
        axes[1].set_title('After')

        fig.show()

    def saveto(self, path):
        try:
            f = open(path, 'wb')
            np.savez(f,
                     costs=self.costs,
                     amplitudes=self.amplitudes,
                     images=self.images,
                     exposure_times=self.exposure_times,
                     timestamps=self.timestamps,
                     random_average_cost=self.random_average_cost,
                     best_cost=self.best_cost,
                     best_amps=self.best_amps,
                     best_image=self.best_image
                     )
            f.close()
        except Exception as e:
            print("ERROR!!")
            print(e)
            traceback.print_exc()

    def loadfrom(self, path):
        # path = path or self.DEFAULT_PATH
        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        self.costs = data['costs']
        self.amplitudes = data['amplitudes']
        self.images = data['images']
        self.exposure_times = data['exposure_times']
        self.timestamps = data['timestamps']
        self.random_average_cost = data.get('random_average_cost', None)
        if self.random_average_cost:
            self.random_average_cost = self.random_average_cost.item()

        self.normaloztion_to_one = self.images[0].max() / self.exposure_times[0]
        self.normalized_images = self._get_normalized_images()

        self.best_cost = data['best_cost']
        self.best_amps = data['best_amps']
        self.best_image = data['best_image']

