import numpy as np
import matplotlib.pyplot as plt

import traceback


class PianoPSOOptimizationResult(object):
    def __init__(self):
        # Results of experiment
        self.costs = []
        self.amplitudes = []
        self.images = []
        self.exposure_times = []
        self.timestamps = []

        # Dac params
        self.good_piezo_indexes = None
        self.max_piezo_voltage = None

        # Optimization params
        self.roi = None
        self.n_pop = None
        self.n_iterations = None
        self.stop_after_n_const_iters = None
        self.reduce_at_iterations = None

        self.normalized_images = []
        self.normaliztion_to_one = None
        self.random_average_cost = None
        self.cam_type = None

    def _get_normalized_images(self):
        norm_ims = []
        for i, im in enumerate(self.images):
            norm_im = im / self.exposure_times[i]
            norm_im = norm_im / self.normaliztion_to_one
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

    @property
    def enhancement(self):
        if self.random_average_cost:
            return np.min(self.costs) / self.random_average_cost
        else:
            return -1

    @property
    def power_in_L_Pol_before_after(self):
        im1 = self.normalized_images[0]
        middle = im1.shape[1] // 2
        percentage1 = im1[:, :middle].sum() / im1.sum()
        # print(f'initial percentage: {percentage1}')

        i = np.argmin(self.costs)
        im2 = self.normalized_images[i]
        percentage2 = im2[:, :middle].sum() / im2.sum()
        # print(f'final percentage: {percentage2}')

        return percentage1, percentage2

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
                     good_piezo_indexes=self.good_piezo_indexes,
                     max_piezo_voltage=self.max_piezo_voltage,
                     roi=self.roi,
                     n_pop=self.n_pop,
                     n_iterations=self.n_iterations,
                     stop_after_n_const_iters=self.stop_after_n_const_iters,
                     reduce_at_iterations=self.reduce_at_iterations,
                     cam_type=self.cam_type
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

        self.normaliztion_to_one = self.images[0].max() / self.exposure_times[0]
        self.normalized_images = self._get_normalized_images()

        self.good_piezo_indexes = data.get('good_piezo_indexes', None)
        self.max_piezo_voltage = data.get('max_piezo_voltage', None)

        self.roi = data.get('roi', None)
        self.n_pop = data.get('n_pop', None)
        self.n_iterations = data.get('n_iterations', None)
        self.stop_after_n_const_iters = data.get('stop_after_n_const_iters', None)
        self.reduce_at_iterations = data.get('reduce_at_iterations', None)
        self.cam_type = data.get('cam_type', None)
