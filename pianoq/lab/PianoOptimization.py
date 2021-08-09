import numpy as np

from pianoq.lab.Edac40 import Edac40
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.misc.borders import Borders

import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx


class PianoOptimization(object):
    def __init__(self):
        self.dac = Edac40(max_piezo_voltage=30, ip=Edac40.DEFAULT_IP)
        self.cam = VimbaCamera(2, exposure_time=500)
        borders = Borders(0, 0, 1280, 1024)
        self.cam.set_borders(borders)
        self.window_size = 3
        self.roi = np.index_exp[600 - self.window_size: 600 + self.window_size,
                                400 - self.window_size: 400 + self.window_size]

        self.optimizer = None
        self.best_cost = None
        self.best_pos = None

        self.good_piezo_indexes = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        # self.good_piezo_indexes = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        #                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
        self.num_good_piezos = len(self.good_piezo_indexes)

    def optimize_pso(self):
        options = {'c1': 1.5, 'c2': 2, 'w': 0.95}
        lower_bound = np.zeros(self.num_good_piezos)
        upper_bound = np.ones(self.num_good_piezos)
        bounds = (lower_bound, upper_bound)

        self.optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=self.num_good_piezos,
                                                 options=options, bounds=bounds)
        # Perform optimization
        self.best_cost, self.best_pos = self.optimizer.optimize(self.vectorial_cost_function, iters=20)

    def vectorial_cost_function(self, amps_times_n_particles):
        res = []
        for amps in amps_times_n_particles:
            assert len(amps) == self.num_good_piezos
            r = self.cost_function(amps)
            res.append(r)

        return np.array(res)

    def cost_function(self, amps):
        """
        amps is the array from the optimization algorithm, with only working amps
        :param amps:
        :return:
        """
        real_amps = np.zeros(40)
        real_amps[self.good_piezo_indexes] = amps
        self.dac.set_amplitudes(real_amps)

        # TODO: change exposure time dynamically like in optimize.py
        im = self.cam.get_image()[self.roi]
        cost = -int(im.sum())
        print(f"cost: {cost}")
        return cost

    def close(self):
        self.cam.close()
        self.dac.close()
