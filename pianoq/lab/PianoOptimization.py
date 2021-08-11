import time
import datetime
import numpy as np

from pianoq.lab.Edac40 import Edac40
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.lab.optimizations.my_pso import MyPSOOptimizer
from pianoq.misc.borders import Borders

from pianoq.results.PianoOptimizationResult import PianoPSOOptimizationResult

import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from scipy.optimize import differential_evolution
from sko.PSO import PSO

LOGS_DIR = 'C:\\temp'


class PianoOptimization(object):
    def __init__(self, initial_exposure_time=900, saveto_path=None):
        self.dac = Edac40(max_piezo_voltage=30, ip=Edac40.DEFAULT_IP)
        self.cam = VimbaCamera(2, exposure_time=initial_exposure_time)
        self.initial_exposure_time = initial_exposure_time
        self.scaling_exposure_factor = 1

        self.saveto_path = saveto_path
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        self.start_time = datetime.datetime.now()

        # TODO: this is kinf of silly... it will rather be better to communicate less bits with the camera...
        # Should probably get as parameter the (x, y) and then define the borders around that part
        borders = Borders(0, 0, 1280, 1024)
        self.cam.set_borders(borders)
        self.window_size = 1
        self.roi = np.index_exp[600 - self.window_size: 600 + self.window_size,
                                400 - self.window_size: 400 + self.window_size]

        self.optimizer = None
        self.best_cost = None
        self.best_pos = None

        # self.good_piezo_indexes = [2, 3, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.good_piezo_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                   20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
        self.num_good_piezos = len(self.good_piezo_indexes)

        self.current_micro_iter = 0
        self.current_macro_iter = 0
        self.res = PianoPSOOptimizationResult()  # TODO: add here good_piezos, and other general experimental parameters

    def optimize_pso(self):
        options = {'c1': 1.5, 'c2': 2, 'w': 0.99}
        lower_bound = np.zeros(self.num_good_piezos)
        upper_bound = np.ones(self.num_good_piezos)
        bounds = (lower_bound, upper_bound)

        self.optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=self.num_good_piezos,
                                                 options=options, bounds=bounds)
        # Perform optimization
        self.best_cost, self.best_pos = self.optimizer.optimize(self.vectorial_cost_function, iters=10)

    def optimize_differential_evollution(self):
        bounds = [[0, 1]] * self.num_good_piezos

        self.diff_res = differential_evolution(self.cost_function, bounds,
                                               strategy='best1bin', maxiter=10,
                                               popsize=10, recombination=0.5, mutation=(0.01, 0.1))

    def optimize_pso_sko(self):
        self.pso = PSO(func=self.cost_function, n_dim=self.num_good_piezos, pop=20, max_iter=20,
                       lb=[0]*self.num_good_piezos, ub=[1]*self.num_good_piezos, w=0.9, c1=1.5, c2=2)
        self.pso.run()

    def optimize_my_pso(self, n_pop, n_iterations, stop_after_n_const_iters, reduce_at_iterations=()):
        self.o = MyPSOOptimizer(self.cost_function, n_pop=n_pop, n_var=self.num_good_piezos, n_iterations=n_iterations,
                                post_iteration_callback=self.post_iteration_callback,
                                w=1, wdamp=0.99, c1=1.5, c2=2,
                                timeout=60*60*2,
                                stop_early=True, stop_after_n_const_iter=stop_after_n_const_iters,
                                vary_popuation=True, reduce_at_iterations=reduce_at_iterations)
        self.o.optimize()

    def vectorial_cost_function(self, amps_times_n_particles):
        # Was needed for optimize_pso from the pyswarms package
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
        real_amps = np.ones(40) * self.dac.REST_AMP
        real_amps[self.good_piezo_indexes] = amps
        self.dac.set_amplitudes(real_amps)

        im = self.cam.get_image()[self.roi]
        cost = -int(im.mean())
        cost *= self.scaling_exposure_factor
        print(f"{self.current_micro_iter}. cost: {cost}")
        self.current_micro_iter += 1

        self._fix_exposure(im)

        return cost

    def _fix_exposure(self, im):
        mx = im.max()
        if mx > 240:
            print('Lowering exposure time!')
            exp_time = self.cam.get_exposure_time()
            if exp_time * (4 / 5) > 45:
                self.cam.set_exposure_time(exp_time * (4 / 5))
                self.scaling_exposure_factor *= (5 / 4)
            else:
                print('**At shortest exposure and still saturated... You might want to add an ND to the camera..**')

    def post_iteration_callback(self, global_best_cost, global_best_positions):
        now = datetime.datetime.now()
        delta = now - self.start_time
        self.res.timestamps.append(delta.total_seconds())

        self.res.amplitudes.append(global_best_positions)
        cost = self.cost_function(global_best_positions)
        # TODO: think about this cost and the reported global_best_cost (see two lines below)

        # self.res.costs.append(global_best_cost)
        self.res.costs.append(cost)

        im = self.cam.get_averaged_image(amount=10)
        self.res.images.append(im)

        # self.o.default_post_iteration(global_best_cost, global_best_positions)
        self.current_macro_iter += 1
        print(f'{self.current_macro_iter}.\t cost: {global_best_cost}\t time: {delta.total_seconds()} seconds')
        self._save_result()

    def _save_result(self):
        saveto_path = self.saveto_path or f"{LOGS_DIR}\\{self.timestamp}.pqoptimizer"
        self.res.saveto(saveto_path)

    def close(self):
        self.cam.close()
        self.dac.close()
