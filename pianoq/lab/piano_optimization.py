import time
import datetime
import numpy as np

from pianoq.misc.borders import Borders
from pianoq.lab.Edac40 import Edac40
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.lab.optimizations.my_pso import MyPSOOptimizer
from pianoq.misc.consts import DEFAULT_BORDERS, DEFAULT_CAM_NO, DEFAULT_BORDERS2

from pianoq.results.piano_optimization_result import PianoPSOOptimizationResult

LOGS_DIR = 'C:\\temp'


class PianoOptimization(object):

    def __init__(self, initial_exposure_time=900, saveto_path=None, roi=None):
        self.dac = Edac40(max_piezo_voltage=70, ip=Edac40.DEFAULT_IP)
        self.cam = VimbaCamera(DEFAULT_CAM_NO, exposure_time=initial_exposure_time)
        self.initial_exposure_time = initial_exposure_time
        self.scaling_exposure_factor = 1
        # Should probably get as parameter the (x, y) and then define the borders around that part
        borders = DEFAULT_BORDERS
        # borders = Borders(330, 520, 800, 615)
        self.cam.set_borders(borders)

        self.saveto_path = saveto_path
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        self.start_time = datetime.datetime.now()
        self.window_size = 2
        self.roi = roi or np.index_exp[150 - self.window_size: 150 + self.window_size,
                                       150 - self.window_size: 150 + self.window_size]

        self.optimizer = None
        self.best_cost = None
        self.best_pos = None

        # self.good_piezo_indexes = [2, 3, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        # self.good_piezo_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        #                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
        self.good_piezo_indexes = np.arange(40)
        self.num_good_piezos = len(self.good_piezo_indexes)

        self.current_micro_iter = 0
        self.current_macro_iter = 0

        self.res = PianoPSOOptimizationResult()
        self.res.good_piezo_indexes = np.array(self.good_piezo_indexes)
        self.res.max_piezo_voltage = self.dac.max_piezo_voltage
        self.res.roi = self.roi

    def optimize_my_pso(self, n_pop, n_iterations, stop_after_n_const_iters, reduce_at_iterations=()):
        self.optimizer = MyPSOOptimizer(self.cost_function, n_pop=n_pop, n_var=self.num_good_piezos,
                                        n_iterations=n_iterations,
                                        post_iteration_callback=self.post_iteration_callback,
                                        w=1, wdamp=0.99, c1=1.5, c2=2,
                                        timeout=60*60*2,
                                        stop_early=True, stop_after_n_const_iter=stop_after_n_const_iters,
                                        vary_popuation=True, reduce_at_iterations=reduce_at_iterations)

        self.res.random_average_cost = self.optimizer.random_average_cost
        self.res.n_pop = n_pop
        self.res.n_iterations = n_iterations
        self.res.stop_after_n_const_iters = stop_after_n_const_iters
        self.res.reduce_at_iterations = reduce_at_iterations

        print(f"Actual amount of iterations is: {self.optimizer.amount_of_micro_iterations()}.\n"
              f"It should take {self.optimizer.amount_of_micro_iterations() * self.dac.SLEEP_AFTER_SEND / 60} minutes")
        self.optimizer.optimize()

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
        self.res.exposure_times.append(self.cam.get_exposure_time())

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
