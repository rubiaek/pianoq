import time
import datetime
import numpy as np
from pianoq.lab.photon_counter import PhotonCounter

from pianoq.lab.asi_cam import ASICam
from pianoq.lab.time_tagger import QPTimeTagger
from pianoq.misc.borders import Borders
from pianoq.lab.Edac40 import Edac40
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.lab.optimizations.my_pso import MyPSOOptimizer
from pianoq.misc.consts import DEFAULT_BORDERS, DEFAULT_CAM_NO, DEFAULT_BORDERS2, DEFAULT_BORDERS_MMF

from pianoq_results.piano_optimization_result import PianoPSOOptimizationResult

LOGS_DIR = 'C:\\temp'


class PianoOptimization(object):

    def __init__(self, initial_exposure_time=450, saveto_path=None, roi=None, cost_function=None, cam_type='vimba', dac=None, cam=None,
                 good_piezo_indexes=np.arange(40), is_double_spot=False):
        ##########   CAREFULL CHANGING THIS VOLTAGE!!! #########
        self.dac = dac or Edac40(max_piezo_voltage=120, ip=Edac40.DEFAULT_IP)

        self.cam_type = cam_type
        self.is_double_spot = is_double_spot
        if self.cam_type == 'vimba':
            self.cam = VimbaCamera(DEFAULT_CAM_NO, exposure_time=initial_exposure_time)
        elif self.cam_type == 'ASI':
            self.dac.SLEEP_AFTER_SEND = 0.3  # wait a bit less since the exposure is so long...
            self.cam = ASICam(exposure=0.7, binning=3, image_bits=16, roi=(1065, 700, 96, 96))
            self.cam.set_gain(400)
        elif self.cam_type == 'SPCM':
            self.cam = cam or PhotonCounter(integration_time=initial_exposure_time)
        elif self.cam_type == 'timetagger':
            if cam:
                self.cam = cam or QPTimeTagger(integration_time=initial_exposure_time)
            else:
                PhotonCounter(integration_time=initial_exposure_time)

        self.initial_exposure_time = initial_exposure_time
        self.scaling_exposure_factor = 1
        # Should probably get as parameter the (x, y) and then define the borders around that part
        # borders = Borders(400, 278, 1024, 634)
        # borders = Borders(330, 520, 800, 615)
        # self.cam.set_borders(borders)

        self.saveto_path = saveto_path
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        self.start_time = datetime.datetime.now()
        self.window_size = 2
        self.roi = roi or np.index_exp[150 - self.window_size: 150 + self.window_size,
                                       150 - self.window_size: 150 + self.window_size]

        self.cost_function = cost_function or self.cost_function_roi

        self.optimizer = None
        self.best_cost = None
        self.best_pos = None

        # self.good_piezo_indexes = [2, 3, 4, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        # self.good_piezo_indexes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        #                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
        self.good_piezo_indexes = good_piezo_indexes
        # self.good_piezo_indexes = [0, 1, 3, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        #                                                        23, 24, 25, 28, 30, 32, 33, 35, 36]
        self.num_good_piezos = len(self.good_piezo_indexes)

        self.current_micro_iter = 0
        self.current_macro_iter = 0

        self.res = PianoPSOOptimizationResult()
        self.res.good_piezo_indexes = np.array(self.good_piezo_indexes)
        self.res.max_piezo_voltage = self.dac.max_piezo_voltage
        self.res.roi = self.roi
        self.res.cam_type = self.cam_type

    def optimize_my_pso(self, n_pop, n_iterations, stop_after_n_const_iters, reduce_at_iterations=(), success_cost=None):
        self.optimizer = MyPSOOptimizer(self.cost_function_callback, n_pop=n_pop, n_var=self.num_good_piezos,
                                        n_iterations=n_iterations,
                                        new_best_callback=self.new_best_callback,
                                        # w is inertia, c1 to self beet, c2 to global best
                                        w=1, wdamp=0.97, c1=1.5, c2=2.2,
                                        timeout=60*60*2,
                                        stop_early=True, stop_after_n_const_iter=stop_after_n_const_iters,
                                        vary_popuation=True, reduce_at_iterations=reduce_at_iterations,
                                        success_cost=success_cost)

        self.res.random_average_cost = self.optimizer.random_average_cost
        self.res.n_for_average_cost = self.optimizer.n_for_average_cost
        self.res.n_pop = n_pop
        self.res.n_iterations = n_iterations
        self.res.stop_after_n_const_iters = stop_after_n_const_iters
        self.res.reduce_at_iterations = reduce_at_iterations

        time_per_iteration = self.dac.SLEEP_AFTER_SEND
        if self.cam_type == 'ASI':
            time_per_iteration += self.cam.get_exposure_time()
        elif self.cam_type in ('SPCM', 'timetagger'):
            time_per_iteration += self.cam.integration_time

        print(f"Actual amount of iterations is: {self.optimizer.amount_of_micro_iterations()}.\n"
              f"It should take {self.optimizer.amount_of_micro_iterations() * time_per_iteration / 60} minutes")
        self.optimizer.optimize()

    def optimize_partitioning(self, n_iterations):
        # Tried for fun - it doesn't immediately work well...
        choosing_vector = np.zeros(self.num_good_piezos)
        choosing_vector[:self.num_good_piezos//2] = 1
        current_amps = np.zeros(self.num_good_piezos)

        for n in range(n_iterations):
            np.random.shuffle(choosing_vector)
            best_cost = 0
            best_amps = np.zeros(self.num_good_piezos)
            for i in np.linspace(0, 1, 11):
                amps = i*choosing_vector + current_amps
                amps = np.mod(amps, 1)
                cost, im = self.cost_function_callback(i*choosing_vector)
                if cost < best_cost:
                    best_cost = cost
                    best_amps = amps
            self.new_best_callback(best_cost, best_amps)
            current_amps = best_amps

    def cost_function_callback(self, amps):
        """
        amps is the array from the optimization algorithm, with only working amps
        :param amps:
        :return:
        """
        real_amps = np.ones(40) * self.dac.REST_AMP
        real_amps[self.good_piezo_indexes] = amps
        self.dac.set_amplitudes(real_amps)
        cost = 0; im = None; cost_std = 0

        if self.cam_type in ['ASI', 'vimba']:
            im = self.cam.get_image()
            cost = self.cost_function(im)
            print(f"{self.current_micro_iter}. cost: {cost:.3f}")

        elif self.cam_type == 'SPCM':
            datas, stds, actual_exp_time = self.cam.read()
            datas = datas/actual_exp_time
            single1, single2, coincidence = datas[0], datas[1], datas[4]

            accidentals = single1 * single2 * 2 * self.cam.coin_window
            real_coin = coincidence - accidentals
            real_coin_std = (np.sqrt(coincidence * actual_exp_time)) / actual_exp_time
            cost_std = real_coin_std
            cost = -real_coin
            im = None
            print(f"{self.current_micro_iter}. cost: {cost:.2f}+-{cost_std:.2f}")
        elif self.cam_type == 'timetagger':
            if not self.is_double_spot:
                single1, single2, coincidence = self.cam.read_interesting()
                accidentals = single1 * single2 * 2 * self.cam.coin_window
                real_coin = coincidence - accidentals
                real_coin_std = (np.sqrt(coincidence * self.cam.integration_time)) / self.cam.integration_time
                cost_std = real_coin_std
                cost = -real_coin
                im = None
                print(f"{self.current_micro_iter}. cost: {cost:.2f}+-{cost_std:.2f}")
            else:  # double spot
                single1, single2, single4, coincidence12, coincidence14 = self.cam.read_interesting()
                accidentals12 = single1 * single2 * 2 * self.cam.coin_window
                accidentals14 = single1 * single4 * 2 * self.cam.coin_window
                real_coin12 = coincidence12 - accidentals12
                real_coin14 = coincidence14 - accidentals14
                real_coin12_std = (np.sqrt(coincidence12 * self.cam.integration_time)) / self.cam.integration_time
                real_coin14_std = (np.sqrt(coincidence14 * self.cam.integration_time)) / self.cam.integration_time

                # to avoid NANs
                real_coin12 = max(real_coin12, 0.1)
                real_coin14 = max(real_coin14, 0.1)
                cost = np.sqrt(real_coin12) + np.sqrt(real_coin14)

                cost_std1 = 0.5*real_coin12**(-0.5)*real_coin12_std
                cost_std2 = 0.5*real_coin14**(-0.5)*real_coin14_std
                cost_std = np.sqrt(cost_std1**2 + cost_std2**2)
                cost = -cost

                im = None
                print(f'real_coin12: {real_coin12:.2f}; real_coin14: {real_coin14:.2f}')
                print(f"{self.current_micro_iter}. cost: {cost:.2f}+-{cost_std:.2f}")

        self.res.all_costs.append(cost)
        self.res.all_costs_std.append(cost_std)
        self.res.all_amplitudes.append(amps)
        self._save_result()

        self.current_micro_iter += 1
        self._fix_exposure(im)

        return cost, cost_std, im

    def cost_function_roi(self, im):
        im = im[self.roi]
        cost = -int(im.mean())
        cost *= self.scaling_exposure_factor
        return cost

    @staticmethod
    def cost_function_H_pol(im):
        # get the most light into H
        mid_col = im.shape[1] // 2

        pol1_energy = im[:, :mid_col].sum()
        pol2_energy = im[:, mid_col:].sum()
        tot_energy = pol1_energy + pol2_energy
        return -(pol1_energy / tot_energy)

    def _fix_exposure(self, im):
        if self.cam_type == 'vimba':
            mx = im.max()
            if mx > 240:
                print('Lowering exposure time!')
                exp_time = self.cam.get_exposure_time()
                if exp_time * (4 / 5) > 45:
                    self.cam.set_exposure_time(exp_time * (4 / 5))
                    self.scaling_exposure_factor *= (5 / 4)
                else:
                    print('**At shortest exposure and still saturated... You might want to add an ND to the camera..**')
        else:
            pass

    def new_best_callback(self, global_best_cost, cost_std, global_best_positions, im):
        now = datetime.datetime.now()
        delta = now - self.start_time
        self.res.timestamps.append(delta.total_seconds())

        self.res.costs.append(global_best_cost)
        self.res.costs_std.append(cost_std)
        self.res.amplitudes.append(global_best_positions)
        self.res.images.append(im)

        if self.cam_type in ['vimba', 'ASI']:
            self.res.exposure_times.append(self.cam.get_exposure_time())

        self.current_macro_iter += 1
        print(f'{self.current_macro_iter}.\t cost: {global_best_cost}\t time: {delta.total_seconds()} seconds')
        self._save_result()

    def _save_result(self):
        saveto_path = self.saveto_path or f"{LOGS_DIR}\\{self.timestamp}.pqoptimizer"
        self.res.saveto(saveto_path)

    def close(self):
        self.cam.close()
        self.dac.close()
