import random
import time
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import matrix_power

from pianoq.lab.optimizations.my_pso import MyPSOOptimizer
from pianoq.results import PopoffPRXResult


class PianoPopoffSimulation(object):
    def __init__(self, piezo_num=30, N_bends=30,
                 normalize_cost_to_tot_power=True, prop_random_phases=True, remove_higher_modes=True):
        self.piezo_num = piezo_num
        self.N_bends = N_bends
        self.normalize_cost_to_tot_power = normalize_cost_to_tot_power
        self.prop_random_phases = prop_random_phases

        self.pop = PopoffPRXResult(path=PopoffPRXResult.DEFAULT_PATH)
        self.pop.normalize_TMs(method='svd1')
        if remove_higher_modes:
            self.pop.remove_higher_modes()

        # We will let the PSO algorithm live in a continuous [0,1] world, and translate it to one of the discreet set
        # of dxs, which will make us choose the relevant TM
        self.dxs = self.pop.dxs[self.pop.index_dx0:]
        self.TMs = self.pop.TM_modes[self.pop.index_dx0:]
        self.Nmodes = self.pop.Nmodes

        self.TM_fiber = self.generate_fiber_TM(self.N_bends)

        self.optimizer = None
        self.n_pop = None
        self.in_modes = (1/np.sqrt(self.Nmodes)) * np.ones(self.Nmodes)

        self.amps_history = []

    def generate_fiber_TM(self, N_bends):
        mat = np.eye(self.Nmodes)

        for _ in range(N_bends):
            mat = mat @ random.choice(self.TMs)
            if self.prop_random_phases:
                mat = mat @ np.diag(np.exp(1j*np.random.uniform(0, 2*np.pi, self.Nmodes)))

        return mat

    def run(self, n_pop=30, n_iterations=50, cost_function=None,
            stop_after_n_const_iters=20, reduce_at_iterations=(2, 5)):

        cost_function = cost_function or self.cost_function_focus
        self.n_pop = n_pop
        self.optimizer = MyPSOOptimizer(cost_function, n_pop=n_pop, n_var=self.piezo_num,
                                        n_iterations=n_iterations,
                                        post_iteration_callback=self.post_iteration_callback,
                                        w=1, wdamp=0.99, c1=1.5, c2=2,
                                        timeout=30*60,
                                        stop_early=True, stop_after_n_const_iter=stop_after_n_const_iters,
                                        vary_popuation=True, reduce_at_iterations=reduce_at_iterations)

        self.optimizer.optimize()

    def _amps_to_indexes(self, amps):
        # dx values are between 0 to 70 with jumps of 2: [0, 2, 4, ..., 68, 70]
        amps = amps * 35  # now between [0, 35]
        TM_indexes = np.around(amps).astype(int)
        return TM_indexes

    def get_pixels(self, amps):
        # (1) initialize some input beam in the mode basis
        in_modes = self.in_modes.copy()

        # (2) translate the amps to dicreet dx values
        TM_indexes = self._amps_to_indexes(amps)

        # (3) propagate the input beam through the relevant TMs (according to (2) dxs)
        curr_TMs = [self.TMs[i] for i in TM_indexes]
        tot_piano_TM = reduce(lambda x, y: x @ y, curr_TMs)
        tot_curr_TM = tot_piano_TM @ self.TM_fiber

        # (4) translate the output in mode basis to pixel basis
        pix1, pix2 = self.pop.propagate(in_modes, tot_curr_TM)

        return pix1, pix2

    def play_N_bends(self, N_bends):
        in_modes = (1/np.sqrt(self.Nmodes)) * np.ones(self.Nmodes)
        TM = self.generate_fiber_TM(N_bends)
        pix1, pix2 = self.pop.propagate(in_modes, TM)
        pixs = np.concatenate((pix1, pix2), axis=1)
        fig, ax = plt.subplots()
        im0 = ax.imshow(np.abs(pixs)**2)
        fig.colorbar(im0, ax=ax)
        fig.show()

    def _power_at_area(self, pix1):
        Nx, Ny = pix1.shape
        window_size = 2
        roi = np.index_exp[(Nx // 2) - window_size: (Nx // 2) + window_size,
                           (Ny // 2) - window_size: (Ny // 2) + window_size]
        powers = np.abs(pix1[roi]) ** 2
        return powers.mean()

    def cost_function_focus(self, amps):
        """ amps are between 0 and 1 """

        pix1, pix2 = self.get_pixels(amps)

        # (5) "measure" the intensity in some point
        if self.normalize_cost_to_tot_power:
            tot_power = (np.abs(pix1) ** 2).sum() + (np.abs(pix2) ** 2).sum()
            cost = -self._power_at_area(pix1) / tot_power
        else:
            cost = -self._power_at_area(pix1)
        return cost

    def post_iteration_callback(self, global_best_cost, global_best_positions):
        print(f'{self.optimizer.curr_iteration}.\t cost: {global_best_cost:2f}\t time: {(time.time()-self.optimizer.start_time):2f} seconds')
        self.amps_history.append(global_best_positions)

    def show_before_after(self):
        fig, axes = plt.subplots(2, 1, figsize=(5, 5.8), constrained_layout=True)

        pix1_0, pix2_0 = self.get_pixels(self.amps_history[0])
        pix1_1, pix2_1 = self.get_pixels(self.amps_history[-1])
        pixs0 = np.concatenate((pix1_0, pix2_0), axis=1)
        pixs1 = np.concatenate((pix1_1, pix2_1), axis=1)

        im0 = axes[0].imshow(np.abs(pixs0)**2)
        im1 = axes[1].imshow(np.abs(pixs1)**2)
        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
        axes[0].set_title('Before')
        axes[1].set_title('After')

        fig.show()


if __name__ == "__main__":
    piano_sim = PianoPopoffSimulation(piezo_num=30, N_bends=80, normalize_cost_to_tot_power=True,
                                      prop_random_phases=True)
    piano_sim.run(n_pop=30, n_iterations=50)
    piano_sim.show_before_after()
    plt.show()
