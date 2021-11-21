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
                 normalize_cost_to_tot_power=True, prop_random_phases=True,
                 Nmodes=None, normalize_TMs_method='mean',
                 quiet=False):

        self.piezo_num = piezo_num
        self.N_bends = N_bends
        self.normalize_cost_to_tot_power = normalize_cost_to_tot_power
        self.prop_random_phases = prop_random_phases

        self.pop = PopoffPRXResult(path=PopoffPRXResult.DEFAULT_PATH)

        self.pop.set_Nmodes(Nmodes)
        self.pop.normalize_TMs(method=normalize_TMs_method)  # 'svd1'
        self.quiet = quiet

        # We will let the PSO algorithm live in a continuous [0,1] world, and translate it to one of the discreet set
        # of dxs, which will make us choose the relevant TM
        self.dxs = self.pop.dxs[self.pop.index_dx0:]
        self.TMs = self.pop.TM_modes[self.pop.index_dx0:]
        self.Nmodes = self.pop.Nmodes

        self.TM_fiber = self.generate_fiber_TM(self.N_bends)

        self.optimizer = None
        self.n_pop = None

        self._init_in_modes()
        self.amps_history = []

    def _init_in_modes(self):
        # Each mode comes in H polarization
        self.H_in_modes = np.zeros(self.Nmodes)
        self.H_Nmodes = self.Nmodes//2
        self.H_in_modes[:self.H_Nmodes] = 1/np.sqrt(self.H_Nmodes)

        # Each mode comes in V polarization
        self.V_in_modes = np.zeros(self.Nmodes)
        V_Nmodes = self.Nmodes // 2
        self.V_in_modes[V_Nmodes:] = 1 / np.sqrt(V_Nmodes)

        # Each mode comes in "+" polarization
        self.plus_in_modes = (1 / np.sqrt(self.Nmodes)) * np.ones(self.Nmodes)

        # Each mode comes in "-" polarization
        self.minus_in_modes = (1 / np.sqrt(self.Nmodes)) * np.ones(self.Nmodes)
        V_Nmodes = self.Nmodes // 2
        self.minus_in_modes[V_Nmodes:] *= -1

        # Is same as plus, used to use this in older version
        self.unif_in_modes = (1 / np.sqrt(self.Nmodes)) * np.ones(self.Nmodes)

    def generate_fiber_TM(self, N_bends):
        if isinstance(N_bends, int):
            mat = np.eye(self.Nmodes)

            for _ in range(N_bends):
                mat = mat @ random.choice(self.TMs)
                if self.prop_random_phases:
                    mat = mat @ np.diag(np.exp(1j*np.random.uniform(0, 2*np.pi, self.Nmodes)))

        elif N_bends == 'fiber1':
            indexes = [10, 15, 20, 25, 30, 35, 20, 10, 11, 14, 17, 19, 25, 29, 27, 13, 5]
            rng = np.random.RandomState(2) # So fiber TM will be given deterministically
            mat = np.eye(self.Nmodes)
            for ind in indexes:
                mat = mat @ self.TMs[ind]
                mat = mat @ self._get_random_phase_TM(self.Nmodes, rng)

        else:
            raise NotImplementedError

        return mat

    def _get_random_phase_TM(self, Nmodes, rng=None):
        rng = rng or np.random  # rng is a random number generator
        return np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, Nmodes)))

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
                                        vary_popuation=True, reduce_at_iterations=reduce_at_iterations,
                                        quiet=self.quiet)

        self.optimizer.optimize()

    def _amps_to_indexes(self, amps):
        # dx values are between 0 to 70 with jumps of 2: [0, 2, 4, ..., 68, 70]
        amps = amps * 35  # now between [0, 35]
        TM_indexes = np.around(amps).astype(int)
        return TM_indexes

    def _amps_to_tot_TM(self, amps):
        TM_indexes = self._amps_to_indexes(amps)

        # (3) propagate the input beam through the relevant TMs (according to (2) dxs)
        curr_TMs = [self.TMs[i] for i in TM_indexes]
        tot_piano_TM = reduce(lambda x, y: x @ y, curr_TMs)
        tot_curr_TM = tot_piano_TM @ self.TM_fiber
        return tot_curr_TM

    def get_pixels(self, amps, in_modes=None):
        if in_modes is None:
            # (1) initialize some input beam in the mode basis
            in_modes = self.unif_in_modes.copy()

        if amps is not None:
            # (2) translate the amps to dicreet dx values
            TM_indexes = self._amps_to_indexes(amps)

            # (3) propagate the input beam through the relevant TMs (according to (2) dxs)
            curr_TMs = [self.TMs[i] for i in TM_indexes]
            tot_piano_TM = reduce(lambda x, y: x @ y, curr_TMs)
            tot_curr_TM = tot_piano_TM @ self.TM_fiber
        else:
            tot_curr_TM = self.TM_fiber

        # (4) translate the output in mode basis to pixel basis
        pix1, pix2 = self.pop.propagate(in_modes, tot_curr_TM)

        return pix1, pix2

    def get_initial_pixels(self, in_modes=None):
        if in_modes is None:
            # (1) initialize some input beam in the mode basis
            in_modes = self.unif_in_modes.copy()

        pix1, pix2 = self.pop.propagate(in_modes, self.TM_fiber)
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

    def cost_function_pol2(self, amps):
        pix1, pix2 = self.get_pixels(amps)

        pix1_power = (np.abs(pix1) ** 2).sum()
        pix2_power = (np.abs(pix2) ** 2).sum()
        tot_power = pix1_power + pix2_power

        # We want all power in pix1, so we want pix2_power to be small
        cost = -pix1_power / tot_power

        return cost

    def cost_function_H(self, amps):
        in_modes = self.H_in_modes.copy()

        pix1, pix2 = self.get_pixels(amps, in_modes=in_modes)
        pix1_power = (np.abs(pix1) ** 2).sum()
        pix2_power = (np.abs(pix2) ** 2).sum()
        tot_power = pix1_power + pix2_power

        # We want all power in pix1, so we want pix2_power to be small
        cost = -pix1_power / tot_power

        return cost

    def cost_function_V(self, amps):
        in_modes = self.V_in_modes.copy()

        pix1, pix2 = self.get_pixels(amps, in_modes=in_modes)
        pix1_power = (np.abs(pix1) ** 2).sum()
        pix2_power = (np.abs(pix2) ** 2).sum()
        tot_power = pix1_power + pix2_power

        # We want all power in pix1, so we want pix2_power to be small
        cost = -pix2_power / tot_power

        return cost

    def cost_function_plus(self, amps):
        # Each mode comes in "+" polarization
        in_modes = self.plus_in_modes.copy()

        pixH, pixV = self.get_pixels(amps, in_modes=in_modes)
        pix_plus = (1 / np.sqrt(2)) * (pixH + pixV)
        pix_minus = (1 / np.sqrt(2)) * (pixH - pixV)

        plus_power = (np.abs(pix_plus) ** 2).sum()
        minus_power = (np.abs(pix_minus) ** 2).sum()
        tot_power = plus_power + minus_power

        # We want all power in pix1, so we want pix2_power to be small
        cost = -(plus_power / tot_power)

        return cost

    def cost_function_minus(self, amps):
        # Each mode comes in "-" polarization
        in_modes = self.minus_in_modes.copy()

        pixH, pixV = self.get_pixels(amps, in_modes=in_modes)
        pix_plus = (1 / np.sqrt(2)) * (pixH + pixV)
        pix_minus = (1 / np.sqrt(2)) * (pixH - pixV)

        plus_power = (np.abs(pix_plus) ** 2).sum()
        minus_power = (np.abs(pix_minus) ** 2).sum()
        tot_power = plus_power + minus_power

        # We want all power in pix1, so we want pix2_power to be small
        cost = -(minus_power / tot_power)

        return cost

    def cost_function_HV(self, amps):
        cost_H = self.cost_function_H(amps)
        cost_V = self.cost_function_V(amps)
        return np.mean([cost_H, cost_V])

    def cost_function_mean_HVPM(self, amps):
        cost_H = self.cost_function_H(amps)
        cost_V = self.cost_function_V(amps)
        cost_plus = self.cost_function_plus(amps)
        cost_minus = self.cost_function_minus(amps)
        return np.mean([cost_H, cost_V, cost_plus, cost_minus])

    def cost_function_max_HVPM(self, amps):
        cost_H = self.cost_function_H(amps)
        cost_V = self.cost_function_V(amps)
        cost_plus = self.cost_function_plus(amps)
        cost_minus = self.cost_function_minus(amps)
        # These are all negative numbers, and i want them all to be as small as possible.
        # Here instead of minimizing their mean, I try and minimize their maximum
        return np.max([cost_H, cost_V, cost_plus, cost_minus])

    def cost_function_degree_of_pol(self, amps):
        Ax, Ay = self.get_pixels(amps)

        S0 = (np.abs(Ax) ** 2) + (np.abs(Ay) ** 2)
        S1 = (np.abs(Ax) ** 2) - (np.abs(Ay) ** 2)
        S2 = 2 * (Ax.conj() * Ay).real
        S3 = 2 * (Ax.conj() * Ay).imag

        dop = np.sqrt(S1.sum() ** 2 + S2.sum() ** 2 + S3.sum() ** 2) / S0.sum()
        return -dop

    def post_iteration_callback(self, global_best_cost, global_best_positions):
        if not self.quiet:
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
    piano_sim = PianoPopoffSimulation(piezo_num=30, N_bends='fiber1',
                                      normalize_cost_to_tot_power=True, prop_random_phases=True,
                                      Nmodes=6, normalize_TMs_method='svd1')

    # piano_sim.run(n_pop=60, n_iterations=500, cost_function=piano_sim.cost_function_focus, stop_after_n_const_iters=30)
    # TODO: play with this. and maybe make a func that does amps->DOP, for external usage also
    # TODO: make also a show script + registry file
    piano_sim.run(n_pop=40, n_iterations=1000, cost_function=piano_sim.cost_function_max_HVPM, stop_after_n_const_iters=50)
    # piano_sim.show_before_after()
    # plt.show()
