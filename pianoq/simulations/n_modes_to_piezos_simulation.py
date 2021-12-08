import numpy as np
import matplotlib.pyplot as plt
import datetime

from pianoq.misc.consts import LOGS_DIR
from pianoq.simulations import PianoPopoffSimulation
from pianoq.results.nmodes_to_piezos_result import NmodesToPiezosResult, PiezosForSpecificModeResult


class NmodesToPiezosSimulation(object):
    def __init__(self, cost_function):
        # [1, 3, 6, 10, 15, 21, 28, 36, 45, 55] * 2
        self.generic_piezo_nums = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40)
        self.less_piezo_nums = (0, 2, 4, 6, 8, 12, 16, 20)

        self.piezos_to_try = self.generic_piezo_nums
        self.modes_to_try = [6, 12, 20, 30]
        # self.modes_to_try = [6, 12, 20, 30, 42, 56]

        self.res = NmodesToPiezosResult()
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.res.timestamp = self.timestamp
        self.res.cost_func = cost_function
        self.res.normalize_TMs_method = 'svd1'
        self.res.pso_n_pop = 40
        self.res.pso_n_iterations = 1000
        self.res.pso_stop_after_n_const_iterations = 50
        self.res.version = 1.1

        self.saveto_path = None

    def run(self, n_mean=5):
        for Nmodes in self.modes_to_try:
            self.populate_specific_n_modes(Nmodes, n_mean=n_mean)

    def populate_specific_n_modes(self, Nmodes, n_mean):
        r = PiezosForSpecificModeResult()
        r.Nmodes = Nmodes
        r.piezo_nums = self.piezos_to_try

        for piezo_num in self.piezos_to_try:
            cost_mean, cost_std, sample_before, sample_after = self.get_cost(n_mean, piezo_num, Nmodes)
            r.costs.append(cost_mean)
            r.cost_stds.append(cost_std)
            r.example_befores.append(sample_before)
            r.example_afters.append(sample_after)

            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(f'Nmodes: {Nmodes}, piezo_num: {piezo_num}, ratio: {cost_mean:.3f} +- {cost_std:.3f}')
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        self.res.different_modes.append(r)
        self._save_result()

    def get_cost(self, n_mean, piezo_num, Nmodes):
        costs = np.zeros(n_mean)
        sample_before, sample_after = None, None

        for i in range(n_mean):
            print(f'##### {i} ####')
            piano_sim = PianoPopoffSimulation(piezo_num=piezo_num, N_bends='fiber1',
                                              normalize_cost_to_tot_power=True, prop_random_phases=True,
                                              Nmodes=Nmodes, normalize_TMs_method=self.res.normalize_TMs_method,
                                              quiet=True)

            if self.res.cost_func == 'pol2':
                cost_func = piano_sim.cost_function_pol2
            elif self.res.cost_func == 'focus':
                cost_func = piano_sim.cost_function_focus
            elif self.res.cost_func == 'HV':
                cost_func = piano_sim.cost_function_HV
            elif self.res.cost_func == 'mean_HVPM':
                cost_func = piano_sim.cost_function_mean_HVPM
            else:
                raise Exception()

            sample_before = piano_sim.get_initial_pixels(in_modes=piano_sim.H_in_modes.copy())

            if piezo_num == 0:
                sample_after = piano_sim.get_initial_pixels(in_modes=piano_sim.H_in_modes.copy())
                cost = cost_func(None)
            else:
                piano_sim.run(n_pop=self.res.pso_n_pop, n_iterations=self.res.pso_n_iterations, cost_function=cost_func,
                              stop_after_n_const_iters=self.res.pso_stop_after_n_const_iterations)
                sample_after = piano_sim.get_pixels(piano_sim.amps_history[-1], in_modes=piano_sim.H_in_modes.copy())

                cost = cost_func(piano_sim.amps_history[-1])

            print(f'cost_func value with {Nmodes} Nmodes and {piezo_num} piezos: {cost}')
            costs[i] = -cost

        return costs.mean(), costs.std(), sample_before, sample_after

    def _save_result(self):
        saveto_path = self.saveto_path or f"{LOGS_DIR}\\{self.timestamp}.nmtnp"
        self.res.saveto(saveto_path)


if __name__ == "__main__":
    # a, b, c = check_cost_functions_for_pol(50)
    # 'mean_HVPM' , 'pol2' , 'focus' , 'degree of polarization'
    n = NmodesToPiezosSimulation(cost_function='mean_HVPM')
    n.run(n_mean=10)

    plt.show()
