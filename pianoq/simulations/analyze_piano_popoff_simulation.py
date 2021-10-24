import numpy as np
import matplotlib.pyplot as plt
import datetime

from pianoq.misc.consts import LOGS_DIR
from pianoq.simulations import PianoPopoffSimulation
from pianoq.results.nmodes_to_piezos_result import NmodesToPiezosResult, PiezosForSpecificModeResult


def check_cost_functions_for_pol(n):
    power_percent_in_pol_focus = []
    power_percent_in_pol_pol1 = []
    power_percent_in_pol_pol2 = []
    for i in range(n):
        print(f'#### {i} ####')
        a, b, c = check_cost_functions_for_pol_once()
        power_percent_in_pol_focus.append(a)
        power_percent_in_pol_pol1.append(b)
        power_percent_in_pol_pol2.append(c)

    power_percent_in_pol_focus = np.array(power_percent_in_pol_focus)
    power_percent_in_pol_pol1 = np.array(power_percent_in_pol_pol1)
    power_percent_in_pol_pol2 = np.array(power_percent_in_pol_pol2)

    print(power_percent_in_pol_focus, power_percent_in_pol_pol1, power_percent_in_pol_pol2)

    print(f'power_percent_in_pol_focus: {power_percent_in_pol_focus.mean()} +- {power_percent_in_pol_focus.std()}')  # 0.87 +- 0.05 at n = 50
    print(f'power_percent_in_pol_pol1: {power_percent_in_pol_pol1.mean()} +- {power_percent_in_pol_pol1.std()}')  # 0.83 +- 0.04  at n =50
    print(f'power_percent_in_pol_pol2: {power_percent_in_pol_pol2.mean()} +- {power_percent_in_pol_pol2.std()}')  # 0.93 +- 0.03  at n = 50

    return power_percent_in_pol_focus, power_percent_in_pol_pol1, power_percent_in_pol_pol2


def check_cost_functions_for_pol_once():
    cost_funcs = ['cost_function_focus', 'cost_function_pol1', 'cost_function_pol2']
    ratios = []

    for cost_func_name in cost_funcs:
        piano_sim = PianoPopoffSimulation(piezo_num=15, N_bends='fiber1',
                                          normalize_cost_to_tot_power=True, prop_random_phases=True,
                                          Nmodes=30, normalize_TMs_method='mean',
                                          quiet=True)
        cost_func = getattr(piano_sim, cost_func_name)
        piano_sim.run(n_pop=40, n_iterations=500, cost_function=cost_func, stop_after_n_const_iters=30)
        pix1, pix2 = piano_sim.get_pixels(piano_sim.amps_history[-1])

        tot_power = (np.abs(pix1)**2).sum() + (np.abs(pix2)**2).sum()
        power_percent_in_pol = (np.abs(pix1)**2).sum() / tot_power
        print(f'power_percent_in_pol when using {cost_func_name} is: {power_percent_in_pol}')
        ratios.append(power_percent_in_pol)

    return ratios


class NmodesToPiezosSimulation(object):
    def __init__(self):
        # [1, 3, 6, 10, 15, 21, 28, 36, 45, 55] * 2
        self.generic_piezo_nums = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40)
        self.less_piezo_nums = (0, 2, 4, 6, 8, 12, 16, 20)

        self.piezos_to_try = self.generic_piezo_nums
        # self.modes_to_try = [6, 12, 20]
        self.modes_to_try = [6, 12, 20, 30, 42, 56]

        self.res = NmodesToPiezosResult()
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.res.timestamp = self.timestamp
        self.res.cost_func = 'pol2'  # 'focus' , 'degree of polarization'
        self.res.normalize_TMs_method = 'svd1'

        self.saveto_path = None

    def run(self, n_mean=5):
        for Nmodes in self.modes_to_try:
            self.populate_specific_n_modes(Nmodes, n_mean=n_mean)

    def populate_specific_n_modes(self, Nmodes, n_mean):
        r = PiezosForSpecificModeResult()
        r.Nmodes = Nmodes
        r.piezo_nums = self.piezos_to_try

        for piezo_num in self.piezos_to_try:
            ratio_mean, ratio_err, sample_before, sample_after = self.get_ratio(n_mean, piezo_num, Nmodes)
            r.ratios.append(ratio_mean)
            r.ratio_stds.append(ratio_err)
            r.example_befores.append(sample_before)
            r.example_afters.append(sample_after)

            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(f'Nmodes: {Nmodes}, piezo_num: {piezo_num}, ratio: {ratio_mean:.3f} +- {ratio_err:.3f}')
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        self.res.different_modes.append(r)
        self._save_result()

    def get_ratio(self, n_mean, piezo_num, Nmodes):
        ratios = np.zeros(n_mean)
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
            else:
                raise Exception()

            sample_before = piano_sim.get_initial_pixels()

            if piezo_num == 0:
                pix1, pix2 = piano_sim.get_initial_pixels()
            else:
                piano_sim.run(n_pop=40, n_iterations=1000, cost_function=cost_func, stop_after_n_const_iters=50)
                pix1, pix2 = piano_sim.get_pixels(piano_sim.amps_history[-1])

            sample_after = (pix1, pix2)

            tot_power = (np.abs(pix1)**2).sum() + (np.abs(pix2)**2).sum()
            power_percent_in_pol = (np.abs(pix1)**2).sum() / tot_power
            print(f'power_percent_in_pol with {Nmodes} Nmodes and {piezo_num} piezos: {power_percent_in_pol}')
            ratios[i] = power_percent_in_pol

        return ratios.mean(), ratios.std(), sample_before, sample_after

    def _save_result(self):
        saveto_path = self.saveto_path or f"{LOGS_DIR}\\{self.timestamp}.nmtnp"
        self.res.saveto(saveto_path)


if __name__ == "__main__":
    # a, b, c = check_cost_functions_for_pol(50)
    n = NmodesToPiezosSimulation()
    n.run(n_mean=10)

    plt.show()
