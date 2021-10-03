import random
import time
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np

from pianoq.simulations import PianoPopoffSimulation


def check_cost_functions_for_pol():

    cost_funcs = ['cost_function_focus', 'cost_function_pol1', 'cost_function_pol2']

    for cost_func_name in cost_funcs:

        piano_sim = PianoPopoffSimulation(piezo_num=30, N_bends=80,
                                          normalize_cost_to_tot_power=True, prop_random_phases=True,
                                          Nmodes=30, normalize_TMs_method='mean',
                                          quiet=True)
        cost_func = getattr(piano_sim, cost_func_name)
        piano_sim.run(n_pop=30, n_iterations=50, cost_function=cost_func)
        pix1, pix2 = piano_sim.get_pixels(piano_sim.amps_history[-1])

        tot_power = (np.abs(pix1)**2).sum() + (np.abs(pix2)**2).sum()
        power_percent_in_pol = (np.abs(pix1)**2).sum() / tot_power
        print(f'power_percent_in_pol when using {cost_func_name} is: {power_percent_in_pol}')


if __name__ == "__main__":
    check_cost_functions_for_pol()
