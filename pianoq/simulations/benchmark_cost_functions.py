import numpy as np
import matplotlib.pyplot as plt

from pianoq.simulations import PianoPopoffSimulation


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


if __name__ == "__main__":
    a, b, c = check_cost_functions_for_pol(50)

    plt.show()
