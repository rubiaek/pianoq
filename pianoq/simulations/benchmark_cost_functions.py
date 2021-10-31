import numpy as np
import matplotlib.pyplot as plt

from pianoq.simulations import PianoPopoffSimulation


def check_cost_functions_for_pol(n):
    cost_res_dop = []
    power_percent_in_pol_pol2 = []
    for i in range(n):
        print(f'#### {i} ####')
        a, b = check_cost_functions_for_pol_once()
        cost_res_dop.append(-a)
        power_percent_in_pol_pol2.append(-b)

    cost_res_dop = np.array(cost_res_dop)
    power_percent_in_pol_pol2 = np.array(power_percent_in_pol_pol2)

    print(cost_res_dop, power_percent_in_pol_pol2)

    print(f'cost_res_dop: {cost_res_dop.mean()} +- {cost_res_dop.std()}')  # 0.87 +- 0.05 at n = 50
    print(f'power_percent_in_pol_pol2: {power_percent_in_pol_pol2.mean()} +- {power_percent_in_pol_pol2.std()}')  # 0.93 +- 0.03  at n = 50
    # cost_res_dop: 0.85 +- 0.04
    # power_percent_in_pol_pol2: 0.86 +- 0.02
    return cost_res_dop,  power_percent_in_pol_pol2


def check_cost_functions_for_pol_once():
    cost_funcs = ['cost_function_degree_of_pol', 'cost_function_pol2']
    ratios = []

    for cost_func_name in cost_funcs:
        piano_sim = PianoPopoffSimulation(piezo_num=15, N_bends='fiber1',
                                          normalize_cost_to_tot_power=True, prop_random_phases=True,
                                          Nmodes=30, normalize_TMs_method='svd1',
                                          quiet=True)
        cost_func = getattr(piano_sim, cost_func_name)
        piano_sim.run(n_pop=40, n_iterations=500, cost_function=cost_func, stop_after_n_const_iters=30)
        res = cost_func(piano_sim.amps_history[-1])

        # pix1, pix2 = piano_sim.get_pixels(piano_sim.amps_history[-1])
        #
        # tot_power = (np.abs(pix1)**2).sum() + (np.abs(pix2)**2).sum()
        # power_percent_in_pol = (np.abs(pix1)**2).sum() / tot_power
        print(f'cost when using {cost_func_name} is: {res}')
        ratios.append(res)

    return ratios


if __name__ == "__main__":
    a, b = check_cost_functions_for_pol(50)

    plt.show()
