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

def get_ratio(n, piezo_num, Nmodes):
    ratios = np.zeros(n)
    for i in range(n):
        print(f'##### {i} ####')
        piano_sim = PianoPopoffSimulation(piezo_num=piezo_num, N_bends='fiber1',
                                          normalize_cost_to_tot_power=True, prop_random_phases=True,
                                          Nmodes=Nmodes, normalize_TMs_method='mean',
                                          quiet=True)
        cost_func = piano_sim.cost_function_pol2
        piano_sim.run(n_pop=40, n_iterations=500, cost_function=cost_func, stop_after_n_const_iters=30)
        pix1, pix2 = piano_sim.get_pixels(piano_sim.amps_history[-1])

        tot_power = (np.abs(pix1)**2).sum() + (np.abs(pix2)**2).sum()
        power_percent_in_pol = (np.abs(pix1)**2).sum() / tot_power
        print(f'power_percent_in_pol with {Nmodes} Nmodes and {piezo_num} piezos: {power_percent_in_pol}')
        ratios[i] = power_percent_in_pol

    return ratios.mean(), ratios.std()


def n_piezos_for_given_Nmodes(Nmodes, piezo_nums, n_mean=5):
    ratio_means = []
    ratio_errs = []
    for piezo_num in piezo_nums:
        ratio_mean, ratio_err = get_ratio(n_mean, piezo_num, Nmodes)
        ratio_means.append(ratio_mean)
        ratio_errs.append(ratio_err)

        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print(f'Nmodes: {Nmodes}, piezo_num: {piezo_num}, ratio: {ratio_mean:.3f} +- {ratio_err:.3f}')
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    return ratio_means, ratio_errs

def Nmodes_to_piezo_num(n_mean=10):
    # [1, 3, 6, 10, 15, 21, 28, 36, 45, 55] * 2
    modes6_piezo_nums = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    modes6_means, modes6_errs = n_piezos_for_given_Nmodes(Nmodes=6, piezo_nums=modes6_piezo_nums, n_mean=n_mean)

    modes12_piezo_nums = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    modes12_means, modes12_errs = n_piezos_for_given_Nmodes(Nmodes=12, piezo_nums=modes12_piezo_nums, n_mean=n_mean)

    modes20_piezo_nums = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    modes20_means, modes20_errs = n_piezos_for_given_Nmodes(Nmodes=20, piezo_nums=modes20_piezo_nums, n_mean=n_mean)

    modes30_piezo_nums = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    modes30_means, modes30_errs = n_piezos_for_given_Nmodes(Nmodes=30, piezo_nums=modes30_piezo_nums, n_mean=n_mean)

    modes42_piezo_nums = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    modes42_means, modes42_errs = n_piezos_for_given_Nmodes(Nmodes=42, piezo_nums=modes42_piezo_nums, n_mean=n_mean)

    modes56_piezo_nums = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    modes56_means, modes56_errs = n_piezos_for_given_Nmodes(Nmodes=56, piezo_nums=modes56_piezo_nums, n_mean=n_mean)

    fig, ax = plt.subplots()
    ax.set_xlabel('piezo_num')
    ax.set_ylabel('percent in wanted polarization')

    ax.errorbar(modes6_piezo_nums, modes6_means, yerr=modes6_errs, fmt='.--', label='6 modes')
    ax.errorbar(modes12_piezo_nums, modes12_means, yerr=modes12_errs, fmt='.--', label='12 modes')
    ax.errorbar(modes20_piezo_nums, modes20_means, yerr=modes20_errs, fmt='.--', label='20 modes')
    ax.errorbar(modes30_piezo_nums, modes30_means, yerr=modes30_errs, fmt='.--', label='30 modes')
    ax.errorbar(modes42_piezo_nums, modes42_means, yerr=modes42_errs, fmt='.--', label='42 modes')
    ax.errorbar(modes56_piezo_nums, modes56_means, yerr=modes56_errs, fmt='.--', label='56 modes')
    ax.legend()
    fig.show()


if __name__ == "__main__":
    # a, b, c = check_cost_functions_for_pol(50)
    Nmodes_to_piezo_num()

    plt.show()
