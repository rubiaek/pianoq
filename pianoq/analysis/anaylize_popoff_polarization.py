import matplotlib.pyplot as plt
from pianoq.results.PopoffPolarizationRotationResult import PopoffPolarizationRotationResult

PATH = "../data/popoff_polarization_data.npz"


def main():
    pop = PopoffPolarizationRotationResult(path=PATH)
    pop._initialize(method='TM')
    # pop._initialize(method='pixel')

    # pop.show_mode(10)
    pop.show_TM(pop.TM_modes[pop.index_dx0])
    # pop.show_all_polarizations_ratios()
    # pop.show_polarizations_ratios_per_mode(range(0, 55, 2), logscale=True, legend=True)
    # pop.show_polarizations_ratios_bar_plots([0, 20, 30, 40])
    # pop.show_mixing_of_mode(pop.TM_modes[5], 10)


if __name__ == "__main__":
    main()
    plt.show()
