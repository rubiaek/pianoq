from PianoQ.Results.PopoffPolarizationRotationResult import PopoffPolarizationRotationResult

PATH = "../Data/popoff_polarization_data.npz"


def main():
    pop = PopoffPolarizationRotationResult(PATH)
    pop.show_TM(pop.TM_modes[pop.index_dx0])
    pop.show_polarizations_ratios_per_mode(range(0, 55, 2), logscale=True, legend=True)
    pop.show_polarizations_ratios_bar_plots([0, 20, 30, 40])


if __name__ == "__main__":
    main()
