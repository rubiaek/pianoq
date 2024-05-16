import glob
import matplotlib.pyplot as plt

from pianoq_results.slm_optimization_result import SLMOptimizationResult

BETWEEN_DIR = r'G:\My Drive\Projects\ScalingPropertiesQWFS\Results\KlyshkoSetup\Try1\Between2Diffusers'
BEFORE_DIR = r'G:\My Drive\Projects\ScalingPropertiesQWFS\Results\KlyshkoSetup\Try1\SLMBeforeDiffuser'


def show_scaling():
    paths_between = glob.glob(BETWEEN_DIR + '\\202*.optimizer2')
    paths_before = glob.glob(BEFORE_DIR + '\\202*.optimizer2')

    ress_between = [SLMOptimizationResult(path) for path in paths_between]
    ress_before = [SLMOptimizationResult(path) for path in paths_before]

    Ns_between = []
    Es_between = []
    Ns_before = []
    Es_before = []
    for res in ress_between:
        res.reload()
        Ns_between.append(len(res.best_phase_mask))
        Es_between.append(res.enhancement)

    for res in ress_before:
        res.reload()
        Ns_before.append(len(res.best_phase_mask))
        Es_before.append(res.enhancement)

    fig, ax = plt.subplots()
    ax.plot(Ns_between, Es_between, '*', label='SLM between')
    ax.plot(Ns_before, Es_before, '*', label='SLM before')
    ax.set_xlabel('$N_{DOF}$')
    ax.set_ylabel('Enhancement')
    ax.legend()
    fig.show()

    return ress_between, ress_before


show_scaling()
plt.show()
