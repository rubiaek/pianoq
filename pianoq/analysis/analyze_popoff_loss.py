import numpy as np
import matplotlib.pyplot as plt
from pianoq.results.popoff_prx_result import PopoffPRXResult


def main():
    pop = PopoffPRXResult()
    pop.loadfrom(pop.DEFAULT_PATH)

    fig, ax = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for i, TM_ind in enumerate([10, 20, 30, 40]):

        ax = fig.axes[i]
        singular_values = np.linalg.svd(pop.TM_modes[TM_ind], compute_uv=False)
        # singular_values /= singular_values[0]
        ax.plot(singular_values)
        ax.set_xlabel('modes')
        ax.set_ylabel('relative loss')
        ax.set_title(f'dx={pop.dxs[TM_ind]:.2f} $ \mu m $')
    fig.show()


if __name__ == "__main__":
    main()
    plt.show()
