import numpy as np

from pianoq.lab.piano_optimization import PianoOptimization


def main():
    n = 1
    for i in range(n):
        wind = 2
        roi = np.index_exp[45 - wind: 45 + wind, 75 - wind: 75 + wind]
        po = PianoOptimization(saveto_path=None, initial_exposure_time=350, roi=roi)
        po.optimize_my_pso(n_pop=10, n_iterations=10, stop_after_n_const_iters=25, reduce_at_iterations=(1,))
        # po.optimize_my_pso(n_pop=25, n_iterations=150, stop_after_n_const_iters=20, reduce_at_iterations=(4, 10))
        po.close()


if __name__ == "__main__":
    main()
