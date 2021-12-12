import numpy as np

from pianoq.lab.piano_optimization import PianoOptimization


def main():
    n = 1
    for i in range(n):
        whole_area_L = 40
        whole_speckle_L = 20
        small_L = 6

        wind = small_L // 2
        roi_L = np.index_exp[70 - wind: 70 + wind, 60 - wind: 60 + wind]
        roi_R = np.index_exp[70 - wind: 70 + wind, 335 - wind: 335 + wind]

        roi = roi_L
        po = PianoOptimization(saveto_path=None, initial_exposure_time=300, roi=roi)
        po.optimize_my_pso(n_pop=20, n_iterations=20, stop_after_n_const_iters=5, reduce_at_iterations=(3,))
        # po.optimize_my_pso(n_pop=25, n_iterations=150, stop_after_n_const_iters=5, reduce_at_iterations=(4, 10))
        po.close()


if __name__ == "__main__":
    main()
