import numpy as np

from pianoq.lab.piano_optimization import PianoOptimization
from pianoq.simulations.calc_fiber_modes import get_modes_FG010LDA


def focus_partitioning():
    small_L = 4
    y = 200
    x = 200

    wind = small_L // 2
    roi = np.index_exp[y - wind: y + wind, x - wind: x + wind]

    # Don't pass here PianoOptimization.cost_function_roi since it isn't a staticmethod so it will do trouble
    po = PianoOptimization(saveto_path=None, initial_exposure_time=150, roi=roi)
    po.optimize_partitioning(30)
    po.close()


def focus():
    n = 1
    for i in range(n):
        whole_area_L = 40
        whole_speckle_L = 20
        small_L = 6

        wind = small_L // 2
        # roi_L = np.index_exp[70 - wind: 70 + wind, 60 - wind: 60 + wind]
        y = 190
        x = 200
        roi_L = np.index_exp[y - wind: y + wind, x - wind: x + wind]
        # roi_R = np.index_exp[70 - wind: 70 + wind, 335 - wind: 335 + wind]

        roi = roi_L
        # Don't pass here PianoOptimization.cost_function_roi since it isn't a staticmethod so it will do trouble
        po = PianoOptimization(saveto_path=None, initial_exposure_time=150, roi=roi)
        po.optimize_my_pso(n_pop=20, n_iterations=50, stop_after_n_const_iters=5, reduce_at_iterations=(3,))
        # po.optimize_my_pso(n_pop=25, n_iterations=150, stop_after_n_const_iters=10, reduce_at_iterations=(3,))
        po.close()


def pol():
    po = PianoOptimization(saveto_path=None, initial_exposure_time=5.5e3,
                           cost_function=PianoOptimization.cost_function_H_pol)
    # po.optimize_my_pso(n_pop=20, n_iterations=20, stop_after_n_const_iters=5, reduce_at_iterations=(3,))
    po.optimize_my_pso(n_pop=25, n_iterations=150, stop_after_n_const_iters=25, reduce_at_iterations=(4, 10))
    po.close()


def reference():
    cost_function_lp = get_cost_function_LP()
    po = PianoOptimization(saveto_path=None, initial_exposure_time=700,
                           cost_function=cost_function_lp)
    # po.optimize_my_pso(n_pop=20, n_iterations=20, stop_after_n_const_iters=5, reduce_at_iterations=(3,))
    po.optimize_my_pso(n_pop=25, n_iterations=150, stop_after_n_const_iters=5, reduce_at_iterations=(2, 10))
    po.close()


def get_cost_function_LP():

    profiles = get_modes_FG010LDA(npoints=75)
    lp1 = np.reshape(profiles[1], (75, 75))
    REFERENCE1 = np.abs(lp1)**2
    REFERENCE1 = REFERENCE1 / REFERENCE1.sum()

    lp2 = np.reshape(profiles[0], (75, 75))  # [5]
    REFERENCE2 = np.abs(lp2)**2
    REFERENCE2 = REFERENCE2 / REFERENCE2.sum()

    def cost_function_LP1(im):
        im1 = im[32:107, 27:102]
        assert im1.shape == REFERENCE1.shape
        im1 = im1 / im1.sum()
        cost1 = np.abs(REFERENCE1 - im1).mean()

        im2 = im[32:107, 295:370]
        im2 = im2 / im2.sum()
        assert im2.shape == REFERENCE2.shape
        cost2 = np.abs(REFERENCE2 - im2).mean()
        cost2 = 0

        return cost1 + cost2

    return cost_function_LP1


if __name__ == "__main__":
    focus()
