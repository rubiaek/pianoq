from lab import ASICam
from pianoq.misc.mplc_lab_imports import *
from pianoq.misc.misc import detect_gaussian_spots_subpixel
import functools
import time
import os
import traceback

MODES_TO_KEEP = np.array([3, 8, 13, 18, 23, 28, 33, 38, 43, 48])
MASKS_PATH = r"G:\My Drive\Projects\MPLC\results\lab\2024_09_10_fixing_phases_different_Us\U0U_2024_09_10_13_57_31.masks"
PHASES_PATH = r"G:\My Drive\Projects\MPLC\results\lab\2024_09_10_fixing_phases_different_Us\U0U_2024_09_10_22_47_58.phases"
MASKS = get_good_masks(masks_path=MASKS_PATH, modes_to_keep=MODES_TO_KEEP, phases_path=PHASES_PATH)


def cost_function(centers_x, centers_y, cam, roi, mimshow=False):
    mplc = MPLCDevice(mask_centers=list(zip(centers_x, centers_y)))
    mplc.load_masks(MASKS)
    mplc.restore_location()
    time.sleep(0.2)
    mplc.restore_location()
    im = cam.get_image(roi=roi)
    cost = detect_gaussian_spots_subpixel(im, np.arange(im.shape[0]), np.arange(im.shape[1]),
                                          num_spots=10, min_distance=10, get_amps=True, window_size=20).sum()
    if mimshow:
        mimshow(im, title=cost)
    return -cost, im


def local_discrete_search(cost_func, cam, roi, init_guess, step_range=5, max_iters=100,
                          verbose=False, dir_path="C:\\temp", optimize_x=True):
    """
    Perform a simple local integer search:
      - Start at init_guess (list of ints)
      - For each parameter in turn, try stepping up/down by 1..step_range
      - Keep the best move if it improves cost
      - Repeat until no improvement or max_iters is reached
    """
    all_best_params = []
    all_best_costs = []
    best_params_path = os.path.join(dir_path, 'best_params.npz')

    best_params = list(init_guess)
    if optimize_x:
        best_cost, im = cost_func(centers_x=best_params)
    else:
        best_cost, im = cost_func(centers_y=best_params)
    cam.save_image(os.path.join(dir_path, '10_spots_new_best.fits'), im, comment=f'{best_params=}')
    print(f'{best_cost=}')
    np.savez(best_params_path, all_best_costs=all_best_costs, all_best_params=all_best_params)

    for _ in range(max_iters):
        improved = False
        display('foo')
        for i in range(len(best_params)):
            current_val = best_params[i]
            for step in range(-step_range, step_range + 1):
                candidate = best_params.copy()
                candidate[i] = current_val + step
                if optimize_x:
                    c_cost, im = cost_func(centers_x=best_params)
                else:
                    c_cost, im = cost_func(centers_y=best_params)
                if verbose:
                    print(f'{c_cost=}, {candidate=}')
                if c_cost < 1.002 * best_cost:  # improve only if mote than 0.2 percent difference
                    best_cost = c_cost
                    best_params = candidate
                    improved = True
                    print(f"Improved! {best_cost=}, {best_params=}")
                    all_best_params.append(best_params)
                    all_best_costs.append(best_cost)
                    cam.save_image(os.path.join(dir_path, '10_spots_new_best.fits'), im, comment=f'{best_params=}')

                    np.savez(best_params_path, all_best_costs=all_best_costs, all_best_params=all_best_params)

        if not improved:
            break  # no single-parameter move helped, so stop

    return best_params, best_cost


def optimize(init_guess, optimize_x=True, step_range=3, exp_time=2, roi=(700, 300, 1100, 900), dir_path="C:\\temp"):
    try:
        cam = PCOCamera()
        cam.set_exposure_time(exp_time)

        if optimize_x:
            cost_f = functools.partial(cost_function, cam=cam, roi=roi, centers_y=init_guess[1])
            guess = init_guess[0]
        else:
            cost_f = functools.partial(cost_function, cam=cam, roi=roi, centers_x=init_guess[0])
            guess = init_guess[1]

        best_params, best_cost = local_discrete_search(
            cost_func=cost_f,
            cam=cam,
            roi=roi,
            init_guess=guess,
            step_range=step_range,  # check ±1..±3 for each parameter
            max_iters=50,  # up to 50 iterations
            verbose=True,
            dir_path=dir_path,
            optimize_x=optimize_x
        )

        cam.close()

        print("Best parameters found:", best_params)
        print("Cost at best:", best_cost)
    except Exception as e:
        print(e)

        traceback.print_exc()
        cam.close()