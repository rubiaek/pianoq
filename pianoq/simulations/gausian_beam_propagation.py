"""
Copyting from Yaron's matlab code which can be found also here:
    G:\My Drive\Projects\Quantum Piano\Matlab Code\GaussianPropagation
"""

import numpy as np
import matplotlib.pyplot as plt


def gaussian_prop_lense(wavelength, old_waist, old_s, f):
    """

    see [1] https://opg.optica.org/ao/fulltext.cfm?uri=ao-22-5-658&id=26503#e12
    and [2] https://github.com/adinatan/Gaussian_Propagation/blob/master/Gaussian_Propagation.m
    [3] https://www.edmundoptics.com/knowledge-center/application-notes/lasers/gaussian-beam-propagation/

    Returns new waist and s_new - position of waist relative to lens
    """
    zR = (np.pi * old_waist**2) / wavelength  # original rayleigh range Eq. 5 in [3]
    s_new = f * (1 + (old_s / f - 1) / ((old_s / f - 1) ** 2 + (zR / f) ** 2))  # new waist location (Eq. (9b)) in [1]
    # w_new = old_waist / np.sqrt(1 - (old_s/f)**2 + (zR/f)**2)  # eq.11 in [1] which seems wrong?+
    w_new = old_waist / np.sqrt((1 - old_s/f)**2 + (zR/f)**2)  # eq.11 in [3]
    return w_new, s_new


def main():
    # working in MKS
    wl = 632.8e-9
    initial_waist = 1e-3
    initial_waist_pos = -25e-3
    f_positions = [0, 500e-3]
    f_s = [300e-3, 200e-3]

    waists = np.zeros(len(f_s) + 1)
    waist_poss = np.zeros(len(f_s) + 1)

    waists[0] = initial_waist
    waist_poss[0] = initial_waist_pos

    for i, f in enumerate(f_s):
        # like 0 position waist which is initial from the first lense which is 0 also
        waist_distance_from_next_lens = f_positions[i] - waist_poss[i]
        new_w, new_rel_s = gaussian_prop_lense(wl, waists[i], waist_distance_from_next_lens, f_s[i])
        waists[i+1] = new_w
        waist_poss[i+1] = new_rel_s + f_positions[i]
    

if __name__ == "__main__":
    main()
