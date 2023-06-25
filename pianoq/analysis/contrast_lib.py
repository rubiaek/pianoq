import numpy as np
from uncertainties import ufloat

def calc_contrast(V):
    contrast = V.std() / V.mean()
    N = V.size
    contrast_err = contrast * np.sqrt(1/(2*N-2) + (contrast**2)/N)
    return ufloat(contrast, contrast_err)

def contrast_to_N_modes(C):
    return 1/C**2
