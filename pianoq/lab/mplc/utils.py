import numpy as np
from pianoq.lab.mplc.consts import SLM_DIMS, MASK_DIMS


def mask_centers_to_mask_slices(mask_centers):
    mask_slices = []
    for mask_center in mask_centers:
        x_c, y_c = mask_center
        sl = np.index_exp[x_c - MASK_DIMS[0]//2: x_c + MASK_DIMS[0]//2,
                          y_c - MASK_DIMS[1]//2: y_c + MASK_DIMS[1]//2]
        mask_slices.append(sl)
