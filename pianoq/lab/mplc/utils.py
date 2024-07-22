import numpy as np
from pianoq.lab.mplc.consts import SLM_DIMS, MASK_DIMS


def mask_centers_to_mask_slices(mask_centers):
    mask_slices = []
    for mask_center in mask_centers:
        sl = np.index_exp[mask_center[0] - MASK_DIMS[0]//2: mask_center[0] + MASK_DIMS[0]//2,
                          mask_center[1] - MASK_DIMS[1]//2: mask_center[1] + MASK_DIMS[1]//2]
        mask_slices.append(sl)
