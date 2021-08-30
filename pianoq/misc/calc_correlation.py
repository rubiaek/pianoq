import os
import numpy as np
import matplotlib.pyplot as plt

from pianoq.lab.Edac40 import Edac40
from pianoq.lab.VimbaCamera import VimbaCamera
from pianoq.misc.consts import DEFAULT_BORDERS
from pianoq.misc.mplt import mimshow

cur_dir = os.path.dirname(os.path.abspath(__file__))
CORR_MASK_PATH = os.path.join(cur_dir, "correlations_mask.npz")


def get_correlation(im1, im2, use_mask=True):
    if use_mask:
        mask = get_correlations_mask()
        im1 = im1[mask]
        im2 = im2[mask]

    numerator = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2)))

    A = (im1 - np.mean(im1))**2
    B = (im2 - np.mean(im2))**2
    denumerator = np.sqrt(np.sum(A) * np.sum(B))
    dist_ncc = numerator / denumerator
    return dist_ncc


def get_correlations_mask():
    mask = np.load(CORR_MASK_PATH)['mask']
    return mask


def generate_correlations_mask():
    # When changing defulat borders we need to rerun this
    cam = VimbaCamera(2, exposure_time=900)
    cam.set_borders(DEFAULT_BORDERS)

    edac = Edac40(max_piezo_voltage=30, ip=Edac40.DEFAULT_IP)

    im = cam.get_image()
    n = 100
    for i in range(n):
        print(f'{i}/{n}')
        amps = np.random.uniform(0, 1, 40)
        edac.set_amplitudes(amps)
        im += cam.get_image()

    im = im / n

    mask = im > 2

    cam.close()
    edac.close()

    np.savez('correlations_mask', mask=mask)

    mimshow(mask.astype(int))

    return mask


if __name__ == "__main__":
    generate_correlations_mask()
    plt.show()
