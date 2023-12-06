import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from pianoq_results.misc import my_mesh
from pianoq_results.scan_result import ScanResult
from pianoq_results.fits_image import FITSImage

PATH_THICK_MEMORY = r'G:\My Drive\Projects\Klyshko Optimization\Results\temp\2023_09_20_09_52_22_klyshko_very_thick_with_memory_meas\Memory'


def get_correlation(im1, im2):
    numerator = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2)))
    A = (im1 - np.mean(im1))**2
    B = (im2 - np.mean(im2))**2
    denumerator = np.sqrt(np.sum(A) * np.sum(B))
    dist_ncc = numerator / denumerator
    return dist_ncc


def get_memory_classical(dir_path=PATH_THICK_MEMORY, option='PCC'):
    X_pixs = 266  # == 1mm which is FOV of scan, divided by 3.76um for cam pix size
    Y_pixs = 266
    paths = sorted(glob.glob(f'{dir_path}\\*d=*um.fits'))
    all_ds = np.array([re.findall('.*d=(.*)um', path)[0] for path in paths]).astype(int)
    ims = [FITSImage(path) for path in paths]
    ind_row, ind_col = np.unravel_index(np.argmax(ims[0].image, axis=None), ims[0].image.shape)
    fixed_ims = []
    for i, im in enumerate(ims):
        delta_d = all_ds[i] - all_ds[0]
        factor = int(10*delta_d / 3.76)
        fixed_im = im.image[ind_row - X_pixs // 2         : ind_row + X_pixs // 2,
                            ind_col - Y_pixs // 2 - factor: ind_col + Y_pixs // 2 - factor]
        fixed_ims.append(fixed_im)

    corrs = []
    for im in fixed_ims:
        if option == 'PCC':
            corr = get_correlation(im, fixed_ims[0])
        elif option == 'max_pix':
            corr = im.max()
        elif option == 'max_speckle':
            corr = 2  # TODO
        else:
            corr = 2

        corrs.append(corr)

    return all_ds, corrs


def get_memory_coin(dir_path=PATH_THICK_MEMORY, option='PCC'):
    paths = sorted(glob.glob(f'{dir_path}\\*d=*um.scan'))
    all_ds = np.array([re.findall('.*d=(.*)um', path)[0] for path in paths]).astype(int)
    scans = [ScanResult(path) for path in paths]
    corrs = []
    for scan in scans:
        # TODO: we need to correlate to the original picture but moving. This is almost what happens here.
        # Look at [(s.X[0], s.Y[0]) for s in scans] next to all_ds
        if option == 'PCC':
            corr = get_correlation(scan.real_coins, scans[0].real_coins)
        elif option == 'max_pix':
            corr = scan.real_coins.max()
        elif option == 'max_speckle':
            corr = 2  # TODO
        else:
            corr = 2
        corrs.append(corr)

    return all_ds, corrs


def show_memories(dir_path=PATH_THICK_MEMORY):
    fig, ax = plt.subplots()
    diode_ds, diode_corrs = get_memory_classical(dir_path=dir_path)
    diode_corrs = np.array(diode_corrs) / max(diode_corrs)
    coin_ds, coin_corrs = get_memory_coin(dir_path=dir_path)
    coin_corrs = np.array(coin_corrs) / max(coin_corrs)
    ax.plot(diode_ds, diode_corrs, '*--', label='diode')
    ax.plot(coin_ds, coin_corrs, '*--', label='SPDC')
    fig.legend()
    fig.show()


def main():
    show_memories()


if __name__ == "__main__":
    main()
    plt.show()
