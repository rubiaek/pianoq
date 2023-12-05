import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from pianoq_results.misc import my_mesh
from pianoq_results.scan_result import ScanResult
from pianoq_results.fits_image import FITSImage

PATH_THICK_MEMORY = r'E:\Google Drive\Projects\Klyshko Optimization\Results\temp\2023_09_20_09_52_22_klyshko_very_thick_with_memory_meas\Memory'


def get_correlation(im1, im2):
    numerator = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2)))
    A = (im1 - np.mean(im1))**2
    B = (im2 - np.mean(im2))**2
    denumerator = np.sqrt(np.sum(A) * np.sum(B))
    dist_ncc = numerator / denumerator
    return dist_ncc


def get_memory_classical(dir_path=PATH_THICK_MEMORY):
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
        corr = get_correlation(im, fixed_ims[0])
        corrs.append(corr)

    return all_ds, corrs


def get_memory_coin(dir_path=PATH_THICK_MEMORY):
    paths = sorted(glob.glob(f'{dir_path}\\*d=*um.scan'))
    all_ds = np.array([re.findall('.*d=(.*)um', path)[0] for path in paths]).astype(int)
    scans = [ScanResult(path) for path in paths]
    corrs = []
    for scan in scans:
        # TODO: we need to correlate to the original picture but moving. This is almost what happens here.
        # Look at [(s.X[0], s.Y[0]) for s in scans] next to all_ds
        corr = get_correlation(scan.real_coins, scans[0].real_coins)
        corrs.append(corr)

    return all_ds, corrs


"""
    fig, axes = plt.subplots(1, len(show_ds), figsize=(len(show_ds)*3.5, 3), constrained_layout=True)
    for i in range(len(show_ds)):
        ind = np.where(all_ds == show_ds[i])[0][0]
        if not classic:
            scan = ScanResult(paths[ind])
            my_mesh(scan.X, scan.Y, scan.real_coins, axes[i])
            axes[i].invert_xaxis()
        else:
            im = FITSImage(paths[ind])
            imm = axes[i].imshow(im.image)

            axes[i].set_xlim(left=ind_col - X_pixs / 2, right=ind_col + X_pixs / 2)
            axes[i].set_ylim(bottom=ind_row - Y_pixs / 2, top=ind_row + Y_pixs / 2)
            fig.colorbar(imm, ax=axes[i])

        axes[i].set_title(f'd = {show_ds[i]}')

    fig.suptitle(f'classic = {classic}')

    fig.show()
"""

def main():
    pass


if __name__ == "__main__":
    main()
