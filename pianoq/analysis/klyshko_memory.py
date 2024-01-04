import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pianoq_results.scan_result import ScanResult
from pianoq_results.fits_image import FITSImage
from uncertainties import unumpy

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


def get_memory_classical2(dir_path=r'G:\My Drive\Projects\Klyshko Optimization\Results\Off_axis\Memory Diode', option='PCC'):
    X_pixs = 266  # == 1mm which is FOV of scan, divided by 3.76um for cam pix size
    Y_pixs = 266
    paths = glob.glob(f'{dir_path}\\*d=*.fits')
    all_ds = np.array([re.findall('.*d=(.*)\.', path)[0] for path in paths]).astype(int)
    all_ds, paths = list(zip(*sorted(zip(all_ds, paths), key=lambda pair: pair[0])))
    all_ds = np.array(all_ds, dtype=int)
    mid_ind = np.where(all_ds==5)[0][0]
    all_ds *= 10
    ims = [FITSImage(path) for path in paths]
    ind_row, ind_col = np.unravel_index(np.argmax(ims[mid_ind].image, axis=None), ims[0].image.shape)
    fixed_ims = []
    for i, im in enumerate(ims):
        delta_d = all_ds[i] - all_ds[mid_ind]
        factor = int(10*delta_d / 3.76)
        fixed_im = im.image[ind_row - X_pixs // 2         : ind_row + X_pixs // 2,
                            ind_col - Y_pixs // 2 - factor: ind_col + Y_pixs // 2 - factor]
        fixed_ims.append(fixed_im)

    corrs = []
    for im in fixed_ims:
        if option == 'PCC':
            corr = get_correlation(im, fixed_ims[mid_ind])
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


def show_memories(dir_path=PATH_THICK_MEMORY, option='PCC'):
    fig, ax = plt.subplots()
    diode_ds2, diode_corrs2 = get_memory_classical2(dir_path=r'G:\My Drive\Projects\Klyshko Optimization\Results\Off_axis\Memory Diode', option=option)
    diode_corrs2 = np.array(diode_corrs2) / max(diode_corrs2)

    diode_ds, diode_corrs = get_memory_classical(dir_path=dir_path, option=option)
    diode_corrs = np.array(diode_corrs) / max(diode_corrs)

    coin_ds, coin_corrs = get_memory_coin(dir_path=dir_path, option=option)
    coin_corrs = np.array(coin_corrs) / max(coin_corrs)

    ax.plot(diode_ds, diode_corrs, '*--', label='diode')
    ax.plot(diode_ds2, diode_corrs2, '*--', label='diode2')
    ax.plot(coin_ds, coin_corrs, '*--', label='SPDC')
    fig.legend()
    fig.show()


def get_memory_classical3(dir_path, l=4):
    paths = glob.glob(f'{dir_path}\\*d=*.fits')
    dark_im = FITSImage(r"G:\My Drive\Projects\Klyshko Optimization\Results\Off_axis\try6\2024_01_03_11_26_52_dark.fits")
    all_ds = np.array([re.findall('.*d=(.*)\.fits', path)[0] for path in paths]).astype(float)
    all_ds, paths = list(zip(*sorted(zip(all_ds, paths), key=lambda pair: pair[0], reverse=True)))
    all_ds = np.array(all_ds, dtype=float)
    all_ds *= 10
    ims = [FITSImage(path) for path in paths]
    max_speckles = []

    for i, im in enumerate(ims):
        image = im.image.astype(float) - dark_im.image.astype(float)
        image[image < 0] = 0
        # TODO: have some moving window around expected focus that you choose the max pixel from within it, similar to SPDC
        # TODO: though there is no real need - it happens automatically
        ind_row, ind_col = np.unravel_index(np.argmax(image, axis=None), image.shape)
        # altogether 9 pixels, which comes out ~101um
        l = 4
        max_speckle = image[ind_row - l:ind_row + l + 1, ind_col - l:ind_col + l + 1].sum()
        # in SPDC we have 50um fibers, and counting 3 pixels that are 25um apart, so we count the inner 50um twice
        # 5 pixels is ~56um
        l = 2
        inner_50_um = image[ind_row - l:ind_row + l + 1, ind_col - l:ind_col + l + 1].sum()
        max_speckles.append(max_speckle+inner_50_um)

    dx = all_ds[0] - all_ds

    return dx, np.array(max_speckles) / max_speckles[0]


def get_memory_SPDC3(dir_path, l=1):
    paths = glob.glob(f'{dir_path}\\*d=*.scan')
    all_ds = np.array([re.findall('.*d=(.*)\.scan', path)[0] for path in paths]).astype(float)
    all_ds, paths = list(zip(*sorted(zip(all_ds, paths), key=lambda pair: pair[0], reverse=True)))
    all_ds = np.array(all_ds, dtype=float)
    all_ds *= 10
    scans = [ScanResult(path) for path in paths]
    max_speckles = []
    max_speckle_stds = []
    for i, scan in enumerate(scans):
        ind_row, ind_col = np.unravel_index(np.argmax(scan.real_coins, axis=None), scan.real_coins.shape)
        # altogether 3X3 pixels, which comes out ~75umX75um
        coin_area = scan.real_coins[ind_row-l:ind_row+l+1, ind_col-l:ind_col+l+1]
        coin_std_area = scan.real_coins_std[ind_row-l:ind_row+l+1, ind_col-l:ind_col+l+1]
        u_max_speckle = unumpy.uarray(coin_area, coin_std_area).sum()
        max_speckle = u_max_speckle.nominal_value
        max_speckle_std = u_max_speckle.std_dev
        max_speckles.append(max_speckle)
        max_speckle_stds.append(max_speckle_std)

    return all_ds[0] - all_ds, np.array(max_speckles) / max_speckles[0], np.array(max_speckle_stds, dtype=float) / max_speckles[0]


def mem_func(theta, d_theta):
    return ( (theta/d_theta) / (1e-17 + np.sinh(theta/d_theta)) )**2


def show_memories3(dir_path_classical, dir_path_SPDC, d_x=22, l1=3, l2=1):
    fig, ax = plt.subplots()
    diode_ds, diode_corrs = get_memory_classical3(dir_path_classical, l=l1)
    SPDC_ds, SPDC_corrs, SPDC_corr_stds = get_memory_SPDC3(dir_path_SPDC, l=l2)

    diode_thetas = diode_ds * 10 / 100e3 # 10x magnification to SMF, and 100mm lens
    SPDC_thetas = SPDC_ds * 10 / 100e3  # 10x magnification to SMF, and 100mm lens
    theta_err = 2*10/100e3  # 20 um in manual micrometer, and 100mm lens

    ax.errorbar(diode_thetas, diode_corrs, xerr=theta_err, fmt='*', label='diode', color='b')
    ax.errorbar(SPDC_thetas, SPDC_corrs, xerr=theta_err, yerr=SPDC_corr_stds, fmt='o', label='SPDC', color='r')
    dummy_theta = np.linspace(1e-6, 0.007, 1000)
    # ax.plot(dummy_x, mem_func(dummy_x, d_x), '-', label='analytical')
    popt, pcov = curve_fit(mem_func, diode_thetas, diode_corrs, p0=0.02, bounds=(1e-6, 2))
    ax.plot(dummy_theta, mem_func(dummy_theta, *popt), '--', label='diode fit', color='b')
    print(*popt)
    popt, pcov = curve_fit(mem_func, SPDC_thetas, SPDC_corrs, p0=0.02, bounds=(1e-6, 2))
    ax.plot(dummy_theta, mem_func(dummy_theta, *popt), '--', label='SPDC fit', color='r')
    print(*popt)
    # ax.set_title(f'l1={l1}, l2={l2}')
    ax.set_xlabel(r'$\Delta\theta$ (rad)')
    ax.set_ylabel('normalized focus intensity')
    fig.legend()
    fig.show()


def main(d_x=22, l1=4, l2=1):
    dir_path_classical = r'G:\My Drive\Projects\Klyshko Optimization\Results\Off_axis\try6\diode_memory'
    dir_path_SPDC = r'G:\My Drive\Projects\Klyshko Optimization\Results\Off_axis\try6\SPDC_memory'
    show_memories3(dir_path_classical, dir_path_SPDC, d_x=d_x, l1=l1, l2=l2)
    # get_memory_classical3(path_classical)


if __name__ == "__main__":
    pass
    # main()
    # plt.show()
