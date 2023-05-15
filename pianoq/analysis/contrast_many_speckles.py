import numpy as np
import glob
from pianoq_results.scan_result import ScanResult
from uncertainties import ufloat

PATH_HERALDED = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Heralded\Many Speckles'
PATH_NOT_HERALDED = r'G:\My Drive\Projects\Quantum Piano\Paper 1\Data\Not Heralded\Many speckles'


def contrast(A):
    contrast = A.std() / A.mean()
    N = A.size
    contrast_err = contrast * np.sqrt(1 / (2 * N - 2) + (contrast ** 2) / N)
    return ufloat(contrast, contrast_err)


def analyze(dir_path, is_coin=True, mask=None):
    paths = glob.glob(f'{dir_path}\*.scan')
    scans = [ScanResult(path) for path in paths]

    if mask is None:
        mask = np.index_exp[5:-5, 5:-5]
    elif isinstance(mask, tuple):
        mask = mask
    elif isinstance(mask, int):
        mask = np.index_exp[mask:-mask, mask:-mask]

    cs = []
    Ns = []
    for s in scans:
        if is_coin:
            c = contrast(s.real_coins[mask])
        else:
            c = contrast(s.single2s[mask])
        N = 1/c**2
        cs.append(c)
        Ns.append(N)
        print(f'contrast: {c}')
        print(f'N: {N}')

    mean_c = np.mean(cs)
    mean_N = np.mean(Ns)
    # print(f'Average c: {mean_c:.2f}')
    print(f'Average N: {mean_N:.2f}')
    # print(f'N from average C: {1/mean_c**2:.2f}')


def both(mask=None):
    print('## Heralded ##')
    print('real_coin')
    analyze(PATH_HERALDED, is_coin=True, mask=mask)
    print('singles')
    analyze(PATH_HERALDED, is_coin=False, mask=mask)
    print()
    print('## Not Heralded ##')
    print('real_coin')
    analyze(PATH_NOT_HERALDED, is_coin=True, mask=mask)
    print('singles')
    analyze(PATH_NOT_HERALDED, is_coin=False, mask=mask)

