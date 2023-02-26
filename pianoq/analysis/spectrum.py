import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def smooth(x, N):
    return np.convolve(x, np.ones(N)/N, mode='same')


def show_spc(path, N=1, ax=None):
    from pyspectra.readers.read_spc import read_spc
    spc = read_spc(path)
    X = np.array(spc.index)
    Y = spc.values
    Y = smooth(Y, N)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(X, Y, label=f'moving average = {N}')
    ax.figure.show()
    return ax


def show_spc_dir(path):
    from pyspectra.readers.read_spc import read_spc_dir
    df_spc, dict_spc = read_spc_dir(path)
    # display(df_spc.transpose())
    f, ax = plt.subplots(1, figsize=(18, 8))
    ax.plot(df_spc.transpose(), '*--')
    plt.xlabel("nm")
    plt.ylabel("Abs")
    ax.legend(labels=list(df_spc.transpose().columns))
    plt.show()


def gaussian(x, a, x0, sig):
    return a*np.exp(-((x-x0)**2)/(sig**2))


def fit_to_gaussian(x, y, show=False):
    popt, pcov = curve_fit(gaussian, x, y, p0=(0.05, 404, 0.2)) #, bounds=([0.02, 400, 2], [1, 409, 3]))
    perrs = np.sqrt(np.diag(pcov))
    x0 = popt[1]
    x0_err = perrs[1]

    if show:
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(x, gaussian(x, *popt), '--')
    return x0, x0_err


def plot_new_pump_gaussian_fit():
    from pyspectra.readers.read_spc import read_spc
    import glob
    import re
    path = r'G:\My Drive\Projects\Quantum Piano\Results\Calibrations\Spectrum\Pump2\*.spc'
    files = glob.glob(path)
    mws = []
    wls = []
    wl_errs = []
    for f in files:
        mw = re.findall('(.*)mW.spc', f)[0].split('\\')[-1]
        mws.append(mw)
        spc = read_spc(f)
        X = np.array(spc.index)
        Y = spc.values
        x0, x0_err = fit_to_gaussian(X, Y, False)
        wls.append(x0)
        wl_errs.append(x0_err)

    fig, ax = plt.subplots()
    ax.errorbar(mws, wls, wl_errs)
    ax.set_xlabel('pump power (mW)')
    ax.set_ylabel('central wavelength (nm)')
    ax.axhline(403.85, color='g', linestyle='--')
    ax.axvline(8, color='r', linestyle='--')
    ax.axhline(404.8, color='g', linestyle='--')
    ax.axvline(20, color='r', linestyle='--')
    ax.axvline(21, color='r', linestyle='--')
    fig.show()


if __name__ == "__main__":
    """
    # Show best measurement of SPDC spectrum, with a wide filter (810/81), 50s integration time, and no flourecence
    # filter to gain another 30% signal
    path = r'G:\My Drive\Projects\Quantum Piano\Results\Calibrations\Spectrum\SPDC\800-80_filter_50s_no_flourescent_filter.spc'
    ax = show_spc(path, 1)
    show_spc(path, 25, ax)
    ax.legend()

    # Show pump spectrum at different powers
    path = r'G:\My Drive\Projects\Quantum Piano\Results\Calibrations\Spectrum\Pump'
    show_spc_dir(path)
    """

    # plot_new_pump_gaussian_fit()
    # plt.show()

    show_spc_dir(r'G:\My Drive\Projects\Quantum Piano\Results\SLD\Different spectral speckles with SLD and spectrometer')
    plt.show()
