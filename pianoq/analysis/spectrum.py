import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    # Show best measurement of SPDC spectrum, with a wide filter (810/81), 50s integration time, and no flourecence
    # filter to gain another 30% signal
    path = r'G:\My Drive\Projects\Quantum Piano\Results\Calibrations\Spectrum\SPDC\800-80_filter_50s_no_flourescent_filter.spc'
    ax = show_spc(path, 1)
    show_spc(path, 25, ax)
    ax.legend()

    # Show pump spectrum at different powers
    path = r'G:\My Drive\Projects\Quantum Piano\Results\Calibrations\Spectrum\Pump'
    show_spc_dir(path)

    plt.show()

