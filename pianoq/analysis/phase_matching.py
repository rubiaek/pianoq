import numpy as np
import  matplotlib.pyplot as plt
import re
import glob
import sys

from pianoq import mimshow

sys.path.append('C:\\code')
sys.path.append('C:\\code\\pianoq_results')

from pianoq_results import FITSImage


T_FOR_140_MW = 29.5  # C (This way we are collinear, but also loose some signal)
T_FOR_365_MW = 91.8  # C (This way we are collinear, but also loose some signal)


def image(s, e, normalize=False):
    # This image plane is after f=200 and f=300 I think, so the width of sigma=30.7 pixels translates to
    # 30.7*2*3.76*200/300 = ~155um (fit to gaussian of x^2/s^2 in intensity, at 28.2 degrees)
    dir_path = r'G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\PPKTP\New-2022-10\f=250_before\Temperature\Image\*.fit'
    paths = glob.glob(dir_path)

    fig, ax = plt.subplots()

    for path in paths[s:e]:
        T = re.findall('.*gain0_T=(.*).fit', path)[0]
        fi = FITSImage(path)
        V = fi.image[373, 570:770]
        if normalize:
            V = (V - V.min())
            V = V / V.max()

        ax.plot(V, label=f'T={T}')

    ax.legend()
    fig.show()


def smooth(x, N):
    return np.convolve(x, np.ones(N)/N, mode='same')


def farfield(s, e, skip=1, smoothing=5):
    dir_path = r'G:\My Drive\Projects\Quantum Piano\Results\Calibrations\SPDC\PPKTP\New-2022-10\f=250_before\Temperature\Farfield\*.fit'
    paths = glob.glob(dir_path)
    fig, ax = plt.subplots()

    for path in paths[s:e:skip]:
        T = re.findall('.*gain0_T=(.*).fit', path)[0]
        fi = FITSImage(path)
        V = fi.image[525, :]
        mimshow(fi.image, vmin=500, vmax=800, title=f'T={T}')
        V = smooth(V, smoothing)[smoothing:-smoothing]
        ax.plot(V, label=f'T={T}')

    ax.legend()
    fig.show()


if __name__ == "__main__":
    image(0, -1, True)
    # farfield(7, 9, 20)
    pass

plt.show()
