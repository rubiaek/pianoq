import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from pianoq_results.image_result import VimbaImage


paths = sorted(glob.glob(r'G:\My Drive\Projects\Quantum Piano\Results\TemperaturePiano\try2\*.cam'))
ims = [VimbaImage(path) for path in paths]
mask = np.index_exp[90:240, 100:250]
Ts = []
for im in ims:
    T = re.findall('.*T=(.*)C.*', im.path)[0]
    Ts.append(float(T))


def get_correlation(im1, im2):
    numerator = np.sum((im1 - np.mean(im1)) * (im2 - np.mean(im2)))
    A = (im1 - np.mean(im1))**2
    B = (im2 - np.mean(im2))**2
    denumerator = np.sqrt(np.sum(A) * np.sum(B))
    dist_ncc = numerator / denumerator
    return dist_ncc


def plot_correlations(ref_index=1):
    corrs = []
    for im in ims:
        corr = get_correlation(ims[ref_index].image[mask], im.image[mask])
        corrs.append(corr)
    fig, ax = plt.subplots()
    ax.plot(Ts, corrs, '*')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('PCC')
    ax.plot(Ts[ref_index], corrs[ref_index], 'o')
    fig.show()


def plot_reps(indexes=(1, 2, 3, 4, 5, 6), ref_index=1):
    corrs = []
    for ind in indexes:
        corr = get_correlation(ims[ref_index].image[mask], ims[ind].image[mask])
        corrs.append(corr)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for i in range(len(indexes)):
        imm = axes.flatten()[i].imshow(ims[indexes[i]].image[mask])
        axes.flatten()[i].set_title(f'{Ts[indexes[i]]} :: {corrs[i]:.3f}')
        fig.colorbar(imm, ax=axes.flatten()[i])
    fig.show()

plt.show()
