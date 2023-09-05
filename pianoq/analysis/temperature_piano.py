import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pianoq_results.image_result import VimbaImage


paths = glob.glob(r'G:\My Drive\Projects\Quantum Piano\Results\TemperaturePiano\try1\*.cam')
ims = [VimbaImage(path) for path in paths]

def get_correlation(im1, im2, use_mask=False):
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

mask = np.index_exp[90:240, 100:250]
corrs = []
Ts = []
for im in ims:
    corr = get_correlation(ims[0].image[mask], im.image[mask])
    corrs.append(corr)
    T = os.path.basename(im.path)[20:]
    Ts.append(T)
    print(f'{T} :: {corr:.3f}')

fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
for i, ax in enumerate(axes.flatten()):
    imm = ax.imshow(ims[i].image[mask])
    ax.set_title(f'{Ts[i]} :: {corrs[i]:.3f}')
    fig.colorbar(imm, ax=ax)
fig.show()

plt.show()
