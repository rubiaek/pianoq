import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from pianoq.results.image_result import VimbaImage


PATH_TEMPLATE = r"G:\My Drive\Projects\Quantum Piano\Results\Calibrations\Images of Equal in Both Pols\im%d.cam"


def main(path_no=1, x_dist=268, y_dist=0):
    assert path_no in [1, 2, 3, 4]
    path = PATH_TEMPLATE % path_no
    im = VimbaImage(path)

    # understand where first spot starts
    cut_1 = im.image[im.image.shape[0] // 2, :]
    first_non_zero1 = np.where(cut_1 > 2)[0][0]

    cut_2 = im.image[int(im.image.shape[0] // 2.3), :]
    first_non_zero2 = np.where(cut_2 > 2)[0][0]

    cut_3 = im.image[int(im.image.shape[0] // 1.7), :]
    first_non_zero3 = np.where(cut_3 > 2)[0][0]

    start_x = min(first_non_zero1, first_non_zero2, first_non_zero3) - 5

    # understand where first spot ends
    _, cm_col = ndimage.measurements.center_of_mass(im.image)
    cm_col = int(cm_col)

    width = cm_col - start_x

    # Split image to two
    if y_dist > 0:
        im1 = im.image[y_dist:, start_x:start_x+width]
        im2 = im.image[:-y_dist, start_x + x_dist:start_x + x_dist + width]
    elif y_dist < 0:
        im1 = im.image[:y_dist, start_x:start_x + width]
        im2 = im.image[-y_dist:, start_x + x_dist:start_x + x_dist + width]
    else:
        im1 = im.image[:, start_x:start_x + width]
        im2 = im.image[:, start_x + x_dist:start_x + x_dist + width]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    im = axes[0].imshow(im1)
    fig.colorbar(im, ax=axes[0])

    im = axes[1].imshow(im2)
    fig.colorbar(im, ax=axes[1])

    im = axes[2].imshow(im1-im2)
    fig.colorbar(im, ax=axes[2])

    grade = np.abs(im1 - im2).mean()
    print(grade)
    fig.suptitle(f'x={x_dist}, y={y_dist}, cost={grade}')

    fig.show()



# if __name__ == "__main__":
#     main()
#     plt.show()
