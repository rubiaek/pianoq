import numpy as np
from pianoq.lab.mplc.mplc_device import MPLCDevice
from pianoq.lab.mplc.mask_utils import mask_centers_to_mask_slices
from pianoq.lab.mplc.consts import SLM_DIMS, MASK_CENTERS, \
    D_BETWEEN_PLANES, D_BETWEEN_PLANES2, WAVELENGTH, PIXEL_SIZE, K, imaging_configs, CENTERS_X, CENTERS_Y

# dimensions of each plane. Larger than what we use in WFM
Dx = 200
Dy = 460


def lenses_mplc(planes, f, center_x, center_y):
    """
    Adds quadratic phases to the desired places in the mplc.

    Parameters:
    planes : plane numbers to put lenses on, 1-based
    f : Focal length of the lenses (in units of the distance between planes D_BETWEEN_PLANES)
    center_x, center_y : Center coordinates for each plane
    Returns image with added quadratic phases
    """

    img = np.zeros(SLM_DIMS)

    X = np.arange(-Dx//2, Dx//2) * 12.5e-6
    Y = np.arange(-Dy//2, Dy//2) * 12.5e-6
    XX, YY = np.meshgrid(X, Y)

    for j, plane in enumerate(planes):
        phase_lens = -K * (XX ** 2 + YY ** 2) / (2 * f[j] * D_BETWEEN_PLANES)
        phase_lens = phase_lens - np.min(phase_lens) + 0.01

        y_start = center_y[plane-1] - Dy//2
        y_end = center_y[plane-1] + Dy//2
        x_start = center_x[plane-1] - Dx//2
        x_end = center_x[plane-1] + Dx//2

        img[y_start:y_end, x_start:x_end] = phase_lens

    return img


class MPLCAligner:
    def __init__(self):
        self.mplc = MPLCDevice()
        self.centers_x = CENTERS_X
        self.centers_y = CENTERS_Y
        self.XX, self.YY = np.meshgrid(np.arange(1280), np.arange(1024))
        self.final_img = np.zeros(SLM_DIMS)

    def update(self, imaging1='none', imaging2='none', pi_steps_x=(), pi_steps_y=(), pi_steps_plane=1):
        # TODO: enable having pi_steps on multiple planes
        planes, f = imaging_configs.get(imaging1, ([], []))
        planes2, f2 = imaging_configs.get(imaging2, ([], []))

        img_lenses1 = lenses_mplc(planes, f, self.centers_x, self.centers_y)
        img_lenses2 = lenses_mplc(planes2, f2, self.centers_x, self.centers_y)

        img1 = np.zeros(SLM_DIMS)
        img2 = np.zeros(SLM_DIMS)
        # pi_steps
        for pix_no in pi_steps_x:
            # XX - pix_no will be positive/negative for pixels above/below pix_now,
            # and I will have +-pi/2 which is a pi step
            img1 = (np.sign(self.XX - pix_no)) * np.pi / 2
            # stop the pi step at the edge of the plane
            img1 = img1 * ((np.abs(self.YY - self.centers_y[pi_steps_plane - 1])) < Dy//2)

            # It is actually better not to cut off the pi step on the other side, because it effectively
            # creates another unwanted pi-step
            # img1 = img1 * ((np.abs(self.XX - self.centers_x[pi_steps_plane - 1])) < Dx//2)
            img2 += img1

        for pix_no in pi_steps_y:
            img1 = (np.sign(self.YY - pix_no)) * np.pi / 2
            img1 = img1 * ((np.abs(self.XX - self.centers_x[pi_steps_plane - 1])) < Dx//2)
            # It is actually better not to cut off the pi step on the other side, because it effectively
            # creates another unwanted pi-step
            # img1 = img1 * ((np.abs(self.YY - self.centers_y[pi_steps_plane - 1])) < Dy//2)
            img2 += img1

        self.final_img = img_lenses1 + img_lenses2 + img2

        self.mplc.load_slm_mask(self.final_img, add_correction=True)

    def find_x(self, plane_no, begin_guess=None):
        # best ways to do the imaging for each plane
        begin_guess = begin_guess or self.centers_x[plane_no-1]
        if plane_no == 1:
            # TODO: try -2,-1,0,1,2 pi steps, show them all, use ginput to get relevant line, and then show relevant slices so in 1d it is even easier to decide
            # TODO: then get from user with input the fibal choice, and update centers_x, centers_y
            self.update(imaging1='1to10w4', imaging2='none', pi_steps_x=begin_guess)

