import time

import numpy as np
import matplotlib.pyplot as plt
from pianoq.lab.mplc.mplc_device import MPLCDevice
from pianoq.lab.mplc.mask_utils import mask_centers_to_mask_slices
from pianoq.lab.mplc.consts import SLM_DIMS, MASK_CENTERS, \
    D_BETWEEN_PLANES, D_BETWEEN_PLANES2, WAVELENGTH, PIXEL_SIZE, K, imaging_configs, CENTERS_X, CENTERS_Y
from pianoq.lab.pco_camera import PCOCamera

# dimensions of each plane. Larger than what we use in WFM
Dx = 200
Dy = 460
Dx_pi_x = 250
Dy_pi_y = 50


def lenses_mplc(planes, f, centers_x, centers_y):
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

        y_start = centers_y[plane-1] - Dy//2 - 1
        y_end = centers_y[plane-1] + Dy//2 - 1
        x_start = centers_x[plane-1] - Dx//2 - 1
        x_end = centers_x[plane-1] + Dx//2 - 1

        img[y_start:y_end, x_start:x_end] = phase_lens

    return img


class MPLCAligner:
    def __init__(self, use_cam=False):
        self.mplc = MPLCDevice()
        self.centers_x = CENTERS_X
        self.centers_y = CENTERS_Y
        self.XX, self.YY = np.meshgrid(np.arange(1280), np.arange(1024))
        self.final_img = np.zeros(SLM_DIMS)
        self.cam = None

        if use_cam:
            try:
                self.cam = PCOCamera()
                self.cam.set_exposure_time(0.5)
            except Exception:
                print('Could not connect to camera, interactive mode will not work')

    def update(self, imaging1='none', imaging2='none', pi_steps_x=(), pi_steps_y=(), pi_steps_plane=1, add_correction=True):
        # TODO: enable having pi_steps on multiple planes
        planes, f = imaging_configs.get(imaging1, ([], []))
        planes2, f2 = imaging_configs.get(imaging2, ([], []))

        img_lenses1 = lenses_mplc(planes, f, self.centers_x, self.centers_y)
        img_lenses2 = lenses_mplc(planes2, f2, self.centers_x, self.centers_y)

        img1 = np.zeros(SLM_DIMS)
        img2 = np.zeros(SLM_DIMS)

        if isinstance(pi_steps_x, int):
            pi_steps_x = [pi_steps_x]
        elif pi_steps_x is None:
            pi_steps_x = []
        if isinstance(pi_steps_y, int):
            pi_steps_y = [pi_steps_y]
        elif pi_steps_y is None:
            pi_steps_y = []

        # pi_steps
        for pix_no in pi_steps_x:
            # XX - pix_no will be positive/negative for pixels above/below pix_now,
            # and I will have +-pi/2 which is a pi step
            img1 = (np.sign(self.XX - pix_no + 1)) * np.pi / 2
            # stop the pi step at the edge of the plane
            img1 = img1 * ((np.abs(self.YY - self.centers_y[pi_steps_plane - 1] + 1)) < Dx_pi_x)

            # It is actually better not to cut off the pi step on the other side, because it effectively
            # creates another unwanted pi-step
            # img1 = img1 * ((np.abs(self.XX - self.centers_x[pi_steps_plane - 1])) < Dx//2)
            img2 += img1

        for pix_no in pi_steps_y:
            img1 = (np.sign(self.YY - pix_no + 1)) * np.pi / 2
            img1 = img1 * ((np.abs(self.XX - self.centers_x[pi_steps_plane - 1] + 1)) < Dy_pi_y)
            # It is actually better not to cut off the pi step on the other side, because it effectively
            # creates another unwanted pi-step
            # img1 = img1 * ((np.abs(self.YY - self.centers_y[pi_steps_plane - 1])) < Dy//2)
            img2 += img1

        self.final_img = img_lenses1 + img_lenses2 + img2

        self.mplc.load_slm_mask(self.final_img, add_correction=add_correction)

    def update_interactive(self, imaging1='none', imaging2='none', pi_step_x=None, pi_step_y=None, pi_steps_plane=1, D_res=100):
        # Not really using this
        # 24*12.5=300 um so in y if the center is 272 and the magnification to this plane is 1,
        # we will look at 272+-12. With other magnifications it will be not 12, so hard to automate
        if not self.cam:
            raise Exception('Must have camera connected for interactive mode')
        if pi_step_x and pi_step_y:
            raise Exception("Either pi_step_x or pi_steps_y in interactive mode")

        self.update(imaging1=imaging1,
                    imaging2=imaging2,
                    pi_steps_x=pi_step_x,
                    pi_steps_y=pi_step_y,
                    pi_steps_plane=pi_steps_plane)

        # inital find spots
        fig, ax = plt.subplots()
        A = self.cam.get_image()
        ax.imshow(A)
        ax.set_title('left-click around interesting')
        fig.show()
        loc = fig.ginput(n=1, timeout=0)
        x0, y0 = loc[0]
        x0, y0 = int(x0), int(y0)
        plt.close(fig)

        # time.sleep(self.cam.get_exposure_time())
        # Find exact location
        roi = [x0 - D_res, y0 - D_res, x0 + D_res, y0 + D_res]
        A = self.cam.get_image(roi=roi)
        fig, ax = plt.subplots()
        ax.imshow(A)
        ax.set_title('left-click at pi step row/col interesting')
        fig.show()
        loc = fig.ginput(n=1, timeout=0)
        x_cut, y_cut = loc[0]
        x_cut, y_cut = int(x_cut), int(y_cut)
        plt.close(fig)

        initial_guess = pi_step_x if pi_step_x else pi_step_y
        fig, axes = plt.subplots(2, 5)
        for i, pix in enumerate(np.array([-2, -1, 0, 1, 2]) + initial_guess):
            print(f'{i},', end='\t')
            self.update(imaging1=imaging1,
                        imaging2=imaging2,
                        pi_steps_x=[pix] if pi_step_x else [],
                        pi_steps_y=[pix] if pi_step_y else [],
                        pi_steps_plane=pi_steps_plane)

            im = self.cam.get_image(roi)
            axes[0, i].imshow(self.cam.get_image(roi))
            axes[0, i].set_title(f'pix = {pix}')
            if pi_step_x:
                axes[1, i].plot(im[x_cut, :])
            else:
                axes[1, i].plot(im[:, y_cut])
        print('done')
        fig.show()

    def find_x(self, plane_no, begin_guess=None):
        # best ways to do the imaging for each plane
        begin_guess = begin_guess or self.centers_x[plane_no-1]
        if plane_no == 1:
            self.update(imaging1='1to5w4f', imaging2='5to11w8', pi_steps_x=begin_guess, pi_steps_plane=plane_no)
        elif plane_no == 2:
            self.update(imaging1='none', imaging2='2to10w6', pi_steps_x=begin_guess, pi_steps_plane=plane_no)
        elif plane_no == 3:
            ml.update('1to3w2', '3to11w5and9', pi_steps_x=begin_guess, pi_steps_plane=plane_no)




# TODO: might be a off-by-pixel with the lens centers and WFM grid etc.

"""
TEST:
    import scipy.io 
    from pianoq.misc.mplt import *
    from pianoq.lab.mplc.mask_aligner import MPLCAligner
    ml = MPLCAligner()
    
    # Matlab 
    mat = scipy.io.loadmat(r'G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\1to11.mat')['img2']
    ml.mplc.load_slm_mask(mat, add_correction=True)
    matlab_mat = ml.mplc.uint_final_mask
    
    # Python 
    ml.update('1to6w4', '6to10w4f', pi_steps_y=[755, 775], pi_steps_plane=6, add_correction=True)
    py_mat = ml.mplc.uint_final_mask
    
    # Show 
    mimshow(matlab_mat, cmap='gray', title='matlab')
    mimshow(py_mat, cmap='gray', title='python')
    mimshow(matlab_mat - py_mat, cmap='gray', title='diff')
    
    # show 
    # mimshow(mat, cmap='gray', title='matlab')
    # mimshow(ml.mplc.slm_mask, cmap='gray', title='python')
    # mimshow(mat - ml.mplc.slm_mask, cmap='gray', title='diff')
    
    # Python starts at +1 and ends at +1 (1000, 738) 
"""
