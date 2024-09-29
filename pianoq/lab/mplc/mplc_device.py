import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from pianoq.lab.mplc.consts import MASK_CENTERS, MASK_DIMS, SLM_DIMS, D1, D2
from pianoq.lab.mplc.mask_utils import mask_centers_to_mask_slices, get_imaging_masks


CORRECTION_PATH = r"G:\My Drive\Projects\MPLC\Technical\correction_pattern_14_12_22.mat"


class MPLCDevice:
    N_PLANES = 10
    GEOMETRY = '1280x1024+1919+1'  # [width, height, x_offset, y_offset] pixel exact to Ohad MPLC class
    ALPHA = 213

    def __init__(self, mask_centers=MASK_CENTERS, geometry=None):
        self.masks = []  # exp(1j*phase)
        self.slm_mask = np.zeros(SLM_DIMS, dtype=float)  # phase [rads]
        self.uint_final_mask = np.zeros(SLM_DIMS, dtype=float)  # phase [1-255]
        self.mask_centers = mask_centers
        self.mask_slices = mask_centers_to_mask_slices(self.mask_centers)
        self.correction = scipy.io.loadmat(CORRECTION_PATH)['correction_final'].astype(float) * 2 * np.pi / 255
        self.geometry = geometry or self.GEOMETRY
        assert self.correction.shape == SLM_DIMS

        self.fig = None
        self.ax = None
        self.image = None
        self.window = None
        self.init_fig()
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def init_fig(self):
        # copied from SLMDevice
        self.fig = plt.figure(f'MPLC-Figure', frameon=False)

        # Axes that fully fils the figure
        self.ax = self.fig.add_axes([0., 0., 1., 1., ])
        # place figure in good place
        self.restore_location()
        # remove toolbar
        self.fig.canvas.toolbar.pack_forget()
        # This pause is necessary to make sure the location of the windows is actually changed when using TeamViewer
        plt.pause(0.1)
        self.ax.set_axis_off()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal', adjustable='box')
        # Removes menu bar
        self.fig.canvas.manager.window.overrideredirect(True)

        self.image = self.ax.imshow(self.slm_mask, cmap='gray', vmin=0, vmax=255)
        self.fig.canvas.draw()
        self.fig.show()
        self.restore_location()

    def restore_location(self, geometry=None):
        geom = geometry or self.geometry
        self.fig.canvas.manager.window.geometry(geom)

    def load_masks_from_path(self, masks_path, linear_tilts=True, plane_10_tilts=None):
        """
           The sending away of light from unwanted modes is the job of whoever supplies the masks.
           The masks will be of both upper and lower halves (signal and idler).
           The masks will be in radians.
        """

        f = open(masks_path, 'rb')
        data = np.load(f, allow_pickle=True)
        masks = data['masks']
        f.close()
        self.load_masks(masks, linear_tilts=linear_tilts, plane_10_tilts=plane_10_tilts)

    def load_masks(self, masks, linear_tilts=True, plane_10_tilts=None):
        self.masks = masks
        masks = np.angle(masks).astype(float)
        self.slm_mask = self.create_slm_mask(masks=masks, linear_tilts=linear_tilts, plane_10_tilts=plane_10_tilts)
        self.uint_final_mask = self.convert_to_uint8(self.slm_mask)
        self._update_screen(self.uint_final_mask)

    def load_slm_mask_from_path(self, path, add_correction=False):
        """ Load full SLM mask from Ohad WFM code """
        slm_mask = scipy.io.loadmat(path)['mask_total']
        self.load_slm_mask(slm_mask, add_correction=add_correction)

    def load_slm_mask(self, slm_mask, add_correction=False):
        self.slm_mask = slm_mask.copy()
        if add_correction:
            self.slm_mask = self.slm_mask + self.correction
            self.slm_mask = self.slm_mask - self.slm_mask.min() + 0.01

        self.uint_final_mask = self.convert_to_uint8(self.slm_mask)
        self._update_screen(self.uint_final_mask)

    def create_slm_mask(self, masks, linear_tilts=True, plane_10_tilts=None):
        slm_mask = np.zeros(SLM_DIMS, dtype=float)

        # add opposite linear tilts on all SLM
        if linear_tilts:
            # +1 for compatibility with Ohad code
            XX, YY = np.meshgrid(np.arange(SLM_DIMS[1]) + 1, np.arange(SLM_DIMS[0]) + 1)
            lin_tilt = -2 * np.pi * np.vstack((YY[:SLM_DIMS[0]//2, :], -YY[SLM_DIMS[0]//2:, :])) / 8
            lin_tilt = lin_tilt - np.min(lin_tilt) + 0.01

            slm_mask += lin_tilt

        # add actual masks
        assert masks.shape == (10, MASK_DIMS[0], MASK_DIMS[1])
        for i in range(self.N_PLANES):
            mask = masks[i].copy()
            # Moving the phase to start from zero, negative numbers produce jumps
            mask = mask - mask.min() + 0.01

            if i > 4:  # Masks 6:10 are after retro-reflecting the light, but the retro only flips the up-down direction
                mask = np.flipud(mask)

            slm_mask[self.mask_slices[i]] = mask + self.correction[self.mask_slices[i]]
            # Moving the phase to start from zero, negative numbers produce jumps
            slm_mask[self.mask_slices[i]] = slm_mask[self.mask_slices[i]] - slm_mask[self.mask_slices[i]].min() + 0.01

        if plane_10_tilts is not None:
            assert isinstance(plane_10_tilts, (int, float)), 'This should represent the number of pixels per 2-pi'
            XX, YY = np.meshgrid(np.arange(MASK_DIMS[1]) + 1, np.arange(MASK_DIMS[0]) + 1)
            # upper half of mask tilts a bit up, and lower a bit down, to avoid the sharp mirror edge
            lin_tilt10 = -2 * np.pi * np.vstack((YY[:MASK_DIMS[0]//2, :], -YY[MASK_DIMS[0]//2:, :])) / plane_10_tilts
            lin_tilt10 = lin_tilt10 - np.min(lin_tilt10) + 0.01

            slm_mask[self.mask_slices[-1]] += lin_tilt10
            slm_mask[self.mask_slices[-1]] = slm_mask[self.mask_slices[-1]] - slm_mask[self.mask_slices[-1]].min() + 0.01


        return slm_mask

    def _update_screen(self, final_mask):
        # restore background
        self.fig.canvas.restore_region(self.background)
        self.image.set_data(final_mask)

        # redraw just the points
        self.ax.draw_artist(self.image)

        # fill in the axes rectangle
        self.fig.canvas.blit(self.ax.bbox)

        plt.pause(0.001)

    def convert_to_uint8(self, phase_mask):
        # phase -> 255 and alpha
        phase_mask = phase_mask * 255 / (2 * np.pi)
        phase_mask = phase_mask * self.ALPHA / 255
        # TODO: understand this line
        condition = phase_mask > 255
        phase_mask[condition] = np.mod(phase_mask[condition] - (256 - self.ALPHA), self.ALPHA) + (256 - self.ALPHA)

        # original less sophisticated code:
        # phase = np.mod(phase * 255 / (2 * np.pi) + correction, 256)
        # phase = phase * self.alpha / 255
        # The current code makes supposedly better use of the dynamic range

        uint_mask = np.uint8(phase_mask)
        return uint_mask

    def set_imaging(self):
        """ image plane 1 to plane 11 of detectors """
        masks = get_imaging_masks()
        self.load_masks(masks)

    def close(self):
        plt.close(self.fig)
