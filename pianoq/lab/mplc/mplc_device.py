import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from pianoq.lab.mplc.consts import MASK_CENTERS, MASK_DIMS, SLM_DIMS
from pianoq.lab.mplc.utils import mask_centers_to_mask_slices

CORRECTION_PATH = r"G:\My Drive\People\Ronen\PHD\MPLC\correction_pattern_14_12_22.mat"


class MPLCDevice:
    N_PLANES = 10
    GEOMETRY = '1272x1024+1913+-30'
    ALPHA = 213

    def __init__(self, mask_centers=MASK_CENTERS):
        self.mask_centers = mask_centers
        self.mask_slices = mask_centers_to_mask_slices(self.mask_centers)
        self.slm_mask = np.zeros(SLM_DIMS)
        self.correction = scipy.io.loadmat(CORRECTION_PATH)['correction_final']
        assert self.correction.shape == SLM_DIMS

        self.fig = None
        self.ax = None
        self.image = None
        self.init_fig()
        self.background = self.fig.canvas.copy_from_bbox(self.axes.bbox)

    def init_fig(self):
        # copied from SLMDevice
        self.fig = plt.figure(f'MPLC-Figure', frameon=False)
        self.ax = self.fig.add_axes([0., 0., 1., 1., ])
        self.fig.canvas.toolbar.pack_forget()
        # This pause is necessary to make sure the location of the windows is actually changed when using TeamViewer
        plt.pause(0.1)
        self.fig.canvas.manager.window.geometry(self.GEOMETRY)
        self.ax.set_axis_off()
        self.image = self.ax.imshow(self.slm_mask, cmap='gray', vmin=0, vmax=255)
        self.fig.canvas.draw()
        self.fig.show()

    def update(self, masks_path, linear_tilts=True):
        """
           The sending away of light from unwanted modes is the job of whoever supplies the masks.
           The masks will be of both upper and lower halves (signal and idler).
           The masks will be 0-2*pi.
        """
        f = open(masks_path, 'rb')
        data = np.load(f, allow_pickle=True)
        masks = data['masks']

        self.slm_mask = np.zeros(SLM_DIMS)

        # add opposite linear tilts on all SLM
        if linear_tilts:
            _, y = np.meshgrid(np.arange(SLM_DIMS[1]), np.arange(SLM_DIMS[0]))
            lin_tilt = -2 * np.pi * np.vstack((y[:SLM_DIMS[0]//2, :], -y[SLM_DIMS[0]//2:, :])) / 8
            lin_tilt = lin_tilt - np.min(lin_tilt) + 0.01
            lin_tilt = np.mod(lin_tilt, 2*np.pi)

            self.slm_mask += lin_tilt

        # add actual masks
        assert masks.shape == (10, MASK_DIMS[0], MASK_DIMS[1])
        for i in range(self.N_PLANES):
            self.slm_mask[self.mask_slices[i]] = masks[i]

        # phase -> 255, correction, and alpha
        phase = self.slm_mask * 255 / (2 * np.pi) + self.correction
        phase = phase * self.ALPHA / 255
        # TODO: understand this line
        condition = phase > 255
        phase[condition] = np.mod(phase[condition]-(256-self.ALPHA), self.ALPHA) + (256-self.ALPHA)

        # original less sofisticated code:
        # phase = np.mod(phase * 255 / (2 * np.pi) + self.correction, 256)
        # phase = phase * self.alpha / 255
        # The current code makes supposedly better use of the dynamic range

        phase = np.uint8(phase)

        # restore background
        self.fig.canvas.restore_region(self.background)
        self.image.set_data(phase)

        # redraw just the points
        self.ax.draw_artist(self.image)

        # fill in the axes rectangle
        self.fig.canvas.blit(self.ax.bbox)

        plt.pause(0.001)
