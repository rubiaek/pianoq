from pianoq.lab.slm import SLMDevice
from pianoq.lab.mplc.consts import MASK_CENTERS
from pianoq.lab.mplc.utils import mask_centers_to_mask_slices


class MPLCDevice:
    def __init__(self, slm_no=49, mask_centers=MASK_CENTERS):
        self.slm = SLMDevice(slm_no)
        self.mask_centers = mask_centers
        self.mask_slices = mask_centers_to_mask_slices(self.mask_centers)

    def set_masks(self, path):
        pass
        # load masks
        # set them according to the self.mask_centers
        # linear tilt outside of masks to send light away
        # I think the sending away of light from unwanted modes is the job of whoever supplies the masks
        # the masks will be of both upper and lower halves (signal and idler)
