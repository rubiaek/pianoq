from pianoq.lab.slm import SLMDevice
from pianoq.lab.mplc.consts import MASK_CENTERS, SLM_DIMS, MASK_DIMS
from pianoq.lab.mplc.mask_utils import mask_centers_to_mask_slices


class MPLCAligner:
    def __init__(self, slm_no=49, mask_centers=MASK_CENTERS):
        self.slm = SLMDevice(slm_no)
        self.mask_centers = mask_centers
        self.mask_slices = mask_centers_to_mask_slices(self.mask_centers)

    def update(self, lenses='none', pi_step_x=None, pi_step_y=None):
        # ugly string implementation like Ohad for lens configs
        # from pixel_no deduce mask_no
        # perform pi_step such that it will affect only the specific plane
        pass
