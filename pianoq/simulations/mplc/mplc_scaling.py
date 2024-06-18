import numpy as np
from pianoq.simulations.mplc.mplc_result import MPLCResult
from pianoq.simulations.mplc.mplc import MPLC
from pianoq.simulations.mplc.mplc_modes import get_spot_conf
from pianoq.simulations.mplc.mplc_utils import show_field


class MPLCScalingSimulation:
    def __init__(self, path1, path2):
        self.path1 = path1
        self.path2 = path2

        # Klyshko
        self.res = MPLCResult()
        self.res.loadfrom(path1)
        self.res2 = MPLCResult()
        self.res2.loadfrom(path2)
        self.initial_field = None
        self.out_field = None
        self.slm1_phase = None
        self.slm2_phase = None

        # first is from fiber to crystal, second from crystal to collection
        self.mplc = MPLC(conf=self.res.conf)
        self.mplc.res.masks = self.res.masks[::-1]
        self.mplc.dist_after_plane = self.mplc.dist_after_plane[::-1]
        self.mplc2 = MPLC(conf=self.res2.conf)
        self.mplc2.res.masks = self.res2.masks

    def set_intial_spot(self, Dx0=-0.2, Dy0=0.2):
        spot = get_spot_conf(self.res.conf, sig=0.1, Dx0=Dx0, Dy0=Dy0)
        self.initial_field = spot

    def propagate_klyshko(self, use_slm1=False, use_slm2=False):
        # propagate Klyshko
        # 4 is plane 7 backwards
        E_SLM1_plane = self.mplc.propagate_mplc(initial_field=self.initial_field, end_plane=4)
        if use_slm1:
            E_SLM1_plane *= np.exp(+1j*self.slm1_phase)

        E_crystal_plane = self.mplc.propagate_mplc(initial_field=E_SLM1_plane, start_plane=4)

        if use_slm2:
            E_crystal_plane *= np.exp(+1j*self.slm2_phase)

        # flipping from 2f (anti-correlations)
        E_crystal_plane = np.flipud(E_crystal_plane)

        E_out = self.mplc2.propagate_mplc(initial_field=E_crystal_plane)
        self.out_field = E_out

    def get_phase_SLM1(self, degree_of_control=1):

        pass


path1 = "C:\\temp\\speckle_speckle3.mplc"
path2 = "C:\\temp\\speckle_speckle4.mplc"
s = MPLCScalingSimulation(path1, path2)


# show_field(spot, active_slice=res.active_slice)
# spot_power = ((np.abs(spot)**2)[res.active_slice]).sum()
# print(f'{spot_power=}')


# TODO: Plan
#  Implement SLM for focusing in relevant region (Start with SLM1 in backprop)
#  Probably will need some statistics with different MPLC phase realizations
#  Scale incomplete control
#  Practically:
#  * implement SLM WFS `find phase` which should be pretty easy
#  * implement some incomplete control mechanism
