import numpy as np
import matplotlib.pyplot as plt
from pianoq.simulations.mplc_sim.mplc_sim_result import MPLCSimResult, MPLCMasks
from pianoq.simulations.mplc_sim.mplc_sim import MPLCSim
from pianoq.simulations.mplc_sim.mplc_modes import get_spot_conf
from pianoq.simulations.mplc_sim.mplc_utils import show_field, downsample_with_mean, corr


class MPLCScalingSimulation:
    """
        A single MPLC, 0 plane is next to crystal
                   MPLC[0=Diffuser+SLM2,
                       1=diffuser,
                       2=diffuser,
                       ...
                       6==Diffuser+SLM1,
                       7=empty
                       8=lense
                       9=empty
                       10=phases not interesting]
    """
    SLM1_plane = 4
    SLM2_plane = 10

    def __init__(self, path1, path2):
        self.path1 = path1
        self.path2 = path2

        # Klyshko
        self.res = MPLCSimResult()
        self.res.loadfrom(path1)
        self.res2 = MPLCResult()
        self.res2.loadfrom(path2)
        self.initial_field = None
        self.out_desired_spot = None
        self.out_field = None
        self.slm1_phase = 0
        self.slm2_phase = 0

        # first is from fiber to crystal, second from crystal to collection
        self.mplc = MPLCSim(conf=self.res.conf)
        self.mplc.res.masks = self.res.masks
        self.mplc.dist_after_plane = self.mplc.dist_after_plane[::-1]
        self.mplc2 = MPLCSim(conf=self.res2.conf)
        self.mplc2.res.masks = self.res2.masks

    def set_intial_spot(self, sig=0.1, Dx0=0.0, Dy0=0.0):
        spot = get_spot_conf(self.res.conf, sig=sig, Dx0=Dx0, Dy0=Dy0)
        self.initial_field = spot

    def set_out_desired_spot(self, sig=0.1, Dx0=0.0, Dy0=0.0):
        spot = get_spot_conf(self.res.conf, sig=sig, Dx0=Dx0, Dy0=Dy0)
        self.out_desired_spot = spot

    def propagate_klyshko(self, use_slm1=False, use_slm2=False):
        # propagate Klyshko
        # TODO: rethink this
        E_SLM1_plane = self.mplc.propagate_mplc(initial_field=self.initial_field, end_plane=self.SLM1_plane,
                                                prop_last_mask=False)
        if use_slm1:
            E_SLM1_plane *= np.exp(+1j*self.slm1_phase)

        first_plane_plane = self.mplc.propagate_mplc(initial_field=E_SLM1_plane, start_plane=self.SLM1_plane,
                                                     prop_last_mask=True)

        if use_slm2:
            first_plane_plane *= np.exp(+1j*self.slm2_phase)

        # flipping from 2f (anti-correlations)
        E_crystal_plane = np.fliplr(np.flipud(first_plane_plane))

        E_out = self.mplc2.propagate_mplc(initial_field=E_crystal_plane)
        self.out_field = E_out
        return E_out

    def get_fixing_phase_SLM(self, slm_plane=SLM1_plane):
        E_SLM1_plane_forward = self.mplc.propagate_mplc(initial_field=self.initial_field, end_plane=slm_plane,
                                                        prop_last_mask=True)

        # backprop all second MPLC
        E_back_crystal = self.mplc2.propagate_mplc(initial_field=self.out_desired_spot,
                                                   backprop=True, prop_last_mask=True)

        E_back_crystal = np.flipud(E_back_crystal)
        # forward wave props this mask, so the backward wave shouldn't
        E_SLM1_plane_backward = self.mplc.propagate_mplc(initial_field=E_back_crystal, end_plane=slm_plane,
                                                         backprop=True, prop_last_mask=False)

        SLM_phase = np.angle(np.conj(E_SLM1_plane_forward) * E_SLM1_plane_backward)
        display_phase = np.zeros_like(SLM_phase)
        display_phase[self.res.active_slice] = SLM_phase[self.res.active_slice]

        # TODO: why does this return a binary 0 or pi mask???

        return display_phase

    def get_mask_with_degree_of_control(self, mask, degree_of_control):
        phase = np.angle(mask[self.res.active_slice])
        # Nrows, Ncols = phase.shape
        # new_Nrows, new_Ncols = int(degree_of_control*Nrows), int(degree_of_control*Ncols)
        block_size = (int(1/degree_of_control), int(1/degree_of_control))

        downsampled_phase = downsample_with_mean(phase, block_size)

        new_mask = np.zeros_like(mask)
        new_mask[self.res.active_slice] = downsampled_phase
        return np.exp(1j*new_mask)


"""
path1 = "C:\\temp\\speckle_speckle3.mplc_sim"
path2 = "C:\\temp\\speckle_speckle4.mplc_sim"
s = MPLCScalingSimulation(path1, path2)
s.set_intial_spot(sig=0.1, Dx0=-0.3, Dy0=-0.4)
s.set_out_desired_spot(sig=0.6, Dx0=0.3, Dy0=0.5)
speckles = s.propagate_klyshko()
s.slm1_phase = s.get_fixing_phase_SLM(s.SLM1_plane)
s.slm2_phase = s.get_fixing_phase_SLM(s.SLM2_plane)
fixed_SLM1 = s.propagate_klyshko(use_slm1=True)
fixed_SLM2 = s.propagate_klyshko(use_slm2=True)
show_field(s.initial_field, active_slice=s.mplc_sim.res.active_slice, title='initial_field')
show_field(speckles, active_slice=s.mplc_sim.res.active_slice, title='speckles')
show_field(fixed_SLM1, active_slice=s.mplc_sim.res.active_slice, title='fixed_SLM1')
show_field(fixed_SLM2, active_slice=s.mplc_sim.res.active_slice, title='fixed_SLM2')
plt.show()
# show_field(spot, active_slice=res.active_slice)
# spot_power = ((np.abs(spot)**2)[res.active_slice]).sum()
# print(f'{spot_power=}')
"""

# TODO: why do the speckles look weird.
# TODO: given I will have some sort of CR mask - maybe next to the np.flipud also truncate the field?


# TODO: Plan
#  Implement SLM for focusing in relevant region (Start with SLM1 in backprop)
#  Probably will need some statistics with different MPLC phase realizations
#  Scale incomplete control
#  Practically:
#  * implement SLM WFS `find phase` which should be pretty easy
#  * implement some incomplete control mechanism
#  * SLM2 probably has much more active pixels. Check what happens to the spot after the lens in MPLC 1
