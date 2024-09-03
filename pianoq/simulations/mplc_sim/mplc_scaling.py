import numpy as np
import matplotlib.pyplot as plt
from pianoq.simulations.mplc_sim.mplc_sim_result import MPLCSimResult, MPLCMasks
from pianoq.simulations.mplc_sim.mplc_sim import MPLCSim
from pianoq.simulations.mplc_sim.mplc_modes import get_spot_conf
from pianoq.simulations.mplc_sim.mplc_utils import show_field, downsample_with_mean, corr


class MPLCScalingSimulation:
    """
        A single MPLC with upper and lower parts of each plane, 0 plane is at the farfield of the crystal
                   MPLC[0=Diffuser+SLM2,
                       1=diffuser,
                       2=diffuser,
                       ...
                       6==Diffuser+SLM1,
                       7=empty
                       8=lense
                       9=empty
                       10=phases not interesting, detector / Klyshko source here]
    """
    SLM1_plane = 6
    SLM2_plane = 0

    def __init__(self, masks_path):
        self.masks_path = masks_path

        # Klyshko
        self.res = MPLCMasks()
        self.res.loadfrom(self.masks_path)

        self.initial_field = None
        self.out_desired_spot = None
        self.out_field = None
        self.slm1_phase = 0
        self.slm2_phase = 0

        # first is from fiber to crystal, second from crystal to collection
        self.mplc = MPLCSim(conf=self.res.conf)
        self.mplc.res.masks = self.res.big_masks # with the 3 size factor

    def set_intial_spot(self, sig=80e-6, Dx0=0.0, Dy0=0.0):
        spot = get_spot_conf(self.res.conf, sig=sig, Dx0=Dx0, Dy0=Dy0)
        self.initial_field = spot

    def set_out_desired_spot(self, sig=80e-6, Dx0=0.0, Dy0=0.0):
        spot = get_spot_conf(self.res.conf, sig=sig, Dx0=Dx0, Dy0=Dy0)
        self.out_desired_spot = spot

    def propagate_klyshko(self, use_slm1=False, use_slm2=False):
        E_SLM1_plane = self.mplc.propagate_mplc(initial_field=self.initial_field,
                                                start_plane=self.mplc.N_planes-1, # last plane - where detector / laser is
                                                end_plane=self.SLM1_plane,
                                                prop_first_mask=False, # unphysical 11th plane
                                                prop_last_mask=False)  # will do in next line
        if use_slm1:
            E_SLM1_plane *= np.exp(+1j*self.slm1_phase)

        E_SLM2_plane = self.mplc.propagate_mplc(initial_field=E_SLM1_plane,
                                                start_plane=self.SLM1_plane,
                                                end_plane=self.SLM2_plane,
                                                prop_first_mask=True, # propagating this middle mask here and not above
                                                prop_last_mask=True)

        if use_slm2:
            # here we assume SLM2 works only on one photon for simplicity
            E_SLM2_plane *= np.exp(+1j*self.slm2_phase)

        # flipping from 2f (anti-correlations)
        E_flipped = np.fliplr(np.flipud(E_SLM2_plane))
        # simulate Cr mask -> kill light outside the active_slice
        E_flipped[self.res.active_slice] = 0

        E_out = self.mplc.propagate_mplc(initial_field=E_flipped,
                                         start_plane=self.SLM2_plane,
                                         end_plane=self.mplc.N_planes - 1,
                                         prop_first_mask=True,  # again, after the flip
                                         prop_last_mask=False)  # unphysical 11th plane
        self.out_field = E_out
        return E_out

    def get_fixing_phase_SLM(self, slm_plane):
        E_SLM_plane_forward = self.mplc.propagate_mplc(initial_field=self.initial_field,
                                                       start_plane=self.mplc.N_planes - 1,
                                                       end_plane=slm_plane,
                                                       prop_first_mask=False,  # unphysical 11th plane
                                                       prop_last_mask=True)  # propagate phase of meeting plane in forward
                                                                             # "meeting 1um `after` this mask"

        # backprop all second MPLC
        E_0_plane_backward = self.mplc.propagate_mplc(initial_field=self.out_desired_spot,
                                                      start_plane=self.mplc.N_planes - 1,
                                                      end_plane=0,
                                                      prop_first_mask=False,  # unphysical 11th plane
                                                      prop_last_mask=True,
                                                      backprop=True)

        # flipping from 2f (anti-correlations)
        E_0_plane_backward = np.fliplr(np.flipud(E_0_plane_backward))
        # simulate Cr mask -> kill light outside the active_slice
        E_0_plane_backward[self.res.active_slice] = 0

        if slm_plane == 0:
            # No need for any more propagation: no free-space, and no phase, since the forward wave accumulated the phase
            E_SLM_plane_backward = E_0_plane_backward
        else:
            E_SLM_plane_backward = self.mplc.propagate_mplc(initial_field=E_0_plane_backward,
                                                            start_plane=0,
                                                            end_plane=slm_plane,
                                                            prop_first_mask=True,  # why not
                                                            prop_last_mask=False,  # forward wave props this mask, so the backward wave shouldn't
                                                            backprop=True)

        SLM_phase = np.angle(np.conj(E_SLM_plane_forward) * E_SLM_plane_backward)
        display_phase = np.ones_like(SLM_phase, dtype=np.complex64)
        display_phase[self.res.active_slice] = SLM_phase[self.res.active_slice]

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
