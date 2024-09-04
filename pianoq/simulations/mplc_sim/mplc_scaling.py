import numpy as np
from pianoq.simulations.mplc_sim.mplc_sim_result import MPLCSimResult, MPLCMasks
from pianoq.simulations.mplc_sim.mplc_sim import MPLCSim
from pianoq.simulations.mplc_sim.mplc_modes import get_spot_conf
from simulations.mplc_sim.mplc_utils import downsample_phase


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
        self.slm1_phase = np.array([], dtype=np.complex128)
        self.slm2_phase = np.array([], dtype=np.complex128)

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
            E_SLM1_plane *= np.exp(+1j * np.angle(self.slm1_phase))

        E_SLM2_plane = self.mplc.propagate_mplc(initial_field=E_SLM1_plane,
                                                start_plane=self.SLM1_plane,
                                                end_plane=self.SLM2_plane,
                                                prop_first_mask=True, # propagating this middle mask here and not above
                                                prop_last_mask=True)

        if use_slm2:
            # here we assume SLM2 works only on one photon for simplicity
            E_SLM2_plane *= np.exp(+1j * np.angle(self.slm2_phase))

        # flipping from 2f (anti-correlations)
        E_flipped = np.fliplr(np.flipud(E_SLM2_plane))
        E_filtered = self.filter_cr_mask(E_flipped)

        E_out = self.mplc.propagate_mplc(initial_field=E_filtered,
                                         start_plane=self.SLM2_plane,
                                         end_plane=self.mplc.N_planes - 1,
                                         prop_first_mask=True,  # again, after the flip
                                         prop_last_mask=False)  # unphysical 11th plane
        self.out_field = E_out
        return E_out

    def get_overlap_at_plane(self, slm_plane):
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
        E_filtered = self.filter_cr_mask(E_0_plane_backward)

        if slm_plane == 0:
            # No need for any more propagation: no free-space, and no phase, since the forward wave accumulated the phase
            E_SLM_plane_backward = E_filtered
        else:
            E_SLM_plane_backward = self.mplc.propagate_mplc(initial_field=E_filtered,
                                                            start_plane=0,
                                                            end_plane=slm_plane,
                                                            prop_first_mask=True,  # why not
                                                            prop_last_mask=False,  # forward wave props this mask, so the backward wave shouldn't
                                                            backprop=True)

        overlap = np.conj(E_SLM_plane_forward) * E_SLM_plane_backward
        return overlap

    def filter_cr_mask(self, E, type='rect'):
        if type == 'rect':
            # kill light outside the active_slice
            E_filtered = np.zeros_like(E)
            E_filtered[self.res.active_slice] = E[self.res.active_slice]
            return E_filtered
        else:
            # TODO: holes of Ohad mask
            raise NotImplementedError

    def get_fixing_phase_SLM(self, slm_plane):
        overlap = self.get_overlap_at_plane(slm_plane)
        return self.get_mask_with_degree_of_control(overlap, 1)

    def get_mask_with_degree_of_control(self, overlap, degree_of_control, weighted=True):
        block_size = (int(1/degree_of_control), int(1/degree_of_control))
        downsampled_phase = downsample_phase(overlap, block_size, weighted=weighted)

        display_phase = np.ones_like(downsampled_phase, dtype=np.complex64)
        display_phase[self.res.active_slice] = downsampled_phase[self.res.active_slice]

        return display_phase
