import numpy as np
from pianoq.simulations.mplc.mplc_result import MPLCResult
from pianoq.simulations.mplc.mplc import MPLC
from pianoq.simulations.mplc.mplc_modes import get_spot_conf
from pianoq.simulations.mplc.mplc_utils import show_field

res = MPLCResult()
res.loadfrom("C:\\temp\\speckle_speckle3.mplc")

mplc = MPLC(conf=res.conf)
spot = get_spot_conf(res.conf, sig=0.1, Dx0=-0.3, Dy0=0.2)
show_field(spot, active_slice=res.active_slice)
spot_power = ((np.abs(spot)**2)[res.active_slice]).sum()
print(f'{spot_power=}')

mplc.res.masks = res.masks
E_out = mplc.propagate_mplc(initial_field=spot)
show_field(E_out, active_slice=res.active_slice)
forward_speckle_power = ((np.abs(E_out)**2)[res.active_slice]).sum()
print(f'{forward_speckle_power=}')


mplc.res.masks = res.masks[::-1]
E_out2 = mplc.propagate_mplc(initial_field=spot)
show_field(E_out2, active_slice=res.active_slice)
backward_speckle_power = ((np.abs(E_out2)**2)[res.active_slice]).sum()
print(f'{backward_speckle_power=}')


# TODO: Plan
#  Show this forward & backward & also Klyshko (==both)
#  Implement SLM for focusing in relevant region (Start with SLM1 in backprop)
#  Probably will need some statistics with different MPLC phase realizations
#  Scale incomplete control
#  Practically:
#  * implement SLM WFS `find phase` which should be pretty easy
#  * implement some incomplete control mechanism
