from pianoq.simulations.mplc.mplc_result import MPLCResult

res = MPLCResult()
res.loadfrom("C:\\temp\\speckle_speckle.mplc")

# TODO: Plan
#  Show what happens to a spot through the calculated MPLC
#  Show this forward & backward & also Klyshko (==both)
#  Implement SLM for focusing in relevant region (Start with SLM1 in backprop)
#  Probably will need some statistics with different MPLC phase realizations
#  Scale incomplete control
#  Practically:
#  * implement `propagate_MPLC(initial_field=E, backward=True)`, which will need a `freespace(E, L)` helper function
#   - note that "backwards" has two meanings: 1) exp(-1*j), 2) start from plane N-1.
#   - do this in current MPLC class. maybe `propagate_MPLC(initial_field=E, start_plane=N-1, end_plane=6)`
#  * implement `find phase` which should be pretty easy
#  * implement some incomplete control mechanism
