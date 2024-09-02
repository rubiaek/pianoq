
import numpy as np
import matplotlib.pyplot as plt
from pianoq.simulations.mplc_sim.mplc_sim import MPLCSim
from pianoq.simulations.mplc_sim.mplc_modes2 import gen_input_spots_arrayd, gen_output_modes_Unitary

# All in m

# dist_after_plane[i] is distance between planes i and i+1 (counting from 0)
dist_after_plane = 87e-3*np.ones(10)
dist_after_plane[4] = 138e-3
active_planes = np.array([True] * 11)

waist_in = 80e-6
waist_out = 45e-6
D_between_modes_in = 300e-6
D_between_modes_out = 360e-6

dim=5
which_modes = np.array([ 2,  7, 12, 17, 22,
                        27, 32, 37, 42, 47])

#  Based on "All Mutually Unbiased Bases in Dimensions Two to Five" (2018)
#  The columns in the matrix are the basis elements
q = np.exp(2j * np.pi / 5)  # Complex fifth root of unity
MUB = np.array([
    [1, 1, 1, 1, 1],
    [1, q, q**2, q**3, q**4],
    [1, q**2, q**4, q, q**3],
    [1, q**3, q, q**4, q**2],
    [1, q**4, q**3, q**2, q]
]) / np.sqrt(5)  # eq. 33

Matrix_trans1 = MUB.conj().T  # To measure in X basis, we need to act with X^dag on the state
zeros_mat = np.zeros((5, 5))
full_transformation = np.block([[Matrix_trans1, zeros_mat],
                                [zeros_mat, np.conj(Matrix_trans1)]])  # [DFT,0; 0, iDFT] final matrix. the conj in bottom right is for correlation to be on identity

N_modes = 10
# All in mm
conf = {'wavelength': 810e-9,  # mm
        'dist_after_plane': dist_after_plane,  # mm
        'active_planes': active_planes,  # bool
        'N_iterations': 30,
        'Nx': 140,  # Number of grid points x-axis
        'Ny': 360,  # Number of grid points y-axis
        'dx': 12.5e-6,  # mm - SLM pixel sizes
        'dy': 12.5e-6,  # mm
        'max_k_constraint': 0.15,  # Ohad: better than 0.1 or 0.2, but not very fine-tuned
        'N_modes': N_modes,
        'min_log_level': 2,
        'size_factor': 3,  # assumed to be odd. Have physical larger grid than the actual SLM planes
        'use_mask_offset': True,
        }


mplc = MPLCSim(conf=conf)
# input_modes = get_spots_modes_conf(conf, sig=0.1, N_rows=N_N_modes, N_cols=N_N_modes, spacing=0.6)
input_spots, x_modes_in, y_modes_in = gen_input_spots_arrayd(waist_in=waist_in, D_between_modes=D_between_modes_in, XX=mplc.XX, YY=mplc.YY, dim=dim)
input_modes = input_spots[which_modes]
x_modes_in = x_modes_in[which_modes]
y_modes_in = y_modes_in[which_modes]

output_modes, phase_pos_x, phase_pos_y = gen_output_modes_Unitary(waist_out, D_between_modes_out, mplc.XX, mplc.YY, full_transformation, dim, which_modes)

mplc.set_modes(input_modes, output_modes)
mplc.find_phases(show_fidelities=False, iterations=5, fix_initial_phases=True)

mplc.res._calc_normalized_overlap()
print(np.angle(np.diag(mplc.res.forward_overlap)))
mplc.res._calc_fidelity()
print(mplc.res.fidelity)