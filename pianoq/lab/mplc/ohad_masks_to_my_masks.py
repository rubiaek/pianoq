import numpy as np
import scipy.io

from pianoq.lab.mplc.mplc_device import MPLCDevice
from pianoq.lab.mplc.consts import SLM_DIMS, MASK_DIMS, PIXEL_SIZE, N_SPOTS, D_BETWEEN_SPOTS_INPUT, SPOT_WAIST_IN


orig_masks_path = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\Masks_31_10_23_QKD5d_MUB2_mm_33_3_conjbases.mat"
MASKS = scipy.io.loadmat(orig_masks_path)['MASKS']
orig_phases_path = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\phase_align_QKD5d_10_11_23_3.mat"

height, width = (1080, 420)
assert MASKS[0].shape == (height, width)

# Matlab "combine_masks" gets two mask sets, and takes the upper part of the first set
# and the lower part of the second set
# I assume that this concatenation will happen before, because this is just weird

# Matlab "cut_masks" - carves out the middle, and takes only first 10 masks
MASKS = MASKS[:10, 360:720, 140:280]
MASKS = np.angle(MASKS)

## Matlab `remove_input_modes`
# `make_grid_phasemask`
Nx = MASK_DIMS[1]
Ny = MASK_DIMS[2]
dx = PIXEL_SIZE
dy = PIXEL_SIZE
X = (np.arange(Nx) - (Nx//2 - 0.5)) * PIXEL_SIZE  # 0.5 for symmetry
Y = (np.arange(Ny) - (Ny//2 - 0.5)) * PIXEL_SIZE  # 0.5 for symmetry
XX, YY = np.meshgrid(X, Y)

# `calc_pos_modes_in`
n_steps_x = []
n_steps_y = []
dim = int(np.sqrt(N_SPOTS))

# this loop results in n_steps_x a len 25 array, of step sizes from middle,
# according to the numbering from the middle right, upwards, then middle second from right, etc.
# for say 3 it will be: 1,1,1,0,0,0,-1,-1,-1
# because of the numbering convention, in y it will be 0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5
for l in range(dim):
    if dim % 2:
        n_steps_x.extend([(dim - 1) / 2 - l] * dim)
    else:
        n_steps_x.extend([0.5 + (dim / 2 - 1 - l)] * dim)
    n_steps_y.extend([0.5 + i for i in range(dim)])

    # adding the other 25 modes
    n_steps_x = n_steps_x + n_steps_x[::-1]
    n_steps_y = n_steps_y + [-y for y in n_steps_y]

# These are lists of the middle of all 50 modes
x_modes_in = D_BETWEEN_SPOTS_INPUT * np.array(n_steps_x)
y_modes_in = D_BETWEEN_SPOTS_INPUT * np.array(n_steps_y)

# actual `remove_input_modes`
modes_to_keep = np.array([1,6,11,16,21,26,31,36,41,46])
row_num_signal = 3
row_num_idler = 3
modes_to_keep[:5] += (row_num_signal - 1)
modes_to_keep[5:] += (row_num_idler - 1)

lin_mask = 2*np.pi*XX/(8*dx)  # 2 pi within 8 pixels

for k in range(2 * N_SPOTS):
    if k + 1 not in modes_to_keep:  # Python uses 0-based indexing, so we add 1 to match MATLAB's 1-based indexing
        condition = np.sqrt((XX - x_modes_in[k])**2 + (Y - y_modes_in[k])**2) < (2 * SPOT_WAIST_IN)
        MASKS[0][condition] = lin_mask[condition]

# Matlab `add_phase_input_spots`
# phases = np.zeros(50)
phases = scipy.io.loadmat(orig_phases_path)['phases']
for k in range(2 * N_SPOTS):
    condition = np.sqrt((XX - x_modes_in[k])**2 + (YY - y_modes_in[k])**2) < (2 * SPOT_WAIST_IN)
    MASKS[0][condition] = MASKS[0][condition] + phases[k]


masks = MASKS.copy()

final_path = r"C:\temp\ronen.masks"

f = open(final_path, 'wb')
np.savez(f, masks=masks)
f.close()

m = MPLCDevice()

"""

V combined_mask = obj.combine_masks(masks1,masks2); %combine the two masks
V combined_mask(1,:,:) = obj.remove_input_modes(modes_keep,squeeze(combined_mask(1,:,:))); %add linear tilt to unwanted input spots
V combined_mask(1,:,:) = obj.add_phase_input_spots(phases,squeeze(combined_mask(1,:,:))); %add phases to input spots
V masks_final = obj.cut_masks(combined_mask);

V MPLC1.combine_MPLC_mask(masks_final); %combining and displaying the final phase mask

"""