import numpy as np
import scipy.io
from pianoq.lab.mplc.mplc_device import MPLCDevice


orig_path = r"G:\My Drive\Ohad and Giora\MPLC\matlab codes\Ronen stuff 17.7.24\Masks_31_10_23_QKD5d_MUB2_mm_33_3_conjbases.mat"
MASKS = scipy.io.loadmat(orig_path)['MASKS']

height, width = (1080, 420)
assert MASKS[0].shape == (height, width)

# Matlab "combine_masks" gets two mask sets, and takes the upper part of the first set
# and the lower part of the second set
# I assume that this concatenation will happen before, because this is just weird


# Matlab "cut_masks" - carves out the middle, and takes only first 10 masks
MASKS = MASKS[:10, 360:720, 140:280]
MASKS = np.angle(MASKS)

masks = MASKS.copy()

final_path = r"C:\temp\ronen.masks"

f = open(final_path, 'wb')
np.savez(f, masks=masks)
f.close()


m = MPLCDevice()

"""
modes_keep = [1,6,11,16,21,26,31,36,41,46];
modes_keep = [modes_keep(1:5)+mmmmm-1,modes_keep(6:10)+mmmmm-1];

V combined_mask = obj.combine_masks(masks1,masks2); %combine the two masks
combined_mask(1,:,:) = obj.remove_input_modes(modes_keep,squeeze(combined_mask(1,:,:))); %add linear tilt to unwanted input spots
combined_mask(1,:,:) = obj.add_phase_input_spots(phases,squeeze(combined_mask(1,:,:))); %add phases to input spots
V masks_final = obj.cut_masks(combined_mask);

MPLC1.combine_MPLC_mask(masks_final); %combining and displaying the final phase mask

"""