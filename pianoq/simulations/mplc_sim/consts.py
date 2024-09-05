import numpy as np

# dist_after_plane[i] is distance between planes i and i+1 (counting from 0)
dist_after_plane = 87e-3 * np.ones(10)
dist_after_plane[4] = 138e-3

default_wfm_conf = {'wavelength': 810e-9,  # mm
                    'dist_after_plane': dist_after_plane,  # mm
                    'N_iterations': 30,
                    'Nx': 140,  # Number of grid points x-axis
                    'Ny': 360,  # Number of grid points y-axis
                    'dx': 12.5e-6,  # mm - SLM pixel sizes
                    'dy': 12.5e-6,  # mm
                    'max_k_constraint': 0.15,  # Ohad: better than 0.1 or 0.2, but not very fine-tuned
                    'min_log_level': 2,
                    'size_factor': 3,  # assumed to be odd. Have physical larger grid than the actual SLM planes
                    'use_mask_offset': True,
                    'symmetric_masks': False,
                    }
