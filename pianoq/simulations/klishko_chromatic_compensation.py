import re
import pyMMF
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import glob
FOLDER = r'G:\My Drive\Lab Wiki\Optical fibers\Nonlinear-Logan simulation\Logan Simulation\GMMNLSE-Solver-FINAL-master\Fibers\GRIN_ronen2'

def get_OM1_modes(wl=0.810):
    NA = 0.275
    radius = 62.5 / 2  # in microns
    areaSize = 2.4 * radius  # calculate the field on an area larger than the diameter of the fiber
    n_points_modes = 256  # resolution of the window
    n1 = 1.49  # index of refraction at r=0 (maximum)
    curvature = None
    k0 = 2. * np.pi / wl

    r_max = 3.2 * radius
    npoints_search = 2 ** 8
    dh = 2 * radius / npoints_search

    # solver parameters
    change_bc_radius_step = 0.95
    N_beta_coarse = 1000
    degenerate_mode = 'exp'
    min_radius_bc = 1.5

    profile = pyMMF.IndexProfile(
        npoints=n_points_modes,
        areaSize=areaSize
    )
    profile.initParabolicGRIN(n1=n1, a=radius, NA=NA)

    solver = pyMMF.propagationModeSolver()
    solver.setIndexProfile(profile)
    solver.setWL(wl)
    modes = solver.solve(mode='radial',
                         curvature=curvature,
                         r_max=r_max,  # max radius to calculate (and first try for large radial boundary condition)
                         dh=dh,  # radial resolution during the computation
                         min_radius_bc=min_radius_bc,  # min large radial boundary condition
                         change_bc_radius_step=change_bc_radius_step,
                         # change of the large radial boundary condition if fails
                         N_beta_coarse=N_beta_coarse,  # number of steps of the initial coarse scan
                         degenerate_mode=degenerate_mode
                         )

    return modes


def get_logan_modes(folder, wavelength):
    files = glob.glob(folder + f'\\*wavelength{wavelength}.mat')
    d = {}
    for file in files:
        mode_num = re.findall('fieldscalarmode(\d+)', files[0])[0]
        Q = loadmat(file)
        d[mode_num] = {'profile': Q['phi'], 'neff': Q['neff'][0][0]}

    return d


if __name__ == "__main__":
    pass
