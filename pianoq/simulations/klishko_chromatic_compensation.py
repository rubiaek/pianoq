import pyMMF
import pyMMF.modes
import numpy as np
import matplotlib.pyplot as plt
N_POINTS_MODES = 2**8  # resolution of the window


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
        npoints=N_POINTS_MODES,
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


def get_field(V, modes: pyMMF.modes.Modes):
    M0 = modes.getModeMatrix()
    field = np.zeros([N_POINTS_MODES] * 2)
    for i in range(len(modes.betas)):
        Mi = modes.profiles[i]
        profile_i = Mi.reshape([N_POINTS_MODES] * 2)
        field = field + profile_i * V[i]

    return field


def main():
    modes810 = get_OM1_modes(wl=0.810)
    L = 2e6  # um
    TM = modes810.getPropagationMatrix(L)
    V0 = np.ones(len(modes810.betas))
    Vout = TM@V0



if __name__ == "__main__":
    pass
