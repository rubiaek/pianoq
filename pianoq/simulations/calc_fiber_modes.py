# import pyMMF


def get_modes_FG010LDA(npoints=2**7):
    # Following https://github.com/wavefrontshaping/pyMMF/blob/master/example/Benchmark_SI.ipynb
    NA = 0.1
    radius = 10  # in microns
    areaSize = 2.5 * radius  # calculate the field on an area larger than the diameter of the fiber
    npoints = npoints  # resolution of the window
    n1 = 1.45
    wl = 0.6328  # wavelength in microns
    curvature = None

    profile = pyMMF.IndexProfile(npoints=npoints, areaSize=areaSize)
    profile.initStepIndex(n1=n1, a=radius, NA=NA)
    solver = pyMMF.propagationModeSolver()
    solver.setIndexProfile(profile)
    solver.setWL(wl)
    Nmodes_estim = pyMMF.estimateNumModesSI(wl, radius, NA, pola=1)

    print(f"Estimated number of modes using the V number = {Nmodes_estim}")

    # modes_semianalytical = solver.solve(mode='SI', curvature=None)

    modes_eig = solver.solve(nmodesMax=Nmodes_estim + 10, boundary='close', mode='eig',
                             curvature=None, propag_only=True)

    return modes_eig.profiles


if __name__ == "__main__":
    profiles = get_modes_FG010LDA()
