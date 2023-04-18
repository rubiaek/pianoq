import pyMMF
import logging
logging.disable()
import matplotlib.pyplot as plt

SOLVER_N_POINTS_SEARCH = 2**8
SOLVER_N_POINTS_MODE = 2**7
SOLVER_R_MAX_COEFF = 1.8
SOLVER_BC_RADIUS_STEP = 0.95
SOLVER_N_BETA_COARSE = 1000
SOLVER_MIN_RADIUS_BC = .5

class Fiber(object):
    def __init__(self, wl=0.808, n1=1.45, NA=0.2, diameter=50, curvature=None, areaSize=None, npoints=2**7, autosolve=True):
        self.NA = NA
        self.diameter = diameter
        self.radius = self.diameter / 2 # in microns
        self.areaSize = areaSize or 2.5*self.radius # calculate the field on an area larger than the diameter of the fiber
        self.npoints = npoints # resolution of the window
        self.n1 = n1
        self.wl = wl # wavelength in microns
        self.curvature = curvature

        self.profile = pyMMF.IndexProfile(npoints=npoints, areaSize=self.areaSize)
        self.profile.initParabolicGRIN(n1=n1, a=self.radius, NA=NA)

        self.solver = pyMMF.propagationModeSolver()
        self.solver.setIndexProfile(self.profile)
        self.solver.setWL(self.wl)

        self.NmodesMax = pyMMF.estimateNumModesGRIN(self.wl, self.radius, self.NA)
        self.modes = None
        self.Nmodes = None

        if autosolve:
            self.solve()

    def solve(self):
        r_max = SOLVER_R_MAX_COEFF * self.diameter
        k0 = 2 * np.pi / self.wl
        dh = self.diameter / SOLVER_N_POINTS_SEARCH
        mode_repr = 'cos'  # or exp for OAM modes

        self.modes = self.solver.solve(mode='radial_test',
                                       r_max=r_max, # max radius to calculate (and first try for large radial boundary condition)
                                       dh=dh, # radial resolution during the computation
                                       min_radius_bc=SOLVER_MIN_RADIUS_BC, # min large radial boundary condition
                                       change_bc_radius_step=SOLVER_BC_RADIUS_STEP, #change of the large radial boundary condition if fails
                                       N_beta_coarse=SOLVER_N_BETA_COARSE, # number of steps of the initial coarse scan
                                       degenerate_mode=mode_repr,
                                       field_limit_tol=1e-4,)
        self.Nmodes = self.modes.number

    def show_mode(self, m):
        """ m mode number"""
        fig, axes = plt.subplots(2)
        axes[0].imshow(np.real(self.modes.profiles[m]).reshape([self.npoints]*2))
        axes[1].imshow(np.imag(self.modes.profiles[m]).reshape([self.npoints]*2))
        fig.show()


class ManyWavelengthSimulation(object):
    def __init__(self, wl0=0.810, Dwl=0.020, N_wl=21):
        self.wl0 = wl0
        self.Dwl = Dwl
        self.N_wl = N_wl
        self.wls = self._get_wl_range()
        self.fibers = [Fiber(wl=wl) for wl in self.wls]
        self.N_modes_cutoff = self.fibers[0].Nmodes # If N_modes changes with wl - discard last modes
        self.betas = np.zeros((N_wl, self.N_modes_cutoff))
        self._populate_betas()

    def _populate_betas(self):
        for i, f in enumerate(self.fibers):
            self.betas[i, :] = f.modes.betas[:self.N_modes_cutoff]

    def _get_wl_range(self):
        """ in um"""
        # stolen from Logan GMMNLSE-Solver-FINAL-master\solve_for_modes.m
        c = 299792458e6 # um/s
        f0 = c / self.wl0 # center frequency in THz
        frange = c / self.wl0**2 * self.Dwl
        df = frange / self.N_wl
        f = f0 + np.arange(-self.N_wl/2, self.N_wl/2)*df
        l = c / f; # um
        return l

    # TODO: find length that will cause this fiber to have a spectral correlation width of ~3nm, and then check our Kilshko two-photon spectral correlation width
    # TODO: this can be defined via choosing a length L such that max_m{beta_m(w+)-beta_m(w-)}*L = 2*pi for w+ - w- = 3nm
    # TODO: then check for same L the (w+ - w-) value such that max_m{beta_m(w+)+beta_m(w-)-w*beta_m(w0)}*2*L = 2*pi, and hope this is larger than 3nm
    def spectral_correlation_width(self):
        pass

s = ManyWavelengthSimulation()
