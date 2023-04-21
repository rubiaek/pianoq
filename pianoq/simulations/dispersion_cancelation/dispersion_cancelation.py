import pyMMF
import logging
logging.disable()
import matplotlib.pyplot as plt
import numpy as np

SOLVER_N_POINTS_SEARCH = 2**8
SOLVER_N_POINTS_MODE = 2**7
SOLVER_R_MAX_COEFF = 1.8
SOLVER_BC_RADIUS_STEP = 0.95
SOLVER_N_BETA_COARSE = 1000
SOLVER_MIN_RADIUS_BC = .5


class Fiber(object): 
    def __init__(self, wl=0.808, n1=1.453, NA=0.2, diameter=50, curvature=None, areaSize=None, npoints=2**7, autosolve=True, L=2e6):
        """ all in um """
        self.NA = NA
        self.diameter = diameter
        self.radius = self.diameter / 2 # in microns
        self.areaSize = areaSize or 2.5*self.radius # calculate the field on an area larger than the diameter of the fiber
        self.npoints = npoints  # resolution of the window
        self.n1 = n1
        self.wl = wl  # wavelength in microns
        self.curvature = curvature
        self.L = L

        self.profile = pyMMF.IndexProfile(npoints=npoints, areaSize=self.areaSize)
        self.profile.initParabolicGRIN(n1=n1, a=self.radius, NA=NA)

        self.solver = pyMMF.propagationModeSolver()
        self.solver.setIndexProfile(self.profile)
        self.solver.setWL(self.wl)

        self.profile_0 = np.zeros([self.npoints]*2)
        self.modes_0 = None
        self.profile_end = np.zeros([self.npoints]*2)
        self.modes_end = None

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
                                       r_max=r_max,  # max radius to calculate (and first try for large radial boundary condition)
                                       dh=dh,  # radial resolution during the computation
                                       min_radius_bc=SOLVER_MIN_RADIUS_BC,  # min large radial boundary condition
                                       change_bc_radius_step=SOLVER_BC_RADIUS_STEP,  # change of the large radial boundary condition if fails
                                       N_beta_coarse=SOLVER_N_BETA_COARSE,  # number of steps of the initial coarse scan
                                       degenerate_mode=mode_repr,
                                       field_limit_tol=1e-4,)
        self.Nmodes = self.modes.number

    def _get_gausian(self, sig, ravel=True):
        """ sig in pixels """
        X = np.arange(-self.npoints/2, self.npoints/2)
        XX, YY = np.meshgrid(X, X)
        # sqrt in 2d, and also 4*sig**2, because field and not power, so abs(g)**2.sum() = 1
        g = 1 / np.sqrt(sig**2 * 2 * np.pi) * np.exp(-(XX ** 2 + YY ** 2) / (4 * sig ** 2))
        if ravel:
            return g.ravel()
        else:
            return g

    def propagate(self, gaussian=True, sigma=10, shift=(3, 9)):
        if not gaussian:
            raise Exception('no other modes supported yet :P')

        self.profile_0 = self._get_gausian(sig=sigma, ravel=True)
        # excite modes with input of shifted gaussian
        self.modes_0 = self.modes.getModeMatrix(shift=shift).T @ self.profile_0
        # evolute modes
        self.modes_end = self.modes.getPropagationMatrix(distance=self.L) @ self.modes_0
        # modes to profile (not shifted, because it is what it is)
        self.profile_end = self.modes_end.T @ self.modes.getModeMatrix().T
        return self.profile_end

    def show_mode(self, m):
        """ m mode number"""
        fig, axes = plt.subplots(2)
        axes[0].imshow(np.real(self.modes.profiles[m]).reshape([self.npoints]*2))
        axes[1].imshow(np.imag(self.modes.profiles[m]).reshape([self.npoints]*2))
        fig.show()


class ManyWavelengthSimulation(object):
    def __init__(self, wl0=0.810, Dwl=0.020, N_wl=21, Nmodes=50):
        """ all in um """
        self.wl0 = wl0
        self.Dwl = Dwl
        self.N_wl = N_wl
        self.wls = self._get_wl_range()
        self.ns = self._sellmeier_silica(self.wls)
        self.fibers = []
        for i, wl in enumerate(self.wls):
            self.fibers.append(Fiber(wl=wl, n1=self.ns[i]))
        self.N_modes_cutoff = self.fibers[-1].Nmodes  # If N_modes changes with wl - discard last modes
        self.betas = np.zeros((N_wl, self.N_modes_cutoff))
        self._populate_betas()

    def _populate_betas(self):
        for i, f in enumerate(self.fibers):
            self.betas[i, :] = f.modes.betas[:self.N_modes_cutoff]

    def _sellmeier_silica(self, wls):
        a1 = 0.6961663
        a2 = 0.4079426
        a3 = 0.8974794
        b1 = 0.0684043
        b2 = 0.1162414
        b3 = 9.896161

        ns_silica = np.sqrt(1 +
                      a1 * (wls**2) / (wls**2 - b1**2) +
                      a2 * (wls**2) / (wls**2 - b2**2) +
                      a3 * (wls**2) / (wls**2 - b3**2))
        return ns_silica

    def _get_wl_range(self):
        """ in um"""
        # stolen from Logan GMMNLSE-Solver-FINAL-master\solve_for_modes.m
        c = 299792458e6  # um/s
        f0 = c / self.wl0  # center frequency in THz
        frange = c / self.wl0**2 * self.Dwl
        df = frange / self.N_wl
        f = f0 + np.arange(-self.N_wl/2, self.N_wl/2)*df
        l = c / f  # um
        return l[::-1]

    def show_betas(self):
        fig, ax = plt.subplots()
        for f in self.fibers:
            ax.plot(f.modes.betas, label=f'$ \lambda= ${f.wl:.4f}')
            ax.set_xlabel('mode #')
            ax.set_ylabel(r'Propagation constant $\beta$ (in $\mu$m$^{-1}$)')
            ax.legend()
        fig.show()

        fig, ax = plt.subplots()
        for i in np.arange(0, 50, 5):
            ax.plot(self.wls, self.betas[:, i], label=f'mode no. {i}')
            ax.set_xlabel(r'$\lambda (\mu m m)$')
            ax.set_ylabel(r'Propagation constant $\beta$ (in $\mu$m$^{-1}$)')
            ax.legend()
        fig.show()

    def show_PCC_classical_and_quantum(self):
        # TODO: different init conditions
        pccs = np.zeros_like(self.fibers)
        E_end0 = self.fibers[0].propagate(gaussian=True, sigma=10, shift=(3, 9))
        I_end0 = np.abs(E_end0) ** 2

        for i, f in enumerate(self.fibers):
            E_end = f.propagate(gaussian=True, sigma=10, shift=(3, 9))
            I_end = np.abs(E_end)**2
            pccs[i] = np.corrcoef(I_end0.ravel(), I_end.ravel())[1, 0]

        fig, ax = plt.subplots()
        ax.plot(self.wls, pccs, label='classical')
        ax.set_xlabel(r'wl ($\mu m$)')
        ax.set_ylabel(r'PCC')
        ax.legend()
        fig.show()



    # TODO: find length that will cause this fiber to have a spectral correlation width of ~3nm, and then check our Kilshko two-photon spectral correlation width
    # TODO: this can be defined via choosing a length L such that max_m{beta_m(w+)-beta_m(w-)}*L = 2*pi for w+ - w- = 3nm
    # TODO: then check for same L the (w+ - w-) value such that max_m{beta_m(w+)+beta_m(w-)-w*beta_m(w0)}*2*L = 2*pi, and hope this is larger than 3nm
    def spectral_correlation_width(self):
        Dbetas = s.betas[10, :] - s.betas[13, :]  # take 3nm of spectrum
        Dbetas -= np.median(Dbetas) # Global phase. median and not mean because there are outliers
        # everything here is in microns, so 2m of piano is 2e6,
        L = 2e6 # 2m fiber
        # this gives Dbetas~1 at least for the first ~30 modes, except for a few modes specific modes, not sure why
        Dphis = Dbetas*L

        DDbetas = s.betas[0, :] + s.betas[20, :] - 2 * s.betas[10, :] # Klyshko picture, 20nms hoping is OK
        DDbetas -= np.median(DDbetas)
        DDphis = DDbetas*2*L  # seems good! even with 20nm bandwidth we seem to still be far off from 2pi!


s = ManyWavelengthSimulation()
