import cv2
import pyMMF
import logging
logging.disable()
import matplotlib.pyplot as plt
import numpy as np
from colorsys import hls_to_rgb
from pianoq_results.misc import Player


SOLVER_N_POINTS_SEARCH = 2**8
SOLVER_N_POINTS_MODE = 2**7
SOLVER_R_MAX_COEFF = 1.8
SOLVER_BC_RADIUS_STEP = 0.95
SOLVER_N_BETA_COARSE = 1000
SOLVER_MIN_RADIUS_BC = .5


def _colorize(z, theme='dark', saturation=1., beta=1.4, transparent=False, alpha=1., max_threshold=1.):
    r = np.abs(z)
    r /= max_threshold * np.max(np.abs(r))
    arg = np.angle(z)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1. / (1. + r ** beta) if theme == 'white' else 1. - 1. / (1. + r ** beta)
    s = saturation

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    if transparent:
        a = 1. - np.sum(c ** 2, axis=-1) / 3
        alpha_channel = a[..., None] ** alpha
        return np.concatenate([c, alpha_channel], axis=-1)
    else:
        return c


class Fiber(object):
    def __init__(self, wl=0.808, n1=1.453, NA=0.2, diameter=50, curvature=None, areaSize=None, npoints=2**7, autosolve=True, L=2e6):
        """ all in um """
        self.rng = np.random.default_rng(12345)

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

        self.profile_0 = np.zeros(self.npoints**2)
        self.modes_0 = None
        self.profile_end = np.zeros(self.npoints**2)
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

    def _get_gausian(self, sig, X0=0, Y0=0, X_linphase=0.0, Y_linphase=0.0, random_phase=0.0, ravel=True):
        """ sig in pixels """
        X = np.arange(-self.npoints/2, self.npoints/2)
        XX, YY = np.meshgrid(X, X)
        # sqrt in 2d, and also 4*sig**2, because field and not power, so abs(g)**2.sum() = 1
        g = 1 / np.sqrt(sig**2 * 2 * np.pi) * np.exp(-((XX-X0) ** 2 + (YY-Y0) ** 2) / (4 * sig ** 2))

        if X_linphase != 0 or Y_linphase != 0:
            g = np.exp(1j*(XX*X_linphase + YY*Y_linphase)) * g

        if random_phase != 0:
            A = random_phase*self.rng.normal(size=(40, 40))
            A = cv2.resize(A, g.shape, interpolation=cv2.INTER_AREA)
            g *= np.exp(1j*A)

        if ravel:
            return g.ravel()
        else:
            return g

    def set_input_gaussian(self, sigma=10, X0=0, Y0=0, X_linphase=0.0, Y_linphase=0.0, random_phase=0.0):
        self.profile_0 = self._get_gausian(sig=sigma, X0=X0, Y0=Y0,
                                           X_linphase=X_linphase, Y_linphase=Y_linphase,
                                           random_phase=random_phase, ravel=True)

    def set_input_random_modes(self, first_N_modes=50):
        amps = np.zeros(self.Nmodes).astype(np.complex128)
        amps[:first_N_modes] = self.rng.uniform(-1, 1, first_N_modes) + 1j * self.rng.uniform(-1, 1, first_N_modes)
        C = np.sqrt(((np.abs(amps)**2).sum()))
        self.modes_0 = amps / C
        self.profile_0 = self.modes_0.T @ self.modes.getModeMatrix().T

    def propagate(self, show=True):
        self.modes_0 = self.modes.getModeMatrix().T @ self.profile_0
        # evolute modes
        self.modes_end = self.modes.getPropagationMatrix(distance=self.L) @ self.modes_0
        # modes to profile (not shifted, because it is what it is)
        self.profile_end = self.modes_end.T @ self.modes.getModeMatrix().T

        if show:
            fig, ax = plt.subplots(1, 2)
            self.show_profile(self.profile_0, ax[0])
            self.show_profile(self.profile_end, ax[1])
            fig.show()

        return self.profile_end

    def show_profile(self, profile, ax=None):
        if len(profile.shape) == 1:
            n = np.sqrt(profile.size)
            assert n.is_integer()
            n = int(n)
            profile = profile.reshape([n] * 2)

        if ax is None:
            fig, ax = plt.subplots()
        ax.imshow(_colorize(profile))
        power_transmitted = (np.abs(profile)**2).sum()
        ax.set_title(f'total power: {power_transmitted:.3f}')
        ax.figure.show()

    def show_mode(self, m):
        """ m mode number"""
        fig, axes = plt.subplots(2)
        axes[0].imshow(np.real(self.modes.profiles[m]).reshape([self.npoints]*2))
        axes[1].imshow(np.imag(self.modes.profiles[m]).reshape([self.npoints]*2))
        fig.show()

    def animate_modes(self):
        fig = plt.figure(figsize = (10,5))
        ax = fig.add_subplot(1,1,1)
        ax.set_ylim(0, 1)
        palette = ['blue', 'red', 'green',
                   'darkorange', 'maroon', 'black']

        def animation_function(i):
            ax.clear()
            self.show_profile(self.modes.profiles[i], ax=ax)
            ax.set_title(f'mode num: {i}')

        animation = Player(fig, animation_function, interval=500, frames=self.Nmodes)
        plt.show()


class ManyWavelengthSimulation(object):
    def __init__(self, wl0=0.810, Dwl=0.040, N_wl=81):
        """ all in um """
        self.wl0 = wl0
        self.Dwl = Dwl
        self.N_wl = N_wl
        self.wls = self._get_wl_range()
        self.ns = self._sellmeier_silica(self.wls)
        self.fibers = []
        for i, wl in enumerate(self.wls):
            self.fibers.append(Fiber(wl=wl, n1=self.ns[i]))
        self.N_modes_cutoff = min(self.fibers[0].Nmodes, self.fibers[-1].Nmodes)  # If N_modes changes with wl - discard last modes
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

    def set_inputs_gaussian(self, sigma=10, X0=3, Y0=9, X_linphase=0.3, random_phase=0.5):
        for f in self.fibers:
            f.set_input_gaussian(sigma=sigma, X0=X0, Y0=Y0, X_linphase=X_linphase, random_phase=random_phase)

    def set_inputs_random_modes(self, N_random_modes=30):
        for f in self.fibers:
            self.fibers[0].set_input_random_modes(N_random_modes)

    def get_klyshko_PCCs(self):
        i_middle = len(self.fibers) // 2
        N_measurements = (len(self.fibers) // 2) + 1  # for 5 wls: 1 degenerate + 2 non degenerate
        pccs = np.zeros(N_measurements)
        delta_lambdas = np.zeros(N_measurements)
        self.fibers[i_middle].set_input_gaussian(sigma=10, X0=3, Y0=9, X_linphase=0.3, random_phase=0.5)

        # Simply propagate twice for degenerate
        E_end0 = self.fibers[i_middle].propagate(show=False)
        self.fibers[i_middle].profile_0 = E_end0
        E_end0 = self.fibers[i_middle].propagate(show=False)

        I_end0 = np.abs(E_end0) ** 2
        II0 = I_end0.reshape([self.fibers[i_middle].npoints] * 2)[50:80, 50:80]  # todo: better than 50:80
        pccs[0] = 1
        delta_lambdas[0] = 0

        for di in range(1, N_measurements):
            f_plus = self.fibers[i_middle+di]
            f_minus = self.fibers[i_middle-di]
            f_plus.set_input_gaussian(sigma=10, X0=3, Y0=9, X_linphase=0.3, random_phase=0.5)
            E_end = f_plus.propagate(show=False)
            # TODO: add z propagation
            f_minus.profile_0 = E_end
            E_end = f_minus.propagate(show=False)
            I_end = np.abs(E_end)**2
            II = I_end.reshape([self.fibers[i_middle].npoints] * 2)[50:80, 50:80]
            pccs[di] = np.corrcoef(II0.ravel(), II.ravel())[1, 0]
            delta_lambdas[di] = f_plus.wl - f_minus.wl

        return np.array(delta_lambdas), pccs

    def get_classical_PCCs(self):
        # i_middle = len(self.fibers) // 2
        # In classical case I just want the dependance for w0 going to one side.
        i_middle = 0
        pccs = np.zeros_like(self.fibers)

        E_end0 = self.fibers[i_middle].propagate(show=False)
        I_end0 = np.abs(E_end0) ** 2
        II0 = I_end0.reshape([self.fibers[i_middle].npoints] * 2)[50:80, 50:80]  # todo: better than 50:80

        for i, f in enumerate(self.fibers):
            E_end = f.propagate(show=False)
            I_end = np.abs(E_end)**2

            II = I_end.reshape([self.fibers[i_middle].npoints] * 2)[50:80, 50:80]
            pccs[i] = np.corrcoef(II0.ravel(), II.ravel())[1, 0]

        delta_lambdas = [f.wl - self.fibers[i_middle].wl for f in self.fibers]

        return np.array(delta_lambdas), pccs

    def get_classical_PCCs_average(self, N_configs=5):
        pccs = np.zeros((N_configs, len(self.fibers)))
        delta_lambdas = np.zeros(len(self.fibers))
        for i in range(N_configs):
            self.set_inputs_gaussian()
            delta_lambdas, pccs[i, :] = self.get_classical_PCCs()
        return delta_lambdas, pccs.mean(axis=0)

    def show_PCC_classical_and_quantum(self, delta_lambdas_classical, pccs_classical, delta_lambdas_klyshko, pccs_klyshko):
        fig, ax = plt.subplots()
        ax.plot(delta_lambdas_classical * 1e3, pccs_classical, label='classical')
        ax.plot(delta_lambdas_klyshko * 1e3, pccs_klyshko, label='Klyshko')
        ax.set_xlabel(r'wl difference $ \Delta\lambda$ (nm)')
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
        L = 2e6  # 2m fiber
        # this gives Dbetas~1 at least for the first ~30 modes, except for a few modes specific modes, not sure why
        Dphis = Dbetas*L

        DDbetas = s.betas[0, :] + s.betas[20, :] - 2 * s.betas[10, :]  # Klyshko picture, 20nms hoping is OK
        DDbetas -= np.median(DDbetas)
        DDphis = DDbetas*2*L  # seems good! even with 20nm bandwidth we seem to still be far off from 2pi!


s = ManyWavelengthSimulation()
f = s.fibers[0]
a, b = s.get_classical_PCCs_average(10)
c, d = s.get_klyshko_PCCs()
s.show_PCC_classical_and_quantum(a, b, c, d)
# s.show_PCC_classical_and_quantum()
