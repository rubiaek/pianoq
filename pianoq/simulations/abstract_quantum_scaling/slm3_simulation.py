import numpy as np
import random
from scipy.stats import unitary_group
from scipy.optimize import minimize, dual_annealing

class SLM3Simulation:
    def __init__(self, N=256, T_method='gaus_iid'):
        self.N = N
        self.T_method = T_method
        self.T = self.get_diffuser()

        self.v_in = 1/np.sqrt(self.N) * np.ones(self.N, dtype=np.complex128)
        self.slm_phases = np.exp(1j*np.zeros(self.N))
        self.f_calls = 0

    def get_diffuser(self):
        if self.T_method == 'gaus_iid':
            return 1/np.sqrt(self.N) * np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(self.N, self.N, 2)).view(np.complex128)[:, :, 0]
        elif self.T_method == 'unitary':
            return unitary_group.rvs(self.N)
        else:
            raise NotImplementedError()

    def reset_T(self):
        self.T = self.get_diffuser()

    def propagate(self):
        # v_in gets slm phases, is scattered by T, and is reflected by the crystal,
        # then T.T, phases againg, and the FFT for phases to take effect

        after_SLM_second_time = self.slm_phases * (self.T.transpose() @ self.T @ (self.slm_phases * self.v_in))
        v_out = np.fft.fft(after_SLM_second_time) / np.sqrt(self.N)
        return v_out

    def get_intensity(self, slm_phases=None, out_mode=None):
        if slm_phases is not None:
            self.slm_phases = np.exp(1j*slm_phases)
        out_mode = out_mode or self.N // 2
        v_out = self.propagate()
        I = np.abs(v_out[out_mode])**2
        self.f_calls += 1
        return -I

    def optimize(self, method="slsqp"):
        import numpy as np # really weird, don't understand why I need this
        self.f_calls = 0
        # Define initial phases as the current slm_phases
        initial_phases = np.exp(1j*np.zeros(self.N))

        if method == "slsqp" or method == "L-BFGS-B":
            result = minimize(
                self.get_intensity, initial_phases, method=method, bounds=[(0, 2 * np.pi)] * self.N
            )
            self.slm_phases = result.x
            intensity = self.get_intensity(self.slm_phases)
            return intensity, result

        elif method == "simulated_annealing":
            bounds = [(0, 2 * np.pi) for _ in range(self.N)]
            result = dual_annealing(self.get_intensity, bounds=bounds)
            self.slm_phases = result.x
            intensity = self.get_intensity(self.slm_phases)
            return intensity, result

        elif method == "genetic_algorithm":
            from deap import base, creator, tools, algorithms
            import random
            import numpy as np

            # Check if FitnessMax and Individual classes already exist to avoid redefinition warnings
            if not hasattr(creator, "FitnessMax"):
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            if not hasattr(creator, "Individual"):
                creator.create("Individual", list, fitness=creator.FitnessMax)

            # Initialize DEAP toolbox
            toolbox = base.Toolbox()
            toolbox.register("attr_phase", random.uniform, 0, 2 * np.pi)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_phase, n=self.N)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", lambda ind: (self.get_intensity(),))  # Fitness function
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Genetic algorithm parameters
            population = toolbox.population(n=100)
            max_generations = 2000  # Set this to your desired number of generations
            algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=max_generations, verbose=False)

            best_individual = tools.selBest(population, k=1)[0]

            # Update slm_phases with best individual found
            self.slm_phases = np.array(best_individual)
            intensity = self.get_intensity(self.slm_phases)
            return intensity, population

        elif method == "PSO":
            try:
                from pyswarm import pso
            except ImportError:
                raise ImportError("pyswarm library not installed. Please install it using 'pip install pyswarm'.")

            lb = [0] * self.N
            ub = [2 * np.pi] * self.N
            best_phases, _ = pso(self.get_intensity, lb, ub, maxiter=150000)
            self.slm_phases = best_phases
            intensity = self.get_intensity(self.slm_phases)
            return intensity, _

        else:
            raise ValueError("Unsupported optimization method")

    def statistics(self, N_times, method="gradient_descent"):
        """
        Runs the optimization process multiple times to gather statistics.
        """
        intensities = []

        for _ in range(N_times):
            # Reset phases before each optimization
            self.slm_phases = np.zeros(self.N)
            self.optimize(method=method)
            intensity = -self.get_intensity()  # Undo negation to get positive intensity
            intensities.append(intensity)

        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)
        print(f"Optimization Method: {method}")
        print(f"Mean Intensity: {mean_intensity}")
        print(f"Standard Deviation of Intensity: {std_intensity}")