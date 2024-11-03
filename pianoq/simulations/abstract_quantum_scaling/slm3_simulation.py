import numpy as np
from scipy.optimize import minimize, dual_annealing


class SLM3Simulation:
    def __init__(self):
        self.N = 256
        self.T = self.get_diffuser()
        self.v_in = 1/np.sqrt(self.N) * np.ones
        self.slm_phases = np.zeros(self.N)

    def get_diffuser(self):
        return 1/np.sqrt(self.N) * np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(self.N, self.N, 2)).view(np.complex128)[:, :, 0]

    def propagate(self):
        # v_in gets slm phases, is scattered by T, and is reflected by the crystal,
        # then T.T, phases againg, and the FFT for phases to take effect

        v_out = np.fft.fft(self.slm_phases * (self.T.transpose() @ self.T @ (self.slm_phases * self.v_in)))
        return v_out

    def get_intensity(self, slm_phases, out_mode=None):
        self.slm_phases = slm_phases
        out_mode = out_mode or self.N // 2
        v_out = self.propagate()
        I = np.abs(v_out[out_mode])**2
        return -I

    def optimize(self, method="gradient_descent"):
        # Define initial phases as the current slm_phases
        initial_phases = self.slm_phases.copy()

        if method == "gradient_descent":
            result = minimize(
                self.get_intensity, initial_phases, method="L-BFGS-B", bounds=[(0, 2 * np.pi)] * self.N
            )
            self.slm_phases = result.x
            intensity = self.get_intensity(self.slm_phases)
            return intensity, result

        elif method == "simulated_annealing":
            bounds = [(0, 2 * np.pi) for _ in range(self.N)]
            result = dual_annealing(self.get_intensity, bounds=bounds)
            self.slm_phases = result.x

        elif method == "genetic_algorithm":
            from deap import base, creator, tools, algorithms

            # Create necessary classes for DEAP genetic algorithm
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
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
            algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, verbose=False)
            best_individual = tools.selBest(population, k=1)[0]

            # Update slm_phases with best individual found
            self.slm_phases = np.array(best_individual)

        elif method == "particle_swarm":
            try:
                from pyswarm import pso
            except ImportError:
                raise ImportError("pyswarm library not installed. Please install it using 'pip install pyswarm'.")

            lb = [0] * self.N
            ub = [2 * np.pi] * self.N
            best_phases, _ = pso(self.get_intensity, lb, ub)
            self.slm_phases = best_phases

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