import numpy as np
import random
from scipy.stats import unitary_group
from scipy.optimize import minimize, dual_annealing

class QWFSSimulation:
    def __init__(self, N=256, T_method='gaus_iid', config='SLM3'):
        self.N = N
        self.T_method = T_method
        self.T = self.get_diffuser()
        self.config = config

        self.v_in = 1/np.sqrt(self.N) * np.ones(self.N, dtype=np.complex128)
        self.slm_phases = np.exp(1j*np.zeros(self.N, dtype=np.complex128))
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
        if self.config == 'SLM3':
            # v_in gets slm phases, is scattered by T, and is reflected by the crystal,
            # then T.T, phases againg, and the FFT for phases to take effect
            after_SLM_second_time = self.slm_phases * (self.T.transpose() @ self.T @ (self.slm_phases * self.v_in))
            v_out = np.fft.fft(after_SLM_second_time) / np.sqrt(self.N)
        elif self.config == 'SLM1':
            after_T = self.T.transpose() @ self.T @ (self.slm_phases * self.v_in)
            v_out = np.fft.fft(after_T) / np.sqrt(self.N)
            # v_out = after_T
        elif self.config == 'SLM2':
            after_T2 = self.T.transpose() @ (self.slm_phases * (self.T @ self.v_in))
            v_out = np.fft.fft(after_T2) / np.sqrt(self.N)
            # v_out = after_T2
        else:
            raise NotImplementedError('WAT?')

        return v_out

    def get_intensity(self, slm_phases=None, out_mode=None):
        if slm_phases is not None:
            self.slm_phases = np.exp(1j*slm_phases)
        out_mode = out_mode or self.N // 2
        v_out = self.propagate()
        I = np.abs(v_out[out_mode])**2
        self.f_calls += 1
        return -I

    def optimize(self, algo="slsqp"):
        import numpy as np # really weird, don't understand why I need this
        self.f_calls = 0
        # Define initial phases as the current slm_phases
        initial_phases = np.exp(1j*np.zeros(self.N))

        if algo == "slsqp" or algo == "L-BFGS-B":
            result = minimize(
                self.get_intensity, initial_phases, method=algo, bounds=[(0, 2 * np.pi)] * self.N
            )
            self.slm_phases = result.x
            intensity = self.get_intensity(self.slm_phases)
            return intensity, result

        elif algo == "simulated_annealing":
            bounds = [(0, 2 * np.pi) for _ in range(self.N)]
            result = dual_annealing(self.get_intensity, bounds=bounds)
            self.slm_phases = result.x
            intensity = self.get_intensity(self.slm_phases)
            return intensity, result

        elif algo == "genetic_algorithm":
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

        elif algo == "PSO":
            from pianoq.lab.optimizations.my_pso import MyPSOOptimizer

            def cost_func(phases):
                return self.get_intensity(phases), None, None

            o = MyPSOOptimizer(cost_func, n_pop=80, n_var=self.N, n_iterations=1000, stop_early=True,
                               stop_after_n_const_iter=100, quiet=True)
            o.optimize()
            self.slm_phases = o.best_positions
            intensity = self.get_intensity(self.slm_phases)
            return intensity, o

        else:
            raise ValueError("Unsupported optimization method")

    def statistics(self, algos, configs, T_methods, N_tries=1):
        N_algos = len(algos)  #
        N_configs = len(configs)
        N_T_methods = len(T_methods)

        results = np.zeros((N_T_methods, N_configs, N_tries, N_algos))
        best_phases = np.zeros((N_T_methods, N_configs, N_tries, N_algos, self.N))
        Ts = []

        for try_no in range(N_tries):
            print(f'{try_no=}')
            for T_method_no, T_method in enumerate(T_methods):
                self.T_method = T_method
                self.reset_T()
                Ts.append(self.T)
                for config_no, config in enumerate(configs):
                    for algo_no, algo in enumerate(algos):
                        self.config = config
                        self.slm_phases = np.exp(1j * np.zeros(self.N, dtype=np.complex128))
                        I, res = self.optimize(algo=algo)
                        v_out = self.propagate()
                        I_tot = (np.abs(v_out) ** 2).sum()
                        # assert np.abs(I_tot - 1) < 0.05, f'Something weird with normalization! {I_tot=}'
                        I_good = np.abs(v_out[self.N // 2]) ** 2
                        # print(rf'{method=}, {I_tot=:.4f}, {I_good=:.4f}, {s.f_calls=}')
                        results[T_method_no, config_no, try_no, algo_no] = I_good
                        best_phases[T_method_no, config_no, try_no, algo_no] = np.angle(self.slm_phases)

        return results, Ts, best_phases
