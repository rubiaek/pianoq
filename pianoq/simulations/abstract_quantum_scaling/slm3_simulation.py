import numpy as np
import random
from scipy.stats import unitary_group
from scipy.optimize import minimize, dual_annealing
import matplotlib.pyplot as plt
from pianoq.misc.mplc_writeup_imports import tnow
from functools import partial


class QWFSResult:
    def __init__(self, path=None):
        # results.shape == N_T_methods, N_configs, N_tries, N_algos
        self.path = path
        self.T_methods = None
        self.configs = None
        self.N_tries = None
        self.algos = None
        self.results = None
        self.Ts = None
        self.best_phases = None

        if path:
            self.loadfrom(path)

    def saveto(self, path):
        np.savez(path, self.__dict__)

    def loadfrom(self, path):
        data = np.load(path)
        self.__dict__.update(data)

    def show(self):
        # all configurations
        fig, axes = plt.subplots(len(self.configs), len(self.T_methods))
        for config_no, config in enumerate(self.configs):
            for T_method_no, T_method in enumerate(self.T_methods):
                # upper_lim = 1 if T_method == 'unitary' else 2
                # imm = axes[config_no, T_method_no].imshow(results[T_method_no, config_no], clim=(0, upper_lim))
                if len(self.configs) > 1:
                    ax = axes[config_no, T_method_no]
                else:
                    ax = axes[T_method_no]
                imm = ax.imshow(self.results[T_method_no, config_no], aspect='auto')
                ax.set_title(rf'{config}, {T_method}')
                fig.colorbar(imm, ax=ax)
        fig.show()

    def print(self):
        for config_no, config in enumerate(self.configs):
            print(f'---- {config} ----')
            for T_method_no, T_method in enumerate(self.T_methods):
                print(f'-- {T_method} --')
                for algo_no, algo in enumerate(self.algos):
                    avg = self.results[T_method_no, config_no].mean(axis=0)[algo_no]
                    std = self.results[T_method_no, config_no].std(axis=0)[algo_no]

                    print(f'{algo:<25} {avg:.3f}+-{std:.2f}')
            print()

class QWFSSimulation:
    def __init__(self, N=256, T_method='gaus_iid', config='SLM3'):
        self.N = N
        self.DEFAULT_OUT_MODE = self.N // 2
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
        elif self.config == 'SLM1-only-T':
            v_out = self.T @ (self.slm_phases * self.v_in)
        else:
            raise NotImplementedError('WAT?')

        return v_out

    def get_intensity(self, slm_phases_rad=None, out_mode=None):
        if slm_phases_rad is not None:
            self.slm_phases = np.exp(1j*np.array(slm_phases_rad))
        out_mode = out_mode or self.DEFAULT_OUT_MODE
        v_out = self.propagate()
        I = np.abs(v_out[out_mode])**2
        self.f_calls += 1
        return -I

    def optimize(self, algo="slsqp", out_mode=None):
        import numpy as np # really weird, don't understand why I need this
        out_mode = out_mode or self.DEFAULT_OUT_MODE

        cost_func = partial(self.get_intensity, out_mode=out_mode)

        self.f_calls = 0
        # Define initial phases as the current slm_phases
        self.slm_phases = np.exp(1j*np.zeros(self.N))

        if algo == "slsqp" or algo == "L-BFGS-B":
            initial_phases = np.zeros(self.N)
            result = minimize(
                cost_func, initial_phases, method=algo, bounds=[(0, 2 * np.pi)] * self.N
            )
            self.slm_phases = result.x
            intensity = cost_func(self.slm_phases)
            return intensity, result

        elif algo == "simulated_annealing":
            bounds = [(0, 2 * np.pi) for _ in range(self.N)]
            result = dual_annealing(cost_func, bounds=bounds)
            self.slm_phases = result.x
            intensity = cost_func(self.slm_phases)
            return intensity, result

        elif algo == 'analytic':
            if self.config == 'SLM1-only-T':
                desired_out_vec = np.zeros(self.N)
                desired_out_vec[out_mode] = 1
                self.slm_phases = -np.angle(self.T.transpose() @ desired_out_vec)
                intensity = cost_func(self.slm_phases)
                return intensity, None
            else:
                return -0.1, None

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
            toolbox.register("evaluate", lambda ind: (cost_func(ind),))  # Fitness function
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
            intensity = cost_func(self.slm_phases)
            return intensity, population

        elif algo == "PSO":
            from pianoq.lab.optimizations.my_pso import MyPSOOptimizer

            def cost_func2(phases):
                return cost_func(phases), None, None

            o = MyPSOOptimizer(cost_func2, n_pop=80, n_var=self.N, n_iterations=1000, stop_early=True,
                               stop_after_n_const_iter=100, quiet=True)
            o.optimize()
            self.slm_phases = o.best_positions
            intensity = cost_func(self.slm_phases)
            return intensity, o

        else:
            raise ValueError("Unsupported optimization method")

    def statistics(self, algos, configs, T_methods, N_tries=1, saveto_path=None):
        saveto_path = saveto_path or f"C:\\temp\\{tnow()}_qwfs.npz"
        qres = QWFSResult()
        qres.configs = configs
        qres.T_methods = T_methods
        qres.N_tries = N_tries
        qres.algos = algos

        N_algos = len(algos)
        N_configs = len(configs)
        N_T_methods = len(T_methods)

        qres.results = np.zeros((N_T_methods, N_configs, N_tries, N_algos))
        qres.best_phases = np.zeros((N_T_methods, N_configs, N_tries, N_algos, self.N))
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
                        # I_tot = (np.abs(v_out) ** 2).sum()
                        # assert np.abs(I_tot - 1) < 0.05, f'Something weird with normalization! {I_tot=}'
                        I_good = np.abs(v_out[self.N // 2]) ** 2
                        # print(rf'{method=}, {I_tot=:.4f}, {I_good=:.4f}, {s.f_calls=}')
                        qres.results[T_method_no, config_no, try_no, algo_no] = I_good
                        qres.best_phases[T_method_no, config_no, try_no, algo_no] = np.angle(self.slm_phases)

        qres.Ts = np.array(Ts)

        qres.saveto(saveto_path)

        return qres
