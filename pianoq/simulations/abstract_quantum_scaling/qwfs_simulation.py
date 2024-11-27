import numpy as np
import random
from scipy.stats import unitary_group
from scipy.optimize import minimize, dual_annealing
from pianoq.misc.mplc_writeup_imports import tnow
from functools import partial
from pianoq.simulations.abstract_quantum_scaling.qwfs_result import QWFSResult


class QWFSSimulation:
    def __init__(self, N=256, T_method='gaus_iid', config='SLM3'):
        self.N = N
        self.DEFAULT_OUT_MODE = self.N // 2
        self.DEFAULT_ONEHOT_INPUT_MODE = 0
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
        if self.config == 'SLM1':
            after_T = self.T @ self.T.transpose() @ (self.slm_phases * self.v_in)
            v_out = np.fft.fft(after_T) / np.sqrt(self.N)
        elif self.config == 'SLM1-only-T':
            v_out = self.T @ (self.slm_phases * self.v_in)
        elif self.config == 'SLM2':
            after_T2 = self.T @ (self.slm_phases * (self.T.transpose() @ self.v_in))
            v_out = np.fft.fft(after_T2) / np.sqrt(self.N)
        elif self.config == 'SLM2-simple-OPC':
            v_in_one_hot = np.zeros_like(self.v_in)
            v_in_one_hot[self.DEFAULT_OUT_MODE] = 1  # TODO: this assumes that we try to optimize the self.DEFAULT_OUT_MODE
            v_out = self.T @ (self.slm_phases * (self.T.transpose() @ v_in_one_hot))
        elif self.config == 'SLM2-simple':
            v_in_one_hot = np.zeros_like(self.v_in)
            v_in_one_hot[self.DEFAULT_ONEHOT_INPUT_MODE] = 1
            v_out = self.T @ (self.slm_phases * (self.T.transpose() @ v_in_one_hot))
        elif self.config == 'SLM3':
            after_SLM_second_time = self.slm_phases * (self.T @ self.T.transpose() @ (self.slm_phases * self.v_in))
            v_out = np.fft.fft(after_SLM_second_time) / np.sqrt(self.N)
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
        # import numpy as np # really weird, don't understand why I need this
        out_mode = out_mode or self.DEFAULT_OUT_MODE

        cost_func = partial(self.get_intensity, out_mode=out_mode)

        self.f_calls = 0
        # Define initial phases as the current slm_phases
        self.slm_phases = np.exp(1j*np.zeros(self.N))

        if algo == "slsqp" or algo == "L-BFGS-B":
            initial_phases = np.zeros(self.N)

            if algo == "L-BFGS-B":
                # Configure L-BFGS-B to "try harder"
                options = {
                    'maxiter': 30000,  # default: 15000. Increase the maximum number of iterations
                    'ftol': 1e-12,  # default: 2.22e-9. Tighter function tolerance
                    'gtol': 1e-8, # default: 1e-5 Tighter gradient norm tolerance
                    'eps': 1e-8,  # default: 1e-8. Smaller step size for gradient estimation
                    # 'disp': True  # Display convergence messages
                }
            else:
                options = {}

            result = minimize(
                cost_func, initial_phases, method=algo, bounds=[(0, 2 * np.pi)] * self.N, options=options
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
            elif self.config == 'SLM2-simple-OPC' or self.config == 'SLM2-simple':
                # TODO: nicer code
                O1 = self.DEFAULT_OUT_MODE if self.config == 'SLM2-simple-OPC' else self.DEFAULT_ONEHOT_INPUT_MODE
                at_slm = (self.T[O1, :] * self.T[self.DEFAULT_OUT_MODE, :])
                self.slm_phases = -np.angle(at_slm)
                intensity = cost_func(self.slm_phases)
                return intensity, None
            else:
                return -0.1, None
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
                    self.config = config
                    for algo_no, algo in enumerate(algos):
                        self.slm_phases = np.exp(1j * np.zeros(self.N, dtype=np.complex128))
                        I, res = self.optimize(algo=algo)
                        self.slm_phases = res.x
                        v_out = self.propagate()
                        I_good = np.abs(v_out[self.DEFAULT_OUT_MODE]) ** 2
                        qres.results[T_method_no, config_no, try_no, algo_no] = I_good
                        qres.best_phases[T_method_no, config_no, try_no, algo_no] = np.angle(self.slm_phases)
                        # I_tot = (np.abs(v_out) ** 2).sum()
                        # assert np.abs(I_tot - 1) < 0.05, f'Something weird with normalization! {I_tot=}'
                        # print(rf'{method=}, {I_tot=:.4f}, {I_good=:.4f}, {s.f_calls=}')

            qres.Ts = np.array(Ts)

            qres.saveto(saveto_path)

        return qres
