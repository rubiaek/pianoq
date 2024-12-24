import numpy as np
import random
from scipy.stats import unitary_group
from scipy.optimize import minimize, dual_annealing
from pianoq.misc.mplc_writeup_imports import tnow
from functools import partial
from pianoq.simulations.abstract_quantum_scaling.qwfs_result import QWFSResult
try:
    import torch
    # so we can do T.transpose() on torch and np arrays similarly
    def _tensor_transpose(self, *args):
        if len(args) == 0:
            return self.t()
        return self.transpose(*args)

    torch.Tensor.transpose = _tensor_transpose
except ImportError:
    pass


class QWFSSimulation:
    def __init__(self, N=256, T_method='gaus_iid', config='SLM3'):
        self.N = N
        self.DEFAULT_OUT_MODE = self.N // 2
        self.DEFAULT_ONEHOT_INPUT_MODE = 0
        self.T_method = T_method
        self.config = config
        self.sig_for_gauss_iid = np.sqrt(2)/2
        self.cost_function = 'energy'
        self.T = self.get_diffuser()
        self.M = self.get_diffuser()

        self.v_in = 1/np.sqrt(self.N) * np.ones(self.N, dtype=np.complex128)
        self.slm_phases = np.exp(1j*np.zeros(self.N, dtype=np.complex128))
        self.f_calls = 0


    def get_diffuser(self):
        if self.T_method == 'gaus_iid':
            return 1/np.sqrt(self.N) * np.random.normal(loc=0, scale=self.sig_for_gauss_iid, size=(self.N, self.N, 2)).view(np.complex128)[:, :, 0]
        elif self.T_method == 'unitary':
            return unitary_group.rvs(self.N)
        elif self.T_method == 'cue':
            # More sophisticated random unitary generation
            Z = np.random.normal(0, 1, (self.N, self.N)) + 1j * np.random.normal(0, 1, (self.N, self.N))
            Q, R = np.linalg.qr(Z)
            D = np.diag(np.diag(R) / np.abs(np.diag(R)))
            return Q @ D
        else:
            raise NotImplementedError()

    def reset_T(self):
        self.T = self.get_diffuser()
        self.M = self.get_diffuser()

    def propagate(self, use_torch=False):

        fft = torch.fft.fft if use_torch else np.fft.fft

        if self.config == 'SLM1' or self.config == 'SLM1-same-mode':
            after_T = self.T @ self.T.transpose() @ (self.slm_phases * self.v_in)
            v_out = fft(after_T) / np.sqrt(self.N)
        elif self.config == 'SLM1-after':
            after_T = self.slm_phases * (self.T @ self.T.transpose() @ self.v_in)
            v_out = fft(after_T) / np.sqrt(self.N)
        elif self.config == 'SLM1-only-T':
            v_out = self.T @ (self.slm_phases * self.v_in)
        elif self.config == 'SLM1-only-T-after':
            after_T = self.slm_phases * (self.T @ self.v_in)
            v_out = fft(after_T) / np.sqrt(self.N)
        elif self.config == 'SLM2':
            after_T2 = self.T @ (self.slm_phases * (self.T.transpose() @ self.v_in))
            v_out = fft(after_T2) / np.sqrt(self.N)
        elif self.config == 'SLM2-simple-OPC':
            v_in_one_hot = np.zeros_like(self.v_in)
            v_in_one_hot[self.DEFAULT_OUT_MODE] = 1  # TODO: this assumes that we try to optimize the self.DEFAULT_OUT_MODE
            v_out = self.T @ (self.slm_phases * (self.T.transpose() @ v_in_one_hot))
        elif self.config == 'SLM2-simple':
            v_in_one_hot = np.zeros_like(self.v_in)
            v_in_one_hot[self.DEFAULT_ONEHOT_INPUT_MODE] = 1
            v_out = self.T @ (self.slm_phases * (self.T.transpose() @ v_in_one_hot))
        elif self.config == 'SLM3' or self.config == 'SLM3-same-mode':
            after_SLM_second_time = self.slm_phases * (self.T @ self.T.transpose() @ (self.slm_phases * self.v_in))
            v_out = fft(after_SLM_second_time) / np.sqrt(self.N)
        else:
            raise NotImplementedError('WAT?')

        return v_out

    def get_intensity(self, slm_phases_rad=None, out_mode=None, use_torch=False):
        exp = torch.exp if use_torch else np.exp
        abs = torch.abs if use_torch else np.abs
        slm_phases_rad = slm_phases_rad if use_torch else np.array(slm_phases_rad)

        if slm_phases_rad is not None:
            self.slm_phases = exp(1j*slm_phases_rad)
        if out_mode is None:
            out_mode = self.DEFAULT_OUT_MODE
        v_out = self.propagate(use_torch=use_torch)
        I_out = abs(v_out)**2
        I_focus = I_out[out_mode]
        self.f_calls += 1

        if self.cost_function == 'energy':
            return -I_focus
        elif self.cost_function == 'contrast':
            return -I_focus / I_out.sum()
        elif self.cost_function == 'total_energy':
            return -I_out.sum()
        else:
            raise NotImplementedError()

    def optimize(self, algo="slsqp", out_mode=None):
        # import numpy as np # really weird, don't understand why I need this
        if out_mode is None:
            out_mode = self.DEFAULT_OUT_MODE

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
        elif algo == 'autograd':
            return self._autograd(out_mode=out_mode)
        else:
            raise ValueError("Unsupported optimization method")

    def _autograd(self, out_mode=None):
        """ Note that this doesn't work for cost functions other than `energy` """
        # param init
        phases = torch.zeros(self.N, requires_grad=True, dtype=torch.float64)
        optimizer = torch.optim.Adam([phases], lr=0.01)
        N_iters = 10000
        prev_cost = float('inf')
        patience = 10  # Number of iterations to wait for improvement
        patience_counter = 0
        eps_stop = 1e-6

        dtype = torch.complex128
        self.T = torch.tensor(self.T, dtype=dtype, requires_grad=False)
        self.M = torch.tensor(self.M, dtype=dtype, requires_grad=False)
        self.v_in = torch.tensor(self.v_in, dtype=dtype, requires_grad=False)

        for i in range(N_iters):
            optimizer.zero_grad()
            # slm_phases = torch.exp(1j * phases)
            # self.slm_phases = slm_phases
            # v_out = self.propagate(use_torch=True)
            # cost = -torch.abs(v_out[out_mode]) ** 2
            cost = self.get_intensity(phases, out_mode=out_mode, use_torch=True)
            cost.backward()
            optimizer.step()
            with torch.no_grad():
                # important to update the data and not create a new tensor that will be detached from the graph
                phases.data = phases.data % (2 * torch.pi)

            current_cost = cost.item()
            if abs(prev_cost - current_cost) < eps_stop:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            else:
                patience_counter = 0
            prev_cost = current_cost

        self.slm_phases = torch.exp(1j * phases).detach().numpy()
        self.reset_to_numpy()

        return -prev_cost, None

    def reset_to_numpy(self):
        """Ensure all attributes are reset to NumPy arrays."""
        self.T = self.T.detach().numpy() if isinstance(self.T, torch.Tensor) else self.T
        self.M = self.M.detach().numpy() if isinstance(self.M, torch.Tensor) else self.M
        self.v_in = self.v_in.detach().numpy() if isinstance(self.v_in, torch.Tensor) else self.v_in
        self.slm_phases = self.slm_phases.detach().numpy() if isinstance(self.slm_phases,
                                                                         torch.Tensor) else self.slm_phases

    def statistics(self, algos, configs, T_methods, N_tries=1, saveto_path=None):
        saveto_path = saveto_path or f"C:\\temp\\{tnow()}_qwfs.npz"
        qres = QWFSResult()
        qres.configs = configs
        qres.T_methods = T_methods
        qres.N_tries = N_tries
        qres.algos = algos
        qres.sig_for_gauss_iid = self.sig_for_gauss_iid

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
                        if config == 'SLM3-same-mode' or config == 'SLM1-same-mode':
                            # this is the equivalent output mode after fourier to the default input of flat phase ones
                            out_mode = 0
                        else:
                            out_mode = self.DEFAULT_OUT_MODE
                        I, res = self.optimize(algo=algo, out_mode=out_mode)
                        v_out = self.propagate()
                        I_good = np.abs(v_out[out_mode]) ** 2
                        qres.results[T_method_no, config_no, try_no, algo_no] = I_good
                        qres.best_phases[T_method_no, config_no, try_no, algo_no] = np.angle(self.slm_phases)
                        # print(rf'{method=}, {I_tot=:.4f}, {I_good=:.4f}, {s.f_calls=}')

            qres.Ts = np.array(Ts)

            qres.saveto(saveto_path)

        return qres
