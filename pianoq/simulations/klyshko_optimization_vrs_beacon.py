import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from pianoq.misc.mplt import *
from scipy.optimize import curve_fit


class KlyshkoBeaconSimulationResult(object):
    def __init__(self):
        self.ctrls = np.linspace(0, 1, 10)
        self.klyshkos = []
        self.klyshko_stds = []
        self.beacon_one_ways = []
        self.beacon_one_way_stds = []
        self.beacon_two_ways = []
        self.beacon_two_way_stds = []
        self.QOPCs = []
        self.QOPC_stds = []

        self.T_mode = ''
        self.incomplete_control_method = ''

    def show(self, fits=None, show_legend=True):
        dummy_x = np.linspace(self.ctrls[0], self.ctrls[-1], 100)
        fig, ax = plt.subplots(figsize=(9, 4))
        b1_lines = ax.errorbar(self.ctrls, self.beacon_one_ways, yerr=self.beacon_one_way_stds, fmt='o', label='classical')
        klyshko_lines = ax.errorbar(self.ctrls, self.klyshkos, yerr=self.klyshko_stds, fmt='o', label='Klyshko optimization')
        b2_lines = ax.errorbar(self.ctrls, self.beacon_two_ways, yerr=self.beacon_two_way_stds, fmt='o', label='double beacon')
        QOPC_lines = ax.errorbar(self.ctrls, self.QOPCs, yerr=self.QOPC_stds, fmt='o', label='Quantum OPC')

        if fits:
            if isinstance(fits, bool):
                k = True
                b1 = True
                b2 = True
                qopc = True
            else:
                k, b1, b2, qopc = fits

            fit_freedom = 0.02
            pow = (2 if k else 1)
            popt, pcov = curve_fit(lambda a, x: a * x ** pow, self.ctrls, self.klyshkos,
                                   bounds=((np.pi/4)**2 - fit_freedom, (np.pi/4)**2 + fit_freedom))
            ax.plot(dummy_x, popt[0] * dummy_x ** pow, linestyle='--', color=klyshko_lines[0].get_c())  # label='Klyshko sqr_fit',

            pow = (2 if b1 else 1)
            popt, pcov = curve_fit(lambda a, x: a * x ** pow, self.ctrls, self.beacon_one_ways,
                                   bounds=(np.pi/4 - fit_freedom, np.pi/4 + fit_freedom))
            ax.plot(dummy_x, popt[0] * dummy_x ** pow, linestyle='--', color=b1_lines[0].get_c())  # label='beacon 1 way sqr_fit',

            pow = (2 if b2 else 1)
            popt, pcov = curve_fit(lambda a, x: a * x ** pow, self.ctrls, self.beacon_two_ways,
                                   bounds=((np.pi/4)**2 - fit_freedom, (np.pi/4)**2 + fit_freedom))
            ax.plot(dummy_x, popt[0] * dummy_x ** pow, linestyle='--', color=b2_lines[0].get_c())  # , label='beacon 2 ways sqr_fit'

            pow = (2 if qopc else 1)
            popt, pcov = curve_fit(lambda a, x: a * x ** pow, self.ctrls, self.QOPCs,
                                   bounds=(1 - fit_freedom, 1 + fit_freedom))
            ax.plot(dummy_x, popt[0] * dummy_x ** pow, linestyle='--', color=QOPC_lines[0].get_c())  # label='QOPC sqr_fit'

        ax.set_xlabel('Degree of control', size=18)
        ax.set_ylabel('Phase only efficiency', size=18)
        ax.axhline(y=1, color='c', linestyle='--')
        ax.axhline(y=np.pi/4, color='g', linestyle='--')
        ax.axhline(y=(np.pi/4)**2, color='b', linestyle='--')
        ax.annotate(r'$\frac{\pi}{4}$', xy=(0.3, np.pi/4), xytext=(0.2, 0.9), arrowprops=dict(facecolor='black', shrink=0.05, width=2), fontsize=16)
        ax.annotate(r'$\left(\frac{\pi}{4}\right)^{2}$', xy=(0.3, (np.pi/4)**2), xytext=(0.2, 0.4), arrowprops=dict(facecolor='black', shrink=0.05, width=2), fontsize=16)
        ax.tick_params(axis='both', labelsize=16)

        if show_legend:
            fig.legend(loc='upper left')  # plt.rcParams['legend.loc'] = 'upper left'
        # fig.suptitle(f'incomplete_method: {self.incomplete_control_method}, T_mode: {self.T_mode}')
        fig.show()
        return fig, ax


def fit_sqr_and_linear(x, y):
    lin_popt, lin_pcov = curve_fit(lambda a, x: a * x, x, y, bounds=(0, 5))
    sqr_popt, sqr_pcov = curve_fit(lambda a, x: a * x**2, x, y, bounds=(0, 5))

    fig, ax = plt.subplots()
    ax.plot(x, y, 'o', label='klyshkos')
    dummy_x = np.linspace(x[0], x[-1], 100)
    ax.plot(dummy_x, lin_popt[0]*dummy_x, '--', label='lin_fit')
    ax.plot(dummy_x, sqr_popt[0]*dummy_x**2, '--', label='sqr_fit')

    fig.legend()
    fig.show()


class KlyshkoBeaconSimulation(object):
    """
        SLM is in real basis, also the diffuser is in real basis, and also measurement.
        The input state should be in Fourier basis, so the SLM will do something nontrivial.

        In Klyshko: Out1 -> U^dag -> slm -> F -> mirror -> F^-1 -> slm -> U -> Out2, so we can just forget the Fourier,
        everything is in real basis, so we just want to enhance the total matrix at [O1, O2]

        In beacon: I1 -> Fourier -> SLM -> U -> O1, and here we want to have only at O1. So there isn't a single index
        to enhance (since we have many inputs), so analytically I feel a bit stuck. I guess I want the phases at SLM
        plane to be the back-prop from O1, so take the phases from the backprop + phases from input, and conjugate that.
    """

    # TODO: both phase and amplitude

    def __init__(self, N=256, T_mode='unitary'):
        """ N the dimension. T_mode could be 'unitary' or 'gaus_iid' """
        # dimension
        self.N = N
        # TM of the thick random medium
        self.T_mode = T_mode
        self.T = None
        self.T2 = None
        self.reset_T()
        # The two output modes optimizing coincidences to
        self.I1 = 0
        self.I2 = 1
        self.O1 = self.N//3
        self.O2 = 2*self.N//3

        self.N_phases = 20
        self.incomplete_control_method = 'macro_pixels'  # 'zero'

    # TODO: see scaling also with thickness / amount of memory. Simulate with help from here: https://www.nature.com/articles/nphys3373
    def optimize_beacon(self):
        best_cost1 = 0
        best_S1 = np.zeros(self.N)
        all_costs1 = []

        for mode_index in range(self.N * 4):
            S = best_S1.copy()
            for phase in np.linspace(0, 2 * np.pi, self.N_phases):
                S[mode_index % self.N] = phase
                final_mat = self.T @ np.diag(np.exp(1j * S))
                complex_cost = final_mat[self.O1, self.O1]
                cost = np.abs(complex_cost) ** 2
                all_costs1.append(cost)
                if cost > best_cost1:
                    best_cost1 = cost
                    best_S1 = S.copy()

        best_cost2 = 0
        best_S2 = np.zeros(self.N)
        all_costs2 = []

        for mode_index in range(self.N * 4):
            S = best_S2.copy()
            for phase in np.linspace(0, 2 * np.pi, self.N_phases):
                S[mode_index % self.N] = phase
                final_mat = self.T @ np.diag(np.exp(1j * S))  # make sure order TS. Also that this shouldn't be dagger
                complex_cost = final_mat[self.O2, self.O2]
                cost = np.abs(complex_cost) ** 2
                all_costs2.append(cost)
                if cost > best_cost2:
                    best_cost2 = cost
                    best_S2 = S.copy()

        return (best_S1 + best_S2)/2, [all_costs1, all_costs2]

    def optimize_klyshko(self):
        best_cost = 0
        best_S = np.zeros(self.N)
        all_costs = []

        for mode_index in range(self.N*4):
            S = best_S.copy()
            for phase in np.linspace(0, 2*np.pi, self.N_phases):
                S[mode_index % self.N] = phase
                final_mat = self.T @ np.diag(np.exp(1j*S))**2 @ self.T.transpose().conjugate()
                complex_cost = final_mat[self.O1, self.O2]
                cost = np.abs(complex_cost)**2
                all_costs.append(cost)
                if cost > best_cost:
                    best_cost = cost
                    best_S = S.copy()
            # print(f'iter: {mode_index}, cost={best_cost}')

        return best_S, all_costs

    def prop_beacon(self, S, in1=True):
        """
        if in1:
            I = I1
        else:
            I = I2

        in_vec = np.zeros(N, complex)
        in_vec[I] = 1
        in_vec = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(in_vec)))
        """
        in_vec = np.ones(self.N, complex) / np.sqrt(self.N)
        out = self.T @ np.diag(np.exp(1j*S)) @ in_vec

        return out

    def prop_klyshko(self, S):
        vec_from_O1 = np.zeros(self.N, complex)
        vec_from_O1[self.O1] = 1  # TODO: there is something here with O1/O2 that I need to understand better
        out = self.T @ np.diag(np.exp(1j*S))**2 @ self.T.transpose() @ vec_from_O1
        return out

    def get_optimal_klyshko_conj(self, deg_of_control=1):
        """
            Returns phases to put on SLM (0, 2*pi)
            Explanation:
            (TS^2T^t)_ij = T_ik S^2_kl T_jl
            ij = O1,O2 -> T_O1k T_O2l S^2_kl
            -> s_kl = (T_O1k T_O2l)^*
            s digonal -> k=l
        """

        at_slm = (self.T[self.O1, :] * self.T[self.O2, :])
        S = -np.angle(at_slm)

        if 0 < deg_of_control < 1:
            if self.incomplete_control_method == 'zero':
                N_to_remove = round(self.N * (1-deg_of_control))
                S[self.N-N_to_remove:] = 0
            elif self.incomplete_control_method == 'macro_pixels':
                # see doc in similar below at self.get_optimal_beacon_conj()
                macro_pixel_size = round(1 / deg_of_control)
                max_index = (len(at_slm)//macro_pixel_size)*macro_pixel_size
                truncated = at_slm[:max_index]
                averaged = truncated.reshape((-1, macro_pixel_size)).sum(axis=1)
                repeated = np.repeat(averaged, macro_pixel_size)
                end_val = at_slm[max_index:].sum()
                padded = np.pad(repeated, (0, len(at_slm) % macro_pixel_size), constant_values=(0, end_val))
                S = -np.angle(padded)
        elif deg_of_control == 0:
            # Global phase does nothing
            S = np.zeros(self.N)


        # dividing by 2 because we go twice on the SLM in Klyshko picture
        return S / 2

    def get_optimal_beacon_conj(self, out1=True, deg_of_control=1):
        if out1:
            O = self.O1
            I = self.I1
        else:
            O = self.O2
            I = self.I2

        # returns angles (0, 2*pi)
        desired_vec = np.zeros(self.N, complex)
        desired_vec[O] = 1
        at_slm = self.T.transpose() @ desired_vec
        angles_from_out = np.angle(at_slm)

        """
            # This is redundant, since when going both directions these linear tilts will cancel out or be trivial 
            # so it is a waste of time 
            in_vec = np.zeros(N, complex)
            in_vec[I] = 1
            in_vec = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(in_vec)))
            angles_from_in = np.angle(in_vec)
            in_vec = np.ones(N, complex)
            S = -(angles_from_in + angles_from_out)
        """

        S = -angles_from_out

        if 0 < deg_of_control < 1:
            if self.incomplete_control_method == 'zero':
                N_to_remove = round(self.N * (1-deg_of_control))
                S[self.N-N_to_remove:] = 0
            elif self.incomplete_control_method == 'macro_pixels':
                # Naively for a 2-pix macro-pixel I would put the average phase, but in practice I should give them
                # weights according to their contribution to the final mode (their phasor arrow lengths). To do this I
                # sum their complex field contributions, and take the conjugate of that angle
                macro_pixel_size = round(1/deg_of_control)
                max_index = (len(at_slm)//macro_pixel_size)*macro_pixel_size

                # inspiration from here: https://stackoverflow.com/questions/18582544/sum-parts-of-numpy-array by Jaime
                # shape of -1 means it is inferred for the length + other dimension lengths
                truncated = at_slm[:max_index]
                averaged = truncated.reshape((-1, macro_pixel_size)).sum(axis=1)
                repeated = np.repeat(averaged, macro_pixel_size)
                end_val = at_slm[max_index:].sum()
                padded = np.pad(repeated, (0, len(at_slm) % macro_pixel_size), constant_values=(0, end_val))
                S = -np.angle(padded)

        elif deg_of_control == 0:
            # Global phase does nothing
            S = np.zeros(self.N)

        return S % (2*np.pi)

    def reset_T(self):
        if self.T_mode == 'unitary':
            self.T = unitary_group.rvs(self.N)
            # self.T2 = unitary_group.rvs(self.N)
        elif self.T_mode == 'gaus_iid':
            # Following this: https://stackoverflow.com/questions/55700338/how-to-generate-a-complex-gaussian-white-noise-signal-in-pythonor-numpy-scipy
            # with a var of 1
            self.T = 1/np.sqrt(self.N) * np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(self.N, self.N, 2)).view(np.complex128)[:, :, 0]
            # self.T2 = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(self.N, self.N, 2)).view(np.complex128)[:, :, 0]
        else:
            raise Exception('Need to choose s.T_mode of either "unitary" or "gaus_iid"')

    def test(self, ctrl=1):
        self.reset_T()
        print('Klyshko')
        eff, std = self.get_klyshko_efficiency(N_average=10, deg_of_control=1)
        print(f'{eff}+-{std}')

        print('beacon one way O1')
        eff, std = self.get_beacon_1way_efficiency(N_average=10, deg_of_control=1, out1=True)
        print(f'{eff}+-{std}')

        print('beacon one way O2')
        eff, std = self.get_beacon_1way_efficiency(N_average=10, deg_of_control=1, out1=False)
        print(f'{eff}+-{std}')

        print('beacon both ways O1->O2')
        eff, std = self.get_beacon_2way_efficiency(N_average=10, deg_of_control=1)
        print(f'{eff}+-{std}')

        print('Klyshko QOPC')
        old_O2 = self.O2
        self.O2 = self.O1
        eff, std = self.get_klyshko_efficiency(N_average=10, deg_of_control=1)
        print(f'{eff}+-{std}')
        self.O2 = old_O2

    def get_klyshko_efficiency(self, N_average=10, deg_of_control=1):
        effs = np.zeros(N_average)

        for i in range(N_average):
            self.reset_T()
            S_k = self.get_optimal_klyshko_conj(deg_of_control=deg_of_control)
            amps = self.prop_klyshko(S_k)
            efficiency = np.abs(amps[self.O2]) ** 2
            effs[i] = efficiency

        return effs.mean(), effs.std()

    def get_beacon_1way_efficiency(self, N_average=10, deg_of_control=1, out1=True):
        effs = np.zeros(N_average)

        for i in range(N_average):
            self.reset_T()
            S1 = self.get_optimal_beacon_conj(out1=out1, deg_of_control=deg_of_control)
            amps = self.prop_beacon(S1)
            efficiency = np.abs(amps[self.O1 if out1 else self.O2]) ** 2
            effs[i] = efficiency

        return effs.mean(), effs.std()

    def get_beacon_2way_efficiency(self, N_average=10, deg_of_control=1):
        effs = np.zeros(N_average)

        for i in range(N_average):
            self.reset_T()
            S1 = self.get_optimal_beacon_conj(out1=True, deg_of_control=deg_of_control)
            S2 = self.get_optimal_beacon_conj(out1=False, deg_of_control=deg_of_control)
            amps = self.prop_klyshko((S1 + S2) / 2)
            efficiency = np.abs(amps[self.O2]) ** 2
            effs[i] = efficiency

        return effs.mean(), effs.std()

    def run(self, N_average=10, N_ctrls=10, ctrls=()):
        res = KlyshkoBeaconSimulationResult()
        res.T_mode = self.T_mode
        res.incomplete_control_method = self.incomplete_control_method

        if len(ctrls) == 0:
            ctrls = np.linspace(0, 1, N_ctrls)
        res.ctrls = ctrls

        for ctrl in ctrls:
            print(ctrl)
            eff, std = self.get_klyshko_efficiency(N_average=N_average, deg_of_control=ctrl)
            res.klyshkos.append(eff)
            res.klyshko_stds.append(std)

            eff, std = self.get_beacon_1way_efficiency(N_average=N_average, deg_of_control=ctrl, out1=True)
            res.beacon_one_ways.append(eff)
            res.beacon_one_way_stds.append(std)

            # eff, std = self.get_beacon_1way_efficiency(N_average=10, deg_of_control=1, out1=False)

            eff, std = self.get_beacon_2way_efficiency(N_average=N_average, deg_of_control=ctrl)
            res.beacon_two_ways.append(eff)
            res.beacon_two_way_stds.append(std)

            old_O2 = self.O2
            self.O2 = self.O1
            eff, std = self.get_klyshko_efficiency(N_average=N_average, deg_of_control=ctrl)
            self.O2 = old_O2
            res.QOPCs.append(eff)
            res.QOPC_stds.append(std)

        return res

    def plot_res(self, S, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        final_mat = self.T @ np.diag(np.exp(1j*S))**2 @ self.T.transpose().conjugate()

        X_MARKER_COLOR = '#929591'
        X_MARKER_EDGEWITDH = 1.5

        axes[0].set_title('original $T*T^t$')
        imm = axes[0].imshow(np.abs(self.T @ self.T.transpose())**2)
        axes[0].plot(self.O2, self.O1, '+', markeredgecolor=X_MARKER_COLOR, markersize=11, markeredgewidth=X_MARKER_EDGEWITDH)
        axes[0].figure.colorbar(imm, ax=axes[0])

        axes[1].set_title('optimized $T*S^2*T^t$')
        imm = axes[1].imshow(np.abs(final_mat)**2)
        axes[1].plot(self.O2, self.O1, '+', markeredgecolor=X_MARKER_COLOR, markersize=11, markeredgewidth=X_MARKER_EDGEWITDH)
        axes[1].figure.colorbar(imm, ax=axes[1])
        axes[1].figure.show()

        print(S)
        plt.show(block=False)


if __name__ == "__main__":
    # S, all_costs = optimize_klyshko()
    # fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    # plot_res(S, all_costs, axes)

    # S2, all_costs = optimize_beacon()
    # fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    # plot_res(S2, all_costs, axes)
    s = KlyshkoBeaconSimulation(T_mode='gaus_iid')
    res = s.run(3, 20, 1 / np.linspace(1, 20, 20))
    res.show((False, False, True, False), show_legend=False)
    plt.show()