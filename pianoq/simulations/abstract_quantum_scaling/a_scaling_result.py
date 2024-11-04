import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class QScalingSimulationResult(object):
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
        # fig.show()
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

