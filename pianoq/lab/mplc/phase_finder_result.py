import numpy as np
import matplotlib.pyplot as plt
import traceback
from pianoq.lab.mplc.consts import N_SPOTS
import datetime
from scipy.optimize import curve_fit


class PhaseFinderResult(object):
    def __init__(self, path=None):
        self.path = path
        if path:
            self.loadfrom()
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.coincidences = None
        self.single1s = None
        self.single2s = None
        self.integration_time = -1
        self.coin_window = -1

        self.phases = np.zeros(N_SPOTS*2)
        self.modes_to_keep = np.array([])
        self.N_phases = 0
        self.phase_vec_step = 0
        self.phase_vec = np.array([])
        self.initial_phases = np.array([])

    def plot_best_phases(self):
        for i, mode_to_keep in enumerate(self.modes_to_keep):
            print(mode_to_keep)
            self.fit_and_plot_cosine(self.phase_vec, self.coincidences[i, :], True)

    def saveto(self, path):
        try:
            f = open(path, 'wb')
            np.savez(f,
                     coincidences=self.coincidences,
                     single1s=self.single1s,
                     single2s=self.single2s,
                     integration_time=self.integration_time,
                     coin_window=self.coin_window,
                     timestamp=self.timestamp,
                     phases=self.phases,
                     modes_to_keep=self.modes_to_keep,
                     N_phases=self.N_phases,
                     phase_vec_step=self.phase_vec_step,
                     phase_vec=self.phase_vec,
                     initial_phases=self.initial_phases)
            f.close()
        except Exception as e:
            print("ERROR!!")
            print(e)
            traceback.print_exc()

    def loadfrom(self, path=None):
        if path is None:
            path = self.path
        if path is None:
            raise Exception("No path")
        path = path.strip('"')
        path = path.strip("'")
        self.path = path

        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        self.coincidences = data['coincidences']
        self.single1s = data['single1s']
        self.single2s = data['single2s']
        self.integration_time = data.get('integration_time', None)
        self.coin_window = data.get('coin_window', 4e-9)
        self.timestamp = data['timestamp']

        self.phases = data['phases']
        self.modes_to_keep = data['modes_to_keep']
        self.N_phases = data['N_phases']
        self.phase_vec_step = data['phase_vec_step']
        self.phase_vec = data['phase_vec']
        self.initial_phases = data.get('initial_phases', np.zeros(50))
        f.close()

    def fit_and_plot_cosine(self, phase_vec, coincidences, should_plot=True):
        """
        Cluade generated code.
        This is to prove to ourselves that the lock-in does a good job, and that it agrees with a fit to a cosine.
        """
        def cosine_func(x, amplitude, offset, phase):
            return amplitude * np.cos(x + phase) + offset

        # Perform the fit
        popt, _ = curve_fit(cosine_func, phase_vec, coincidences,
                            p0=[np.max(coincidences) - np.min(phase_vec), np.mean(coincidences), 0])
        amplitude, offset, phase = popt

        # Generate points for the fitted curve
        x_fit = np.linspace(0, 2*np.pi, 1000)
        y_fit = cosine_func(x_fit, *popt)

        # Calculate phase for maximal amplitude
        phase_max = -phase % (2 * np.pi)

        if should_plot:
            CC = (coincidences * np.exp(1j * phase_vec)).sum()
            phi_best_lock_in = np.mod(np.angle(CC)+2*np.pi, 2*np.pi)

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot the data and the fit
            ax.scatter(phase_vec, coincidences, label='Data')
            ax.plot(x_fit, y_fit, 'r--', label='Fit', linewidth=0.3)

            ax.set_xlabel('Phase')
            ax.set_ylabel('Coincidences')
            ax.set_title(f'Cosine Fit for Coincidences')
            ax.legend()

            # Add vertical line at phase_max
            ax.axvline(x=phase_max, color='g', linestyle='--', label='Max Amplitude Phase fit')
            ax.axvline(x=phi_best_lock_in, color='c', linestyle='--', label='Max Amplitude Phase lock-in')
            ax.legend()

            fig.tight_layout()
            plt.show(block=True)

        return phase_max

    def reload(self):
        self.loadfrom(self.path)

