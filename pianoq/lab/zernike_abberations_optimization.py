"""
 credit to the idea and a lot of the code:
 https://www.wavefrontshaping.net/post/id/23
"""
import numpy as np
from aotools.functions import phaseFromZernikes

from pianoq.lab.optimizations.my_pso import MyPSOOptimizer
from pianoq.lab.photon_scan import PhotonScanner
from pianoq.lab.slm import SLMDevice
from pianoq.lab.thorlabs_motor import ThorlabsKcubeDC, ThorlabsKcubeStepper
from pianoq.lab.time_tagger import QPTimeTagger
import datetime


class ZernikeOptimizer(object):
    N_PIXELS = 800
    ACTIVE_MASK = np.index_exp[50:850, 50:850]

    def __init__(self, integration_time=1.0, N_zernike=15):
        self.integration_time = integration_time
        self.N_zernike = N_zernike
        self.tt = None
        self.slm = None
        self.optimizer = None
        self.best_vec = np.zeros(N_zernike)
        self._init_hardware()

    def _init_hardware(self):
        self.slm = SLMDevice(0)
        self.tt = QPTimeTagger(integration_time=self.integration_time)

    def optimize_pso(self):
        self.optimizer = MyPSOOptimizer(self.get_cost, n_pop=30, n_var=self.N_zernike,
                                        n_iterations=1000,
                                        w=1, wdamp=0.99, c1=1.5, c2=2,
                                        timeout=45 * 60,
                                        stop_early=True, stop_after_n_const_iter=20,
                                        vary_popuation=True, reduce_at_iterations=(2,),
                                        quiet=False, n_for_average_cost=3)
        self.optimizer.optimize()
        self.best_vec = self.pos01_to_real(self.optimizer.best_positions)


    def optimize_naive(self):
        cur_vec = np.zeros(self.N_zernike)
        for i in range(self.N_zernike):
            best_pos = 0
            best_cost_iter = 0
            for pos in np.linspace(0, 1, 21):
                cur_vec[i] = pos
                cost, _, _ = self.get_cost(cur_vec)
                if cost <  best_cost_iter:
                    best_pos = pos
                    best_cost_iter = cost
                    print(self.pos01_to_real(cur_vec))
                    print(f'best cost: {best_cost_iter}')

            cur_vec[i] = best_pos
        self.best_vec = self.pos01_to_real(cur_vec)

    def pos01_to_real(self, vec):
        return (vec * 4) - 2

    def get_cost(self, zern_vec):
        # zern_vec is in [0,1], and I want it in [-2, 2]
        vec = self.pos01_to_real(zern_vec)
        self.slm.update_phase_in_active(phaseFromZernikes(vec, self.N_PIXELS), self.ACTIVE_MASK)
        s1, s2, c = self.tt.read_interesting()
         #cost = c - s1*s2*2*self.tt.coin_window
        print(c)
        return -c, np.sqrt(c), None

    def scan(self, name=''):
        mid_x = 8.7
        mid_y = 14.15
        start_x = 11.3
        start_y = 14.45
        x_pixels = 10
        y_pixels = 10
        pixel_size_x = 0.05
        pixel_size_y = 0.05

        scanner = PhotonScanner(self.integration_time, start_x, start_y, x_pixels, y_pixels, pixel_size_x, pixel_size_y,
                                run_name=name, is_timetagger=True)

        x_motor = ThorlabsKcubeDC(27600573)
        print('got x_motor')
        y_motor = ThorlabsKcubeStepper(26003411)
        print('got y_motor')
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        saveto_path = None
        single1s, single2s, coincidences = scanner.scan(ph=self.tt, x_motor=x_motor, y_motor=y_motor)

    def close(self):
        self.slm.close()
        self.tt.close()


def main():
    z = ZernikeOptimizer(integration_time=4, N_zernike=15)
    z.optimize_pso()
    z.optimize_naive()
    z.close()

