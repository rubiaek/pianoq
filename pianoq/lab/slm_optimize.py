import time
import datetime
import traceback

import SLMlayout
from scipy.optimize import differential_evolution

from pianoq.lab.power_meter100 import PowerMeterPM100
from pianoq.lab.slm import SLMDevice
from pianoq.lab.time_tagger import QPTimeTagger
from pianoq.lab.asi_cam import ASICam
from pianoq_results.slm_optimization_result import SLMOptimizationResult
import matplotlib.pyplot as plt
import numpy as np

LOGS_DIR = 'C:\\temp'


def spiral(X, Y):
    yield 0, 0


class SLMOptimizer(object):
    POINTS_FOR_LOCK_IN = 8

    PARTITIONING = 'partitioning'
    CONTINUOUS = 'continuous'
    GENETIC = 'genetic'
    PARTITIONING_HEX = 'partitioning_hex'
    CONTINUOUS_HEX = 'continuous_hex'

    def __init__(self, macro_pixels=None, sleep_period=0.1, run_name='optimizer_result', saveto_path=None):
        self.macro_pixels = macro_pixels
        self.sleep_period = sleep_period

        self.power_scaling_factor = 1

        self.run_name = run_name
        self.saveto_path = saveto_path

        self.res = SLMOptimizationResult()
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        self.cur_best_slm_phase = np.zeros([self.macro_pixels, self.macro_pixels])
        self.micro_iter_num = 0
        self.macro_iter_num = 0

        self.is_cam = None
        self.slm = None
        self.cam = None
        self.roi = None
        self.timetagger = None
        self.power_meter = None
        self.best_phi_method = None

        self.hexs = None

    # Optimization functions
    def optimize(self, method='partitioning', iterations=1000, slm=None, cam=None, timetagger=None, power_meter=None,
                 roi=None, best_phi_method='lock_in', cell_size=20):
        self.slm = slm
        self.cam = cam
        self.timetagger = timetagger
        self.power_meter = power_meter
        self.roi = roi
        self.best_phi_method = best_phi_method

        mask_generator = None
        lock_in_method = True
        self.res.opt_method = method
        self.res.best_phi_method = best_phi_method
        self.res.roi = roi
        try:
            if method == self.PARTITIONING:
                mask_generator = self._partitioning()
            elif method == self.GENETIC:
                lock_in_method = False
                bounds = [[0, 2 * np.pi]] * self.macro_pixels * self.macro_pixels
                res = differential_evolution(self.get_cost_genetic, bounds,
                                             strategy='best1bin', maxiter=iterations,
                                             popsize=0.1, recombination=0.5, mutation=(0.01, 0.1))
                self._save_result()
                self.genetic_res = res
            elif method == self.PARTITIONING_HEX or method == self.CONTINUOUS_HEX:
                print("Getting Hexs...")
                self.hexs = SLMlayout.Hexagons(radius=self.slm.radius, cellSize=cell_size,
                                               resolution=self.slm.correction.shape, center=self.slm.center,
                                               method='equal')  # TODO: maybe grid?
                print("Got it!")
                self.cur_best_slm_phase = np.zeros(self.hexs.nParts)
                if method == self.PARTITIONING_HEX:
                    mask_generator = self._partitioning_hex()
                elif method == self.CONTINUOUS_HEX:
                    mask_generator = self._continuous_hex()
            else:
                raise NotImplemented()

            if lock_in_method:
                for i in range(iterations):
                    mask_to_play = next(mask_generator)
                    start_time = time.time()
                    self.macro_iter_num += 1
                    print(f'\niteration: {self.macro_iter_num}')
                    self.do_iteration(mask_to_play)
                    duration = round(time.time()-start_time, 2)
                    print(f'took {duration} seconds')
                    self._save_result()

                    # yield

        except Exception:
            print('==>ERROR!<==')
            traceback.print_exc()

        self._save_result()

    def _continuous(self):
        while True:
            for i, j in spiral(self.macro_pixels, self.macro_pixels):
                # new_i = (i + 2) % self.macro_pixels_y_slm
                mask_to_play = np.zeros([self.macro_pixels, self.macro_pixels])
                mask_to_play[i, j] = 1
                yield mask_to_play

    def _partitioning(self):
        while True:
            mask_to_play = np.random.randint(2, size=(self.macro_pixels, self.macro_pixels))
            yield mask_to_play

    def _continuous_hex(self):
        i = 0
        while True:
            mask_to_play = np.zeros(self.hexs.nParts)
            mask_to_play[i % self.hexs.nParts] = 1
            i += 1
            yield mask_to_play

    def _partitioning_hex(self):
        while True:
            mask_to_play = np.random.randint(2, size=self.hexs.nParts)
            yield mask_to_play

    def do_iteration(self, mask_to_play):
        current_iter_costs = []
        # TODO: export 1pi option (relevant for Klyshko where we hit the SLM twice)
        # TODO: maybe not 1*pi?
        phis = np.linspace(0, np.pi, self.POINTS_FOR_LOCK_IN+1)
        phis = phis[:-1]  # Don't need both 0 and 2*pi

        for phi in phis:
            phase_mask = self.cur_best_slm_phase.copy() + phi * mask_to_play
            self.update_slm(phase_mask)
            cost, cost_witness = self.get_cost()
            self.res.all_phase_masks.append(phase_mask)
            self.res.all_costs.append(cost)
            # self.res.all_cost_witnesses.append(cost_witness)
            self.res.all_cost_witnesses.append(None)
            current_iter_costs.append(cost)

        best_phi = self._get_best_phi(phis, current_iter_costs)
        best_phase_mask = self.cur_best_slm_phase.copy() + best_phi * mask_to_play
        self.update_slm(best_phase_mask)
        best_cost, best_cost_witness = self.get_cost()

        # Scaling factor comes from changing of exposure time
        print(f'best cost: {best_cost}')
        self.cur_best_slm_phase = np.mod(best_phase_mask, 2 * np.pi)

        self.res.phase_masks.append(self.cur_best_slm_phase)
        self.res.costs.append(best_cost)
        self.res.cost_witnesses.append(best_cost_witness)

    def update_slm(self, phase_mask):
        if self.res.opt_method == self.PARTITIONING_HEX:
            patt = self.hexs.getImageFromVec(phase_mask, dtype=float)
            self.slm.update_phase(patt)
        else:
            self.slm.update_phase_in_active(phase_mask)

    def _fix_exposure(self, powers):
        mx = powers.max()
        if mx > 50e3:
            print('Lowering exposure time!')
            exp_time = self.cam.get_exposure_time()
            if exp_time * (4 / 5) > 0: # TODO: check this 0
                self.cam.set_exposure_time(exp_time * (4 / 5))
                self.power_scaling_factor *= (5 / 4)
            else:
                print('**At shortest exposure and still saturated... You might want to add an ND to the camera..**')

    def _get_best_phi(self, phis, powers, plot_cos=False):
        if self.best_phi_method == 'lock_in':
            # "Lock in"
            C1 = powers * np.cos(phis)
            C = np.sum(C1)
            S1 = powers * np.sin(phis)
            S = np.sum(S1)
            A = C + 1j * S
            best_phi = np.angle(A)
            if best_phi < 0:
                best_phi += 2 * np.pi
        elif self.best_phi_method == 'silly_max':
            best_phi = phis[np.argmax(powers)]
        elif self.best_phi_method == 'cos_fit':
            raise Exception('not implemented yet')
        else:
            raise Exception('say wat?')

        if plot_cos:
            fig, ax = plt.subplots()
            ax.plot(phis, powers, '*')
            ax.axvline(x=best_phi, linestyle='--')
            fig.show()

        return best_phi

    def get_cost_genetic(self, phase_mask):
        if phase_mask.shape == (self.macro_pixels*self.macro_pixels,):
            phase_mask = phase_mask.reshape(self.macro_pixels, self.macro_pixels)
        else:
            raise Exception("something unexpected!")

        self.slm.update_phase_in_active(phase_mask)
        self.slm.update_phase_in_active(phase_mask)
        cost, cost_witness = self.get_cost()
        self.res.all_phase_masks.append(phase_mask)
        self.res.all_costs.append(cost)
        self.res.all_cost_witnesses.append(cost_witness)

        if self.micro_iter_num % 10 == 0:
            print(f'\niteration: {self.micro_iter_num}. cost={cost}')
            self._save_result()

        return -cost

    def get_cost(self):
        self.micro_iter_num += 1
        time.sleep(self.sleep_period)
        if self.timetagger:
            assert isinstance(self.timetagger, QPTimeTagger)
            s1, s2, c = self.timetagger.read_interesting()
            cost = c-2*s1*s2*self.timetagger.coin_window
            cost_witness = None
        elif self.power_meter:
            cost = self.power_meter.get_power()
            cost_witness = None
        else:
            cost_witness = self.cam.get_image()
            self._fix_exposure(cost_witness)
            if self.roi:
                imm = cost_witness[self.roi]
                cost = imm.sum()
            else:
                cost = cost_witness.sum()

        return cost, cost_witness

    def _save_result(self):
        saveto_path = self.saveto_path or f"{LOGS_DIR}\\{self.timestamp}_{self.run_name}.optimizer2"
        self.res.saveto(saveto_path)


if __name__ == '__main__':
    macro_pixels = 20
    sleep_period = 0.02
    run_name = f'radius_150_type_mirror'

    asi_exposure_time = 3e-3
    roi = (3040, 1746, 600, 600)
    l = 3
    cost_roi = np.index_exp[300-l:300+l, 300-l:300+l]

    slm = SLMDevice(0, use_mirror=True)
    slm.set_pinhole(radius=150, center=(530, 500), pinhole_type='mirror')  # pinhole_type='rand'

    cam = ASICam(asi_exposure_time, binning=1, roi=roi, gain=0)
    power_meter = PowerMeterPM100()

    # tt = QPTimeTagger(integration_time=1, coin_window=2e-9, single_channel_delays=(0, 1600))

    o = SLMOptimizer(macro_pixels=macro_pixels, sleep_period=sleep_period, run_name=run_name, saveto_path=None)
    # g = o.optimize(method=SLMOptimizer.PARTITIONING, iterations=(macro_pixels**2)*2, slm=slm, timetagger=tt)
    # g = o.optimize(method=SLMOptimizer.PARTITIONING, iterations=(macro_pixels**2)*2, slm=slm, cam=cam, roi=cost_roi,
    #                best_phi_method='silly_max')
    # g = o.optimize(method=SLMOptimizer.GENETIC, iterations=(macro_pixels**2)*2, slm=slm, cam=cam, roi=cost_roi)

    g = o.optimize(method=SLMOptimizer.CONTINUOUS_HEX, iterations=150, slm=slm, cam=cam, power_meter=power_meter,
                   roi=cost_roi, best_phi_method='silly_max', cell_size=30)

    power_meter.close()
    cam.close()
    slm.close()

