import time
import datetime
import traceback

from scipy.optimize import differential_evolution

from pianoq.lab.slm import SLMDevice
from pianoq.lab.time_tagger import QPTimeTagger
from pianoq.lab.asi_cam import ASICam
from pianoq_results.slm_optimization_result import SLMOptimizationResult
import matplotlib.pyplot as plt
import numpy as np

LOGS_DIR = 'C:\\temp'


class SLMOptimizer(object):
    POINTS_FOR_LOCK_IN = 6

    PARTITIONING = 'partitioning'
    CONTINUOUS = 'continuous'
    GENETIC = 'genetic'

    def __init__(self, macro_pixels=None, sleep_period=0.1, run_name='optimizer_result', saveto_path=None):
        self.macro_pixels = macro_pixels
        self.sleep_period = sleep_period

        self.power_scaling_factor = 1

        self.run_name = run_name
        self.saveto_path = saveto_path

        self.res = SLMOptimizationResult()
        self.res.powers = []
        self.res.mid_results = {}
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        self.cur_best_slm_phase = np.zeros([self.macro_pixels, self.macro_pixels])
        self.micro_iter_num = 0
        self.macro_iter_num = 0

        self.is_cam = None
        self.slm = None
        self.cam = None
        self.roi = None
        self.timetagger = None

    # Optimization functions
    def optimize(self, method='continuous', iterations: int = 1000, slm=None, cam=None, timetagger=None, roi=None):
        self.slm = slm
        self.cam = cam
        self.timetagger = timetagger
        self.roi = roi

        mask_generator = None
        lock_in_method = True
        self.res.opt_method = method
        try:
            if method == self.PARTITIONING:
                mask_generator = self._partitioning()
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

                    yield

        except Exception:
            print('==>ERROR!<==')
            traceback.print_exc()

        self._save_result()

    def _continuous(self):
        while True:
            for i, j in spiral(self.macro_pixels_y_slm, self.macro_pixels_x_slm):
                # new_i = (i + 2) % self.macro_pixels_y_slm
                mask_to_play = np.zeros([self.macro_pixels_y_slm, self.macro_pixels_x_slm])
                mask_to_play[i, j] = 1
                yield mask_to_play

    def _partitioning(self):
        while True:
            mask_to_play = np.random.randint(2, size=(self.macro_pixels, self.macro_pixels))
            yield mask_to_play

    def do_iteration(self, mask_to_play):
        current_iter_costs = []
        phis = np.linspace(0, 2 * np.pi, self.POINTS_FOR_LOCK_IN+1)
        phis = phis[:-1]  # Don't need both 0 and 2*pi

        for phi in phis:
            phase_mask = self.cur_best_slm_phase.copy() + phi * mask_to_play
            self.slm.update_phase_in_active(phase_mask)
            cost, cost_witness = self.get_cost()
            self.res.all_phase_masks.append(phase_mask)
            self.res.all_costs.append(cost)
            self.res.all_cost_witnesses.append(cost_witness)
            current_iter_costs.append(cost)

        best_phi = self._get_best_phi(phis, current_iter_costs)
        best_phase_mask = self.cur_best_slm_phase.copy() + best_phi * mask_to_play
        self.slm.update_phase_in_active(best_phase_mask)
        best_cost, best_cost_witness = self.get_cost()

        # Scaling factor comes from changing of exposure time
        print(f'best cost: {best_cost}')
        self.cur_best_slm_phase = np.mod(best_phase_mask, 2 * np.pi)

        self.res.phase_masks.append(self.cur_best_slm_phase)
        self.res.costs.append(best_cost)
        self.res.cost_witnesses.append(best_cost_witness)

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
        # "Lock in"
        C1 = powers * np.cos(phis)
        C = np.sum(C1)
        S1 = powers * np.sin(phis)
        S = np.sum(S1)
        A = C + 1j * S
        best_phi = np.angle(A)
        if best_phi < 0:
            best_phi += 2 * np.pi

        if plot_cos:
            fig, ax = plt.subplots()
            ax.plot(phis, powers, '*')
            ax.axvline(x=best_phi, linestyle='--')
            fig.show()

        return best_phi

    def get_cost(self):
        self.micro_iter_num += 1
        time.sleep(self.sleep_period)
        if self.timetagger:
            assert isinstance(self.timetagger, QPTimeTagger)
            s1, s2, c = self.timetagger.read_interesting()
            cost = c-2*s1*s2*self.timetagger.coin_window
            cost_witness = None
        else:
            cost_witness = cam.get_image()
            self._fix_exposure(cost_witness)
            if self.roi:
                imm = cost_witness[self.roi]
                cost = imm.sum()
            else:
                cost = cost_witness.sum()

        return cost, cost_witness

    def _save_result(self):
        saveto_path = self.saveto_path or f"{LOGS_DIR}\\{self.timestamp}_{self.run_name}.optimizer"
        self.res.saveto(saveto_path)


if __name__ == '__main__':
    if True:  # Lab
        macro_pixels = 30
        sleep_period = 0.1
        run_name = 'optimizer_result'

        asi_exposure_time = 1e-3
        roi = (None, None, None, None)

        slm = SLMDevice(0)
        cam = ASICam(asi_exposure_time, binning=1, roi=roi, gain=0)
        # tt = QPTimeTagger(integration_time=1, coin_window=1e-9, single_channel_delays=(0, 1600))

        o = SLMOptimizer(macro_pixels=macro_pixels, sleep_period=sleep_period, run_name=run_name, saveto_path=None)
        g = o.optimize(method=SLMOptimizer.PARTITIONING, iterations=(macro_pixels**2)*2, slm=slm, cam=cam)

        for i in g:
            if i == 5:  # Just so there will be lines of code to break from while debugging
                pass
