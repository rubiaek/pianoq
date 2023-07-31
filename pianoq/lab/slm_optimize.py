import time
import datetime
import traceback

from scipy.optimize import differential_evolution

from pianoq.lab.slm import SLMDevice
from pianoq.lab.time_tagger import QPTimeTagger
from pianoq.lab.asi_cam import ASICam
from opticalsimulator.results import OptimizerResult
from opticalsimulator.simulations.ronen.misc_utils import spiral
import matplotlib.pyplot as plt
import numpy as np
import cv2

LOGS_DIR = 'C:\\temp'


class Optimizer(object):
    """
        This optimizer can work both in lab and in simulation!
        It assumes this configuration: Beam->SLM1-4f-SLM2-2f->camera.
        We use SLM2 as a diffuser, and try optimizing algorithms to use SLM1 to fix the diffusion.
        At the beginning we see a speckle pattern, and we try to get to a small spot.
        The results are saved using the OptimizerResult object.
    """
    MICRO_PIXELS_X = 150
    MICRO_PIXELS_Y = 270
    MAG = 60
    MACRO_PIXELS_X_SLM = int(np.round(MICRO_PIXELS_X / MAG))
    MACRO_PIXELS_Y_SLM = int(np.round(MICRO_PIXELS_Y / MAG))
    MACRO_PIXELS_X_DIFFUSER = int(np.round(MICRO_PIXELS_X / MAG))
    MACRO_PIXELS_Y_DIFFUSER = int(np.round(MICRO_PIXELS_Y / MAG))
    FIELD_PIXELS_X = 1000
    FIELD_PIXELS_Y = 1000
    SLICES_OF_INTEREST = [np.index_exp[99:101, 99:101]]
    OUTPUT_OF_INTEREST = np.index_exp[97:105, 97:105]
    POINTS_FOR_LOCK_IN = 6
    N = 1024  # For simulation grid size

    BORDERS = Borders(-0.010, -0.010, 0.010, 0.010)
    WAVELENGTH = 404e-9
    FOCAL_LENGTH = 0.300

    PARTITIONING = 'partitioning'
    CONTINUOUS = 'continuous'
    GENETIC = 'genetic'

    def __init__(self, is_simulation=True, slices_of_interest=None,
                 macro_pixels_x_diffuser=None, macro_pixels_y_diffuser=None, diffuser_mask=None,
                 macro_pixels_x_slm=None, macro_pixels_y_slm=None, sleep_period=0.0,
                 exposure_time=1000, mid_results_diff=50,
                 run_name='optimizer_result', saveto_path=None):

        self.is_simulation = is_simulation
        self.slices_of_interest = slices_of_interest or self.SLICES_OF_INTEREST
        if not isinstance(self.slices_of_interest, list):
            self.slices_of_interest = [self.slices_of_interest]

        self.macro_pixels_x_diffuser = macro_pixels_x_diffuser or self.MACRO_PIXELS_X_DIFFUSER
        self.macro_pixels_y_diffuser = macro_pixels_y_diffuser or self.MACRO_PIXELS_Y_DIFFUSER
        self.diffuser_mask = diffuser_mask
        self.macro_pixels_x_slm = macro_pixels_x_slm or self.MACRO_PIXELS_X_SLM
        self.macro_pixels_y_slm = macro_pixels_y_slm or self.MACRO_PIXELS_Y_SLM
        self.sleep_period = sleep_period
        self.exposure_time = exposure_time
        self.power_scaling_factor = 1

        self.run_name = run_name
        self.saveto_path = saveto_path

        self.result = OptimizerResult()
        self.result.powers = []
        self.result.mid_results = {}
        self.mid_results_diff = mid_results_diff
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        if self.is_simulation:
            self.light_source = GaussianLightSource(sigma=0.0014, wavelength=self.WAVELENGTH)
            self.slm = SLMMask(self.BORDERS, pixels_x=self.MICRO_PIXELS_X, pixels_y=self.MICRO_PIXELS_Y)
            self.diffuser = SLMMask(self.BORDERS, pixels_x=self.MICRO_PIXELS_X, pixels_y=self.MICRO_PIXELS_Y)
            self.farfield = FourierLens() #f=self.FOCAL_LENGTH)
        else:
            self.slm = RealSLMDevice(1)
            # self.diffuser = RealSLMDevice(2, use_mirror=True)
            self.camera = VimbaCamera(0, exposure_time=self.exposure_time)  # micro-s

        # self._init_init_power()
        # self.show_input_field()

        self._init_good_before()
        self._init_diffuser(should_save=True, force_mask=self.diffuser_mask)
        self.cur_best_slm_phase = np.zeros([self.macro_pixels_y_slm, self.macro_pixels_x_slm])
        self.micro_iter_num = 0
        self.macro_iter_num = 0

    # Preparation functions
    def _generate_diffuser_mask(self):
        phase_mask = 2 * np.pi * np.random.rand(self.macro_pixels_y_diffuser, self.macro_pixels_x_diffuser)
        return phase_mask

    def _init_init_power(self):
        if self.diffuser_mask:
            print("Not checking initial power because specific diffuser given")
            self.init_power = 0
            return

        print('Checking Mean initial power')
        N = 8
        powers = []
        for i in range(N):
            self._init_diffuser(should_save=False)
            im = self.get_image()
            self.show_power(self.orig_speckles_ax, im, 'Original Speckle Pattern')
            power = self.power_at_target(im)
            print(f'Power before optimization: {power}')
            powers.append(power)

        self.init_power = np.mean(powers[1:])  # Weird thing happens with first...
        print(f'Mean power before optimization: {self.init_power}')

    def _init_good_before(self):
        if not self.is_simulation:
            im = self.get_image()
        else:  # is simulation
            im = self.simulate_propagation_new(fixer_mask=-self.diffuser_mask)
        self.result.original_good = im

    def _init_diffuser(self, should_save=False, force_mask=None):
        if force_mask is not None:
            diffuser_mask = force_mask
        else:
            diffuser_mask = self._generate_diffuser_mask()

        # TODO: add to optimize result a "before" picture before setting diffuser
        # self.diffuser.update_phase_in_active(diffuser_mask)
        self.result.diffuser_phase_grid = diffuser_mask

        if should_save:
            plt.pause(0.2)
            if not self.is_simulation:
                im = self.get_image()
            else: # is simulation
                fixer_mask = np.zeros((4, 4))
                im = self.simulate_propagation_new(fixer_mask)
            self.result.original_speckle_pattern = im

    # Optimization functions
    def optimize(self, method='continuous', iterations: int = 1000):

        mask_generator = None
        lock_in_method = True
        self.result.opt_method = method
        try:
            if method == self.PARTITIONING:
                mask_generator = self._partitioning()
            elif method == self.CONTINUOUS:
                mask_generator = self._continuous()
            elif method == self.GENETIC:
                lock_in_method = False
                bounds = [[0, 2*np.pi]] * self.macro_pixels_x_slm * self.macro_pixels_y_slm
                res = differential_evolution(self._mask_to_power_ge, bounds,
                                             strategy='best1bin', maxiter=iterations,
                                             popsize=6, recombination=0.5, mutation=(0.01, 0.1))
                self._save_result()
                self.genetic_res = res
            else:
                raise NotImplemented()

            if lock_in_method:
                for i in range(iterations):
                    mask_to_play = next(mask_generator)
                    start_time = time.time()
                    self.macro_iter_num += 1
                    print(f'\niteration: {self.macro_iter_num}')
                    self.find_best_phase(mask_to_play)
                    duration = round(time.time()-start_time, 2)
                    print(f'took {duration} seconds')

                    self._save_result()

                    yield

        except Exception:
            print('==>ERROR!<==')
            traceback.print_exc()

        self._save_result()
        # self.plot_powers(powers)

    def _continuous(self):
        while True:
            for i, j in spiral(self.macro_pixels_y_slm, self.macro_pixels_x_slm):
                # new_i = (i + 2) % self.macro_pixels_y_slm
                mask_to_play = np.zeros([self.macro_pixels_y_slm, self.macro_pixels_x_slm])
                mask_to_play[i, j] = 1
                yield mask_to_play

    def _partitioning(self):
        while True:
            mask_to_play = np.random.randint(2, size=(self.macro_pixels_y_slm, self.macro_pixels_x_slm))
            yield mask_to_play

    def find_best_phase(self, mask_to_play):
        powers = []
        phis = np.linspace(0, 2 * np.pi, self.POINTS_FOR_LOCK_IN)
        phis = phis[:-1]  # Don't need both 0 and 2*pi

        for phi in phis:
            phase_mask = self.cur_best_slm_phase.copy() + phi * mask_to_play
            power = self._mask_to_power(phase_mask, is_best=False)
            powers.append(power)

        best_phi = self._get_best_phi(phis, powers)

        phase_mask = self.cur_best_slm_phase.copy() + best_phi * mask_to_play
        power = self._mask_to_power(phase_mask, is_best=True)

        """
        # Yaron though this isn't a good idea, and rather trust in the force of many meas. and not in the force of
        # chance in single meas.
        old_max_power = np.max(powers)
        if old_max_power > power:
            # TODO: is this a bug? added
            best_phi = phis[np.where(np.array(powers) == old_max_power)[0][0]]
            # print(f'=>now<=. best phi was {best_phi}')
            best_power = max(old_max_power, power)
        """
        best_power = power
        # Scaling factor comes from changing of exposure time
        print(f'best power: {best_power * self.power_scaling_factor}')

        self.cur_best_slm_phase += best_phi * mask_to_play
        self.cur_best_slm_phase = np.mod(self.cur_best_slm_phase, 2 * np.pi)

    def _mask_to_power_ge(self, phase_mask):
        if phase_mask.shape == (self.macro_pixels_x_slm*self.macro_pixels_y_slm,):
            phase_mask = phase_mask.reshape(self.macro_pixels_y_slm, self.macro_pixels_x_slm)

        is_best = False
        if self.micro_iter_num % 10 == 0:
            print(f'micro-iteration:{self.micro_iter_num}')
            is_best = True

        return -self._mask_to_power(phase_mask, is_best)

    def _mask_to_power(self, phase_mask, is_best=False):
        self.micro_iter_num += 1
        self.slm.update_phase_in_active(phase_mask)

        time.sleep(self.sleep_period)

        if self.is_simulation:
            # field = self.simulate_propagation()
            # powers = self._field_to_powers(field)
            powers = self.simulate_propagation_new(fixer_mask=phase_mask)
            interest_size = 3
            N = self.N
            interest_powers = powers[N // 2 - interest_size: N // 2 + interest_size,
                             N // 2 - interest_size: N // 2 + interest_size]
            power = sum(sum(interest_powers))

        else:
            powers = self.camera.get_image()
            power = self.power_at_target(powers)

        # If lock in method - save only when in best. if in ge - once in a while
        if is_best:
            self.result.best_result = powers
            self.result.slm_phase_grid = phase_mask

            if self.macro_iter_num % self.mid_results_diff == 0:
                self.result.mid_results[self.macro_iter_num] = (phase_mask, power * self.power_scaling_factor)

            self.result.powers.append(power * self.power_scaling_factor)
            self._save_result()

            self._fix_exposure(powers)
        return power

    def _fix_exposure(self, powers):
        mx = powers[self.slices_of_interest[0]].max()
        if mx > 240 and not self.is_simulation:
            print('Lowering exposure time!')
            exp_time = self.camera.get_exposure_time()
            if exp_time * (4 / 5) > 45:
                self.camera.set_exposure_time(exp_time * (4 / 5))
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

        return best_phi

    def get_image(self):
        if self.is_simulation:
            field = self.simulate_propagation()
            im = self._field_to_powers(field)
        else:
            im = self.camera.get_image()
        return im

    def power_at_target(self, powers):
        power = 0
        for sl in self.slices_of_interest:
            power += np.sum(powers[sl].flatten())
        return power

    def _save_result(self):
        saveto_path = self.saveto_path or f"{LOGS_DIR}\\{self.timestamp}_{self.run_name}.optimizer"
        self.result.saveto(saveto_path)

    # Simulation functions
    def simulate_propagation(self):
        assert self.is_simulation
        assert isinstance(self.slm, SLMMask)
        assert isinstance(self.diffuser, SLMMask)

        # TODO: do this once in init and deep copy each time instead of recreate
        field = create_field_grid_by_light_source(self.BORDERS, self.FIELD_PIXELS_X, self.FIELD_PIXELS_Y,
                                                  self.light_source)
        field = self.slm.prop_through(field)
        field = self.diffuser.prop_through(field)
        field = self.farfield.prop_through(field)
        return field

    def simulate_propagation_new(self, fixer_mask):
        sigma = 1e-3
        N = self.N
        dx = 12.5e-6
        dy = 12.5e-6
        X = np.arange(-N/2, N/2) * dx
        Y = np.arange(-N/2, N/2) * dy
        Xs, Ys = np.meshgrid(X, Y)
        G = np.exp(-(Xs**2 + Ys**2) / (sigma**2))

        tot_mask = np.zeros((N, N))
        d_lines, d_rows = self.diffuser_mask.shape

        fixer_mask = cv2.resize(fixer_mask, (d_rows, d_lines), interpolation=cv2.INTER_AREA)
        phase_mask = self.diffuser_mask + fixer_mask

        tot_mask[N//2 - d_lines//2 : N//2 + d_lines//2, N//2 - d_rows//2 : N//2 + d_rows//2] = phase_mask

        G_proped_phase = G * np.exp(1j * tot_mask)
        farfield = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(G_proped_phase)))

        farfield_powers = np.abs(farfield) ** 2
        return farfield_powers * 1e-7

    def _field_to_powers(self, field):
        return np.abs(field.Es) ** 2

    # Plotting functions
    @staticmethod
    def show_phase_mask(phase_mask, ax, title='Phase Mask'):
        ax.clear()
        ax.set_title(title)
        ax.imshow(phase_mask, cmap='gray', vmin=0, vmax=2 * np.pi)
        ax.figure.show()
        ax.figure.canvas.flush_events()

    def plot_powers(self, powers):
        fig, ax = plt.subplots()
        ax.set_title("Power over Iterations")
        ax.plot(powers)
        if self.is_simulation:
            # When not simulation - it won't really work...
            ax.axhline(y=self.theoretical_power, linestyle='--')
        fig.show()

    def show_power(self, ax, im, name='power'):
        ax.clear()
        ax.pcolormesh(im, shading='nearest', vmin=0)
        ax.set_title(name)
        ax.figure.show()
        ax.figure.canvas.flush_events()

    def show_input_field(self):
        fig, ax = plt.subplots()
        cam = BasicCamera('input beam', ax)
        start_field = create_field_grid_by_light_source(self.BORDERS, self.FIELD_PIXELS_X, self.FIELD_PIXELS_Y,
                                                        self.light_source)
        cam.show(start_field)


if __name__ == '__main__':
    if False:  # Lab
        exposure_time = 4000
        slm_macro_pixels = 30

        sleep_period = 0.01
        b = 460
        path = "C:\\temp\\cn2=1e-15,L=1e3.e1.clouds"
        diffuser_mask = np.load(path)['diffuser']
        # diffuser_mask = 2*np.pi*np.random.rand(12, 12)

        o = Optimizer(is_simulation=False, slices_of_interest=np.index_exp[a:a+2, b:b+2],
                      diffuser_mask=diffuser_mask,
                      macro_pixels_x_slm=slm_macro_pixels, macro_pixels_y_slm=slm_macro_pixels,
                      sleep_period=sleep_period, exposure_time=exposure_time,
                      run_name='kolmogorov_fixer', saveto_path=r'C:\Users\Owner\Google Drive\People\Ronen\results\temp\part_30X30_cn2=1e-15L=1e3.e1.optimizer')
        g = o.optimize(method=Optimizer.PARTITIONING, iterations=(slm_macro_pixels**2)*5)
        # g = o.optimize(method=Optimizer.GENETIC, iterations=10)
        for i in g:
            # print(o.genetic_res)
            if i == 5:  # Just so there will be lines of code to break from while debugging
                pass
        # print(o.genetic_res)

    else:  # Simulation
        path = r"G:\My Drive\Projects\Shaping Entangled Photons Through Atmosphere\Paper\Data For Figures\figs_4_5_all_data\diffuser.diffuser"
        A = np.load(path)['diffuser']
        # A = np.random.rand(10, 10)*2*np.pi
        o = Optimizer(is_simulation=True, run_name='40X40',
                      diffuser_mask=A,
                      macro_pixels_x_slm=40, macro_pixels_y_slm=40,
                      sleep_period=0.001)
        g = o.optimize(method=Optimizer.PARTITIONING, iterations=(o.macro_pixels_x_slm**2)*5)
        # g = o.optimize(method=Optimizer.GENETIC, iterations=(o.macro_pixels_x_slm**2)*2)
        for i in g:
            pass
    plt.show()
