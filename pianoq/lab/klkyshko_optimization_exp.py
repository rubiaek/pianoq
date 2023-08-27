import os
import sys
import time
import traceback

import numpy as np
import datetime
import json

from pianoq.lab.photon_counter import PhotonCounter
from pianoq.lab.asi_cam import ASICam
from pianoq.lab.power_meter100 import PowerMeterPM100
from pianoq.lab.slm import SLMDevice
from pianoq.lab.photon_scan import PhotonScanner
from pianoq.lab.slm_optimize import SLMOptimizer
from pianoq.lab.thorlabs_motor import ThorlabsKcubeStepper, ThorlabsKcubeDC
from pianoq.lab.time_tagger import QPTimeTagger
from pianoq.misc.flushing_file import FlushingPrintingFile

LOGS_DIR = r'E:\Google Drive\Projects\Klyshko Optimization\Results\temp'


class KlyshkoExperiment(object):
    def __init__(self, config):
        self.config = config
        self.log_file = None
        self.orig_stdout = sys.stdout

        self.get_hardware()
        self.optimizer = None

    def get_hardware(self):
        # Hardware
        self.x_motor = ThorlabsKcubeDC(27600573)
        print('got x_motor')
        self.y_motor = ThorlabsKcubeStepper(26003411)
        print('got y_motor')

        self.asi_cam = ASICam(exposure=self.config['cam_exposure'], binning=1, roi=self.config['cam_roi'], gain=0)
        print('got ASI camera')

        self.photon_counter = QPTimeTagger(integration_time=self.config['optimized_integration_time'],
                                           coin_window=self.config['coin_window'],
                                           single_channel_delays=(0, 1600))
        print('got photon counter')

        self.slm = SLMDevice(0, use_mirror=True)
        self.slm.set_pinhole(radius=self.config['slm_pinhole_radius'], center=self.config['slm_pinhole_center'],
                             pinhole_type=self.config['slm_pinhole_type'])

        self.power_meter = PowerMeterPM100()

    def make_dir(self):
        # dirs and paths
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.dir_path = f"{LOGS_DIR}\\{self.timestamp}_klyshko"
        os.mkdir(self.dir_path)
        print(f"Results will be saved in Here: {self.dir_path}")

        more_info_file_path = f'{self.dir_path}\\{self.timestamp}_log_more.txt'
        self.more_info_file = open(more_info_file_path, 'w')
        self.more_info_file.write('Hi!\n')

        klyshk_file_path = f'{self.dir_path}\\{self.timestamp}_clickme.klyshk'
        self.klyshk_file = open(klyshk_file_path, 'w')
        self.klyshk_file.write('Hi!\n')
        self.klyshk_file.close()

    def redirect_stdout(self):
        log_path = f'{self.dir_path}\\{self.timestamp}_log.txt'
        self.log_file = FlushingPrintingFile(log_path, 'w', self.orig_stdout)
        sys.stdout = self.log_file

    def save_config(self, comment=''):
        config_path = f'{self.dir_path}\\{self.timestamp}_config.json'
        self.config['comment'] = comment
        open(config_path, 'w').write(json.dumps(self.config, indent=3))

    def _input(self, msg):
        print(msg)
        input()

    def log_power(self, msg):
        power = self.power_meter.get_power()
        to_write = f'{msg}: {power}'
        print(to_write)
        self.more_info_file.write(to_write)

    def log_coin(self, msg):
        # have some statistic, it happens only 3 times during the whole experiment...
        for i in range(3):
            s1, s2, c = self.photon_counter.read_interesting()
            real_c = c - 2*s1*s2*self.photon_counter.coin_window
            to_write = f'{msg}: singles1: {s1}, singles2: {s2}, coincidences: {c}, real_c: {real_c}'
            print(to_write)
            self.more_info_file.write(to_write)

    def run_virtual_speckles(self, comment=''):
        self.make_dir()
        self.save_config(comment)
        self.redirect_stdout()

        self.slm.normal()
        self.take_dark_pic()
        self.take_asi_pic('normal')

        phase_kolmogorov1 = self.slm.set_kolmogorov(cn2=1e-16, L=1e3)
        self.save_phase_screen(phase_kolmogorov1, 'kolmogorov1')
        self.take_asi_pic('kolmogorov1')

        phase_kolmogorov2 = self.slm.set_kolmogorov(cn2=1e-16, L=1e3)
        self.save_phase_screen(phase_kolmogorov2, 'kolmogorov2')
        self.take_asi_pic('kolmogorov2')

        phase_10_macros_1 = self.slm.set_diffuser(10)
        self.save_phase_screen(phase_10_macros_1, '10_macros_1')
        self.take_asi_pic('10_macros_1')

        phase_10_macros_2 = self.slm.set_diffuser(10)
        self.save_phase_screen(phase_10_macros_2, '10_macros_2')
        self.take_asi_pic('10_macros_2')

        ###### SPCMs ######
        self._input('Press enter when changed to SPCMs')
        self.slm.normal()
        self.scan_coincidence('normal')

        self.slm.update_phase_in_active(phase_kolmogorov1)
        self.scan_coincidence('kolmogorov1')

        self.slm.update_phase_in_active(phase_kolmogorov2)
        self.scan_coincidence('kolmogorov2')

        final_mask = np.zeros(self.slm.correction.shape)
        final_mask[self.slm.active_mask_slice] = phase_10_macros_1
        self.slm.update_phase(phase_10_macros_1)
        self.scan_coincidence('10_macros_1')

        final_mask = np.zeros(self.slm.correction.shape)
        final_mask[self.slm.active_mask_slice] = phase_10_macros_2
        self.slm.update_phase(phase_10_macros_2)
        self.scan_coincidence('10_macros_2')

        print('Done!')

    def save_phase_screen(self, A, title):
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        path = f'{self.dir_path}\\{timestamp}_{title}.npz'

        f = open(path, 'wb')
        np.savez(f, diffuser=A)
        f.close()

    def run(self, comment=''):
        try:
            self.make_dir()
            self.save_config(comment)
            self.redirect_stdout()

            self.set_motors_to_optimization()

            ###### diode ######
            self._input('Press enter when you changed to diode+power_meter...')
            self.log_power(f'Power no diffuser')
            self.take_dark_pic()

            self._input('Press enter when you changed to diode+cam...')
            self.take_asi_pic('diode_no_diffuser')

            self._input('Press enter when you added the diffuser...')
            self.take_asi_pic('diode_speckles')

            self._input('Press enter when you changed to diode+power_meter...')
            self.log_power(f'Power yes diffuser')
            self.slm_optimize()
            self.set_slm_optimized()
            self.log_power(f'Power optimized')

            self._input('Press enter when you changed to diode+_cam...')
            self.take_asi_pic('diode_optimized')

            ###### SPDC ######
            self._input('####### Press enter when you changed to SPCMs... #######')
            self.log_coin('coin optimized')
            self.slm.normal()
            self.log_coin('coin yes diffuser')

            flag = False
            if flag:
                self._input('Press enter when you removed diffuser...')
                self.log_coin('coin no diffuser')
                self._input('Press enter when you added diffuser...')

            self.set_slm_optimized()
            self.set_photon_integration_time(self.config['optimized_integration_time'])
            self.scan_coincidence('corr_optimized')

            self.slm.normal()
            self.set_photon_integration_time(self.config['speckle_integration_time'])
            self.scan_coincidence('two_photon_speckle')
            self._input('Press enter when you removed the diffuser...')

            self.set_photon_integration_time(self.config['focus_integration_time'])
            self.scan_coincidence('corr_no_diffuser')

            print("done!")

        except Exception as e:
            print('Exception!!!')
            print(e)
            traceback.print_exc()

    def take_dark_pic(self):
        im = self.asi_cam.get_image()
        darks = np.zeros((10, im.shape[0], im.shape[1]))
        for i in range(10):
            time.sleep(0.1)
            darks[i] = self.asi_cam.get_image()

        dark_im = np.mean(darks, axis=0)
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        image_path = f'{self.dir_path}\\{timestamp}_dark.fits'
        self.asi_cam.save_image(image_path, im=dark_im, comment='dark', add_timestamp_to_name=False)
        time.sleep(0.1)

    def set_motors_to_optimization(self):
        self.x_motor.move_absolute(self.config['optimization_x_loc'])
        self.y_motor.move_absolute(self.config['optimization_y_loc'])

    def set_photon_integration_time(self, time_sec):
        if self.photon_counter.integration_time == time_sec:
            return
        self.photon_counter.close()
        time.sleep(2)

        if self.config['is_time_tagger']:
            self.photon_counter = QPTimeTagger(integration_time=time_sec,
                                               coin_window=self.config['coin_window'],
                                               single_channel_delays=(0, 1600))
        else:
            self.photon_counter = PhotonCounter(integration_time=time_sec)

    def set_slm_optimized(self):
        self.optimizer.update_slm(self.optimizer.res.best_phase_mask)

    def take_asi_pic(self, title=''):
        time.sleep(0.2)
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        image_path = f'{self.dir_path}\\{timestamp}_{title}.fits'
        self.asi_cam.save_image(image_path, comment=title, add_timestamp_to_name=False)
        time.sleep(0.2)

    def slm_optimize(self):
        optimization_result_path = f'{self.dir_path}\\{self.timestamp}_optimized.optimizer2'
        self.optimizer = SLMOptimizer(macro_pixels=self.config['macro_pixels'],
                                      sleep_period=self.config['sleep_period'], run_name='run_name',
                                      saveto_path=optimization_result_path)

        y = self.config['cost_roi_mid'][0]
        x = self.config['cost_roi_mid'][1]
        l = 4
        cost_roi = np.index_exp[y-l: y+l, x-l: x+l]
        self.optimizer.optimize(method=SLMOptimizer.CONTINUOUS_HEX, iterations=self.config['n_iterations'],
                                slm=self.slm, power_meter=self.power_meter,
                                roi=cost_roi, best_phi_method=self.config['best_phi_method'],
                                cell_size=self.config['cell_size'])

    def scan_coincidence(self, title=''):
        print(f'### Scanning {title} ###')
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        saveto_path = f"{self.dir_path}\\{timestamp}_{title}.scan"
        scanner = PhotonScanner(self.photon_counter.integration_time,
                                self.config['start_x'],
                                self.config['start_y'],
                                self.config['x_pixels'],
                                self.config['y_pixels'],
                                self.config['pix_size'],
                                self.config['pix_size'],
                                saveto_path=saveto_path,
                                coin_window=self.photon_counter.coin_window)
        single1s, single2s, coincidences = scanner.scan(ph=self.photon_counter,
                                                        x_motor=self.x_motor,
                                                        y_motor=self.y_motor)

    def close(self):
        self.x_motor.close()
        self.y_motor.close()
        self.photon_counter.close()
        self.asi_cam.close()
        self.slm.close()
        self.more_info_file.close()
        if self.log_file:
            self.log_file.close()

        # print('Done closing')


if __name__ == "__main__":

    config = {
        # hardware
        # 'cam_roi': (2846, 1808, 400, 400),
        'cam_roi': (2910, 1780, 800, 800),
        'cam_exposure': 10e-3,
        'slm_pinhole_radius': 150,
        'slm_pinhole_center': (530, 500),
        'slm_pinhole_type': 'mirror',
        'cell_size': 15,

        # optimization
        'n_iterations': 400,
        'cost_roi_mid': (400, 400),
        'best_phi_method': 'silly_max',
        'macro_pixels': 25,
        'optimization_x_loc': 8.6,
        'optimization_y_loc': 13.6,

        # scan areas
        # mid_x = 8.6
        # mid_y = 13.6
        'start_x': 8.1,
        'start_y': 13.1,
        # 'x_pixels': 30,
        # 'y_pixels': 30,
        # 'pix_size': 0.025,
        'x_pixels': 20,
        'y_pixels': 20,
        'pix_size': 0.05,

        # Integration times
        'optimized_integration_time': 2,
        'speckle_integration_time': 1,
        'focus_integration_time': 1,
        'sleep_period': 0.1,  # after SLM update

        # Timetagger
        'is_time_tagger': True,
        'coin_window': 2e-9,
    }

    ke = KlyshkoExperiment(config)
    # ke.run('two_diffusers_0.25_0.5_power_meter_continuous_hex')
    ke.run_virtual_speckles('first try')
    ke.close()

"""
    ke.slm.close()
    ke.make_dir()
    ke.save_config('comment')

    ###### SPDC ######
    input('Press enter when you changed to SPCMs...')
    ke.set_photon_integration_time(ke.config['optimized_integration_time'])
    ke.scan_coincidence('corr_optimized')

    input('now put phase on SLM back to normal')
    ke.set_photon_integration_time(ke.config['speckle_integration_time'])
    ke.scan_coincidence('two_photon_speckle')

    input('Press enter when you removed the diffuser...')
    ke.set_photon_integration_time(ke.config['focus_integration_time'])
    ke.scan_coincidence('corr_no_diffuser')
"""