import os
import sys
import time
import traceback

import numpy as np
import datetime
import json

from pianoq.lab.photon_counter import PhotonCounter
from pianoq.lab.asi_cam import ASICam
from pianoq.lab.slm import SLMDevice
from pianoq.lab.photon_scan import PhotonScanner
from pianoq.lab.piano_optimization import PianoOptimization
from pianoq.lab.slm_optimize import SLMOptimizer
from pianoq.lab.thorlabs_motor import ThorlabsKcubeStepper, ThorlabsKcubeDC
from pianoq.lab.time_tagger import QPTimeTagger
from pianoq.misc.flushing_file import FlushingPrintingFile

LOGS_DIR = r'G:\My Drive\Projects\Quantum Piano\Results\temp'


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

        self.asi_cam = ASICam(exposure=self.config['cam_exposure'], binning=1, image_bits=16,
                              roi=self.config['cam_roi'], gain=0)
        print('got ASI camera')

        self.photon_counter = QPTimeTagger(integration_time=self.config['optimized_integration_time'],
                                           coin_window=self.config['coin_window'],
                                           single_channel_delays=(0, 1600))
        print('got photon counter')

        self.slm = SLMDevice(0, use_mirror=True)

    def make_dir(self):
        # dirs and paths
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.dir_path = f"{LOGS_DIR}\\{self.timestamp}"
        os.mkdir(self.dir_path)
        print(f"Results will be saved in Here: {self.dir_path}")

    def redirect_stdout(self):
        log_path = f'{self.dir_path}\\{self.timestamp}_log.txt'
        self.log_file = FlushingPrintingFile(log_path, 'w', self.orig_stdout)
        sys.stdout = self.log_file

    def save_config(self, comment=''):
        config_path = f'{self.dir_path}\\{self.timestamp}_config.json'
        self.config['comment'] = comment
        open(config_path, 'w').write(json.dumps(self.config, indent=3))

    def run(self, comment=''):
        try:
            self.make_dir()
            self.save_config(comment)
            self.redirect_stdout()

            ###### diode ######
            input('Press enter when you changed to diode...')
            self.take_asi_pic('diode_no_diffuser')
            input('Press enter when you added the diffuser...')
            self.take_asi_pic('speckles')
            self.slm_optimize()
            self.set_slm_optimized()
            self.take_asi_pic('diode_optimized')

            ###### SPDC ######
            input('Press enter when you changed to SPCMs...')
            self.set_photon_integration_time(self.config['optimized_integration_time'])
            self.scan_coincidence('corr_optimized')

            self.slm.normal()
            self.set_photon_integration_time(self.config['speckle_integration_time'])
            self.scan_coincidence('two_photon_speckle')
            input('Press enter when you removed the diffuser...')
            self.set_photon_integration_time(self.config['focus_integration_time'])
            self.scan_coincidence('corr_no_diffuser')

            print("done!")

        except Exception as e:
            print('Exception!!!')
            print(e)
            traceback.print_exc()

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
        self.slm.update_phase_in_active(self.optimizer.res.best_phase_mask)

    def take_asi_pic(self, title=''):
        time.sleep(0.2)
        timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        image_path = f'{self.dir_path}\\{timestamp}_{title}.fits'
        self.asi_cam.save_image(image_path, comment=title, add_timestamp_to_name=False)
        time.sleep(0.2)

    def slm_optimize(self):
        optimization_result_path = f'{self.dir_path}\\{self.timestamp}_optimized.optimizer2'
        self.optimizer = SLMOptimizer(macro_pixels=self.config['macro_pixels'], sleep_period=0.001, run_name='run_name',
                                      saveto_path=optimization_result_path)
        self.optimizer.optimize(method=SLMOptimizer.PARTITIONING, iterations=self.config['n_iterations'],
                                slm=self.slm, cam=self.asi_cam,
                                roi=self.config['cost_roi'], best_phi_method=self.config['best_phi_method'])

    def scan_coincidence(self, title=''):
        print(f'### Scanning {title} two-photon speckle ###')
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
        if self.log_file:
            self.log_file.close()


if __name__ == "__main__":

    l = 3
    config = {
        # hardware
        'cam_roi': (1400, 780, 400, 500),
        'cam_exposure': 3e-3,

        # optimization
        'n_iterations': 200,
        'cost_roi': np.index_exp[200-l:200+l, 200-l:200+l],
        'best_phi_method': 'silly_max',
        'macro_pixels': 20,

        # scan areas
        # TODO: the real scanning params
        'start_x': 16.2,
        'start_y': 16.025,
        'x_pixels': 30,
        'y_pixels': 30,
        'pix_size': 0.025,
        # 'x_pixels': 15,
        # 'y_pixels': 15,
        # 'pix_size': 0.05,

        # Integration times
        'optimized_integration_time': 2,
        'speckle_integration_time': 2,
        'focus_integration_time': 5,
        'ASI_exposure': 8,

        # Timetagger
        'is_time_tagger': True,
        'coin_window': 1e-9,
    }

    is_test = False
    if is_test:
        config['focus_scan_integration_time'] = 1
        config['speckle_scan_integration_time'] = 1
        config['piano_integration_time'] = 1
        config['x_pixels'] = 3
        config['y_pixels'] = 3
        config['n_iterations'] = 3
        config['is_time_tagger'] = True

    ke = KlyshkoExperiment(config)
    ke.run('first_try')
    ke.close()
