import os
import time
import numpy as np
import datetime
import json

from pianoq.lab.elliptec_stage import ElliptecSwitcher
from pianoq.lab.photon_counter import PhotonCounter
from pianoq.lab.asi_cam import ASICam
from pianoq.lab.photon_scan import PhotonScanner
from pianoq.lab.piano_optimization import PianoOptimization
from pianoq.lab.thorlabs_motor import ThorlabsKcubeStepper, ThorlabsKcubeDC
from pianoq.simulations.calc_fiber_modes import get_modes_FG010LDA
from pianoq.lab.Edac40 import Edac40
from pianoq.misc.misc import retry_if_exception

LOGS_DIR = r'G:\My Drive\Projects\Quantum Piano\Results\temp'

class OptimizationExperiment(object):
    def __init__(self, config=None):
        self.config = config

        # dirs and paths
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.dir_path = f"{LOGS_DIR}\\{self.timestamp}"
        os.mkdir(self.dir_path)
        print(f"Results will be saved in Here: {self.dir_path}")

        # Hardware
        self.switcher = ElliptecSwitcher()
        print('got elliptec switcher')
        self.x_motor = ThorlabsKcubeStepper()
        print('got x_motor')
        self.y_motor = ThorlabsKcubeDC()
        print('got y_motor')
        self.dac = Edac40(max_piezo_voltage=self.config['DAC_max_piezo_voltage'])
        print('got DAC')
        self.dac.SLEEP_AFTER_SEND = self.config['DAC_SLEEP_AFTER_SEND']
        self.asi_cam = ASICam(exposure=self.config['ASI_exposure'], binning=2, image_bits=16, roi=(None, None, None, None))
        print('got ASI camera')
        self.asi_cam.set_gain(0)
        # self.asi_cam.set_roi(*self.config['ASI_ROI']) # the elliptec stage moves the correct place to look at
        self.photon_counter = PhotonCounter(integration_time=self.config['ph_integration_time'])
        print('got photon counter')

        # Technical
        self.optimization_no = 1

    def run(self, comment=''):
        self.save_config(comment)
        self.take_asi_pic('singles_before')
        self.scan_coincidence('speckles')
        self.set_motors()
        self.piano_optimization()
        self.take_asi_pic('singles_after_optimization')
        self.scan_coincidence('optimized')
        self.take_asi_pic('singles_after_scan')

    def save_config(self, comment=''):
        config_path = f'{self.dir_path}\\{self.timestamp}_config.json'
        self.config['comment'] = comment
        open(config_path, 'w').write(json.dumps(self.config, indent=3))

    def set_motors(self):
        print(f'### Moving motors to optimization position: {self.config["optimized_xy"]} ###')
        self.x_motor.move_absolute(self.config['optimized_xy'][0])
        self.y_motor.move_absolute(self.config['optimized_xy'][1])

    def take_asi_pic(self, title=''):
        self.switcher.backwards()
        time.sleep(0.5)
        image_path = f'{self.dir_path}\\{self.timestamp}_{title}.fits'
        self.asi_cam.save_image(image_path, comment=title)
        self.switcher.forwards()
        time.sleep(0.5)

    def piano_optimization(self):
        print('### Running piano optimization ###')
        po = self._single_optimization()
        while True:
            self.optimization_no += 1
            if np.abs(po.res.costs).max() > self.config['least_optimization_res']:
                break
            print('!## optimization not good enough - trying again ##!')
            print(f'{np.abs(po.res.costs).max()} < {self.config["least_optimization_res"]}!')
            self._single_optimization()

    def _single_optimization(self):
        saveto_path = f'{self.dir_path}\\{self.timestamp}_{self.optimization_no}.pqoptimizer'
        po = PianoOptimization(saveto_path=saveto_path,
                               initial_exposure_time=self.config['ph_integration_time'],
                               cost_function=lambda x: -x,
                               cam_type=self.config['cam_type'],  # SPCM or timetagger
                               dac=self.dac,
                               cam=self.photon_counter)

        po.optimize_my_pso(n_pop=self.config['n_pop'],
                           n_iterations=self.config['n_iterations'],
                           stop_after_n_const_iters=self.config['stop_after_n_const_iters'],
                           reduce_at_iterations=self.config['reduce_at_iterations'])
        po.dac.set_amplitudes(po.res.amplitudes[-1])
        return po

    def scan_coincidence(self, title=''):
        print(f'### Scanning {title} two-photon speckle ###')
        # TODO: scan spiral
        saveto_path=f"{self.dir_path}\\{self.timestamp}_{title}.scan"
        scanner = PhotonScanner(self.config['ph_integration_time'],
                                self.config['start_x'],
                                self.config['start_y'],
                                self.config['x_pixels'],
                                self.config['y_pixels'],
                                self.config['pix_size'],
                                self.config['pix_size'],
                                saveto_path=saveto_path)
        single1s, single2s, coincidences = scanner.scan(ph=self.photon_counter,
                                                        x_motor=self.x_motor,
                                                        y_motor=self.y_motor)

    def close(self):
        self.dac.close()
        self.x_motor.close()
        self.y_motor.close()
        self.photon_counter.close()
        # self.asi_cam.close()


if __name__ == "__main__":

    config = {
        # set_motors params
        'optimized_xy': (16.4, 16.2),

        # piano_optimization params
        'n_pop': 15,
        'n_iterations': 25,
        'stop_after_n_const_iters': 6,
        'reduce_at_iterations': (2,),
        'cam_type': 'SPCM',
        'least_optimization_res': 450,

        # scan_optimized params
        'start_x' : 16.2,
        'start_y': 15.9,
        'x_pixels': 20,
        'y_pixels': 20,
        'pix_size': 0.025,
        # TODO: timetagger option

        # hardware params
        'ASI_exposure': 2,
        'ASI_ROI': (2900, 1800, 600, 600),
        'DAC_max_piezo_voltage': 120,
        'DAC_SLEEP_AFTER_SEND' : 0.3,
        'ph_integration_time' : 3, # TODO: different integration times for different parts
    }

    is_test = True
    if is_test:
        config['ph_integration_time'] = 1
        config['x_pixels'] = 3
        config['y_pixels'] = 3
        config['n_pop'] = 3
        config['n_iterations'] = 3
        config['least_optimization_res'] = 150


    oe = OptimizationExperiment(config)
    oe.run()
    oe.close()
