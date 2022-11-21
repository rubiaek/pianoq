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

        self.get_hardware()
        # Technical
        self.optimization_no = 1

    def get_hardware(self):
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
        self.photon_counter = PhotonCounter(integration_time=self.config['piano_integration_time'])
        print('got photon counter')


    def make_dir(self):
        # dirs and paths
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.dir_path = f"{LOGS_DIR}\\{self.timestamp}"
        os.mkdir(self.dir_path)
        print(f"Results will be saved in Here: {self.dir_path}")


    def run(self, comment=''):
        self.optimization_no = 1
        self.make_dir()
        self.save_config(comment)

        self.randomize_dac()
        self.take_asi_pic('singles_before')

        if self.config['should_scan_speckles']:
            self.set_photon_integration_time(self.config['speckle_scan_integration_time'])
            self.scan_coincidence('speckles')

        self.set_motors()
        self.set_photon_integration_time(self.config['piano_integration_time'])
        self.piano_optimization()
        self.take_asi_pic('singles_after_optimization')

        self.set_photon_integration_time(self.config['focus_scan_integration_time'])
        self.scan_coincidence('optimized')
        self.take_asi_pic('singles_after_scan')

    def set_photon_integration_time(self, time_sec):
        if hasattr(self, 'photon_counter'):
            if self.photon_counter.integration_time == time_sec:
                return
            else:
                self.photon_counter.close()
                time.sleep(1)

        self.photon_counter = PhotonCounter(integration_time=time_sec)

    def randomize_dac(self):
        amps = np.random.rand(40)
        self.dac.set_amplitudes(amps)
        time.sleep(2)

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
            po = self._single_optimization()

    def _single_optimization(self):
        saveto_path = f'{self.dir_path}\\{self.timestamp}_{self.optimization_no}.pqoptimizer'
        po = PianoOptimization(saveto_path=saveto_path,
                               initial_exposure_time=self.config['piano_integration_time'],
                               cost_function=lambda x: -x,
                               cam_type=self.config['cam_type'],  # SPCM or timetagger
                               dac=self.dac,
                               cam=self.photon_counter,
                               good_piezo_indexes=self.config['good_piezo_indexes'])

        po.optimize_my_pso(n_pop=self.config['n_pop'],
                           n_iterations=self.config['n_iterations'],
                           stop_after_n_const_iters=self.config['stop_after_n_const_iters'],
                           reduce_at_iterations=self.config['reduce_at_iterations'],
                           success_cost=self.config['success_cost'])

        amps = np.ones(40) * self.dac.REST_AMP
        amps[po.res.good_piezo_indexes] = po.res.amplitudes[-1]
        po.dac.set_amplitudes(amps)
        return po

    def scan_coincidence(self, title=''):
        print(f'### Scanning {title} two-photon speckle ###')
        # TODO: scan spiral
        saveto_path=f"{self.dir_path}\\{self.timestamp}_{title}.scan"
        scanner = PhotonScanner(self.photon_counter.integration_time,
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
        # TODO: there is a pesky bug with the KcubeDC motor, which I wasn't quite able to solve.
        # TODO: we might at some point move to pylablib, but there I don't completely understand the scale and coudn't make it actually work...
        # self.y_motor.close()
        self.photon_counter.close()
        self.asi_cam.close()


if __name__ == "__main__":


    good_piezzos = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                             10, 11, 12, 13, 14, 15,     17, 18,
                                 21, 22, 23, 24, 25, 26, 27, 28, 29,
                             30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    config = {
        # general params
        'optimized_xy': (16.4, 16.1),
        'should_scan_speckles' : True,

        # piano_optimization params
        'n_pop': 25,
        'n_iterations': 120,
        'stop_after_n_const_iters': 10,
        'reduce_at_iterations': (2,),
        'cam_type': 'SPCM',
        'good_piezo_indexes': good_piezzos[:],  # TODO: choose only a subset
        'least_optimization_res': 600,
        'piano_integration_time': 1,
        'success_cost' : 850,

        # scan_optimized params
        'start_x' : 16.2,
        'start_y': 15.9,
        'x_pixels': 20,
        'y_pixels': 20,
        'pix_size': 0.025,
        # TODO: timetagger option
        # TODO: maybe after stuck so search again for a 90% percent of record and stop when you get there

        # hardware params
        'ASI_exposure': 2,
        'ASI_ROI': (2900, 1800, 600, 600),
        'DAC_max_piezo_voltage': 120,
        'DAC_SLEEP_AFTER_SEND' : 0.3,
        'speckle_scan_integration_time': 7,
        'focus_scan_integration_time': 4,
    }

    is_test = False
    if is_test:
        config['should_scan_speckles'] = True
        config['focus_scan_integration_time'] = 1
        config['speckle_scan_integration_time'] = 1.5
        config['piano_integration_time'] = 1
        config['x_pixels'] = 3
        config['y_pixels'] = 3
        config['n_pop'] = 10
        config['n_iterations'] = 5
        config['least_optimization_res'] = 150
        config['success_cost'] = 170


    oe = OptimizationExperiment(config)
    oe.run('With piano stop at 700')
    oe.close()
