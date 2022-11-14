import os
import time
import numpy as np
import datetime

from pianoq.lab.photon_scan import PhotonScanner
from pianoq.lab.piano_optimization import PianoOptimization
from pianoq.lab.thorlabs_motor import ThorlabsKcubeStepper, ThorlabsKcubeDC
from pianoq.simulations.calc_fiber_modes import get_modes_FG010LDA
from pianoq.lab.Edac40 import Edac40
from pianoq.misc.misc import retry_if_exception

LOGS_DIR = 'C:\\temp'

class OptimizationExperiment(object):
    def __init__(self, config=None):
        self.config = config

        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.dir_path = f"{LOGS_DIR}\\{self.timestamp}"
        os.mkdir(self.dir_path)
        print(f"Results will be saved in Here: {self.dir_path}")
        self.po = None
        self.optimization_no = 1

    def run(self):
        print('waiting a bit for motors...')
        time.sleep(10)
        # self.set_motors()
        print('Letting motors rest...')
        time.sleep(10)
        self.piano_optimization()
        print('Letting serial rest...')
        time.sleep(10)
        self.scan_optimized()

    @retry_if_exception(max_retries=10)
    def set_motors(self):
        x_motor = ThorlabsKcubeStepper()
        y_motor = ThorlabsKcubeDC()

        print(f"moving motors to {self.config['optimized_xy']}")

        x_motor.move_absolute(self.config['optimized_xy'][0])
        y_motor.move_absolute(self.config['optimized_xy'][1])

        x_motor.close()
        y_motor.close()


    def piano_optimization(self):
        self._single_optimization()
        while True:
            self.optimization_no += 1

            # self.po will change in each iteration to currnet / last optimization try
            if np.abs(self.po.res.costs).max() > 450:  # TODO: move the "450" figure to config
                break

            self._single_optimization()

    @retry_if_exception(max_retries=5)
    def _single_optimization(self):
        self.po = PianoOptimization(saveto_path=f'{self.dir_path}\\{self.timestamp}_{self.optimization_no}.pqoptimizer',
                               initial_exposure_time=self.config['initial_exposure_time'],
                               cost_function=lambda x: -x,
                               cam_type=self.config['cam_type'])
        self.po.optimize_my_pso(n_pop=self.config['n_pop'],
                           n_iterations=self.config['n_iterations'],
                           stop_after_n_const_iters=self.config['stop_after_n_const_iters'],
                           reduce_at_iterations=self.config['reduce_at_iterations'])
        self.po.dac.set_amplitudes(self.po.res.amplitudes[-1])
        self.po.cam.close()

    @retry_if_exception(max_retries=5)
    def scan_optimized(self):
        # TODO: scan spiral
        scanner = PhotonScanner(self.config['scan_integration_time'],
                                self.config['start_x'],
                                self.config['start_y'],
                                self.config['x_pixels'],
                                self.config['y_pixels'],
                                self.config['pix_size'],
                                self.config['pix_size'],
                                saveto_path=f"{self.dir_path}\\{self.timestamp}.scan")
        single1s, single2s, coincidences = scanner.scan()

        self.po.close()


if __name__ == "__main__":

    config = {
        # set_motors params
        'optimized_xy': (16, 16),

        # piano_optimization params
        'initial_exposure_time': 4,
        'n_pop': 25,
        'n_iterations': 30,
        'stop_after_n_const_iters': 6,
        'reduce_at_iterations': (3,),
        'cam_type': 'SPCM',

        # scan_optimized_params
        'scan_integration_time' : 4,
        'start_x' : 15.75,
        'start_y': 15.8,
        'x_pixels': 10,
        'y_pixels': 10,
        'pix_size': 0.05
        # TODO: timetagger option
    }
    oe = OptimizationExperiment(config)
    oe.run()
