import os
import sys
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
from pianoq.lab.time_tagger import QPTimeTagger
from pianoq.simulations.calc_fiber_modes import get_modes_FG010LDA
from pianoq.lab.Edac40 import Edac40
from pianoq.misc.misc import retry_if_exception
from pianoq.misc.flushing_file import FlushingPrintingFile

LOGS_DIR = r'G:\My Drive\Projects\Quantum Piano\Results\temp'

class OptimizationExperiment(object):
    def __init__(self, config=None):
        self.config = config
        self.log_file = None
        self.orig_stdout = sys.stdout

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
        if self.config['is_time_tagger']:
            if not self.config['is_double_spot']:
                self.photon_counter = QPTimeTagger(integration_time=self.config['speckle_scan_integration_time'],
                                                   coin_window=self.config['coin_window'],
                                                   single_channel_delays=[400, 0])
            else:
                self.photon_counter = QPTimeTagger(integration_time=self.config['speckle_scan_integration_time'],
                                                   coin_window=self.config['coin_window'],
                                                   single_channels=(1, 2, 4),
                                                   coin_channels=((1, 2), (1, 4)),
                                                   single_channel_delays=[0, -600, 0])
        else:
            self.photon_counter = PhotonCounter(integration_time=self.config['speckle_scan_integration_time'])
        print('got photon counter')

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

    def run(self, comment=''):
        try:
            self.optimization_no = 1
            self.make_dir()
            self.save_config(comment)
            self.redirect_stdout()

            self.randomize_dac()
            self.take_asi_pic('singles_before')

            if self.config['should_scan_speckles']:
                self.set_photon_integration_time(self.config['speckle_scan_integration_time'])
                self.scan_coincidence('speckles')

            self.set_motors()
            self.set_photon_integration_time(self.config['piano_integration_time'])

            if self.config['is_double_spot']:
                self.random_configurations()

            self.piano_optimization()
            self.take_asi_pic('singles_after_optimization')

            self.set_photon_integration_time(self.config['focus_scan_integration_time'])
            self.scan_coincidence('optimized')
            self.take_asi_pic('singles_after_scan')
        except Exception as e:
            print('Exception!!!')
            print(e)

    def only_speckles(self, num, amps=None):
        self.randomize_dac(amps=amps)
        self.take_asi_pic(f'{num}_singles_before')

        self.set_photon_integration_time(self.config['speckle_scan_integration_time'])
        self.scan_coincidence(f'{num}_speckles')

    def only_optimization(self, comment=''):
        self.make_dir()
        self.save_config(comment)

        self.randomize_dac()
        # self.take_asi_pic(f'singles_before')

        self.set_motors()
        # self.set_photon_integration_time(self.config['piano_integration_time'])
        self.piano_optimization()
        # self.take_asi_pic('singles_after_optimization')


    def set_photon_integration_time(self, time_sec):
        if hasattr(self, 'photon_counter'):
            if self.photon_counter.integration_time == time_sec:
                return
            else:
                print("NOOOOOOO I'm recreating the time tagger but without the correct parameters!!!")
                self.photon_counter.close()
                time.sleep(2)

        if self.config['is_time_tagger']:
            # self.photon_counter = QPTimeTagger(integration_time=time_sec,
            #                                    coin_window=self.config['coin_window'])
            # TODO CHANGE BACK!!
            self.photon_counter = QPTimeTagger(integration_time=time_sec,
                                               coin_window=self.config['coin_window'],
                                               single_channels=(1, 2, 4),
                                               coin_channels=((1, 2), (1, 4)),
                                               single_channel_delays=[0, -600, 0])

        else:
            self.photon_counter = PhotonCounter(integration_time=time_sec)

    def randomize_dac(self, amps=None):
        if amps is None:
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
        time.sleep(1)
        image_path = f'{self.dir_path}\\{self.timestamp}_{title}.fits'
        self.asi_cam.save_image(image_path, comment=title)
        self.switcher.forwards()
        time.sleep(1)

    def piano_optimization(self):
        print('### Running piano optimization ###')
        po, is_interupt = self._single_optimization()
        while True:
            if is_interupt:
                break

            self.optimization_no += 1
            if np.abs(po.res.costs).max() > self.config['least_optimization_res']:
                break
            print('!## optimization not good enough - trying again ##!')
            print(f'{np.abs(po.res.costs).max()} < {self.config["least_optimization_res"]}!')

            po, is_interupt = self._single_optimization()

    def random_configurations(self):
        print('random configurations for double spot')
        assert self.config['is_double_spot']

        single1s = []
        single2s = []
        single3s = []
        coin1s = []
        coin2s = []
        for i in range(self.config['n_random_configurations']):
            amps = np.random.rand(40)
            self.dac.set_amplitudes(amps)
            s1, s2, s3, c1, c2 = self.photon_counter.read_double_spot()
            single1s.append(s1)
            single2s.append(s2)
            single3s.append(s3)
            coin1s.append(c1)
            coin2s.append(c2)

        saveto_path = f'{self.dir_path}\\{self.timestamp}_N={self.config["n_random_configurations"]}.randz'
        f = open(saveto_path, 'wb')
        np.savez(f,
                 single1s=single1s,
                 single2s=single2s,
                 single3s=single3s,
                 coin1s=coin1s,
                 coin2s=coin2s,)
        f.close()

    def _single_optimization(self):
        saveto_path = f'{self.dir_path}\\{self.timestamp}_{self.optimization_no}.pqoptimizer'
        cam_type = 'timetagger' if self.config['is_time_tagger'] else 'SPCM'

        po = PianoOptimization(saveto_path=saveto_path,
                               initial_exposure_time=self.config['piano_integration_time'],
                               cost_function=lambda x: -x,
                               cam_type=cam_type,
                               dac=self.dac,
                               cam=self.photon_counter,
                               good_piezo_indexes=self.config['good_piezo_indexes'],
                               is_double_spot=self.config['is_double_spot'])

        try:
            po.optimize_my_pso(n_pop=self.config['n_pop'],
                               n_iterations=self.config['n_iterations'],
                               stop_after_n_const_iters=self.config['stop_after_n_const_iters'],
                               reduce_at_iterations=self.config['reduce_at_iterations'],
                               success_cost=self.config['success_cost'])
        except KeyboardInterrupt:
            amps = np.ones(40) * self.dac.REST_AMP
            amps[po.res.good_piezo_indexes] = po.res.amplitudes[-1]
            po.dac.set_amplitudes(amps)
            return po, True

        amps = np.ones(40) * self.dac.REST_AMP
        amps[po.res.good_piezo_indexes] = po.res.amplitudes[-1]
        po.dac.set_amplitudes(amps)
        return po, False

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
                                saveto_path=saveto_path,
                                coin_window=self.photon_counter.coin_window,
                                is_double_spot=self.config['is_double_spot'])
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
        if self.log_file:
            self.log_file.close()


if __name__ == "__main__":

    good_piezzos = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                    10, 11, 12, 13, 14, 15,     17, 18,
                        21, 22, 23, 24, 25, 26, 27, 28, 29,
                    30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    config = {
        # uninteresting params
        'n_pop': 20,
        'n_iterations': 120,
        'stop_after_n_const_iters': 10,
        'reduce_at_iterations': (1,),
        'good_piezo_indexes': good_piezzos[:],  # TODO: choose only a subset
        'start_x': 16.2,
        'start_y': 16.075,
        'ASI_ROI': (1400, 780, 400, 500),
        'DAC_max_piezo_voltage': 120,
        'DAC_SLEEP_AFTER_SEND': 0.3,

        # optimization
        # 'optimized_xy': (16.45, 16.4),
        'optimized_xy': (16.575, 16.5),
        'least_optimization_res': 34,
        'success_cost': 35,
        'is_double_spot': True,
        'n_random_configurations' : 120,

        # Resolution
        'x_pixels': 30,
        'y_pixels': 30,
        'pix_size': 0.025,
        # 'x_pixels': 15,
        # 'y_pixels': 15,
        # 'pix_size': 0.05,

        # Integration times
        'should_scan_speckles': False,
        'speckle_scan_integration_time': 2,
        'piano_integration_time': 2,
        'focus_scan_integration_time': 2,
        'ASI_exposure': 8,

        # Timetagger
        'is_time_tagger': True,
        'coin_window': 1e-9,
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
        config['least_optimization_res'] = 30
        config['success_cost'] = 30
        config['is_time_tagger'] = True

    oe = OptimizationExperiment(config)

    # Just run full experiment
    # oe.make_dir()
    # oe.save_config('filter=3nm_heralded_timetagger_two_spots_100um_fiber_bundle')
    # oe.only_speckles(1)
    oe.run('filter=3nm_not_heralded_timetagger_two_spots')

    # oe.config['start_y'] = 16.00
    # oe.config['x_pixels'] = 30
    # oe.config['y_pixels'] = 30
    # oe.config['pix_size'] = 0.025
    # oe.run('filter=3nm_heralded_timetagger_two_spots')

    # Only speckles
    # oe.make_dir()
    # oe.save_config('filter=3nm_heralded_timetagger_two_spots_100um_fiber_bundle')
    # oe.only_speckles(1)
    # oe.only_speckles(2)
    # oe['speckle_scan_integration_time'] = 2
    # oe.only_speckles(3)
    # oe.only_speckles(4)

    # only optimization
    # oe.only_optimization('filter=3nm_heralded_timetagger_two_spots')

    oe.close()
