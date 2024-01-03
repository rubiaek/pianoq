import time
import datetime

from pianoq.lab.power_meter100 import PowerMeterPM100
from pianoq.lab.thorlabs_motor import ThorlabsKcubeDC, ThorlabsKcubeStepper
from pianoq.lab.photon_counter import PhotonCounter

import numpy as np
import matplotlib.pyplot as plt
from pianoq.lab.time_tagger import QPTimeTagger

from pianoq_results.scan_result import ScanResult
from pianoq.misc.mplt import my_mesh
# LOGS_DIR = "C:\\temp"
# LOGS_DIR = r'G:\My Drive\Projects\Klyshko Optimization\Results\temp'
LOGS_DIR = r'G:\My Drive\Projects\Klyshko Optimization\Results\same_speckle\try3'


class PhotonScanner(object):
    best_x = 17.5
    best_y = 16.9

    def __init__(self, integration_time, start_x, start_y, x_pixels, y_pixels, pixel_size_x, pixel_size_y,
                 run_name='_scan', saveto_path=None, coin_window=4e-9, is_timetagger=False, is_double_spot=False):
        self.start_x = start_x or self.best_x - ((x_pixels-1)*pixel_size_x) / 2
        self.start_y = start_y or self.best_y - ((y_pixels-1)*pixel_size_y) / 2

        self.x_pixels = x_pixels
        self.y_pixels = y_pixels
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y

        self.integration_time = integration_time
        self.run_name = run_name
        self.saveto_path = saveto_path
        self.coin_window = coin_window
        self.is_timetagger = is_timetagger
        self.is_double_spot = is_double_spot

        self.X = np.linspace(self.start_x, self.start_x+(self.x_pixels-1)*self.pixel_size_x, self.x_pixels)
        self.Y = np.linspace(self.start_y, self.start_y+(self.y_pixels-1)*self.pixel_size_y, self.y_pixels)
        self.Y = self.Y[::-1]

        self.single1s = np.zeros((self.y_pixels, self.x_pixels))
        self.single2s = np.zeros((self.y_pixels, self.x_pixels))
        self.single3s = np.zeros((self.y_pixels, self.x_pixels))
        self.coincidences = np.zeros((self.y_pixels, self.x_pixels))
        self.coincidences2 = np.zeros((self.y_pixels, self.x_pixels))

        self.result = ScanResult(X=self.X, Y=self.Y, integration_time=self.integration_time)
        self.result.coin_window = self.coin_window
        self.result.is_timetagger = self.is_timetagger
        self.result.is_double_spot = self.is_double_spot
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.start_time = time.time()

    def scan(self, spiral=False, ph=None , x_motor=None, y_motor=None, use_power_meter=False):

        close_ph = False
        if ph is None and not use_power_meter:
            print('getting photon counter...')
            if self.is_timetagger:
                ph = QPTimeTagger(integration_time=self.integration_time, coin_window=self.coin_window, single_channel_delays=(0, 0))
            else:
                ph = PhotonCounter(integration_time=self.integration_time)
            close_ph = True
        elif ph is not None:
            assert isinstance(ph, (PhotonCounter, QPTimeTagger))
            self.integration_time = ph.integration_time

        if x_motor is None:
            print('getting x motor...')
            x_motor = ThorlabsKcubeStepper()
            close_x_motor = True
        else:
            close_x_motor = False

        if y_motor is None:
            print('getting y motor...')
            y_motor = ThorlabsKcubeDC()
            close_y_motor = True
        else:
            close_y_motor = False

        try:
            if not spiral:
                self._linear(x_motor, y_motor, ph, use_power_meter)

        except Exception as e:
            print('Exception occurred')
            print(e)
            import traceback
            traceback.print_exc()

        if close_x_motor:
            x_motor.close()
        if close_y_motor:
            pass # pesky y_motor closing bug
            # y_motor.close()
        if close_ph:
            ph.close()

        return self.single1s, self.single2s, self.coincidences

    def _spiral(self):
        middle_x = self.start_x + (self.pixel_size_x*self.x_pixels / 2)
        middle_y = self.start_y + (self.pixel_size_y*self.y_pixels / 2)

        print('Moving to starting position...')
        # x_motor.move_absolute(self.start_x)
        # y_motor.move_absolute(self.start_y)

        # TODO: we need to follow both the absulote value in mm, and also the relevant discreet index in matrix

    def _linear(self, x_motor, y_motor, ph, use_power_meter=False):
        print('Moving to starting position...')
        x_motor.move_absolute(self.start_x)
        y_motor.move_absolute(self.start_y)

        power_meter = None
        if use_power_meter:
            power_meter = PowerMeterPM100()
            power_meter.set_exposure(0.05)

        print('starting scan')
        count = 0
        self.start_time = time.time()
        for i in range(self.y_pixels):
            for j in range(self.x_pixels):
                x_motor.move_relative(self.pixel_size_x)
                duration_till_now = time.time() - self.start_time

                if use_power_meter:
                    self.single1s[i, j], self.single2s[i, j], self.coincidences[i, j] = 0, 0, power_meter.get_power() * 1e6
                    print(f'dur: {int(duration_till_now)}. pix: {i}, {j}. Power: {self.coincidences[i, j]:.2f}.')
                    self._save_result()
                    continue

                if not self.is_double_spot:
                    self.single1s[i, j], self.single2s[i, j], self.coincidences[i, j] = ph.read_interesting()
                    print(f'dur: {int(duration_till_now)}. pix: {i}, {j}. Singles1: {self.single1s[i, j]:.0f}. '
                          f'Singles2: {self.single2s[i, j]:.0f}. Coincidence: {self.coincidences[i, j]:.0f}.')
                else:
                    self.single1s[i, j], self.single2s[i, j], self.single3s[i, j], \
                        self.coincidences[i, j], self.coincidences2[i, j] = ph.read_double_spot()
                    print(f'dur: {int(duration_till_now)}. pix: {i}, {j}. Singles1: {self.single1s[i, j]:.0f}. '
                          f'Singles2: {self.single2s[i, j]:.0f}. Singles3: {self.single3s[i, j]:.0f}. '
                          f'Coincidence: {self.coincidences[i, j]:.0f}. Coincidence2: {self.coincidences2[i, j]:.0f}.')

                self._save_result()

            y_motor.move_relative(self.pixel_size_y)
            x_motor.move_absolute(self.start_x)

    def _save_result(self):
        self.result.coincidences = self.coincidences
        self.result.coincidences2 = self.coincidences2
        self.result.single1s = self.single1s
        self.result.single2s = self.single2s
        self.result.single3s = self.single3s
        saveto_path = self.saveto_path or f"{LOGS_DIR}\\{self.timestamp}_scan_{self.run_name}.scan"
        self.result.saveto(saveto_path)

    def plot(self, mat):
        fig, ax = plt.subplots()
        my_mesh(self.X, self.Y, mat, ax)
        ax.set_title('Coincidence')

    def plot_coincidence(self, name):
        plt.close(1338)
        fig, ax = plt.subplots()
        my_mesh(self.X, self.Y, self.coincidences, ax)
        # ax.invert_yaxis()
        fig.show()
        fig.savefig(f"C:\\temp\\{time.time()}{name}.png")

    def plot_singles(self):
        fig, axes = plt.subplots(1, 2)
        xx, yy = np.meshgrid(self.X, self.Y)
        im0 = axes[0].pcolormesh(xx, yy, self.single1s, shading='nearest', vmin=0)
        im1 = axes[1].pcolormesh(xx, yy, self.single2s, shading='nearest', vmin=0)
        fig.suptitle('single counts')

        fig.colorbar(im0, ax=axes[0])
        fig.colorbar(im1, ax=axes[1])
        # axes[0].invert_yaxis()
        # axes[1].invert_yaxis()
        fig.show()


def klyshko_scan(name='', integration_time=1.0, use_power_meter=False, D0=0.0, D=0.0):
    mid_x = 13.5  # 13.7  # this is with the linear tilt on SLM.
    mid_y = 9.05  # with d=7 which is middle of single counts

    # d=2 -> mid_x = 8.45
    # d=12 -> mid_x = 9.6
    start_x = 13.3
    # 1e-3 for um, 10 because micrometer 7 as actually 70, and 9.5x for different magnification
    start_y = 8.85 - (np.abs(D-D0) * 1e-3 * 10 * 9.5)

    x_pixels = 14
    y_pixels = 14
    pixel_size_x = 0.025
    pixel_size_y = 0.025

    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dir_path = r'G:\My Drive\Projects\Klyshko Optimization\Results\Off_axis\try6\SPDC_memory'
    path = f'{dir_path}\\{timestamp}_{name}.scan'
    scanner = PhotonScanner(integration_time, start_x, start_y, x_pixels, y_pixels, pixel_size_x, pixel_size_y,
                            run_name=name, is_timetagger=True, coin_window=2e-9, saveto_path=path)

    x_motor = ThorlabsKcubeStepper(26003411)
    print('got x_motor')
    y_motor = ThorlabsKcubeDC(27600573)
    print('got y_motor')
    tt = None
    pm = None
    if not use_power_meter:
        tt = QPTimeTagger(integration_time=integration_time, coin_window=2e-9, single_channel_delays=(0, 1600))
        print('got timetagger')

    single1s, single2s, coincidences = scanner.scan(x_motor=x_motor, y_motor=y_motor, ph=tt, use_power_meter=use_power_meter)
    # x_motor.close()
    # y_motor.close()  # pesky bug?
    if not use_power_meter:
        tt.close()


if __name__ == '__main__':
    best_x = 17.5
    best_y = 16.9
    best_z = 10  # Not very accurate, but seems OK

    D0 = 7
    D = 1

    klyshko_scan(integration_time=4, name=f'optimized_d={D}', use_power_meter=False, D0=D0, D=D)
