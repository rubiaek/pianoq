import time
import datetime

from pianoq.lab.thorlabs_motor import ThorlabsKcubeDC, ThorlabsKcubeStepper
from pianoq.lab.photon_counter import PhotonCounter

import numpy as np
import matplotlib.pyplot as plt

from pianoq_results.scan_result import ScanResult
from pianoq.misc.mplt import my_mesh
LOGS_DIR = "C:\\temp"


class PhotonScanner(object):
    best_x = 17.5
    best_y = 16.9

    def __init__(self, integration_time, start_x, start_y, x_pixels, y_pixels, pixel_size_x, pixel_size_y,
                 run_name='_scan', saveto_path=None):
        self.start_x = start_x or self.best_x - ((x_pixels-1)*pixel_size_x) / 2
        self.start_y = start_y or self.best_y - ((y_pixels-1)*pixel_size_y) / 2

        self.x_pixels = x_pixels
        self.y_pixels = y_pixels
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y

        self.integration_time = integration_time
        self.run_name = run_name
        self.saveto_path = saveto_path

        self.X = np.linspace(self.start_x, self.start_x+(self.x_pixels-1)*self.pixel_size_x, self.x_pixels)
        self.Y = np.linspace(self.start_y, self.start_y+(self.y_pixels-1)*self.pixel_size_y, self.y_pixels)
        self.Y = self.Y[::-1]

        self.single1s = np.zeros((self.y_pixels, self.x_pixels))
        self.single2s = np.zeros((self.y_pixels, self.x_pixels))
        self.coincidences = np.zeros((self.y_pixels, self.x_pixels))

        self.result = ScanResult(X=self.X, Y=self.Y, integration_time=self.integration_time)
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    def scan(self, spiral=False):
        print('getting photon counter...')
        ph = PhotonCounter(integration_time=self.integration_time)

        print('getting motors...')
        x_motor = ThorlabsKcubeStepper()
        y_motor = ThorlabsKcubeDC()

        try:
            if not spiral:
                self._linear(x_motor, y_motor, ph)

        except Exception as e:
            print('Exception occurred')
            print(e)
            import traceback
            traceback.print_exc()

        x_motor.close()
        y_motor.close()
        ph.close()

        return self.single1s, self.single2s, self.coincidences

    def _spiral(self):
        middle_x = self.start_x + (self.pixel_size_x*self.x_pixels / 2)
        middle_y = self.start_y + (self.pixel_size_y*self.y_pixels / 2)

        print('Moving to starting position...')
        x_motor.move_absolute(self.start_x)
        y_motor.move_absolute(self.start_y)

        # TODO: we need to follow both the absulote value in mm, and also the relevant discreet index in matrix


    def _linear(self, x_motor, y_motor, ph):
        print('Moving to starting position...')
        x_motor.move_absolute(self.start_x)
        y_motor.move_absolute(self.start_y)

        print('starting scan')
        count = 0
        self.start_time = time.time()
        for i in range(self.y_pixels):
            for j in range(self.x_pixels):
                x_motor.move_relative(self.pixel_size_x)
                self.single1s[i, j], self.single2s[i, j], self.coincidences[i, j] = ph.read_interesting()
                self._save_result()

            y_motor.move_relative(self.pixel_size_y)
            x_motor.move_absolute(self.start_x)

    def _save_result(self):
        duration_till_now = time.time() - self.start_time
        # print(f'dur: {int(duration_till_now)}. pix: {i}, {j}. Singles1: {self.single1s[i, j]:2f}. '
        #   f'Singles2: {self.single2s[i, j]:2f}. Coincidence: {self.coincidences[i, j]:2f}.')

        self.result.coincidences = self.coincidences
        self.result.single1s = self.single1s
        self.result.single2s = self.single2s
        saveto_path = self.saveto_path or f"{LOGS_DIR}\\{self.timestamp}_scan_{self.run_name}.scan"
        print(f'saving result, path = {saveto_path}')
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


def whole_scan(name='whole_area', integration_time=5):
    start_x = 15.2
    start_y = 15.3
    x_pixels = 15
    y_pixels = 15
    pixel_size_x = 0.1
    pixel_size_y = 0.1


    scanner = PhotonScanner(integration_time, start_x, start_y, x_pixels, y_pixels, pixel_size_x, pixel_size_y,
                            run_name=name)
    single1s, single2s, coincidences = scanner.scan()


def middle_scan(name='middle_area', integration_time=20):
    pix_size = 0.05
    if pix_size == 0.025:
        start_x = 15.85
        start_y = 15.85
        x_pixels = 20
        y_pixels = 20

    elif pix_size == 0.05:
        start_x = 15.75
        start_y = 15.8
        x_pixels = 10
        y_pixels = 10

    scanner = PhotonScanner(integration_time, start_x, start_y, x_pixels, y_pixels, pix_size, pix_size,
                            run_name=name)
    single1s, single2s, coincidences = scanner.scan()
    # scanner.plot_coincidence(name)


def small_scan(name='small_area', integration_time=20):
    start_x = 16.85
    start_y = 16.45
    x_pixels = 5
    y_pixels = 5
    pixel_size_x = 0.050
    pixel_size_y = 0.050

    scanner = PhotonScanner(integration_time, start_x, start_y, x_pixels, y_pixels, pixel_size_x, pixel_size_y,
                            run_name=name)
    single1s, single2s, coincidences = scanner.scan()
    # scanner.plot_coincidence(name)


def scan_1D(name='1D', integration_time=1):
    start_x = 15
    start_y = 16.6
    x_pixels = 30
    y_pixels = 1
    pixel_size_x = 0.1
    pixel_size_y = 0.1

    scanner = PhotonScanner(integration_time, start_x, start_y, x_pixels, y_pixels, pixel_size_x, pixel_size_y,
                            run_name=name)
    single1s, single2s, coincidences = scanner.scan()
    # scanner.plot_coincidence(name)


if __name__ == '__main__':
    best_x = 17.5
    best_y = 16.9
    best_z = 10  # Not very accurate, but seems OK

    middle_scan(integration_time=4)
    # small_scan(integration_time=1)
    # whole_scan(integration_time=3)
    # scan_1D(integration_time=0.5)
