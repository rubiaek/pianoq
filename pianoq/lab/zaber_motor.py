import time
import numpy as np
from opticalsimulator.simulations.ronen.misc_utils import spiral_motor

SERIAL_PORT = "COM4"


class ZaberMotors(object):
    MAX_ABS_LOCATION = 300000
    MM_TO_ABS = 20997  # Moving 20997 in absolute moves 1 mm

    def __init__(self, serial_port=SERIAL_PORT, xy=True):

        try:
            from zaber_motion import Library, Units
            from zaber_motion.ascii import Connection
            Library.enable_device_db_store()
        except ImportError:
            print('can\t use zaber motor')
            raise

        self.con = Connection.open_serial_port(serial_port)
        self.device_list = self.con.detect_devices()
        self.ax_list = [dev.get_axis(1) for dev in self.device_list]
        self.xy = xy
        if xy:
            self.x_device, self.y_device = self.device_list
            self.x_ax, self.y_ax = self.ax_list

    def move_relative(self, ax, mms):
        # move in mm units
        ax.move_relative(mms, Units.LENGTH_MILLIMETRES)

    def move_x(self, mms):
        if not self.xy:
            raise Exception('this will work only if self.xy ! ')
        self.move_relative(self.x_ax, mms)

    def move_y(self, mms):
        if not self.xy:
            raise Exception('this will work only if self.xy ! ')
        self.move_relative(self.y_ax, mms)

    def home(self):
        for ax in self.ax_list:
            ax.home()

    def get_position(self):
        return [ax.get_position() for ax in self.ax_list]


    def close(self):
        self.con.close()
