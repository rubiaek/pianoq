import time
import numpy as np

try:
    from zaber_motion import Library, Units
    from zaber_motion.ascii import Connection

    Library.enable_device_db_store()
except ImportError:
    print('can\t use zaber motor')
    raise

SERIAL_PORT = "COM4"


class _ZaberMotor:
    def __init__(self, ax, connection, backlash=0, wait_after_move=0.):
        self.connection = connection
        self.ax = ax
        self.backlash = backlash
        self.wait_after_move = wait_after_move

    def move_relative(self, mms, blocking=True):
        self.ax.move_relative(mms, Units.LENGTH_MILLIMETRES, wait_until_idle=blocking)
        time.sleep(self.wait_after_move)

    def move_absolute(self, mms, blocking=True):
        if self.backlash != 0:
            self.ax.move_absolute(mms - self.backlash, Units.LENGTH_MILLIMETRES, wait_until_idle=blocking)
            time.sleep(self.wait_after_move)
        self.ax.move_absolute(mms, Units.LENGTH_MILLIMETRES, wait_until_idle=blocking)
        time.sleep(self.wait_after_move)

    def get_position(self):
        return self.ax.get_position(Units.LENGTH_MILLIMETRES)

    def home(self):
        self.ax.home()

    def close(self):
        """ This will close the connection also with the other motors, which is almost always fine, just beware... """
        self.connection.close()


class ZaberMotors(object):
    MAX_ABS_LOCATION = 300000
    MM_TO_ABS = 20997  # Moving 20997 in absolute moves 1 mm

    def __init__(self, serial_port=SERIAL_PORT, backlash=0, wait_after_move=0.1):
        self.serial_port = serial_port
        self.backlash = backlash
        self.wait_after_move = wait_after_move
        self.con = Connection.open_serial_port(self.serial_port)
        self.device_list = self.con.detect_devices()
        self.ax_list = [dev.get_axis(1) for dev in self.device_list]
        self.motors = [_ZaberMotor(ax, self.con, backlash=backlash, wait_after_move=wait_after_move) for ax in self.ax_list]

    def close(self):
        self.con.close()
