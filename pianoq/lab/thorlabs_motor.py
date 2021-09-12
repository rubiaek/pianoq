import numpy as np

try:
    import py_thorlabs_ctrl.kinesis
    _KINESIS_PATH = r'C:\Program Files\Thorlabs\Kinesis'
    py_thorlabs_ctrl.kinesis.init(_KINESIS_PATH)
    from py_thorlabs_ctrl.kinesis.motor import KCubeDCServo
    from py_thorlabs_ctrl.kinesis.motor import KCubeStepper
except ImportError:
    print('cant use py_thorlabs_ctrl.kinesis')


SERIAL_1 = 27253522

# Has to do with the screwing angle of the waveplate to the inside of the motor
MY_HWP_ZERO = 3.5505

def get_motor(serial=SERIAL_1):
    # TODO: make a class that inherits KCubeDCServo so you can have the .close() like you like, and always cast to float
    motor = KCubeDCServo(serial)
    motor.create()
    motor.enable()
    # To finish use motor.disable() and then motor.disconnect
    # when using "move_absolute" or "move_relative" always use "float"
    # movement units are typically in mm, but in our case they are in degrees (0-360)
    return motor

class ThorLabsMotor(object):
    """
    Uses this library to communicate with the motors:
    https://github.com/rwalle/py_thorlabs_ctrl
    Note: I (Alon) changed a little bit the library itself

    All units here are in mm
    """

    def __init__(self):
        py_thorlabs_ctrl.kinesis.init(self._KINESIS_PATH)
        from py_thorlabs_ctrl.kinesis.motor import KCubeDCServo
        from py_thorlabs_ctrl.kinesis.motor import KCubeStepper
        self._KCubeDCServo = KCubeDCServo

        self.serial = 27253522
        self._motor = self._create_motor(26001271, KCubeStepper)

        self._moving_timeout = 20 * 1000  # In milliseconds

    def _create_motor(self, serial_number, motor_class):
        kcube = motor_class(serial_number)
        kcube.create()
        kcube.enable()
        return kcube

    def move_relative(self, motor, mms):
        """move in mm units"""
        motor.move_relative(float(mms), self._moving_timeout)

    def move_absolute(self, motor, location):
        """Location in mms"""
        # Note: location may be float or numpy.float64, but if it is numpy.float64, the motor doesn't move to the
        # right place. For example, moving to 12.5 will result in moving to 12.0. This is why we have to convert
        # the type here
        motor.move_absolute(float(location), self._moving_timeout)

    def move_x_relative(self, mms):
        self.move_relative(self.x_motor, mms)

    def move_y_relative(self, mms):
        self.move_relative(self.y_motor, mms)

    def move_x_absolute(self, location):
        self.move_absolute(self.x_motor, location)

    def move_y_absolute(self, location):
        self.move_absolute(self.y_motor, location)

    def move_z_absolute(self, location):
        self.move_absolute(self.z_motor, location)

    def close(self):
        for motor in [self.x_motor, self.y_motor, self.z_motor]:
            motor.disconnect()
