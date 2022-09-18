import numpy as np

try:
    import py_thorlabs_ctrl.kinesis
    _KINESIS_PATH = r'C:\Program Files\Thorlabs\Kinesis'
    py_thorlabs_ctrl.kinesis.init(_KINESIS_PATH)
    from py_thorlabs_ctrl.kinesis.motor import KCubeDCServo
    from System import Decimal
except ImportError:
    print('cant use py_thorlabs_ctrl.kinesis')


try:
    import thorlabs_apt as apt
    # If python crashes - open the kinesis program and close it.
except ImportError:
    print('cant use thorlabs apt')


class ThorlabsRotatingServoMotor(KCubeDCServo):
    """
    Uses this library to communicate with the motors:
        https://github.com/rwalle/py_thorlabs_ctrl
    Note: Alon changed a little bit the library itself to support another type of KCube or something,
    and also tried to add timeout

    """

    SERIAL_1 = 27253522

    # Has to do with the screwing angle of the waveplate to the inside of the motor
    MY_HWP_ZERO = 2.15

    def __init__(self, serial_number=None, zero_angle=None):
        self.zero_angle = zero_angle or self.MY_HWP_ZERO
        serial_number = serial_number or self.SERIAL_1
        super().__init__(serial_number=serial_number)
        self.create()
        self.enable()

    def move_relative(self, degrees, timeout=10000):
        """
        :param degrees: typically in mm, but in our case they are in degrees (0-360)
        :param timeout: in ms. send 0 for non-blocking
        It is important to send float, and not some other np type...
        """
        device = self.get_device()
        device.SetMoveRelativeDistance(Decimal(float(degrees)))
        device.MoveRelative(timeout)

    def move_absolute(self, degrees, timeout=20000):
        """
        :param degrees: typically in mm, but in our case they are in degrees (0-360)
        :param timeout: in ms. send 0 for non-blocking
        It is important to send float, and not some other np type...
        """

        device = self.get_device()
        angle = float(degrees + self.zero_angle) % 360  # can't be more that 360.
        device.MoveTo(Decimal(angle), timeout)

    def close(self):
        self.disable()
        self.disconnect()


class ThorLabsMotorsXY(object):
    """All units here are in mm"""

    def __init__(self, xy=True):
        apt.core._cleanup()
        apt.core._lib = apt.core._load_library()
        self.serials = apt.list_available_devices()

        self.xy = xy
        if xy:
            self.x_motor = apt.Motor(26001271)
            self.y_motor = apt.Motor(27253522)
            # self.z_motor = apt.Motor(27501989)
        else:
            self.motors = [apt.Motor(ser[1]) for ser in self.serials]

    @staticmethod
    def move_relative(motor, mms):
        """move in mm units"""
        motor.move_by(mms, blocking=True)

    @staticmethod
    def move_absolute(motor, location):
        """Location in mms"""
        motor.move_to(location, blocking=True)

    def move_x_relative(self, mms):
        if not self.xy:
            raise Exception('this will work only if self.xy!')
        self.move_relative(self.x_motor, mms)

    def move_y_relative(self, mms):
        if not self.xy:
            raise Exception('this will work only if self.xy!')
        self.move_relative(self.y_motor, mms)

    def move_x_absolute(self, location):
        if not self.xy:
            raise Exception('this will work only if self.xy!')
        self.move_absolute(self.x_motor, location)

    def move_y_absolute(self, location):
        if not self.xy:
            raise Exception('this will work only if self.xy!')
        self.move_absolute(self.y_motor, location)

    @staticmethod
    def close():
        apt.core._cleanup()
        apt.core._lib = apt.core._load_library()


class ManualMotor(object):
    MY_QWP_ZERO = 5

    def __init__(self, zero_angle=None):
        self.zero_angle = zero_angle or self.MY_QWP_ZERO

    def move_absolute(self, degrees):
        qq = input(f"Make sure the fast axis of QWP is on {degrees + self.zero_angle} degrees, and then press enter")
        print(f"Got it! {qq}")

    def close(self):
        pass
