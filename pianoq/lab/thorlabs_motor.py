import numpy as np
import clr
import time

try:
    import py_thorlabs_ctrl.kinesis
    _KINESIS_PATH = r'C:\Program Files\Thorlabs\Kinesis'
    py_thorlabs_ctrl.kinesis.init(_KINESIS_PATH)
    from py_thorlabs_ctrl.kinesis.motor import KCubeDCServo, KCubeStepper
    from System import Decimal
except ImportError:
    print('cant use py_thorlabs_ctrl.kinesis')


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


class ThorlabsKcubeDC(KCubeDCServo):
    SERIAL_1 = 27253522

    def __init__(self, serial_number=None):
        serial_number = serial_number or self.SERIAL_1
        super().__init__(serial_number=serial_number)

        success = False
        for i in range(7):
            try:
                self.create()
                self.enable()
                if self.device.IsEnabled:
                    success = True
                    break
            except Exception:
                print(f'not able to connect to motor, trying again...')
                time.sleep(1)
                if i == 6:
                    raise

    def move_relative(self, mm, timeout=30000):
        """ timeout in ms. send 0 for non-blocking. It is important to send float, and not some other np type..."""
        device = self.get_device()
        if not device.IsEnabled:
            self.create()
            self.enable()

        device.SetMoveRelativeDistance(Decimal(float(mm)))
        device.MoveRelative(timeout)

    def move_absolute(self, mm, timeout=30000):
        """ timeout in ms. send 0 for non-blocking. It is important to send float, and not some other np type..."""
        device = self.get_device()
        if not device.IsEnabled:
            self.create()
            self.enable()

        device.MoveTo(Decimal(mm), timeout)

    def close(self):
        self.disable()
        self.disconnect()


class ThorlabsKcubeStepper(KCubeStepper):
    SERIAL_1 = 26001271

    def __init__(self, serial_number=None):
        serial_number = serial_number or self.SERIAL_1
        super().__init__(serial_number=serial_number)

        success = False
        for i in range(7):
            try:
                self.create()
                self.enable()
                success = True
                break
            except Exception:
                print(f'not able to connect to motor, trying again...')
                time.sleep(1)
                if i == 6:
                   raise


    def move_relative(self, mm, timeout=30000):
        """ timeout in ms. send 0 for non-blocking. It is important to send float, and not some other np type..."""
        device = self.get_device()
        if not device.IsEnabled:
            self.create()
            self.enable()

        device.SetMoveRelativeDistance(Decimal(float(mm)))
        device.MoveRelative(timeout)

    def move_absolute(self, mm, timeout=30000):
        """ timeout in ms. send 0 for non-blocking. It is important to send float, and not some other np type..."""
        device = self.get_device()
        if not device.IsEnabled:
            self.create()
            self.enable()

        device.MoveTo(Decimal(mm), timeout)

    def close(self):
        self.disable()
        self.disconnect()


class ManualMotor(object):
    MY_QWP_ZERO = 5

    def __init__(self, zero_angle=None):
        self.zero_angle = zero_angle or self.MY_QWP_ZERO

    def move_absolute(self, degrees):
        qq = input(f"Make sure the fast axis of QWP is on {degrees + self.zero_angle} degrees, and then press enter")
        print(f"Got it! {qq}")

    def close(self):
        pass

