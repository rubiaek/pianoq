from datetime import datetime
from ctypes import cdll, c_long, c_ulong, c_uint32, byref, create_string_buffer, c_bool, c_char_p, c_int, c_int16, c_double, sizeof, c_voidp
import time
import sys; sys.path.append(r'C:\Program Files (x86)\IVI Foundation\VISA\Winnt\TLPM\Examples\Python')
from TLPM import TLPM


class PowerMeterPM100(object):
    def __init__(self):
        # taken from here: "C:\Program Files (x86)\IVI Foundation\VISA\Winnt\TLPM\Examples\Python\Power Meter Write Read Raw.py"
        self.tlpm = TLPM()

        deviceCount = c_uint32()
        self.tlpm.findRsrc(byref(deviceCount))
        print("devices found: " + str(deviceCount.value))
        resourceName = create_string_buffer(1024)

        for i in range(0, deviceCount.value):
            self.tlpm.getRsrcName(c_int(i), resourceName)
            print(c_char_p(resourceName.raw).value)
            break

        self.tlpm.open(resourceName, c_bool(True), c_bool(True))

        message = create_string_buffer(1024)
        self.tlpm.getCalibrationMsg(message)
        print(c_char_p(message.raw).value)

    def get_power(self):
        power = c_double()
        self.tlpm.measPower(byref(power))
        return power.value

    def close(self):
        self.tlpm.close()
