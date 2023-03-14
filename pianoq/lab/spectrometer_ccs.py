import os
import time
import matplotlib.pyplot as plt
from ctypes import c_double, byref, cdll, c_void_p, c_int

import numpy as np


class Spectrometer(object):
    # https://github.com/Thorlabs/Light_Analysis_Examples/blob/main/Python/Thorlabs%20CCS%20Spectrometers/CCS%20using%20ctypes%20-%20Python%203.py
    # This didn't wok straight away: https://github.com/tz15/ThorLabsCCS200
    def __init__(self, integration_time=50e-3):
        self.integration_time = integration_time
        self.lib = cdll.LoadLibrary(r"C:\Program Files\IVI Foundation\VISA\Win64\Bin\TLCCS_64.dll")

        self.ccs_handle=c_int(0)
        #documentation: C:\Program Files\IVI Foundation\VISA\Win64\TLCCS\Manual
        #Start Scan- Resource name will need to be adjusted
        #windows device manager -> NI-VISA USB Device -> Spectrometer -> Properties -> Details -> Device Instance ID
        self.lib.tlccs_init(b"USB0::0x1313::0x8089::M00426981::RAW", 1, 1, byref(self.ccs_handle))

        self.set_integration_time(self.integration_time)

    def set_integration_time(self, integration_time):
        #set integration time in  seconds, ranging from 1e-5 to 6e1
        integration_time = c_double(integration_time)
        self.lib.tlccs_setIntegrationTime(self.ccs_handle, integration_time)


    def get_data(self):
        #start scan
        self.lib.tlccs_startScan(self.ccs_handle)
        wavelengths=(c_double*3648)()

        self.lib.tlccs_getWavelengthData(self.ccs_handle, 0, byref(wavelengths), c_void_p(None), c_void_p(None))

        #retrieve data
        data_array=(c_double*3648)()
        self.lib.tlccs_getScanData(self.ccs_handle, byref(data_array))

        return np.array(wavelengths), np.array(data_array)

    def show(self):
        wavelengths, data_array = self.get_data()
        #plot data
        plt.plot(wavelengths, data_array)
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Intensity [a.u.]")
        plt.grid(True)
        plt.show()

    def close():
        self.lib.tlccs_close (self.ccs_handle)
