import os
import re
import time
import struct
import socket
import subprocess
import numpy as np

"""
TODO: show zohar: 
    1) the [g1, g1] bug in set_gain 
    2) the 95 instead of 80 amp gain
    3) the [128, 128] weird thing? 
    4) how her usage of 30 and [0:100] ruins the point of max_dac_voltage  
"""

cur_dir = os.path.dirname(os.path.abspath(__file__))
EDAC_LIST_PATH = os.path.join(cur_dir, 'edac40list.exe')


class Edac40(object):
    NUM_OF_PIEZOS = 40
    PORT = 1234
    DISCOVER_PORT = 30303
    AMP_GAIN = 80
    SLEEP_AFTER_SEND = 0.3
    REST_AMP = 0

    # This is from .c code
    _EDAC_PHYS_CHANNELS = np.array([6, 7, 4, 5, 2, 3, 0, 1, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13,
                                   # Analog outputs 1-20 (right connector)
                                   10, 11, 8, 9, 38, 39, 36, 37, 34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 24,
                                   25])  # Analog outputs 21-40 (left connector)

    # Fixes empirically: 0<->1
    EDAC_PHYS_CHANNELS = np.array([6, 7, 4, 5, 2, 3, 1, 0, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13,
                                   # Analog outputs 1-20 (right connector)
                                   10, 11, 8, 9, 38, 39, 36, 37, 34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 24,
                                   25])  # Analog outputs 21-40 (left connector)

    SET_DAC_VALUE_CMDID = 0
    SET_OFFSET_CMDID = 1
    SET_GAIN_CMDID = 2
    SET_GLOBAL_OFFSET_CMDID = 3
    SET_SAVE_TO_NVRAM_CMDID = 4

    DEFAULT_IP = '169.254.124.204'

    def __init__(self, max_piezo_voltage=30, ip=None):
        self.max_piezo_voltage = max_piezo_voltage
        self.max_dac_voltage = self.max_piezo_voltage / self.AMP_GAIN
        assert 0 < self.max_dac_voltage < 12, "DAC max voltage must be between 0V and 12V"

        self.ip = ip or self.find_ip()

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect((self.ip, self.PORT))

        # Dummy values that are set in the set_* functions
        self.gain_all = -1
        self.offset_all = -1
        self.global_offset_all = -1
        self.current_amplitudes = np.ones(self.NUM_OF_PIEZOS) * -1

        self.set_gain()
        self.set_global_offset(0)
        self.set_offset(0x8000)
        self.save_to_nvram()
        self.set_amplitudes(self.REST_AMP * np.ones(self.NUM_OF_PIEZOS))

        self.print_voltage_range()

    def find_ip(self):
        print('searching for ip...')
        process = subprocess.Popen([EDAC_LIST_PATH], stdout=subprocess.PIPE)
        stdout = process.communicate()[0].decode()
        ip = re.findall('.*IP Address: (.*), MAC.*', stdout)[0]
        print(f'found IP! {ip}')
        return ip

    def _send_buff(self, buff):
        self.sock.send(buff)
        time.sleep(self.SLEEP_AFTER_SEND)

    def _reorder_to_phys(self, volts):
        # this reorders the volts, so instead of being by the software numbering, they go to the physical numbering
        # according to self.EDAC_PHYS_CHANNELS
        # TODO: check this again
        return volts[self.EDAC_PHYS_CHANNELS]

    def _percentage_to_inner_val(self, percentages):
        # allowing a bit off if algorithm overthrows
        assert np.all(-0.05 <= percentages) and np.all(percentages <= 1.05), 'percentages must be between 0 and 1!'
        percentages = np.clip(percentages, 0, 1)
        # volts = self.max_dac_voltage * percentages
        amps = (percentages * (2 ** 16 - 1)).astype(int)
        amps = self._reorder_to_phys(amps)
        return amps

    def set_amplitudes(self, amps):
        """
        amps should be between 0 and 1, and amplitudes to DAC will be relative to self.max_dac_voltage
        """
        # Should I add "gradual change" option? and "alpha" option? is this important?

        if type(amps) in [int, float]:
            amps = np.ones(self.NUM_OF_PIEZOS) * amps
        assert len(amps) == self.NUM_OF_PIEZOS

        self.current_amplitudes = amps
        amps = self._percentage_to_inner_val(amps)

        # 255s for all channels. Theretically you can decide to update only some of the channels,
        # but i didn't implement it. see edac40_prepare_packet in edac40.c
        arr = np.ones(5, dtype=int) * 255
        channel_mask = struct.pack('<5B', *arr)
        command = struct.pack('<B', self.SET_DAC_VALUE_CMDID)

        values = struct.pack(f'<{self.NUM_OF_PIEZOS}H', *amps)
        buff = channel_mask + command + values
        self._send_buff(buff)

    def set_gain(self, gain=None):
        # max_dac_voltage in units between 1 - 65535
        # This calc is not really clear to me, but it gives reasonable range at the end...
        if not gain:
            gain = (self.max_dac_voltage / 12) * (2 ** 16 - 1)
        self.set_globally(self.SET_GAIN_CMDID, int(gain))
        self.gain_all = gain

    def set_offset(self, offset):
        self.set_globally(self.SET_OFFSET_CMDID, offset)
        self.offset_all = offset

    def set_global_offset(self, offset=0):
        self.set_globally(self.SET_GLOBAL_OFFSET_CMDID, offset)
        self.global_offset_all = offset

    def save_to_nvram(self):
        self.set_globally(self.SET_SAVE_TO_NVRAM_CMDID, None)

    def set_globally(self, command_code, value):
        """ set all channels to given command code parameter to given value.
         Command codes are:
         0 - set DAC value
         1 - set offset
         2 - set gain
         3 - set global offset
         4 - save current setting to NVRAM
         """
        # 255s for all channels
        arr = np.ones(5, dtype=int) * 255
        channel_mask = struct.pack('<5B', *arr)
        command = struct.pack('<B', command_code)
        if command_code in [self.SET_SAVE_TO_NVRAM_CMDID, self.SET_GLOBAL_OFFSET_CMDID]:
            values = b'\x00\x00'
        else:
            arr = np.ones(self.NUM_OF_PIEZOS, dtype=int) * int(value)
            values = struct.pack(f'<{self.NUM_OF_PIEZOS}H', *arr)
        buff = channel_mask + command + values

        self._send_buff(buff)

    def _code_to_V(self, code):
        dacCode = code * (self.gain_all + 1) / 65535.0 + self.offset_all - 32768.0
        if dacCode > 65535:
            dacCode = 65535
        if dacCode < 0:
            dacCode = 0

        v = 12.0 * (dacCode - 4 * self.global_offset_all) / 65535.0
        return v

    def print_voltage_range(self):
        minV = self._code_to_V(0)
        maxV = self._code_to_V(2 ** 16 - 1)
        print(f"Voltage range between {minV} and {maxV} Volts")

    def close(self):
        self.set_amplitudes(self.REST_AMP * np.ones(self.NUM_OF_PIEZOS))
        self.sock.close()
