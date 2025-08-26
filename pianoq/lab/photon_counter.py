import time
import serial

import numpy as np


class PhotonCounter(object):
    PKT_LEN = 41
    PKT_PER_SEC = 10
    SEC_PER_PKT = 0.1
    coin_window = 4e-9

    W = np.zeros((8, 41))
    for j in range(0, 8):
        W[j, j * 5 + 1: (j + 1) * 5] = [1, 128, 16384, 2097152]

    def __init__(self, serial_port='COM8', integration_time=1):
        self.serial_port = serial_port
        assert integration_time >= 0.2, 'PhotonCounter doesn\'t support integration_time<0.2'
        self.integration_time = integration_time

        # buffer size should place for enough packets in single integration time
        self.buffer_size = int((self.PKT_PER_SEC * self.integration_time + 1) * self.PKT_LEN)
        self.ser = self.open_serial()

    def open_serial(self):
        success = False
        for i in range(7):
            try:
                ser = serial.Serial(baudrate=19200, timeout=2)
                ser.set_buffer_size(self.buffer_size)
                ser.setPort(self.serial_port)
                ser.open()
                success = True
                break
            except Exception:
                raise
                print(f'not able to open port {self.serial_port}, trying again...')
                time.sleep(1)
                if i == 3:
                    raise
        return ser

    def read_interesting(self):
        datas, stds, actual_exp_time = self.read()
        datas = datas/actual_exp_time
        single1, single2, coincidence = datas[0], datas[1], datas[4]

        return single1, single2, coincidence

    def read_double_spot(self):
        datas, stds, actual_exp_time = self.read()
        datas = datas/actual_exp_time
        single1, single2, single3, coincidence, coincidence2 = datas[0], datas[1], datas[2], datas[4], datas[6] # TODO: is it really 6?

        return single1, single2, single3, coincidence, coincidence2

    def read(self):
        """returns:
        Format of raw: \xff - 40 bytes of pkt - \xff - 40 bytes of pkt etc.
        Format of packet: unsigned int (4 bytes) \x00 unsigned int (4 bytes) \x00 etc.

        data: the data from each channel
        stds: standard deviations of each channel
        actual_exp_time: actual number of data packets read * 0.1 seconds.
        when calculating rate (counts per second), divide D/actual_exp_time.
        they are output seperately to allow correct averaging in calling function.

        To calculate the count per second the calling function should do data/actual_exp_time
        """

        self.ser.flush()
        self.ser.read_all()
        start = time.time()
        raw = b''
        while len(raw) < self.buffer_size:
            raw += self.ser.read_all()
            time.sleep(0.8 * self.SEC_PER_PKT)
            if time.time() - start > self.integration_time * 2:
                print('Error, not enough info coming in from serial')
                break

        return self._parse(raw)

    @classmethod
    def _parse(cls, raw, option=2):
        # if less than 1 packet was captured - return zeros
        if len(raw) < 41:
            data = np.zeros((8, 1))
            stds = data
            actual_exp_time = 1
            return data, stds, actual_exp_time

        pkts = raw.split(b'\xff')
        pkts = [pkt for pkt in pkts if len(pkt) == 40]  # Remove bad packets
        pkts = pkts[1:]  # Remove first packet - apparently it tends to be bad (TODO: check this)
        actual_exp_time = len(pkts) * 0.1  # each pkt is 0.1 seconds

        if option == 1:
            all_datas = np.zeros((len(pkts), 8))
            for i, pkt in enumerate(pkts):
                data = cls._pkt_to_data(pkt)  # Return 8-tuple
                all_datas[i, :] = data
                """
                if np.max([pkt[4], pkt[9], pkt[14], pkt[19], pkt[24], pkt[29], pkt[34], pkt[39]]) > 0:
                    print(f'{i}: wierd')
                """

            final_data = np.sum(all_datas, 0)
            stds = np.std(all_datas, 0)
            return final_data, stds, actual_exp_time

        elif option == 2:
            final_data, stds = cls._all_pkts_to_datas(pkts)
            return final_data, stds, actual_exp_time

    @classmethod
    def _pkt_to_data(cls, pkt):
        # Assuming 4 bytes are relevant, and it's byte1*1 + byte2*16384 + byte3*16384 etc.
        # This is more readable, but probably less efficient than matrix_way below
        arr = np.frombuffer(pkt, dtype=np.byte)
        channels = arr.reshape(8, 5)
        A, _ = np.meshgrid([1, 128, 16384, 2097152, 0], np.arange(8))
        channels = channels * A
        data = np.sum(channels, 1)
        return data

    @classmethod
    def _all_pkts_to_datas(cls, pkts):
        # Copied from matlab - don't understand exactly, but works.
        buff = b'\xff' + b'\xff'.join(pkts)
        arr = np.frombuffer(buff, dtype=np.uint8)
        d_array = np.reshape(arr, (41, len(pkts)), order='F')

        D_0 = cls.W @ d_array  # Matrix multiplication
        D = np.sum(D_0, 1)
        s = np.std(D_0, 1, ddof=1)  # This is the way it was in matlab, and the default behavior of matlab

        return D, s

    def close(self):
        self.ser.close()


# This for debugging
# data = open('c:\\temp\\ronen_raw_stream_photon_counter.bin', 'rb').read()
# data = open('c:\\temp\\ronen_raw_stream2.bin', 'rb').read()
# data = open('c:\\temp\\ronen_raw_stream3.bin', 'rb').read()
# print(PhotonCounter._parse(data, 2))
# print(PhotonCounter._parse(data, 1))
