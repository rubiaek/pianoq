import datetime
import traceback

import numpy as np
from pianoq.lab.thorlabs_motor import ThorlabsKcubeDC, ThorlabsKcubeStepper
from pianoq.lab.zaber_motor import ZaberMotors
from pianoq.lab.time_tagger import QPTimeTagger

LOGS_DIR  = "C:\\temp"


class DiscreetScanResult:
    def __init__(self):
        self.path = None
        self.timestamp = None
        self.locs_signal = None
        self.locs_idler = None
        self.integration_time = None
        self.coin_window = None
        self.single1s = None
        self.single2s = None
        self.coincidences = None

    @property
    def real_coins(self):
        return self.coincidences - self.accidentals

    @property
    def accidentals(self):
        return 2*self.single1s*self.single2s*self.coin_window

    def saveto(self, path):
        try:
            f = open(path, 'wb')
            np.savez(f,
                     coincidences=self.coincidences,
                     single1s=self.single1s,
                     single2s=self.single2s,
                     integration_time=self.integration_time,
                     coin_window=self.coin_window,
                     locs_signal=self.locs_signal,
                     locs_idler=self.locs_idler,
                     timestamp=self.timestamp)
            f.close()
        except Exception as e:
            print("ERROR!!")
            print(e)
            traceback.print_exc()

    def loadfrom(self, path=None):
        if path is None:
            path = self.path
        if path is None:
            raise Exception("No path")
        path = path.strip('"')
        path = path.strip("'")
        self.path = path

        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        self.coincidences = data['coincidences']
        self.single1s = data['single1s']
        self.single2s = data['single2s']
        self.integration_time = data.get('integration_time', None)
        self.coin_window = data.get('coin_window', 4e-9)
        self.locs_signal = data['locs_signal']
        self.locs_idler = data['locs_idler']
        self.timestamp = data['timestamp']
        f.close()

    def reload(self):
        self.loadfrom(self.path)



class DiscretePhotonScanner:
    def __init__(self, locs_signal, locs_idler, integration_time=1, coin_window=1e-9, remote_tagger=True,
                 saveto_path='', run_name=''):
        self.res = DiscreetScanResult()
        self.res.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.res.path = saveto_path or f"{LOGS_DIR}\\{self.res.timestamp}_{run_name}.dscan"

        self.res.locs_signal = locs_signal
        self.res.locs_idler = locs_idler
        self.res.integration_time = integration_time
        self.res.coin_window = coin_window

        self.zaber_ms = None
        self.m_sig_x = None
        self.m_sig_y = None
        # idler with lower y values
        self.m_idl_x = None
        self.m_idl_y = None
        self.time_tagger = None
        self._get_hardware()


    def _get_hardware(self):
        self.zaber_ms = ZaberMotors()
        self.m_sig_x = self.zaber_ms.motors[1]
        self.m_sig_y = self.zaber_ms.motors[0]
        print("Got Zaber motors!")

        self.m_idl_x = ThorlabsKcubeDC(27501989)
        self.m_idl_y = ThorlabsKcubeStepper(26003414)
        print("Got Thorlabs motors!")

        self.time_tagger = QPTimeTagger(integration_time=self.res.integration_time, remote=True)
        print("Got TimeTagger!")

    def scan(self):
        self.res.single1s = np.zeros((len(self.res.locs_signal), len(self.res.locs_idler)))
        self.res.single2s = np.zeros_like(self.res.single1s)
        self.res.coincidences = np.zeros_like(self.res.single1s)


        for i, loc_sig in enumerate(self.res.locs_signal):
            self.m_sig_x.move_absolute(loc_sig[0])
            self.m_sig_y.move_absolute(loc_sig[1])

            for j, loc_idl in enumerate(self.res.locs_idler):
                self.m_idl_x.move_absolute(loc_idl[0])
                self.m_idl_y.move_absolute(loc_idl[1])
                s1, s2, c12 = self.time_tagger.read_interesting()
                self.res.single1s[i, j] = s1
                self.res.single2s[i, j] = s2
                self.res.coincidences[i, j] = c12
                print(rf'{i}, {j}: {s1:.2f}, {s2:.2f}, {c12:.2f}')

            self.res.saveto(self.res.path)


    def close(self):
        self.zaber_ms.close()
        self.time_tagger.close()
        self.m_sig_x.close()
        self.m_sig_y.close()


locs_x_idler = [9.1, 9.082, 9.042, 9.0173, 8.98]
locs_y_idler = [2.8, 2.42, 2.07, 1.68, 1.3]
locs_idler = list(zip(locs_x_idler, locs_y_idler))

locs_x_signal = [11.559, 11.59, 11.6256, 11.652, 11.68]
locs_y_signal = [8.784, 9.1338, 9.524, 9.884, 10.24]
locs_signal = list(zip(locs_x_signal, locs_y_signal))

dps = DiscretePhotonScanner(locs_signal, locs_idler, integration_time=1, remote_tagger=True, run_name='QKD_row3')
dps.scan()
dps.close()
