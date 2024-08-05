import numpy as np
import matplotlib.pyplot as plt
import traceback
from pianoq.lab.mplc.consts import N_SPOTS


class PhaseFinderResult(object):
    def __init__(self, path=None):
        self.path = path
        if path:
            self.loadfrom()
        self.timestamp = None
        self.coincidences = None
        self.single1s = None
        self.single2s = None
        self.integration_time = -1
        self.coin_window = -1

        self.phases = np.zeros(N_SPOTS*2)
        self.modes_to_keep = np.array([])
        self.N_phases = 0
        self.phase_vec_step = 0
        self.phase_vec = np.array([])

    def saveto(self, path):
        try:
            f = open(path, 'wb')
            np.savez(f,
                     coincidences=self.coincidences,
                     single1s=self.single1s,
                     single2s=self.single2s,
                     integration_time=self.integration_time,
                     coin_window=self.coin_window,
                     timestamp=self.timestamp,
                     phases=self.phases,
                     modes_to_keep=self.modes_to_keep,
                     N_phases=self.N_phases,
                     phase_vec_step=self.phase_vec_step,
                     phase_vec=self.phase_vec)
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
        self.timestamp = data['timestamp']

        self.phases = data['phases']
        self.modes_to_keep = data['modes_to_keep']
        self.N_phases = data['N_phases']
        self.phase_vec_step = data['phase_vec_step']
        self.phase_vec = data['phase_vec']
        f.close()

    def reload(self):
        self.loadfrom(self.path)

