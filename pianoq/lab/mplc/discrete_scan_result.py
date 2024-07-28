import numpy as np
import matplotlib.pyplot as plt
import traceback


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
        self.wait_after_move = None
        self.backlash = None

    @property
    def real_coins(self):
        return self.coincidences - self.accidentals

    @property
    def accidentals(self):
        return 2*self.single1s*self.single2s*self.coin_window

    def show(self, remove_accidentals=False):
        fig, ax = plt.subplots()
        if remove_accidentals:
            imm = ax.imshow(self.real_coins)
        else:
            imm = ax.imshow(self.coincidences)
        fig.colorbar(imm, ax=ax)
        fig.show()

    def show_singles(self):
        fig, axes = plt.subplots(1, 2)
        imm = axes[0].imshow(self.single1s)
        fig.colorbar(imm, ax=axes[0])
        imm = axes[1].imshow(self.single2s)
        fig.colorbar(imm, ax=axes[1])
        fig.show()

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
                     timestamp=self.timestamp,
                     wait_after_move=self.wait_after_move,
                     backlash=self.backlash)
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
        self.wait_after_move = data['wait_after_move']
        self.backlash = data['backlash']
        f.close()

    def reload(self):
        self.loadfrom(self.path)

