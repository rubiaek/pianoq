import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import typing
import pyperclip
import traceback
import glob
from pianoq.misc.mplt import my_mesh

LOGS_DIR = "C:\\temp"


class ScanResult(object):
    coin_window = 4e-9

    def __init__(self, coincidences=None, single1s=None, single2s=None, X=None, Y=None, integration_time=None):
        self.coincidences = coincidences
        self.single1s = single1s
        self.single2s = single2s
        self.accidentals = None
        self.X = X
        self.Y = Y
        self.integration_time = integration_time

    def show(self, show_singles=False, title='', remove_accidentals=True) -> typing.Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        if remove_accidentals:
            coin = self.coincidences - self.accidentals
        else:
            coin = self.coincidences

        fig, ax = plt.subplots()
        ax.set_title(f'Coincidences {title}')
        my_mesh(self.X, self.Y, coin, ax)
        ax.invert_xaxis()
        fig.show()

        if show_singles:
            fig, axes = plt.subplots(1, 2)
            my_mesh(self.X, self.Y, self.single1s, axes[0])
            my_mesh(self.X, self.Y, self.single2s, axes[1])
            axes[0].invert_xaxis()
            axes[1].invert_xaxis()
            axes[0].set_title(f'Single counts 1 {title}')
            axes[1].set_title(f'Single counts 2 {title}')
            fig.show()

        return fig, ax

    def saveto(self, path):
        try:
            f = open(path, 'wb')
            np.savez(f,
                     coincidences=self.coincidences,
                     single1s=self.single1s,
                     single2s=self.single2s,
                     X=self.X,
                     Y=self.Y,
                     integration_time=self.integration_time)
            f.close()
        except Exception as e:
            print("ERROR!!")
            print(e)
            traceback.print_exc()

    def loadfrom(self, path=None, BAK=True):
        if path is None:
            BAK = '' if not BAK else '.BAK'
            paths = glob.glob(f"{LOGS_DIR}\\*.scan")
            for i, path in enumerate(paths):
                print(f'{i}: {path}')
            choice = int(input('which one?'))
            path = paths[choice]
        elif path == 0:
            path = pyperclip.paste()

        path = path.strip('"')
        path = path.strip("'")

        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        self.coincidences = data['coincidences']
        self.single1s = data['single1s']
        self.single2s = data['single2s']
        self.X = data['X']
        self.Y = data['Y']
        self.integration_time = data.get('integration_time', None)

        self.accidentals = self.single1s * self.single2s * 2 * self.coin_window

        f.close()
