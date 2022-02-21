import pickle
import datetime
import matplotlib.pyplot as plt


class WavePlateOptimizationResult(object):
    def __init__(self, path=None):
        self.H_angles = None
        self.Q_angles = None
        self.heatmap = None

        self.path = path
        self.timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        if path:
            self.loadfrom(path)

    def saveto(self, path=None):
        f = open(path or self.path, 'wb')
        pickle.dump(self, f)
        f.close()

    def loadfrom(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            self.__dict__ = obj.__dict__
            self.__class__ = obj.__class__

    def show_heatmap(self):
        # different rows are different HWP angles
        # zero is on the top left
        fig, ax = plt.subplots()
        ax.set_title("Energy in H polarization")
        ax.imshow(self.heatmap, extent=[0, 360, 360, 0])
        ax.set_xlabel(r'QWP angle')
        ax.set_ylabel(r'HWP angle')
        fig.show()
