import pickle
import datetime


class QPPickleResult(object):
    def __init__(self, path=None, **kwargs):
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
