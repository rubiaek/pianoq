

class PSOOptimizationResult(object):
    def __init__(self):
        # TODO: add active piezos, active piezo amounts, piezo range, etc...
        # TODO: think how to do it generically also for non-piano related optimization
        # TODO: (Also the images isn't very general...)
        self.costs = []
        self.amplitudes = []
        self.images = None
        self.best_cost = None
        self.best_amps = None
        self.best_image = None


    def saveto(self, path):
        try:
            np.savez(path,
                     costs=self.costs,
                     amplitudes=self.amplitudes,
                     images=self.images,
                     best_cost=self.best_cost,
                     best_amps=self.best_amps,
                     best_image=self.best_image
                     )
        except Exception as e:
            print("ERROR!!")
            print(e)
            traceback.print_exc()

    def loadfrom(self, path):
        # path = path or self.DEFAULT_PATH
        f = open(path, 'rb')
        data = np.load(f, allow_pickle=True)
        self.costs = data['costs']
        self.amplitudes = data['amplitudes']
        self.images = data['images']
        self.best_cost = data['best_cost']
        self.best_amps = data['best_amps']
        self.best_image = data['best_image']

