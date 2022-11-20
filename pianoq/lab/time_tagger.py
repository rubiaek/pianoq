import TimeTagger


class QPTimeTagger(object):
    def __init__(self, coin_window=1e-9):
        self.tagger = TimeTagger.createTimeTagger()
        self.tagger.setInputDelay(1, 0)
        self.tagger.setInputDelay(2, 0)
        self.countrate1 = TimeTagger.Countrate(tagger=self.tagger, channels=[1])
        self.countrate2 = TimeTagger.Countrate(tagger=self.tagger, channels=[2])
        self.coin_measure = correlation = TimeTagger.Correlation(tagger=self.tagger,
                                                                 channel_1=1,
                                                                 channel_2=2,
                                                                 binwidth=1,
                                                                 n_bins=2*10**6)

    def read(self):
        pass

    def close(self):
        TimeTagger.freeTimeTagger(self.tagger)
