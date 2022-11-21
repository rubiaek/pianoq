import time
import TimeTagger


class QPTimeTagger(object):
    # TODO: try this out with a real box and see that things do what I think they do...
    def __init__(self, integration_time=1, coin_window=1000):
        """
        :param coin_window: in ps
        :param integration_time: in seconds
        """
        self.integration_time = integration_time
        self.coin_window = coin_window
        self.tagger = TimeTagger.createTimeTagger()
        self.tagger.setInputDelay(1, 0)  # Set if have different cable lengths etc.
        self.tagger.setInputDelay(2, 0)
        # see here https://www.swabianinstruments.com/static/documentation/TimeTagger/api/VirtualChannels.html#coincidence
        self.coin_virtual_channel = TimeTagger.Coincidence(self.tagger, [1, 2], self.coin_window)


        self.counter = TimeTagger.Counter(tagger=self.tagger, channels=[1, 2, self.coin_virtual_channel.getChannel()],
                                          binwidth=self.integration_time*1e12, n_values=1)

    def read(self):
        self.counter.clear()
        time.sleep(self.integration_time)
        data = self.counter.getDataNormalized()
        single1, single2, coin = data
        return single1, single2, coin


    def close(self):
        TimeTagger.freeTimeTagger(self.tagger)
