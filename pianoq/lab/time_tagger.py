import time
import TimeTagger


class QPTimeTagger(object):
    def __init__(self, integration_time=1, coin_window=1e-9):
        """
        :param coin_window: in seconds
        :param integration_time: in seconds
        """
        self.integration_time = integration_time
        self.coin_window = coin_window
        self.tagger = TimeTagger.createTimeTagger()
        self.tagger.setInputDelay(1, 0)  # Set if have different cable lengths etc.
        self.tagger.setInputDelay(2, 0)
        # see here https://www.swabianinstruments.com/static/documentation/TimeTagger/api/VirtualChannels.html#coincidence
        self.coin_virtual_channel = TimeTagger.Coincidence(self.tagger, [1, 2], self.coin_window * 1e12)

        self.counter = TimeTagger.Counter(tagger=self.tagger, channels=[1, 2, self.coin_virtual_channel.getChannel()],
                                          binwidth=self.integration_time*1e12, n_values=1)

    def set_integration_time(self, integration_time):
        self.integration_time = integration_time
        # TODO: close previous counter somehow, or maybe we can somehow simply change the binwidth?
        self.counter = TimeTagger.Counter(tagger=self.tagger, channels=[1, 2, self.coin_virtual_channel.getChannel()],
                                          binwidth=self.integration_time * 1e12, n_values=1)

    def read_interesting(self):
        self.counter.clear()
        time.sleep(0.1)  # Need to sleep a bit more than him so the data will get here
        time.sleep(self.integration_time)
        data = self.counter.getDataNormalized()
        single1, single2, coin = data
        return single1[0], single2[0], coin[0]

    def close(self):
        TimeTagger.freeTimeTagger(self.tagger)
