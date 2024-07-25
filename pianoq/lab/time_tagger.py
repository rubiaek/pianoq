import time
try:
    import TimeTagger
except ImportError:
    print('can\'t import TimeTagger')
import numpy as np


class QPTimeTagger(object):
    def __init__(self, integration_time=1, coin_window=1e-9,
                 single_channels=(1, 2), single_channel_delays=None, coin_channels=((1, 2),),
                 remote=False, address='132.64.81.93:41101'):
        """
        :param coin_window: in seconds
        :param integration_time: in seconds
        single_channel_delays: if have different cable lengths etc.
        """

        self.integration_time = integration_time
        self.coin_window = coin_window
        self.single_channels = single_channels
        self.coin_channels = coin_channels
        self.coin_virtual_channels = []

        if single_channel_delays is None:
            single_channel_delays = np.zeros_like(single_channels)
        else:
            assert len(single_channels) == len(single_channel_delays)

        self.single_channel_delays = single_channel_delays
        if remote:
            self.tagger = TimeTagger.createTimeTaggerNetwork(address)
        else:
            self.tagger = TimeTagger.createTimeTagger()

        for i, chan in enumerate(single_channels):
            self.tagger.setInputDelay(chan, single_channel_delays[i])

        for pair in coin_channels:
            # see here https://www.swabianinstruments.com/static/documentation/TimeTagger/api/VirtualChannels.html#coincidence
            self.coin_virtual_channels.append(TimeTagger.Coincidence(self.tagger, pair, self.coin_window * 1e12))

        coin_channel_nos = [ch.getChannel() for ch in self.coin_virtual_channels]
        all_channels = list(single_channels) + coin_channel_nos
        self.counter = TimeTagger.Counter(tagger=self.tagger, channels=all_channels,
                                          binwidth=self.integration_time*1e12, n_values=1)

    def set_integration_time(self, integration_time):
        self.integration_time = integration_time
        # TODO: close previous counter somehow, or maybe we can somehow simply change the binwidth?
        self.counter = TimeTagger.Counter(tagger=self.tagger, channels=[1, 2, self.coin_virtual_channel.getChannel()],
                                          binwidth=self.integration_time * 1e12, n_values=1)

    def read(self):
        while True:
            # This is so it will have the same API as the regular photon counter
            self.counter.clear()
            time.sleep(0.15)  # Need to sleep a bit more than him so the data will get here
            time.sleep(self.integration_time)
            data = self.counter.getDataNormalized()
            data = [i[0] for i in data]
            single1, single2, coin = data
            if np.isnan(data).any():
                print('nan issue in timetagger')
                continue
            break

        return [single1, single2, None, None, coin, None, None, None]

    def read_interesting(self):
        while True:
            self.counter.clear()
            time.sleep(0.1)  # Need to sleep a bit more than him so the data will get here
            time.sleep(self.integration_time)
            data = self.counter.getDataNormalized()
            data = [i[0] for i in data]
            if np.isnan(data).any():
                print('nan issue in timetagger')
                continue
            break
        return data

    def read_double_spot(self):
        # This here assumes that whoever created the timetagger instance created it with the correct channels
        # (e.g. 3 single counts, and two coincidence counters), so what we expect to return here is:
        # s1, s2, s3, c1, c2
        return self.read_interesting()

    def close(self):
        TimeTagger.freeTimeTagger(self.tagger)
        time.sleep(0.2)
