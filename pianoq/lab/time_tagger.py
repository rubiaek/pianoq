try:
    import TimeTagger
except ImportError:
    print('can\'t import TimeTagger')

import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt


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
        self.counter = TimeTagger.Counter(tagger=self.tagger, channels=[1, 2, self.coin_virtual_channels[0].getChannel()],
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
            time.sleep(0.15)  # Need to sleep a bit more than him so the data will get here
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

    def show_gui(self, history_sec=60, refresh_hz=15):
        """
        Minimal 4-panel GUI with just the plots and current values
        """
        import matplotlib.pyplot as plt
        from collections import deque
        from threading import Thread, Event
        import time

        import matplotlib
        fig_name = '4chan'
        if fig_name in plt.get_figlabels():
            print('It is already running!')
            return
        matplotlib.rcParams['keymap.fullscreen'] = []

        # Number formatting function
        def format_number(x):
            if abs(x) >= 1e6: return f'{x/1e6:.1f}M'
            elif abs(x) >= 1e3: return f'{x/1e3:.1f}K'
            else: return f'{x:.1f}'

        # Data buffers - initialize with current readings
        pts = max(10, int(history_sec / max(self.integration_time, 1e-3)))
        
        # Get initial reading
        try:
            s1_init, s2_init, coin_init = self.read_interesting()
            real_init = coin_init - (2* s1_init * s2_init * self.coin_window)
        except:
            s1_init, s2_init, coin_init, real_init = 0.0, 0.0, 0.0, 0.0
        
        b1 = deque([s1_init]*pts, maxlen=pts)
        b2 = deque([s2_init]*pts, maxlen=pts)
        bc = deque([coin_init]*pts, maxlen=pts)
        br = deque([real_init]*pts, maxlen=pts)

        # Setup figure
        plt.figure(fig_name)      # create or activate named figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), num=fig_name)
        fig.patch.set_facecolor('black')
        axs = axs.ravel()
        
        titles = ['S1:', 'S2:', 'Coincidences:', 'Real coin:']
        lines = []
        value_texts = []
        
        for i, (ax, title) in enumerate(zip(axs, titles)):
            ax.set_facecolor('black')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.tick_params(colors='white', labelsize=10)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5, color='gray')
            
            # Title and value above the plot - show initial values
            initial_vals = [s1_init, s2_init, coin_init, real_init]
            value_txt = ax.text(0.5, 1.08, f'{title} {format_number(initial_vals[i])}', transform=ax.transAxes,
                            color='white', fontsize=24, weight='bold',
                            ha='center', va='bottom')
            value_texts.append(value_txt)
            
            # Yellow plot
            ln, = ax.plot([], [], color='yellow', linewidth=2)
            lines.append(ln)

        # Zero line for real coincidence
        zero_ln, = axs[3].plot([], [], '--', color='white', linewidth=0.5, alpha=0.7)

        # Data reader thread
        stop_event = Event()
        
        def reader():
            while not stop_event.is_set():
                try:
                    s1, s2, coin = self.read_interesting()
                    real_coin = coin - (2 * s1 * s2 * self.coin_window)
                    b1.append(s1)
                    b2.append(s2)
                    bc.append(coin)
                    br.append(real_coin)
                except Exception as e:
                    print(f"Read error: {e}")
                time.sleep(max(0.05, self.integration_time))
        
        reader_thread = Thread(target=reader, daemon=True)
        reader_thread.start()

        # Display update
        def update_display():
            if not b1: return
            x_data = range(len(b1))
            
            for line, buf, txt, title in zip(lines, [b1, b2, bc, br], value_texts, titles):
                y_data = list(buf)
                line.set_data(x_data, y_data)
                if y_data:
                    txt.set_text(f'{title} {format_number(y_data[-1])}')
                    y_min, y_max = min(y_data), max(y_data)
                    if y_min == y_max: y_min, y_max = y_min - 1, y_max + 1
                    margin = (y_max - y_min) * 0.1
                    line.axes.set_xlim(0, len(y_data) - 1)
                    line.axes.set_ylim(y_min - margin, y_max + margin)
            
            zero_ln.set_data(x_data, [0] * len(br))
            fig.canvas.draw_idle()

        timer = fig.canvas.new_timer(interval=int(1000/refresh_hz))
        timer.add_callback(update_display)
        timer.start()

        def on_close(_):
            stop_event.set()
            timer.stop()
            try:
                reader_thread.join(timeout=1.1)
            except Exception:
                pass
        fig.canvas.mpl_connect('close_event', on_close)

        # Fullscreen on 'f' key
        def on_key(e):
            if e.key == 'f':
                mng = fig.canvas.manager
                if hasattr(mng, 'full_screen_toggle'):
                    mng.full_screen_toggle()
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.tight_layout()
        plt.show()