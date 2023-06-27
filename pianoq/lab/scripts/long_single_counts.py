from pianoq.lab.time_tagger import QPTimeTagger
import matplotlib.pyplot as plt
import datetime
import time
import csv

LOGS_DIR = r'G:\My Drive\Projects\Quantum Piano\Results\temp'


def main():
    tt = QPTimeTagger(integration_time=1, coin_window=1e-9, single_channels=[1, 2, 3, 4], coin_channels=[])
    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    path = f"{LOGS_DIR}\\{now}_4_channels.csv"

    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)

        while True:
            t = time.time()
            data = tt.read_interesting()
            line = [t] + data
            writer.writerow(line)
            f.flush()


def show_intensities(path):
    from numpy import genfromtxt
    data = genfromtxt(path, delimiter=',')
    fig, ax = plt.subplots()

    times = data[:, 0]
    times = times - times[0]
    times = times / 3600

    for i in range(4):
        ax.plot(times, data[:, i+1], label=f'channel {i+1}')

    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('counts')
    ax.legend()
    fig.show()


if __name__ == "__main__":
    main()
