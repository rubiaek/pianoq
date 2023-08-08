import time
import tqdm
from scipy.optimize import curve_fit
from pianoq.lab.slm import SLMDevice
from pianoq.lab.asi_cam import ASICam
import matplotlib.pyplot as plt
import numpy as np


def _cos(x, a, w, phi, c):
    return a * np.cos(w * x + phi) + c


def fit_cos(phis, costs):
    min_bounds = [(costs.max() - costs.min()) / 3, 1.8, 0, costs.mean() / 3]
    max_bounds = [(costs.max() - costs.min()) * 3, 2.2, 2 * np.pi, costs.mean() * 3]

    popt, pcov = curve_fit(_cos, phis, costs, bounds=(min_bounds, max_bounds))
    return popt


def main():
    macro_pixels = 20
    sleep_period = 0.001

    asi_exposure_time = 3e-3
    roi = (2800, 1950, 400, 400)
    l = 3
    cost_roi = np.index_exp[200 - l:200 + l, 200 - l:200 + l]

    slm = SLMDevice(0, use_mirror=True)
    cam = ASICam(asi_exposure_time, binning=1, roi=roi, gain=0)

    fig, ax = plt.subplots()
    phis = np.linspace(0, 2 * np.pi, 15)
    dummy_y = _cos(phis, 1000, 2, 0, 5000)
    line, = ax.plot(phis, dummy_y, '*')

    dummy_popt = fit_cos(phis, dummy_y)
    dummy_x = np.linspace(0, 2 * np.pi, 100)
    fit_line, = ax.plot(dummy_x, _cos(dummy_x, *dummy_popt), '--', alpha=0.5)

    mask_to_play = np.random.randint(2, size=(macro_pixels, macro_pixels))
    while True:
        costs = []
        for phi in tqdm.tqdm(phis):
            slm.update_phase_in_active(mask_to_play*phi)
            time.sleep(sleep_period)
            cost = cam.get_image()[cost_roi].sum()
            costs.append(cost)

        costs = np.array(costs)
        # line.set_xdata(phis)
        line.set_ydata(costs)
        popt = fit_cos(phis, costs)

        # fit_line.set_xdata(dummy_x)
        fit_line.set_ydata(_cos(dummy_x, *popt))

        # ax.set_xlim(left=None, right=None)
        ax.set_ylim(bottom=0.9*costs.min(), top=1.1*costs.max())

        fig.canvas.draw()
        fig.canvas.flush_events()

        fig.show()
        ans = input('change mask? (y or n))')
        if ans == 'y':
            print('changing mask')
            mask_to_play = np.random.randint(2, size=(macro_pixels, macro_pixels))


if __name__ == "__main__":
    main()
