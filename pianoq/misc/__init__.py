import traceback
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pianoq.misc.borders import Borders
from pianoq.misc.consts import DEFAULT_BORDERS
from pianoq.misc.mplt import mplot, mimshow
from pianoq.misc.calc_contrast import calc_contrast

try:
    from pianoq.lab.scripts.live_camera import live_cam
except Exception as e:
    print("ERROR!!")
    print(e)
    traceback.print_exc()


def color_generator():
    i = 1
    while True:
        # https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
        # n = 20
        # cmap = plt.cm.get_cmap(plt.cm.viridis, 140)
        # i += 1
        # yield cmap(i*n % 140)

        for name, color in mcolors.TABLEAU_COLORS.items():
            yield color

        # for item in ['r', 'g', 'b', 'c', 'm', 'y', 'k']:
        #     yield item


color_gen = color_generator()
