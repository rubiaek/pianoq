import traceback
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
    while True:
        for item in ['r', 'g', 'b', 'c', 'm', 'y', 'k']:
            yield item


color_gen = color_generator()
