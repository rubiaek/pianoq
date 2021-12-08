import traceback
from pianoq.misc.borders import Borders
from pianoq.misc.consts import DEFAULT_BORDERS
from pianoq.misc.mplt import mplot, mimshow

try:
    from pianoq.lab.scripts.live_camera import live_cam
except Exception as e:
    print("ERROR!!")
    print(e)
    traceback.print_exc()
