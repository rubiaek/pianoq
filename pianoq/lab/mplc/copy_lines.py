from pianoq.lab.mplc.consts import thorlabs_x_serial, thorlabs_y_serial
from pianoq.lab.thorlabs_motor import ThorlabsKcubeDC, ThorlabsKcubeStepper
from pianoq.lab.zaber_motor import ZaberMotors
from pianoq.lab.time_tagger import QPTimeTagger

backlash = 0.0
wait_after_move = 0.0

mxi = ThorlabsKcubeDC(thorlabs_x_serial, backlash=backlash, wait_after_move=wait_after_move)
myi = ThorlabsKcubeStepper(thorlabs_y_serial, backlash=backlash, wait_after_move=wait_after_move)

zaber_ms = ZaberMotors(backlash=backlash, wait_after_move=wait_after_move)
mxs = zaber_ms.motors[1]
mys = zaber_ms.motors[0]


integration_time = 1
tt = QPTimeTagger(integration_time=integration_time, remote=True)


def move_i(x, y):
    mxi.move_absolute(x)
    myi.move_absolute(y)

def move_s(x, y):
    mxs.move_absolute(x)
    mys.move_absolute(y)


mxi.close()
myi.close()
mxs.close()
mys.close()
